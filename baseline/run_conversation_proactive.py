import asyncio
from contextlib import asynccontextmanager, AsyncExitStack

import json
import logging
import os
import pathlib
import traceback
from typing import List, Optional, Tuple

import dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tqdm import tqdm
import argparse
import uuid
import time

from utils.clogger import _set_logger
from utils.llm_api import ChatModel
from dataclasses import dataclass

_set_logger(
    exp_dir=pathlib.Path("./logs"),
    logging_level_stdout=logging.INFO,
    logging_level=logging.DEBUG,
    file_name="baseline.log",
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

INPUT_QUERIES_FILE = "./baseline/data/example_queries.json"
CONVERSATION_RESULTS_FILE = f"./baseline/output/{os.getenv('MODEL', 'None').replace('/', '_')}_{os.getenv('EMBEDDING_MODEL', 'None').replace('/', '_')}.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default=INPUT_QUERIES_FILE,
        help="Path to the input queries file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=CONVERSATION_RESULTS_FILE,
        help="Path to the output conversation results file.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of concurrent threads to use.",
    )
    return parser.parse_args()


@dataclass
class SessionCoro:
    session: ClientSession
    exit_stack: AsyncExitStack


class SessionPoolManager:
    def __init__(self, create_session_coro, size: int):
        self._create = create_session_coro  # async () -> ClientSession
        self._q = asyncio.Queue(maxsize=size)
        self._size = size
        self._init_lock = asyncio.Lock()
        self._initialized = False

    def is_initialized(self):
        return self._initialized

    async def init(self):
        async with self._init_lock:
            if self._initialized:
                return

            sessions = await asyncio.gather(
                *[self._create() for _ in range(self._size)],
                return_exceptions=False,
            )
            for s in sessions:
                await self._q.put(s)

            self._initialized = True

    @asynccontextmanager
    async def acquire(self):
        ps = await self._q.get()
        try:
            yield ps.session
        except Exception:
            await self._close_session(ps)
            ps = await self._create()
            raise
        finally:
            await self._q.put(ps)

    async def _close_session(self, ps):
        # 视 mcp ClientSession 的关闭方式实现
        try:
            await ps.exit_stack.aclose()
        except Exception:
            pass

    async def aclose(self):
        while not self._q.empty():
            s = await self._q.get()
            await self._close_session(s)


class LoggingMCPClient:

    def __init__(self, num_threads=16):
        self.chat_model = ChatModel(
            model_name=os.getenv("MODEL", "Qwen/Qwen3-8B-sft"),
            # api_key=os.getenv("OPENAI_API_KEY"),
            model_url=os.getenv("BASE_URL", ""),
        )
        self.session_pool = SessionPoolManager(
            self.create_mcp_copilot_session, size=num_threads
        )
        self.timeout = 180

    def build_tools(self, response):
        available_tools = []
        for tool in response.tools:
            if tool.name == "route":
                available_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "tool_request",
                            "description": "Submit a request describing what tool or functionality is needed.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "description": {
                                        "type": "string",
                                        "description": "A detailed description of the tool or function the user needs.",
                                    }
                                },
                                "required": ["description"],
                            },
                        },
                    }
                )
            else:
                available_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return available_tools

    async def init(self):
        await self.session_pool.init()
        logger.info("MCP Copilot session pool initialized.")

        # init available tools
        async with self.session_pool.acquire() as session:
            self.available_tools = self.build_tools(await session.list_tools())

    async def cleanup(self):
        await self.session_pool.aclose()

    async def create_mcp_copilot_session(self) -> ClientSession:
        config = {
            "mcpServers": {
                "mcp-copilot": {
                    "command": "python",
                    "args": ["-m", "baseline.mcp_copilot"],
                },
            }
        }
        config = config["mcpServers"]
        command = config["mcp-copilot"]["command"]
        args = config["mcp-copilot"].get("args", [])
        env = None
        PROXY_ENV_LIST = [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "no_proxy",
        ]
        for proxy_env in PROXY_ENV_LIST:
            if proxy_env in os.environ:
                env = env or {}
                env[proxy_env] = os.environ[proxy_env]

        exit_stack = AsyncExitStack()
        try:
            server_params = StdioServerParameters(command=command, args=args, env=env)
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            session = await exit_stack.enter_async_context(ClientSession(stdio, write))
            await asyncio.wait_for(session.initialize(), timeout=self.timeout)
            return SessionCoro(session=session, exit_stack=exit_stack)
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to server")
            raise
        except Exception as e:
            logger.error(f"Error connecting to server: {e}")
            await exit_stack.aclose()
            raise

    async def process_query(
        self,
        query: str,
        history: Optional[list] = None,
        max_tool_tokens: int = 10000,
    ) -> Tuple[str, List[dict]]:
        if history is None:
            messages = [
                {
                    "role": "system",
                    "content": """\
You are an agent designed to assist users with daily tasks by using external tools. You have access to two tools: a retrieval tool and an execution tool. The retrieval tool allows you to search a large toolset for relevant tools, and the execution tool lets you invoke the tools you retrieved. Whenever possible, you should use these tools to get accurate, up-to-date information and to perform file operations.

Note that you can only response to user once, so you should try to provide a complete answer in your response.
""",
                }
            ]
        else:
            messages = history.copy()

        messages.append({"role": "user", "content": query})

        # get session from pool
        assert self.session_pool.is_initialized(), "Session pool is not initialized."
        available_tools = []
        async with self.session_pool.acquire() as session:
            # get available tools from the session
            available_tools = self.available_tools
            final_text = []
            stop_flag = False
            try:
                while not stop_flag:
                    request_payload = {
                        "messages": messages,
                        "tools": available_tools,
                    }
                    t0 = time.time()
                    response = await asyncio.to_thread(
                        self.chat_model.complete_with_retry, **request_payload
                    )
                    logger.info(f"[TIME] LLM took {time.time()-t0:.2f}s")

                    if hasattr(response, "error"):
                        raise Exception(
                            f"Error in OpenAI response: {response.error['metadata']['raw']}"
                        )

                    response_message = response.choices[0].message
                    if response_message.tool_calls:
                        tool_call_list = []
                        for tool_call in response_message.tool_calls:
                            if not tool_call.id:
                                tool_call.id = str(uuid.uuid4())
                            tool_call_list.append(tool_call)
                        response_message.tool_calls = tool_call_list
                    messages.append(response_message.model_dump(exclude_none=True))
                    content = response_message.content
                    if (
                        content
                        and not response_message.tool_calls
                        and not response_message.function_call
                    ):
                        final_text.append(content)
                        stop_flag = True
                    else:
                        tool_calls = response_message.tool_calls
                        if not tool_calls:
                            logger.warning(
                                "Received empty response from LLM without content or tool calls."
                            )
                            break

                        t1 = time.time()
                        for tool_call in tool_calls:
                            try:
                                tool_name = tool_call.function.name
                                tool_args = json.loads(tool_call.function.arguments)
                                tool_id = tool_call.id

                                logger.info(
                                    f"LLM is calling tool: {tool_name}({tool_args}), \n with session {session}"
                                )
                                if tool_name == "tool_request":
                                    description = tool_args["description"]
                                    result = await asyncio.wait_for(
                                        session.call_tool(
                                            "route", {"query": description}
                                        ),
                                        timeout=300,
                                    )
                                else:
                                    result = await asyncio.wait_for(
                                        session.call_tool(tool_name, tool_args),
                                        timeout=300,
                                    )
                            except asyncio.TimeoutError:
                                logger.error(f"Tool call {tool_name} timed out.")
                                result = "Tool call timed out."
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_id,
                                        "content": str(result),
                                    }
                                )
                                raise

                            except Exception as e:
                                logger.error(f"Error calling tool {tool_name}: {e}")
                                result = f"Error: {str(e)}"
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_id,
                                        "content": str(result),
                                    }
                                )
                                raise
                            result = str(result)
                            result = result[:max_tool_tokens]
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "content": str(result),
                                }
                            )
                        logger.info(
                            f"[TIME] tool {tool_name} took {time.time()-t1:.2f}s"
                        )
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                final_text.append(f"Error: {str(e)}")
                messages.append({"role": "assistant", "content": str(e)})
        return "\n".join(final_text), messages


async def main(args):
    if not pathlib.Path(args.input_path).exists():
        logger.error(f"Input queries file {args.input_path} does not exist.")
        return
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"len(queries): {len(data)}")
    client = LoggingMCPClient(num_threads=args.threads)
    await client.init()

    logger.info(
        "pathlib.Path(args.output_path).exists(): {}".format(
            pathlib.Path(args.output_path).exists()
        )
    )
    if os.path.exists(args.output_path):
        print("output exists", args.output_path)
        with open(args.output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        exist_ids = {entry["task_id"] for entry in all_results}
    else:
        all_results = []
        exist_ids = set()
    error_queries = set()

    sem = asyncio.Semaphore(args.threads)

    async def work(entry, sem):
        async with sem:
            task_id = entry["task_id"]
            if task_id in exist_ids:
                return None
            query = entry["Question"]
            try:
                logger.info("Before process_query")
                response, messages = await client.process_query(query, None)
                logger.info(f"{response}")
                entry["response"] = response
                entry["messages"] = messages
                # all_results.append(entry)
                return entry
            except Exception:
                error_queries.add(query)
                logger.error(traceback.format_exc())
                return None

    try:
        tasks = [asyncio.create_task(work(entry, sem)) for entry in data]

        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await fut
            if result:
                all_results.append(result)
    finally:
        await client.cleanup()
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
