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
from collections import defaultdict

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
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_url=os.getenv("BASE_URL", ""),
        )
        self.summary_model = ChatModel(
            model_name=os.getenv("SUMMARY_MODEL", "Qwen/Qwen3-8B-sft"),
            api_key=os.getenv("SUMMARY_OPENAI_API_KEY"),
            model_url=os.getenv("SUMMARY_BASE_URL", ""),
        )
        self.session_pool = SessionPoolManager(
            self.create_mcp_copilot_session, size=num_threads
        )
        self.timeout = 180

    def build_tools(self, response):
        available_tools = []
        for tool in response.tools:
            # if tool.name == "route":
            #     available_tools.append(
            #         {
            #             "type": "function",
            #             "function": {
            #                 "name": "tool_request",
            #                 "description": (
            #                     "Request ONE specific tool function for ONE atomic action. "
            #                     "The request must correspond to a single concrete tool capability. "
            #                 ),
            #                 "parameters": {
            #                     "type": "object",
            #                     "properties": {
            #                         "description": {
            #                             "type": "string",
            #                             "description": (
            #                                 "Describe EXACTLY ONE tool capability to search for. "
            #                                 "This should map to a single function.\n\n"
            #                                 "Valid examples:\n"
            #                                 "- 'Search nearby restaurants within a radius'\n"
            #                                 "- 'Get current weather for a city'\n\n"
            #                                 "Invalid examples:\n"
            #                                 "- 'Search places and summarize results'\n"
            #                                 "- 'Plan a trip and recommend restaurants'"
            #                             ),
            #                         }
            #                     },
            #                     "required": ["description"],
            #                 },
            #             },
            #         }
            #     )

            # else:
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

    async def _summarize_messages(self, messages: List[dict]) -> str:
        """
        Summarize the conversation messages into a concise workflow summary
        for internal agent use.
        """

        prompt = f"""
You are an assistant summarizing an ongoing agent workflow.

The conversation messages include:
- User questions
- Assistant intermediate reasoning
- Tool calls and tool results

Your task is to produce a concise working summary of the process so far.

Focus on:
- The user's original goal
- What has already been completed
- Key conclusions from tool results (not raw data)
- The current state of the task
- What is still missing or needs to be done next

Rules:
- Be concise and factual
- Do NOT include chain-of-thought or detailed reasoning
- Do NOT quote messages verbatim
- Do NOT add new assumptions

Output the summary using the following format:

User Goal:
What Has Been Done:
Tool Results:
Current State:
Pending / Next Step:

Conversation:
{json.dumps(messages, ensure_ascii=False, indent=2)}
""".strip()

        request_payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise workflow summarization assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": 2400,
            "temperature": 0.6,
        }

        summary = await asyncio.to_thread(
            self.summary_model.complete_with_retry, **request_payload
        )
        return summary.choices[0].message.content.strip()

    async def _classify_result(self, question: str, response: str, summary: str) -> str:
        """Use LLM to classify the response into predefined categories."""

        prompt = """You are a result classifier. Your job is ONLY to classify, not to explain.

You are given:
- The user's original question
- The assistant's latest response
- A workflow summary describing what has happened so far

Based on ALL of this information, determine which ONE
of the following categories the current response belongs to:

1. SUCCESS - The user's problem is fully and correctly solved
2. PARTIAL - The task is only partially completed and more work is required
3. NEED_CLARIFICATION - The task cannot continue without more input from the user

Rules:
- Output EXACTLY one label from the list above
- Do NOT output explanations or additional text
- If uncertain, choose PARTIAL

User Question:
{question}

Workflow Summary:
{summary}

Assistant Response:
{response}
"""

        prompt_filled = prompt.format(
            question=question, summary=summary, response=response
        )

        request_payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict result classifier.",
                },
                {"role": "user", "content": prompt_filled},
            ],
            "max_completion_tokens": 200,
            "temperature": 0.0,
        }
        result = await asyncio.to_thread(
            self.chat_model.complete_with_retry, **request_payload
        )

        label = result.choices[0].message.content.strip().upper()

        # allowed = {"SUCCESS", "PARTIAL", "NEED_CLARIFICATION"}
        import re

        raw = (result.choices[0].message.content or "").strip()
        text = raw.upper()

        # 1) 先用正则从文本里抓出三个标签之一（支持前后有多余文字）
        m = re.search(r"\b(SUCCESS|PARTIAL|NEED_CLARIFICATION)\b", text)
        if m:
            label = m.group(1)
            return label

        # 2) 兼容一些模型可能输出的变体（可选）
        # 例如 "NEED CLARIFICATION"、"NEED-CLARIFICATION"
        m = re.search(r"\bNEED[\s\-_]*CLARIFICATION\b", text)
        if m:
            return "NEED_CLARIFICATION"

        # 3) 抽不到就 NULL，并把原始输出打出来方便你排查
        logger.info(f"classifier raw output: {raw!r}")
        return "NULL"

    async def _fix_partial_response(
        self, question: str, response: str, summary: str, available_tools
    ):
        """
        Fix a PARTIAL response by completing missing steps or tool calls
        based on the current workflow summary.
        """

        prompt = """You are an autonomous assistant continuing an ongoing task.

The current response is PARTIAL and does not fully solve the user's problem.

You are given:
- The user's original question
- A workflow summary describing what has already been done
- The latest (partial) assistant response

Your task:
1. Use the workflow summary to understand the current state
2. Identify what is still missing to fully solve the user's problem
3. COMPLETE the missing steps or required tool usage
4. Produce a COMPLETE answer for the user

Rules:
- Do NOT repeat steps that are already completed
- Do NOT restate the workflow summary
- Do NOT mention that the response was partial
- Do NOT explain your reasoning
- Output ONLY the answer to the user

User Question:
{question}

Workflow Summary:
{summary}

Partial Assistant Response:
{response}
"""

        prompt_filled = prompt.format(
            question=question, summary=summary, response=response
        )

        request_payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a reliable autonomous agent that completes unfinished work.",
                },
                {"role": "user", "content": prompt_filled},
            ],
            "tools": available_tools,
            "max_completion_tokens": 4000,
            "temperature": 0.3,
        }
        response = await asyncio.to_thread(
            self.chat_model.complete_with_retry, **request_payload
        )

        return response

    async def _generate_clarification_question(
        self, question: str, assistant_response: str
    ) -> str:
        """
        Based on the user question and the assistant's response,
        generate a single clarification question needed to continue.
        """

        prompt = """You are generating a clarification question for the user.

The assistant cannot continue the task because required information is missing.

Based on:
- The user's original question
- The assistant's latest response

Your task:
- Identify the SINGLE most important missing piece of information
- Ask ONE clear and specific clarification question to the user

Rules:
- Ask ONLY one question
- Be concise and user-friendly
- Do NOT explain why the question is needed
- Do NOT mention internal reasoning or system limitations
- Output ONLY the question text

User Question:
{question}

Assistant Response:
{assistant_response}
    """

        prompt_filled = prompt.format(
            question=question,
            assistant_response=assistant_response,
        )
        request_payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You generate minimal and precise clarification questions.",
                },
                {"role": "user", "content": prompt_filled},
            ],
            "max_completion_tokens": 800,
            "temperature": 0.3,
        }

        clarification = await asyncio.to_thread(
            self.chat_model.complete_with_retry, **request_payload
        )

        return clarification.choices[0].message.content.strip()

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
""",
                }
            ]
        else:
            messages = history.copy()

        messages.append({"role": "user", "content": query})

        # get session from pool
        assert self.session_pool.is_initialized(), "Session pool is not initialized."
        available_tools = []
        tool_error_num = defaultdict(int)

        async with self.session_pool.acquire() as session:
            # 给定一个session，在里面方便调用MCP服务来使用
            available_tools = self.available_tools
            final_text = []
            stop_flag = False
            MAX_FIX_ROUNDS = 3
            last_content = None
            MAX_TURNS = 20
            turns = 0
            fix_round = 0
            try:
                while not stop_flag and turns < MAX_TURNS:
                    turns += 1
                    request_payload = {
                        "messages": messages,
                        "tools": available_tools,
                    }

                    t0 = time.time()

                    content = None
                    # ====== A. 正常 LLM + tool 流程 ======
                    response = await asyncio.to_thread(
                        self.chat_model.complete_with_retry, **request_payload
                    )

                    response_message = response.choices[0].message
                    messages.append(response_message.model_dump(exclude_none=True))
                    content = response_message.content
                    if content:
                        last_content = content

                    # ====== B. 模型停止输出（无 tool call） ======
                    if (
                        content
                        and not response_message.tool_calls
                        and not response_message.function_call
                    ):

                        summary = await self._summarize_messages(messages[:-1])
                        logger.info(f"Summary:\n{summary}")
                        while True:
                            label = await self._classify_result(query, content, summary)

                            if label == "SUCCESS":
                                logger.info("SUCCESS → exit")
                                final_text.append(content)
                                stop_flag = True
                                break

                            if label == "NEED_CLARIFICATION":
                                logger.info("NEED_CLARIFICATION → ask user")
                                # generate a clarification question
                                user_question = (
                                    await self._generate_clarification_question(
                                        question=query,
                                        assistant_response=content,
                                    )
                                )
                                messages.append(
                                    {"role": "user", "content": user_question}
                                )
                                stop_flag = True
                                break

                            if label == "PARTIAL":
                                logger.info("PARTIAL → fixing")
                                fix_round += 1

                                if fix_round > MAX_FIX_ROUNDS:
                                    logger.warning("Too many PARTIAL fixes, stopping")
                                    final_text.append(content)
                                    stop_flag = True
                                    break

                                response = await self._fix_partial_response(
                                    question=query,
                                    response=content,
                                    summary=summary,
                                    available_tools=available_tools,
                                )
                                response_message = response.choices[0].message

                                messages[-1] = response_message.model_dump(
                                    exclude_none=True
                                )
                                content = response_message.content
                                break
                            if label == "NULL":
                                logger.warning(
                                    "LLM returned invalid label, treating as PARTIAL"
                                )
                                # treat as PARTIAL
                                final_text.append(content)
                                stop_flag = True
                                break

                    # 包含了工具的调用请求，需要进一步处理
                    tool_calls = response_message.tool_calls
                    if not tool_calls:
                        logger.warning(
                            "Received empty response from LLM without content or tool calls."
                        )
                        break

                    if turns >= MAX_TURNS:
                        final_text.append(last_content or "Reached max turns.")
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
                            # if tool_name == "tool_request":
                            #     description = tool_args["description"]
                            #     result = await asyncio.wait_for(
                            #         session.call_tool("route", {"query": description}),
                            #         timeout=300,
                            #     )
                            # else:
                            result = await asyncio.wait_for(
                                session.call_tool(tool_name, tool_args),
                                timeout=300,
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"Tool call {tool_name} timed out.")
                            result = "Tool call timed out."
                            tool_error_num[tool_name] += 1
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "content": str(result),
                                }
                            )
                            continue

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
                            tool_error_num[tool_name] += 1
                            continue
                        result = str(result)
                        result = result[:max_tool_tokens]
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": str(result),
                            }
                        )
                    for tool_name in tool_error_num:
                        if tool_error_num[tool_name] >= 3:
                            logger.error(
                                f"Tool {tool_name} has failed 3 times, skipping further calls."
                            )
                            raise RuntimeError(
                                f"Tool {tool_name} failed 3 times, aborting."
                            )

                    logger.info(f"[TIME] tool {tool_name} took {time.time()-t1:.2f}s")
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                final_text.append(f"Error: {str(e)}")
                messages.append({"role": "assistant", "content": str(e)})
        return ("\n".join(final_text) or last_content or ""), messages


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
