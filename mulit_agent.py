import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GTIFF_SRS_SOURCE"]="EPSG"
import json
import logging
import asyncio
import time
from enum import auto
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from logging.handlers import RotatingFileHandler

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.tools import tool
from langchain_core.callbacks import AsyncCallbackHandler
import base64
# Pprint for debugging
from pprint import pprint

# Change to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Global variables
logger = None
temp_dir_path = None
debug_trace_path = None  # set once temp_dir_path exists; live LLM/tool/delegate trace goes here

# Configuration
model_name = 'eve'
autoplanning = True
# Set to a list of question IDs to re-run only specific questions; None runs all
RETRY_IDS = None
# Parallel batching: set BATCH_TOTAL > 1 and launch BATCH_TOTAL copies with BATCH_INDEX 0..N-1
# Each copy handles a non-overlapping slice; merge results_summary.json files afterwards
BATCH_TOTAL = 1   # total number of parallel workers (1 = no batching)
BATCH_INDEX = 0   # which slice this worker handles (0-indexed)
# Debugging: cap the run to the first N questions (after RETRY_IDS/batch filtering) and
# print a live trace (timestamps + durations) of every LLM call and tool call as it
# happens, so a slow run can be diagnosed without waiting for it to finish.
DEBUG_MODE = False
MAX_QUESTIONS = 1 if DEBUG_MODE else None


def _ts() -> str:
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]


def _preview(obj, max_chars: int = 300) -> str:
    s = str(obj)
    return s if len(s) <= max_chars else s[:max_chars].rstrip() + '...'


def _log_trace(msg: str) -> None:
    """Append one line to the dedicated debug-trace file, kept separate from stdout so
    it isn't interleaved with FastMCP's own console logging."""
    if debug_trace_path is None:
        print(msg, flush=True)
        return
    with open(debug_trace_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


class VerboseCallbackHandler(AsyncCallbackHandler):
    """Prints a live, timestamped trace of every LLM call made by a given agent
    (supervisor or one specialist), so you can see exactly what the system is doing
    (and how long each model round-trip takes) while a run is in progress, instead of
    only finding out after the whole question completes."""

    def __init__(self, label: str):
        self.label = label
        self._start_times = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start_times[run_id] = time.monotonic()
        n_msgs = sum(len(batch) for batch in messages)
        _log_trace(f"[{_ts()}] [{self.label}] LLM call START ({n_msgs} messages)")

    async def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start_times[run_id] = time.monotonic()
        _log_trace(f"[{_ts()}] [{self.label}] LLM call START")

    async def on_llm_end(self, response, *, run_id, **kwargs):
        dur = time.monotonic() - self._start_times.pop(run_id, time.monotonic())
        usage = getattr(response, 'llm_output', None) or {}
        tokens = usage.get('token_usage') if isinstance(usage, dict) else None
        tok_str = f", {tokens}" if tokens else ""
        _log_trace(f"[{_ts()}] [{self.label}] LLM call END ({dur:.1f}s{tok_str})")

    async def on_llm_error(self, error, *, run_id, **kwargs):
        dur = time.monotonic() - self._start_times.pop(run_id, time.monotonic())
        _log_trace(f"[{_ts()}] [{self.label}] LLM call ERROR after {dur:.1f}s: {error}")
# Tool categories: each corresponds to one MCP server in agent/config.json and becomes
# its own specialist agent, coordinated by a supervisor agent (see create_multi_agent_system).
CATEGORIES = ["Index", "Inversion", "Perception", "Analysis", "Statistics"]

CATEGORY_INFO = {
    "Index": {
        "description": "Computes spectral indices from remote sensing imagery: NDVI, NDWI, NDBI, EVI, NBR, FVC, WRI, NDTI, FRP, NDSI, TVDI, extreme snow-loss percentage, and their batch variants.",
    },
    "Inversion": {
        "description": "Performs physical parameter inversion/retrieval from remote sensing data: land surface temperature (single/multi-channel, split-window, temperature-emissivity separation, day/night, TTM), band ratio, ATI, SAR dual-polarization/dual-frequency/multi-frequency features, sea-ice concentration, water turbidity.",
    },
    "Perception": {
        "description": "Performs image perception tasks: thresholding/segmentation, bounding-box geometry (expansion, area), object counting, centroid extraction and centroid-to-centroid distance (closest/farthest pair), skeleton analysis, foundation-model inference (RemoteCLIP, Strip R-CNN, SM3Det, RemoteSAM, InstructSAM, SAM2), and change detection (ChangeOS). This is the ONLY category with spatial/geometric tools -- any task involving distances, areas, or positions derived from detections or bounding boxes belongs here, not Statistics.",
    },
    "Analysis": {
        "description": "Performs statistical/time-series trend analysis: linear trend, Mann-Kendall test, Sen's slope, STL decomposition, change-point detection, autocorrelation, seasonality detection, and hotspot (Getis-Ord Gi*) analysis and direction.",
    },
    "Statistics": {
        "description": "Computes descriptive statistics over flat numeric lists and raster/image pixel values (mean, std, median, min, max, sum, skewness, kurtosis, coefficient of variation, hotspot percentage/maps), simple scalar arithmetic (difference, division, multiply, percentage change, Kelvin-to-Celsius), and generic file utilities including get_filelist for discovering filenames in a data directory. Has NO addition or square-root tool, so it cannot compute Euclidean distances or sums of two values -- and has no spatial/geometric tools at all (use Perception for anything involving bounding boxes, centroids, or distances between detected objects).",
    },
}

COMMON_ATTENTION = '''ATTENTION:
1. When a tool returns "Result saved at /path/to/file", you must use the full returned path "/path/to/file" in all subsequent tool calls.
2. If a tool returns an error, you may only retry that exact call once.
3. Before accessing any file in a data directory, always call get_filelist on that directory first to discover the actual filenames. Never assume or guess filenames.
4. You must only report numeric values that were actually returned by one of your tool calls. If none of your available tools can compute a value you need, say so explicitly and stop -- do not compute it yourself in prose. A supervisor will re-route the sub-task if your toolset can't cover it.'''


def format_tool_entry(t, max_desc_chars=160):
    """One line 'name(arg: type, ...) -- short description' for a single tool, built
    directly from its real MCP-reported args and description, so this can never drift
    out of sync with the underlying server code."""
    try:
        props = t.args or {}
    except Exception:
        props = {}
    arg_strs = []
    for pname, pinfo in props.items():
        ptype = pinfo.get('type', 'any') if isinstance(pinfo, dict) else 'any'
        arg_strs.append(f"{pname}: {ptype}")
    signature = f"{t.name}({', '.join(arg_strs)})"

    # MCP tool descriptions are often multi-line with a leading "Description:" header;
    # pull out the first substantive line for a compact one-liner.
    desc_lines = [ln.strip() for ln in (t.description or '').strip().splitlines() if ln.strip()]
    desc_lines = [ln for ln in desc_lines if ln.lower() not in ('description:',)]
    short_desc = desc_lines[0] if desc_lines else ''
    if len(short_desc) > max_desc_chars:
        short_desc = short_desc[:max_desc_chars].rstrip() + '...'

    return f"- {signature} -- {short_desc}"


def format_tool_inventory(tools) -> str:
    """Bullet-list inventory of every tool available to a specialist: real name, real
    argument names/types, and a short purpose line -- built from live MCP metadata."""
    return '\n'.join(format_tool_entry(t) for t in tools)


def format_tool_name_manifest(tools) -> str:
    """Comma-separated tool names only. Used in the supervisor prompt so it knows each
    specialist's exact capabilities without bloating the prompt with full descriptions."""
    return ', '.join(t.name for t in tools)


def build_specialist_prompt(category: str, tools) -> str:
    """System prompt for the specialist agent that owns one tool category. `tools` is
    the specialist's actual (untraced) tool list, used to render a live inventory."""
    return f'''You are a geoscientist specialized in the "{category}" toolset for Earth observation data analysis.
{CATEGORY_INFO[category]["description"]}

Your available tools (exact name, arguments, and purpose):
{format_tool_inventory(tools)}

You have been delegated a specific sub-task by a supervising agent. Before doing any
calculation yourself, check whether one of the tools above -- possibly chained across
several calls -- already produces the value directly (e.g. converting detections to
centroids and then to distances is two chained tool calls, not manual math). Prefer
chaining your own tools over improvising arithmetic in prose. Use your tools to
complete the task as accurately and completely as possible, then report your findings
(including any computed values and output file paths) in clear prose. You do not need
to produce a final multiple-choice answer yourself, just report what you found.
{COMMON_ATTENTION}'''


def build_supervisor_prompt(tools_by_category) -> str:
    """System prompt for the supervisor. `tools_by_category` maps each category to its
    actual (untraced) tool list, used to render a per-specialist tool-name manifest so
    the supervisor can route sub-tasks to whichever specialist can actually finish
    them, instead of splitting a task across specialists based on category vibes."""
    delegate_lines = []
    for category in CATEGORIES:
        names = format_tool_name_manifest(tools_by_category[category])
        delegate_lines.append(
            f'- delegate_to_{category.lower()}_agent: {CATEGORY_INFO[category]["description"]}\n'
            f'  Tools it has access to: {names}'
        )
    delegate_block = '\n'.join(delegate_lines)

    return f'''You are a lead geoscientist coordinating a team of specialist agents to answer multiple-choice questions about Earth observation data analysis. You have no data-processing tools of your own, you can only delegate sub-tasks to these specialist agents, each wrapping one category of tools:

{delegate_block}

Specialists cannot see the original question or each other's work, only what you pass them. For each delegate call, include all the context that specialist needs (the data directory/paths, prior results it should use, and the exact sub-task).

ROUTING RULE: check each specialist's tool list above before delegating. If one specialist's own tools can carry a sub-task all the way through to the final derived value (for example: detecting objects, converting them to centroids, and computing the distance between the two farthest all live in Perception), delegate that whole sub-task to them in a single call rather than splitting it across specialists. Splitting forces the receiving specialist to improvise arithmetic instead of using a purpose-built tool, which produces unreliable numbers. Only split a task across specialists when the categories genuinely don't overlap (e.g. computing NDVI in Index, then trend-testing the NDVI time series in Analysis).

Chain calls as needed, feeding one specialist's results into the next.
ATTENTION:
1. If a delegate call returns an error, you may only retry that exact call once.
2. Once you have enough information, your FINAL turn must be plain text only, with no further tool calls, containing exactly:
<Answer>Your choice</Answer>'''

username = os.getenv("EVE_USERNAME")
password = os.getenv("EVE_PASSWORD")

token = base64.b64encode(f"{username}:{password}".encode()).decode()
headers = {"Authorization": f"Basic {token}"}


def init_global_params():
    """Initialize global parameters and logging"""
    global temp_dir_path, logger, debug_trace_path

    if temp_dir_path is None:
        batch_suffix = f'_b{BATCH_INDEX}of{BATCH_TOTAL}' if BATCH_TOTAL > 1 else ''
        temp_dir_path = Path('./evaluate_langchain/{}_{}_{}{}' .format(
            model_name,
            'AP' if autoplanning else "IF",
            datetime.now().strftime('%y-%m-%d_%H-%M'),
            batch_suffix
        )).absolute()
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    # Dedicated, FastMCP-free trace of every LLM/tool/delegate call, written live.
    debug_trace_path = temp_dir_path / "debug_trace.txt"
    debug_trace_path.write_text("", encoding='utf-8')

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            # Simplified logging compatible with original format
            log_record = {
                "question_index": record.args[0] if record.args else "unknown",
                "timestamp": self.formatTime(record, self.datefmt),
                "conversations": record.args[1] if len(record.args) > 1 else [],
                "final_answer": record.args[2] if len(record.args) > 2 else None,
                "tool_call_order": record.args[3] if len(record.args) > 3 else [],
            }
            return json.dumps(log_record, ensure_ascii=False, indent=4)

    logger = logging.getLogger("text_logger")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        temp_dir_path / "{}_{}_langchain.log".format(
            model_name, 'AP' if autoplanning else "IF"
        )
    )
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    return temp_dir_path, logger


def init_chat_logger():
    """Initialize chat logger for .chat file like AgentScope"""
    global temp_dir_path
    chat_log_path = temp_dir_path / "{}_{}_langchain.chat".format(
        model_name, 'AP' if autoplanning else "IF"
    )
    return chat_log_path


def save_chat_message(chat_log_path, message_data):
    """Save a single chat message to .chat file in AgentScope format"""
    import time
    from datetime import datetime
    import uuid

    # Format message in AgentScope style
    chat_record = {
        "__module__": "langchain.schema.messages",
        "__name__": "ChatMessage",
        "id": str(uuid.uuid4()).replace('-', ''),
        "name": message_data.get('name', 'langchain_agent'),
        "role": message_data.get('role', 'assistant'),
        "content": message_data.get('content', []),
        "metadata": message_data.get('metadata', None),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Append to chat file (one JSON per line, like AgentScope)
    with open(chat_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(chat_record, ensure_ascii=False) + '\n')


def load_langchain_config(config_path='./agent/config.json'):
    """Load configuration and initialize LangChain components"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize OpenAI model with stricter parameters
    model_config = config['models'][0]
    llm_kwargs = {
        'model': "EVE-Instruct",
        'api_key': "EMPTY",
        'base_url': os.getenv("EVE_ENDPOINT"),
        'temperature': 0,  # Lower temperature for more focused responses
        'request_timeout': 300,  # 5 minute timeout per request
        'default_headers': headers
    }

    # Add generate_args via extra_body if present in config
    if 'generate_args' in model_config:
        llm_kwargs['extra_body'] = model_config['generate_args']

    llm = ChatOpenAI(**llm_kwargs)

    # Prepare MCP servers configuration
    mcp_servers = {}
    for server_name, server_config in config['mcpServers'].items():
        # Update paths to use current temp directory
        updated_args = []
        for arg in server_config['args']:
            if 'tmp/tmp/out' in arg:
                updated_args.append(str(temp_dir_path / 'out'))
            elif arg.startswith('tools/'):
                updated_args.append('agent/' + arg)
            else:
                updated_args.append(arg)

        server_env = dict(server_config.get('env') or {})
        ld = os.environ.get("LD_LIBRARY_PATH")
        if ld:
            server_env.setdefault("LD_LIBRARY_PATH", ld)

        mcp_servers[server_name] = {
            "command": server_config['command'],
            "args": updated_args,
            "transport": "stdio",
            "env": server_env,
        }

    return llm, mcp_servers


def wrap_tool_with_trace(original_tool, category: str, tool_trace: list):
    """Wrap a real MCP tool so every invocation is recorded, in true call order, into
    tool_trace. This is what lets us reconstruct "the order the tools were called in"
    reliably even though the tools now live behind several specialist agents instead of
    one flat agent, and even if a supervisor turn fans out to more than one specialist
    concurrently (list.append below happens synchronously at call time, before the
    await, so entry order always matches invocation order regardless of which call
    finishes first)."""
    from langchain_core.tools import StructuredTool

    async def _traced(**kwargs):
        record = {"agent": category, "name": original_tool.name, "input": kwargs, "output": None, "error": False}
        tool_trace.append(record)
        _log_trace(f"[{_ts()}] [{category}] TOOL START: {original_tool.name}({_preview(kwargs, 200)})")
        start = time.monotonic()
        try:
            result = await original_tool.ainvoke(kwargs)
            record["output"] = result
            dur = time.monotonic() - start
            _log_trace(f"[{_ts()}] [{category}] TOOL END: {original_tool.name} in {dur:.1f}s -> {_preview(result)}")
            return result
        except Exception as e:
            record["output"] = str(e)
            record["error"] = True
            dur = time.monotonic() - start
            _log_trace(f"[{_ts()}] [{category}] TOOL ERROR: {original_tool.name} in {dur:.1f}s -> {e}")
            raise

    return StructuredTool.from_function(
        coroutine=_traced,
        name=original_tool.name,
        description=original_tool.description,
        args_schema=original_tool.args_schema,
    )


def make_delegate_tool(category: str, specialist_agent):
    """Wrap a specialist agent as a tool the supervisor can call by category name."""
    tool_name = f"delegate_to_{category.lower()}_agent"
    description = CATEGORY_INFO[category]["description"]

    async def _delegate(task: str) -> str:
        _log_trace(f"[{_ts()}] [supervisor] DELEGATE START -> {category}: {_preview(task, 200)}")
        start = time.monotonic()
        try:
            result = await specialist_agent.ainvoke(
                {"messages": [HumanMessage(content=task)]},
                config={
                    "recursion_limit": 50,
                    "max_execution_time": 300,
                    "callbacks": [VerboseCallbackHandler(category)],
                },
            )
        except Exception as e:
            dur = time.monotonic() - start
            _log_trace(f"[{_ts()}] [supervisor] DELEGATE ERROR -> {category} in {dur:.1f}s: {e}")
            return f"Error: {category} agent failed: {e}"

        dur = time.monotonic() - start
        answer = extract_answer_from_response(result)
        _log_trace(f"[{_ts()}] [supervisor] DELEGATE END -> {category} in {dur:.1f}s: {_preview(answer)}")
        return answer

    return tool(tool_name, description=description)(_delegate)


async def create_multi_agent_system(llm, mcp_servers):
    """Create a supervisor agent plus one specialist ReAct agent per tool category
    (Index, Inversion, Perception, Analysis, Statistics), each restricted to its own MCP server's tools."""
    client = MultiServerMCPClient(mcp_servers)

    try:
        raw_tools_by_category = {}
        for category in CATEGORIES:
            raw_tools_by_category[category] = await client.get_tools(server_name=category)
            print(f"Loaded {len(raw_tools_by_category[category])} tools for the {category} agent")

        # Share the generic file-listing utility (lives in the Statistics server) with every
        # specialist, since each of them is instructed to call get_filelist before touching files.
        filelist_tool = next((t for t in raw_tools_by_category["Statistics"] if t.name == "get_filelist"), None)

        # Every real tool call (regardless of which specialist makes it) is recorded here,
        # in true chronological call order, so downstream logging doesn't have to
        # reconstruct order from the agent-of-agents message structure.
        tool_trace = []

        # Untraced tool lists (real name/args/description) are kept around purely to
        # render accurate prompt text -- the agents themselves run on the traced copies.
        tools_by_category = {}
        traced_tools_by_category = {}
        for category in CATEGORIES:
            category_tools = list(raw_tools_by_category[category])
            if filelist_tool and category != "Statistics":
                category_tools.append(filelist_tool)
            tools_by_category[category] = category_tools
            traced_tools_by_category[category] = [wrap_tool_with_trace(t, category, tool_trace) for t in category_tools]

        specialists = {
            category: create_react_agent(
                llm,
                traced_tools_by_category[category],
                prompt=build_specialist_prompt(category, tools_by_category[category]),
                name=f"{category.lower()}_agent",
            )
            for category in CATEGORIES
        }

        worker_tools = [make_delegate_tool(category, specialists[category]) for category in CATEGORIES]

        supervisor_prompt = build_supervisor_prompt(tools_by_category)
        supervisor = create_react_agent(llm, worker_tools, prompt=supervisor_prompt, name="supervisor")

        return supervisor, client, tool_trace
    except Exception as e:
        print(f"Error creating multi-agent system: {e}")
        if hasattr(client, 'close'):
            await client.close()
        raise


def load_questions(test_json_path: str = 'benchmark/question.json'):
    """Load evaluation questions"""
    with open(test_json_path, 'r') as f:
        test_json = json.load(f)

    out = []
    for _, (question_idx, question_info) in enumerate(test_json.items()):
        AP_INDEX = 0 if question_info['evaluation'][0]['type'] == 'autonomous planning' else 1
        data = question_info['evaluation'][AP_INDEX].get('data', None)
        data = question_info['evaluation'][1 - AP_INDEX].get('data', None) if data is None else data

        if data is None:
            continue
        out.append({
            "question_id": question_idx,
            "auto": question_info['evaluation'][AP_INDEX]['question'],
            "instruct": question_info['evaluation'][1 - AP_INDEX]['question'],
            "data": data,
            "choices": question_info.get('choices', None)
        })

    return out


def extract_answer_from_response(response):
    """Extract just the parsed choice from the final answer (used for grading /
    results_summary.json, and as the return value fed back to the supervisor by a
    delegate call)."""
    messages = response.get("messages", [])

    for message in reversed(messages):
        if hasattr(message, 'type') and message.type == 'ai':
            content = message.content
            if '<Answer>' in content and '</Answer>' in content:
                start = content.find('<Answer>') + len('<Answer>')
                end = content.find('</Answer>')
                return content[start:end].strip()
            return content

    return "No answer found"


def get_final_message_content(response):
    """Raw content of the last AI message, completely untouched (tags and all). This is
    what goes into the JSON log's `final_answer` field, to match the old single-agent
    log format, which logged the raw final message rather than a parsed choice."""
    messages = response.get("messages", [])
    for message in reversed(messages):
        if hasattr(message, 'type') and message.type == 'ai':
            return message.content
    return ""


def build_conversation_log(query, tool_call_order, final_content):
    """Build the JSON log's `conversations` field to match the old single-agent shape:
    one user turn, then one assistant(tool-call)/tool(result) pair per REAL tool
    invocation (flattened across whichever specialist made it), in true chronological
    order, then a final assistant text turn. This intentionally ignores the
    supervisor/specialist delegate-call framing -- the log should look the same
    whether the question was answered by one flat agent or by several behind the
    scenes."""
    log = [{"role": "user", "content": query}]

    for call in tool_call_order:
        log.append({
            "role": "assistant",
            "content": [{"name": call["name"], "input": call["input"]}]
        })
        log.append({
            "role": "tool",
            "name": call["name"],
            "content": [{"output": [{"text": str(call["output"])}]}]
        })

    log.append({
        "role": "assistant",
        "content": [{"type": "text", "content": final_content}]
    })

    return log


def save_transcript_to_chat(chat_log_path, label, messages):
    """Save a linear message list's AI/tool messages to the .chat file, tagged with `label`."""
    for message in messages:
        if not hasattr(message, 'type') or message.type == 'human':
            continue

        if message.type == 'ai':
            assistant_chat_content = []
            if message.content and message.content.strip():
                assistant_chat_content.append({"type": "text", "text": message.content})

            if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                for tool_call in message.additional_kwargs['tool_calls']:
                    try:
                        arguments = json.loads(tool_call['function']['arguments']) if isinstance(tool_call['function']['arguments'], str) else tool_call['function']['arguments']
                    except:
                        arguments = tool_call['function']['arguments']
                    assistant_chat_content.append({
                        "type": "tool_use",
                        "id": tool_call['id'],
                        "name": tool_call['function']['name'],
                        "input": arguments
                    })

            if assistant_chat_content:
                save_chat_message(chat_log_path, {
                    "name": label,
                    "role": "assistant",
                    "content": assistant_chat_content,
                    "metadata": None
                })

        elif message.type == 'tool':
            save_chat_message(chat_log_path, {
                "name": "system",
                "role": "system",
                "content": [{
                    "type": "tool_result",
                    "id": getattr(message, 'tool_call_id', 'unknown'),
                    "output": [{"type": "text", "text": str(message.content), "annotations": None, "meta": None}],
                    "name": message.name
                }],
                "metadata": {"agent": label}
            })


def save_tool_trace_to_chat(chat_log_path, tool_calls):
    """Save the flat, true-call-order record of real tool invocations (across every
    specialist) to the .chat file, one entry per call, tagged with which specialist
    agent made it."""
    for call in tool_calls:
        save_chat_message(chat_log_path, {
            "name": call["name"],
            "role": "tool",
            "content": [{
                "type": "tool_result",
                "input": call["input"],
                "output": [{"type": "text", "text": str(call["output"])}],
                "is_error": call["error"],
            }],
            "metadata": {"agent": call["agent"]}
        })


async def handle_question(supervisor, question, chat_log_path, tool_trace):
    """Handle a single question with the supervisor + specialist multi-agent system"""
    query = question['auto'] + question['data'] if autoplanning else \
        question['instruct'] + question['data']

    if question['choices']:
        query += '\n'.join([''] + [
            '{}.{}'.format(chr(ord('A') + i), choice)
            for i, choice in enumerate(question['choices'])
        ])

    print(f"\n--- Processing Question {question['question_id']} ---")
    print(f"Query: {query[:200]}...")

    # Save user message to chat log
    user_message = {
        "name": "user",
        "role": "user",
        "content": query,
        "metadata": {"question_id": question['question_id']}
    }
    save_chat_message(chat_log_path, user_message)

    # Each question gets a clean slate of real tool-call traces
    tool_trace.clear()

    try:
        # Invoke the supervisor agent with configuration to prevent infinite loops
        response = await supervisor.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "recursion_limit": 75,  # supervisor may chain several delegate calls
                "max_execution_time": 600,  # 10 minutes timeout
                "callbacks": [VerboseCallbackHandler("supervisor")],
            }
        )

        # Parsed choice (e.g. "C") -- used for grading / results_summary.json
        final_answer = extract_answer_from_response(response)
        # Raw final message content (e.g. "...<Answer>C</Answer>") -- used for the JSON log,
        # to match the old single-agent log format
        final_content = get_final_message_content(response)

        # tool_trace is already in true call order (see wrap_tool_with_trace): this is
        # the order the real tools were invoked in, independent of which specialist agent
        # made the call. Copy it now, before the next question clears it.
        tool_call_order = [
            {"agent": call["agent"], "name": call["name"], "input": call["input"], "output": call["output"], "error": call["error"]}
            for call in tool_trace
        ]

        # JSON log: flattened real tool-call trace in the old single-agent shape
        conversation_log = build_conversation_log(query, tool_call_order, final_content)

        # .chat file: supervisor-level transcript, plus each real tool call in true order
        save_transcript_to_chat(chat_log_path, "supervisor", response.get("messages", []))
        save_tool_trace_to_chat(chat_log_path, tool_call_order)

        # Log the conversation in the old format (raw final_content), plus the tool-call order
        logger.info("Chat Content", question['question_id'], conversation_log, final_content, tool_call_order)

        print(f"Final Answer: {final_answer}")
        print(f"Tool calls in order: {[c['name'] for c in tool_call_order]}")
        return final_answer

    except Exception as e:
        error_msg = f"Error processing question {question['question_id']}: {e}"
        print(error_msg)

        # Save error to chat log
        error_message = {
            "name": "system",
            "role": "system",
            "content": [{"type": "text", "content": error_msg}],
            "metadata": {"error": True, "question_id": question['question_id']}
        }
        save_chat_message(chat_log_path, error_message)

        # Fixed: previously this was logger.info(question['question_id'], [], error_msg),
        # which mis-shifted args (question_id became the log msg, error_msg became
        # `final_answer`... actually landed in `conversations`). Now matches the schema:
        # (msg, question_id, conversations, final_answer, tool_call_order).
        logger.info("Chat Content", question['question_id'], [], error_msg, [])
        return f"Error: {e}"


async def main():
    """Main evaluation function"""
    print("Initializing LangChain-based Earth Science Agent...")

    # Initialize global parameters
    init_global_params()

    # Initialize chat logger
    chat_log_path = init_chat_logger()
    print(f"Chat log will be saved to: {chat_log_path}")
    print(f"Live LLM/tool/delegate trace will be saved to: {debug_trace_path}")

    # Load configuration and create the supervisor + specialist multi-agent system
    llm, mcp_servers = load_langchain_config()
    supervisor, client, tool_trace = await create_multi_agent_system(llm, mcp_servers)

    try:
        # Load questions
        questions = load_questions()
        if RETRY_IDS is not None:
            retry_set = set(RETRY_IDS)
            questions = [q for q in questions if q['question_id'] in retry_set]
        if BATCH_TOTAL > 1:
            questions = [q for i, q in enumerate(questions) if i % BATCH_TOTAL == BATCH_INDEX]
        if MAX_QUESTIONS is not None:
            questions = questions[:MAX_QUESTIONS]
        print(f"Loaded {len(questions)} questions for evaluation"
              + (f" (batch {BATCH_INDEX+1}/{BATCH_TOTAL})" if BATCH_TOTAL > 1 else "")
              + (f" [DEBUG_MODE: capped to {MAX_QUESTIONS}]" if MAX_QUESTIONS is not None else ""))

        # Process questions
        results = []
        for question in tqdm(questions, desc="Processing questions"):
            answer = await handle_question(supervisor, question, chat_log_path, tool_trace)
            results.append({
                "question_id": question['question_id'],
                "answer": answer
            })

        # Save results summary
        results_path = temp_dir_path / "results_summary.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"\nEvaluation completed! Results saved to {results_path}")
        print(f"Detailed logs available at: {temp_dir_path}")
        print(f"Chat history saved to: {chat_log_path}")

    except Exception as e:
        print(f"Error in main evaluation: {e}")
        raise

    finally:
        # Clean up
        if hasattr(client, 'close'):
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())