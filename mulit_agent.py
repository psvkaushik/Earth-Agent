import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
import json
import logging
import asyncio
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field
import base64

# Change to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Global variables
# ============================================================================
logger = None
temp_dir_path = None

# Configuration
model_name = 'eve'
autoplanning = True
RETRY_IDS = None
BATCH_TOTAL = 1
BATCH_INDEX = 0


_current_subagent_traces = []

_kit_message_history = {}


username = os.getenv("EVE_USERNAME")
password = os.getenv("EVE_PASSWORD")
token = base64.b64encode(f"{username}:{password}".encode()).decode()
headers = {"Authorization": f"Basic {token}"}


KIT_SPECS = {
    "index": {
        "match_keywords": ["index"],
        "description": "spectral indices (e.g. NDVI, NDWI, NBR, NDBI, EVI)",
    },
    "inversion": {
        "match_keywords": ["inversion"],
        "description": "geophysical parameter retrieval (e.g. land surface temperature, soil moisture)",
    },
    "perception": {
        "match_keywords": ["perception"],
        "description": "image perception (scene classification, object detection, segmentation)",
    },
    "analysis": {
        "match_keywords": ["analysis", "analytic"],
        "description": "spatiotemporal analysis (trend detection, seasonality, change-point detection, spatial autocorrelation)",
    },
    "statistics": {
        "match_keywords": ["statistic", "stats"],
        "description": "pixel/batch statistics, thresholding, image algebra, cloud masking, and get_filelist",
    },
}


SPECIALIST_PROMPT_TEMPLATE = '''
You are a geoscientist, and you need to use tools to complete a sub-task about Earth observation data analysis given to you by a lead agent. You have access to tools for {tools_desc}, plus get_filelist. Note that if a tool returns an error, you can only try again once. Report your result clearly, including any output file paths, so the lead agent can use it.
ATTENTION:
1. When a tool returns "Result saved at /path/to/file", you must use the full returned path "/path/to/file" in all subsequent tool calls.
2. The lead agent's instructions will always include the data directory path for this question, and, whenever relevant, the exact file paths that have already been retrieved or produced -- either filenames the lead agent already discovered, or paths a specialist previously reported back to it (e.g. "Result saved at ..."). Treat those paths as given: use them exactly as written, never re-verify them, and never assume, guess, or slightly alter a path you were not actually given.
3. Only call get_filelist yourself if the instructions you received contain NO usable file paths at all -- i.e. only a bare directory, with no filenames and no previously retrieved paths to work from. If any paths are already present in what you were given, use those directly instead of calling get_filelist.
4. Before processing multiple files, check whether a batch version of the tool exists (e.g. a tool literally named calculate_batch_X, or a parameter typed as a list/array) and use it to handle all the files in one call. Only call a tool once per file, one by one, if no batch option exists for that specific operation -- don't default to looping if a batch tool is available. Also check whether a tool's parameters are file paths or require the data loaded into memory first. If you are unsure what a tool accepts, check its documentation rather than guessing.
5. If you're unable to answer a question with the available tools, report the limitation clearly, do not guess any answer or hallucinate.
'''

for _kit, _spec in KIT_SPECS.items():
    _spec["system_prompt"] = SPECIALIST_PROMPT_TEMPLATE.format(tools_desc=_spec["description"])

# Same as the original single-agent sys_prompt, with one addition: a list of
# the five delegation tools in place of the flat toolset.
ORCHESTRATOR_SYS_PROMPT = '''
You are a geisoscientist, and you need to use tools to answer multiple-choice questions about Earth observation data analysis. Note that if a tool returns an error, you can only try again once. Ultimately, you only need to explicitly tell me the correct choice.

You have access to five tools, each delegating a sub-task to a specialist agent with its own set of underlying tools:
- call_index_kit: to calculate spectral indices (e.g. NDVI, NDWI, NBR, NDBI, EVI)
- call_inversion_kit: geophysical parameter retrieval (e.g. land surface temperature, soil moisture)
- call_perception_kit: image perception (scene classification, object detection, segmentation)
- call_analysis_kit: spatiotemporal analysis (trend detection, seasonality, change-point detection, spatial autocorrelation)
- call_statistics_kit: pixel/batch statistics, thresholding, image algebra, cloud masking, and get_filelist
Each specialist remembers its own earlier calls within this question (so you can tell it "as before" or point out a prior error), but it does not see your reasoning or other specialists' results unless you include them in the instructions you give it.

ATTENTION:
1. When a tool returns "Result saved at /path/to/file", you must use the full returned path "/path/to/file" in all subsequent tool calls.
2. For each question, you must provide the choice you think is most appropriate. Don't gibe me another format. Your final answer format must be:
<Answer>Your choice<Answer>
3. There are two directories you must keep track of throughout the question and pass to specialists as needed:
   (a) The INPUT data directory for this question (given in the question text) -- pass this to any specialist that needs to read the original source files.
   (b) The OUTPUT directory where specialists save results -- the first time any specialist reports "Result saved at /some/path/file.tif", note the directory that path lives in; specialist outputs generally land in that same output directory for the rest of the question. When a later specialist needs a file another specialist produced, give it that exact full output path, not just a directory -- specialists cannot see each other's results, so you must copy the exact path over yourself. Never invent or modify a path.
4. If a specialist is unable to attend to your query and states its limitation, try a different specialist instead. If none of the specialists can answer, report the limitation clearly, do not get stuck in a loop calling the same specialist, do not guess any answer or hallucinate.
5. Tell each specialist WHAT you need, not HOW to do it. Describe the goal (e.g. "compute the dryness index for this NDVI/LST time series and give me its annual trend") and give it the context it needs (paths, dates, region), but do not specify formulas, algorithms, step-by-step methods, or which exact tool it should call. The specialist knows its own tools and how to use them correctly -- if you specify the method yourself, you risk describing the wrong formula and the specialist following it instead of using the correct tool for the job.
'''

# ============================================================================
# Logging (unchanged in spirit from the single-agent version)
# ============================================================================

def init_global_params():
    global temp_dir_path, logger
    if temp_dir_path is None:
        batch_suffix = f'_b{BATCH_INDEX}of{BATCH_TOTAL}' if BATCH_TOTAL > 1 else ''
        temp_dir_path = Path('./evaluate_langchain/{}_{}_{}{}'.format(
            model_name,
            'AP' if autoplanning else "IF",
            datetime.now().strftime('%y-%m-%d_%H-%M'),
            batch_suffix
        )).absolute()
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "question_index": record.args[0] if record.args else "unknown",
                "timestamp": self.formatTime(record, self.datefmt),
                "conversations": record.args[1] if len(record.args) > 1 else [],
                "final_answer": record.args[2] if len(record.args) > 2 else None
            }
            return json.dumps(log_record, ensure_ascii=False, indent=4)

    logger = logging.getLogger("text_logger")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        temp_dir_path / "{}_{}_langchain.log".format(model_name, 'AP' if autoplanning else "IF")
    )
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return temp_dir_path, logger


def init_chat_logger():
    global temp_dir_path
    return temp_dir_path / "{}_{}_langchain.chat".format(model_name, 'AP' if autoplanning else "IF")


def init_trace_logger():
    """Plain-text, human-readable trace: one entry per question showing the
    full call tree (orchestrator turns -> delegations -> each specialist's
    own tool calls/results -> final answer). Complements the JSON .log
    (machine-parseable) and the AgentScope-style .chat file (replay format)
    with something you can just open and skim."""
    global temp_dir_path
    return temp_dir_path / "{}_{}_trace.txt".format(model_name, 'AP' if autoplanning else "IF")


def render_log_entries(entries, indent=0):
    """Render a conversation_log-shaped list (user/assistant/tool/specialist
    entries, as built in handle_question) into indented plain-text lines.
    Recurses into "specialist" entries since their "content" is itself a
    list of the same-shaped entries (see response_to_trace)."""
    pad = "    " * indent
    lines = []
    for entry in entries:
        role = entry.get("role")
        if role == "user":
            lines.append(f"{pad}[USER] {entry.get('content')}")
        elif role == "assistant":
            for item in entry.get("content", []):
                if item.get("type") == "text":
                    lines.append(f"{pad}[ASSISTANT] {item.get('content')}")
                elif "name" in item:
                    try:
                        args_str = json.dumps(item.get("input", {}), ensure_ascii=False)
                    except TypeError:
                        args_str = str(item.get("input", {}))
                    lines.append(f"{pad}[ASSISTANT -> CALL] {item['name']}({args_str})")
        elif role == "tool":
            try:
                output_text = entry["content"][0]["output"][0]["text"]
            except Exception:
                output_text = str(entry.get("content"))
            lines.append(f"{pad}[TOOL RESULT: {entry.get('name')}] {output_text}")
        elif role == "specialist":
            kit_label = entry.get("kit", "?").upper()
            lines.append(f"{pad}--- {kit_label} SPECIALIST ---")
            lines.extend(render_log_entries(entry.get("content", []), indent=indent + 1))
            lines.append(f"{pad}--- end {kit_label} ---")
        else:
            lines.append(f"{pad}[{role}] {entry}")
    return lines


def format_conversation_log_as_text(question_id, conversation_log, final_answer) -> str:
    """Structured recap written once the question finishes -- comes AFTER
    the real-time [HH:MM:SS] stream (written live by TraceCallbackHandler /
    write_trace as each call happens) and organizes the same events into a
    clean, indented tree for a quicker read after the fact."""
    lines = []
    lines.append("-" * 100)
    lines.append(f"Question ID: {question_id}    SUMMARY  "
                 f"(structured recap of the run above)  "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("-" * 100)
    lines.extend(render_log_entries(conversation_log))
    lines.append("-" * 100)
    lines.append(f"FINAL ANSWER: {final_answer}")
    lines.append("=" * 100)
    lines.append("")  # blank separator line between questions
    return "\n".join(lines)


def write_trace(trace_log_path, text: str):
    with open(trace_log_path, 'a', encoding='utf-8') as f:
        f.write(text + "\n")


class TraceCallbackHandler(BaseCallbackHandler):
    """Writes each tool call and its result to the trace .txt file the
    MOMENT it happens, instead of buffering everything until the whole
    question (or whole specialist delegation) finishes. One instance is
    scoped to a label ("ORCHESTRATOR" or a kit name like "INDEX") so you can
    tell, in real time, who is calling what -- including a specialist's own
    internal tool calls while it's still working, before it has returned
    anything back to the orchestrator."""

    def __init__(self, trace_log_path, label: str):
        self.trace_log_path = trace_log_path
        self.label = label

    def _write(self, line: str):
        write_trace(self.trace_log_path, line)

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name", "unknown_tool")
        ts = datetime.now().strftime('%H:%M:%S')
        self._write(f"[{ts}] [{self.label}] -> CALL {name}({input_str})")

    def on_tool_end(self, output, **kwargs):
        ts = datetime.now().strftime('%H:%M:%S')
        self._write(f"[{ts}] [{self.label}] <- RESULT {output}")

    def on_tool_error(self, error, **kwargs):
        ts = datetime.now().strftime('%H:%M:%S')
        self._write(f"[{ts}] [{self.label}] <- ERROR {error}")


def save_chat_message(chat_log_path, message_data):
    import uuid
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
    with open(chat_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(chat_record, ensure_ascii=False) + '\n')


# ============================================================================
# Config / MCP loading
# ============================================================================

def load_langchain_config(config_path='./agent/config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_config = config['models'][0]
    llm_kwargs = {
        'model': "EVE-Instruct",
        'api_key': "EMPTY",
        'base_url': os.getenv("EVE_ENDPOINT"),
        'temperature': 0,
        'request_timeout': 300,
        'default_headers': headers
    }
    if 'generate_args' in model_config:
        llm_kwargs['extra_body'] = model_config['generate_args']

    llm = ChatOpenAI(**llm_kwargs)

    mcp_servers = {}
    for server_name, server_config in config['mcpServers'].items():
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


def assign_servers_to_kits(mcp_servers: dict):
    """Map each configured MCP server name to a kit based on KIT_SPECS
    keywords. Prints a loud warning for anything that doesn't match so
    mis-mapping is easy to spot before a run burns API calls."""
    kit_to_servers = {kit: [] for kit in KIT_SPECS}
    unmatched = []

    for server_name in mcp_servers:
        matched_kit = None
        lowered = server_name.lower()
        for kit, spec in KIT_SPECS.items():
            if any(kw in lowered for kw in spec["match_keywords"]):
                matched_kit = kit
                break
        if matched_kit:
            kit_to_servers[matched_kit].append(server_name)
        else:
            unmatched.append(server_name)

    print("\n=== Kit <-> MCP server mapping ===")
    for kit, servers in kit_to_servers.items():
        print(f"  {kit:11s} -> {servers if servers else '!!! NO SERVER MATCHED !!!'}")
    if unmatched:
        print(f"  (unmatched servers, available to ALL kits + orchestrator as shared utilities): {unmatched}")
    print("===================================\n")

    for kit, servers in kit_to_servers.items():
        if not servers:
            print(f"WARNING: kit '{kit}' matched no MCP server. Check KIT_SPECS['{kit}']"
                  f"['match_keywords'] against your agent/config.json server names.")

    return kit_to_servers, unmatched


# Tools that should be available to every kit specialist AND the
# orchestrator, regardless of which MCP server happens to host them.
# get_filelist lives in tools/Statistics.py, but every kit's system prompt
# requires calling it before touching a data directory, so it must be
# universally available rather than exclusive to the Statistics specialist.
SHARED_TOOL_NAMES = {"get_filelist"}


async def load_tools_per_kit(client: MultiServerMCPClient, mcp_servers: dict):
    kit_to_servers, unmatched_servers = assign_servers_to_kits(mcp_servers)

    # Load every server's tools once.
    tools_by_server = {}
    for server_name in mcp_servers:
        tools_by_server[server_name] = await client.get_tools(server_name=server_name)

    # Pull out the tools that must be shared everywhere (e.g. get_filelist),
    # no matter which server they live on.
    shared_tools = []
    for server_name, tools in tools_by_server.items():
        for tool in tools:
            if tool.name in SHARED_TOOL_NAMES:
                shared_tools.append(tool)
    if shared_tools:
        print(f"  Shared across all agents (found on their home server, "
              f"reused everywhere): {[t.name for t in shared_tools]}")
    else:
        print("  WARNING: none of SHARED_TOOL_NAMES were found in any server's "
              "tool list — get_filelist (or equivalent) won't be available.")

    # Tools from servers that didn't match any kit keyword are ALSO treated
    # as shared utilities, in case your config.json adds a server this
    # script doesn't know about.
    for server_name in unmatched_servers:
        shared_tools.extend(tools_by_server[server_name])

    kit_tools = {}
    for kit, servers in kit_to_servers.items():
        tools = list(shared_tools)  # everyone gets the shared utilities (incl. get_filelist)
        for server_name in servers:
            tools.extend(tools_by_server[server_name])
        # de-dupe by tool name in case a kit's own server also hosts a
        # "shared" tool (e.g. Statistics hosting get_filelist itself)
        seen = set()
        deduped = []
        for tool in tools:
            if tool.name not in seen:
                seen.add(tool.name)
                deduped.append(tool)
        kit_tools[kit] = deduped

    return kit_tools, shared_tools


# ============================================================================
# Sub-agent construction + "agent as tool" wrapping
# ============================================================================

def response_to_trace(response):
    """Convert a sub-agent's raw message list into the same lightweight
    conversation_log structure used for the top-level log, so it can be
    merged in later."""
    trace = []
    for message in response.get("messages", []):
        if not hasattr(message, 'type'):
            continue
        if message.type == 'human':
            trace.append({"role": "user", "content": message.content})
        elif message.type == 'ai':
            content = []
            if message.content and message.content.strip():
                content.append({"type": "text", "content": message.content})
            if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                for tool_call in message.additional_kwargs['tool_calls']:
                    try:
                        arguments = json.loads(tool_call['function']['arguments']) \
                            if isinstance(tool_call['function']['arguments'], str) \
                            else tool_call['function']['arguments']
                    except Exception:
                        arguments = tool_call['function']['arguments']
                    content.append({"name": tool_call['function']['name'], "input": arguments})
            if content:
                trace.append({"role": "assistant", "content": content})
        elif message.type == 'tool':
            trace.append({
                "role": "tool",
                "name": message.name,
                "content": [{"output": [{"text": str(message.content)}]}]
            })
    return trace


DELEGATION_TOOL_PREFIX = "call_"
DELEGATION_TOOL_SUFFIX = "_kit"


def _is_delegation_wrapper(tool_name: str) -> bool:
    """True for orchestrator-level delegation tools (call_index_kit, etc.)
    -- these aren't real domain tools, they're just the plumbing that
    routes to a specialist. Downstream tooling (extraction/eval scripts
    written against the single-agent log format) expects only real tool
    calls, so these wrappers get dropped from the .log output."""
    return tool_name.startswith(DELEGATION_TOOL_PREFIX) and tool_name.endswith(DELEGATION_TOOL_SUFFIX)


def flatten_tool_calls(conversation_log):
    """Flatten a (possibly nested) multi-agent conversation_log into a
    single chronological list containing only real tool calls/results --
    matching the shape the single-agent script's .log file already has.

    Two things happen here:
    1. "specialist" entries (a kit's nested sub-trace) are recursed into
       and their contents spliced in at the point the delegation occurred,
       instead of being nested under a "specialist" wrapper.
    2. The delegation wrapper calls themselves (call_index_kit,
       call_analysis_kit, ...) and their results are dropped -- they are
       not real tools and would not match ground truth built against a
       single-agent tool sequence. The specialist's own real tool calls
       (get_filelist, compute_tvdi, mann_kendall_test, ...) take their
       place in the flattened sequence.

    User/text-only assistant turns pass through unchanged, same as the
    single-agent log."""
    flat = []
    for entry in conversation_log:
        role = entry.get("role")
        if role == "specialist":
            flat.extend(flatten_tool_calls(entry.get("content", [])))
        elif role == "assistant":
            kept = []
            for item in entry.get("content", []):
                if item.get("type") == "text":
                    kept.append(item)
                elif "name" in item:
                    if _is_delegation_wrapper(item["name"]):
                        continue  # drop the wrapper call itself
                    kept.append(item)
            if kept:
                flat.append({"role": "assistant", "content": kept})
        elif role == "tool":
            if _is_delegation_wrapper(entry.get("name", "")):
                continue  # drop the wrapper's result; real results already spliced in above
            flat.append(entry)
        else:
            flat.append(entry)
    return flat


def format_specialist_return(new_messages) -> str:
    """Format the messages generated during ONE delegation call (this turn's
    AI text, tool calls, and tool results) into a single string for the
    orchestrator. Deliberately close to a raw ReAct trace rather than a
    paraphrased summary, so if a tool call failed, the orchestrator sees the
    actual tool error text directly -- the same signal it would have seen
    itself in the single-agent version -- instead of the specialist's
    interpretation of what went wrong."""
    lines = []
    for message in new_messages:
        if not hasattr(message, 'type'):
            continue
        if message.type == 'ai':
            if message.content and message.content.strip():
                lines.append(message.content.strip())
            if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                for tool_call in message.additional_kwargs['tool_calls']:
                    lines.append(f"[called {tool_call['function']['name']}"
                                 f"({tool_call['function']['arguments']})]")
        elif message.type == 'tool':
            lines.append(f"[{message.name} returned] {message.content}")
    return "\n".join(lines) if lines else "(specialist made no tool calls or produced no output)"


def build_kit_subagents(llm, kit_tools: dict):
    """Create one ReAct agent per kit, each scoped to that kit's tools and
    given its specialist system prompt."""
    subagents = {}
    for kit, tools in kit_tools.items():
        spec = KIT_SPECS[kit]
        # Fold the specialist system prompt in as the leading message via
        # state_modifier / prompt kwarg supported by create_react_agent.
        subagents[kit] = create_react_agent(llm, tools, prompt=spec["system_prompt"])
        print(f"  Built '{kit}' sub-agent with {len(tools)} tools")
    return subagents


class KitDelegationInput(BaseModel):
    instructions: str = Field(
        description=(
            "Full, self-contained instructions for the specialist. Describe "
            "the goal (WHAT you need), not the method (HOW to do it) -- no "
            "formulas, algorithms, or tool names; the specialist knows its "
            "own tools. Must always include: the data directory path for "
            "this question, and the exact file paths relevant to this "
            "sub-task -- either filenames you discovered via get_filelist "
            "or paths a specialist previously reported back to you (e.g. "
            "'Result saved at ...'). Never invent, guess, or alter a path. "
            "If a path came from another specialist's earlier result, copy "
            "it in verbatim, since specialists cannot see each other's work."
        )
    )


def make_kit_tool(kit_name: str, subagent, trace_log_path) -> StructuredTool:
    spec = KIT_SPECS[kit_name]
    kit_callback = TraceCallbackHandler(trace_log_path, label=kit_name.upper())

    async def _run(instructions: str) -> str:
        # Reuse this kit's accumulated message history (if any) from earlier
        # delegations within the same question, so the specialist has real
        # memory: it can see what it already tried, what already failed,
        # and what files/results it already produced, instead of starting
        # from a blank slate every single delegation. What it does with
        # that memory (retry, skip, change approach) is left to the model,
        # same as the single-agent script left "retry only once" to the
        # model rather than enforcing it in code.
        history = _kit_message_history.setdefault(kit_name, [])
        pre_call_len = len(history)
        history.append(HumanMessage(content=instructions))

        write_trace(trace_log_path, f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"[ORCHESTRATOR] delegating to {kit_name.upper()}: {instructions}")

        response = await subagent.ainvoke(
            {"messages": history},
            config={"recursion_limit": 30, "max_execution_time": 180, "callbacks": [kit_callback]},
        )
        _kit_message_history[kit_name] = response["messages"]

        _current_subagent_traces.append({
            "kit": kit_name,
            "trace": response_to_trace(response),
        })

        # Return this turn's tool calls/results, not a paraphrase of them --
        # if a tool call failed, the orchestrator sees the actual error text.
        new_messages = response["messages"][pre_call_len:]
        return format_specialist_return(new_messages)

    return StructuredTool.from_function(
        name=f"call_{kit_name}_kit",
        description=(
            f"Delegate a sub-task to the {kit_name.upper()} specialist. "
            f"Covers {spec['description']}. Pass complete, self-contained "
            f"instructions -- this specialist remembers its OWN earlier "
            f"delegations within this question, but it does NOT see your "
            f"reasoning or other specialists' work unless you include it."
        ),
        args_schema=KitDelegationInput,
        coroutine=_run,
    )


async def create_multi_agent_system(llm, mcp_servers, trace_log_path):
    """Build the full supervisor + 5-specialist system and return the
    orchestrator agent plus the underlying MCP client (for cleanup)."""
    client = MultiServerMCPClient(mcp_servers)
    try:
        kit_tools, shared_tools = await load_tools_per_kit(client, mcp_servers)

        print("Building kit specialist sub-agents...")
        subagents = build_kit_subagents(llm, kit_tools)

        kit_tool_wrappers = [
            make_kit_tool(kit, agent, trace_log_path) for kit, agent in subagents.items()
        ]

        orchestrator_tools = kit_tool_wrappers# + shared_tools
        orchestrator = create_react_agent(llm, orchestrator_tools, prompt=ORCHESTRATOR_SYS_PROMPT)

        total_tools = sum(len(t) for t in kit_tools.values()) #+ len(shared_tools)
        print(f"Successfully loaded {total_tools} underlying MCP tools across "
              f"{len(subagents)} specialists + orchestrator "
              f"({len(orchestrator_tools)} orchestrator-level tools)")

        return orchestrator, client
    except Exception as e:
        print(f"Error creating multi-agent system: {e}")
        if hasattr(client, 'close'):
            await client.close()
        raise


# ============================================================================
# Questions
# ============================================================================

def load_questions(test_json_path: str = 'benchmark/question.json'):
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


# ============================================================================
# Question handling
# ============================================================================

async def handle_question(orchestrator, question, chat_log_path, trace_log_path):
    global _current_subagent_traces, _kit_message_history
    _current_subagent_traces = []
    _kit_message_history = {}

    try:
        query = question['auto'] + question['data'] if autoplanning else \
            question['instruct'] + question['data']

        if question['choices']:
            query += '\n'.join([''] + [
                '{}.{}'.format(chr(ord('A') + i), choice)
                for i, choice in enumerate(question['choices'])
            ])

        full_query = query  # ORCHESTRATOR_SYS_PROMPT is already bound via `prompt=`

        print(f"\n--- Processing Question {question['question_id']} ---")
        print(f"Query: {query[:200]}...")

        write_trace(trace_log_path, "\n" + "=" * 100 +
                    f"\nQuestion ID: {question['question_id']}    "
                    f"START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 100 +
                    f"\n[USER] {full_query}")

        user_message = {
            "name": "user",
            "role": "user",
            "content": full_query,
            "metadata": {"question_id": question['question_id']}
        }
        save_chat_message(chat_log_path, user_message)

        orch_callback = TraceCallbackHandler(trace_log_path, label="ORCHESTRATOR")
        response = await orchestrator.ainvoke(
            {"messages": [HumanMessage(content=full_query)]},
            config={"recursion_limit": 50, "max_execution_time": 600, "callbacks": [orch_callback]},
        )

        final_answer = extract_answer_from_response(response)

        # ---- Build conversation_log: orchestrator turns, with each
        # delegation call's specialist trajectory inlined right after it. ----
        conversation_log = []
        trace_iter = iter(_current_subagent_traces)

        for message in response.get("messages", []):
            if not hasattr(message, 'type'):
                continue
            if message.type == 'human':
                conversation_log.append({"role": "user", "content": message.content})
            elif message.type == 'ai':
                assistant_content = []
                if message.content and message.content.strip():
                    assistant_content.append({"type": "text", "content": message.content})
                delegated_this_turn = False
                if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                    for tool_call in message.additional_kwargs['tool_calls']:
                        try:
                            arguments = json.loads(tool_call['function']['arguments']) \
                                if isinstance(tool_call['function']['arguments'], str) \
                                else tool_call['function']['arguments']
                        except Exception:
                            arguments = tool_call['function']['arguments']
                        assistant_content.append({"name": tool_call['function']['name'], "input": arguments})
                        if tool_call['function']['name'].startswith('call_') and tool_call['function']['name'].endswith('_kit'):
                            delegated_this_turn = True
                if assistant_content:
                    conversation_log.append({"role": "assistant", "content": assistant_content})
                if delegated_this_turn:
                    sub = next(trace_iter, None)
                    if sub is not None:
                        conversation_log.append({
                            "role": "specialist",
                            "kit": sub["kit"],
                            "content": sub["trace"],
                        })
            elif message.type == 'tool':
                conversation_log.append({
                    "role": "tool",
                    "name": message.name,
                    "content": [{"output": [{"text": str(message.content)}]}]
                })

        # Any remaining specialist traces (e.g. parallel/edge cases) appended at the end
        for sub in trace_iter:
            conversation_log.append({"role": "specialist", "kit": sub["kit"], "content": sub["trace"]})

        # ---- .chat file (AgentScope-style) ----
        for message in response.get("messages", []):
            if not hasattr(message, 'type'):
                continue
            if message.type == 'human':
                continue
            elif message.type == 'ai':
                assistant_chat_content = []
                if message.content and message.content.strip():
                    assistant_chat_content.append({"type": "text", "text": message.content})
                if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                    for tool_call in message.additional_kwargs['tool_calls']:
                        try:
                            arguments = json.loads(tool_call['function']['arguments']) \
                                if isinstance(tool_call['function']['arguments'], str) \
                                else tool_call['function']['arguments']
                        except Exception:
                            arguments = tool_call['function']['arguments']
                        assistant_chat_content.append({
                            "type": "tool_use",
                            "id": tool_call['id'],
                            "name": tool_call['function']['name'],
                            "input": arguments
                        })
                if assistant_chat_content:
                    save_chat_message(chat_log_path, {
                        "name": question['question_id'],
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
                    "metadata": None
                })

        # Persist specialist trajectories to the .chat file too
        for sub in _current_subagent_traces:
            save_chat_message(chat_log_path, {
                "name": f"{sub['kit']}_specialist",
                "role": "specialist",
                "content": sub["trace"],
                "metadata": {"question_id": question['question_id'], "kit": sub["kit"]}
            })

        # The JSON .log gets a flattened, tool-calls-only view (matching
        # the single-agent log shape) so downstream extraction/eval
        # scripts see the real domain tool sequence regardless of which
        # specialist made each call. The readable .txt trace keeps the
        # nested "specialist" grouping since that's more useful to skim.
        flattened_log = flatten_tool_calls(conversation_log)
        logger.info("Chat Content", question['question_id'], flattened_log, final_answer)
        write_trace(trace_log_path, format_conversation_log_as_text(
            question['question_id'], conversation_log, final_answer
        ))

        print(f"Final Answer: {final_answer}")
        return final_answer

    except Exception as e:
        error_msg = f"Error processing question {question['question_id']}: {e}"
        print(error_msg)
        save_chat_message(chat_log_path, {
            "name": "system",
            "role": "system",
            "content": [{"type": "text", "content": error_msg}],
            "metadata": {"error": True, "question_id": question['question_id']}
        })
        logger.info(question['question_id'], [], error_msg)
        write_trace(trace_log_path, format_conversation_log_as_text(
            question['question_id'], [], error_msg
        ))
        return f"Error: {e}"


# ============================================================================
# Main
# ============================================================================

async def main():
    print("Initializing multi-agent (supervisor + 5 kit specialists) Earth Science Agent...")

    init_global_params()
    chat_log_path = init_chat_logger()
    trace_log_path = init_trace_logger()
    print(f"Chat log will be saved to: {chat_log_path}")
    print(f"Human-readable trace will be saved to: {trace_log_path}")

    llm, mcp_servers = load_langchain_config()
    orchestrator, client = await create_multi_agent_system(llm, mcp_servers, trace_log_path)

    try:
        questions = load_questions()[:10]
        if RETRY_IDS is not None:
            retry_set = set(RETRY_IDS)
            questions = [q for q in questions if q['question_id'] in retry_set]
        if BATCH_TOTAL > 1:
            questions = [q for i, q in enumerate(questions) if i % BATCH_TOTAL == BATCH_INDEX]
        print(f"Loaded {len(questions)} questions for evaluation"
              + (f" (batch {BATCH_INDEX+1}/{BATCH_TOTAL})" if BATCH_TOTAL > 1 else ""))

        results = []
        for question in tqdm(questions, desc="Processing questions"):
            answer = await handle_question(orchestrator, question, chat_log_path, trace_log_path)
            results.append({"question_id": question['question_id'], "answer": answer})

        results_path = temp_dir_path / "results_summary.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"\nEvaluation completed! Results saved to {results_path}")
        print(f"Detailed logs available at: {temp_dir_path}")
        print(f"Chat history saved to: {chat_log_path}")
        print(f"Human-readable trace saved to: {trace_log_path}")

    except Exception as e:
        print(f"Error in main evaluation: {e}")
        raise

    finally:
        if hasattr(client, 'close'):
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())