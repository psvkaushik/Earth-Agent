"""
agents/tracing.py

Cross-cutting tool-call tracing for the multi-agent system.

WHY THIS EXISTS: ADK's Runner.run_async() event stream only surfaces
top-level events for whichever agent is currently "in control" of that
stream. Our sub-agents are wired in as AgentTool-wrapped tools on the
orchestrator (this is the "fixed in code" mechanism that guarantees a
RESULT + control-return without depending on the model correctly calling
transfer_to_agent itself) — which means a sub-agent's OWN tool calls
(get_filelist, compute_tvdi, calculate_threshold_ratio, etc.) execute in a
fully nested invocation and never appear on the parent Runner's event
stream at all. Only the sub-agent's single collapsed final-text response
does, e.g. one function_call/function_response pair literally named
"index_agent" carrying its whole answer as one opaque string.

To get real tool-level visibility — which sub-agent called which actual
tool with what arguments — every agent (orchestrator AND every sub-agent)
must register the callbacks below. Each agent's own LlmAgent instance
invokes its own registered callbacks at the point IT executes a tool,
independent of which agent is "on top" of the call stack, so this reaches
tool calls no amount of event-stream filtering at the orchestrator level
ever could.

This module also implements the identical-repeated-call loop breaker:
before_tool_callback can return a value ADK will use AS the tool's result
instead of actually running it — so once the same (agent, tool, args)
signature has been seen too many times in one question, the next attempt
is short-circuited with an explicit "you're repeating yourself" message
instead of executing again.

VERIFY BEFORE RELYING ON THIS IN PRODUCTION: the callback parameter name
(before_tool_callback/after_tool_callback) and the exact attribute path
for pulling session_id/agent_name off the callback context vary across
google-adk versions. Everything below is defensive (try/except, several
fallback attribute paths) so a mismatch degrades to "skip this trace
event" rather than crashing a tool call or the whole run — but confirm
against your installed google-adk version's source/docs rather than
trusting this blind. A silent mismatch here means incomplete traces, not
a loud error, which is exactly the kind of failure worth checking for
once rather than discovering mid-run.
"""
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Tool names that are actually other sub-agents wrapped as AgentTool, not
# real MCP tools — excluded from the flattened conversation log so it
# shows the real underlying tool calls instead of this wrapper layer.
# Call register_subagent_names() once, right after ALL_SUB_AGENTS is built.
_SUBAGENT_TOOL_NAMES: set[str] = set()


def register_subagent_names(names) -> None:
    _SUBAGENT_TOOL_NAMES.clear()
    _SUBAGENT_TOOL_NAMES.update(names)


# --- Loop-breaking thresholds -------------------------------------------------
MAX_IDENTICAL_CALLS = 3   # same (agent, tool, args) seen more than this many times -> blocked


@dataclass
class QuestionTrace:
    """One instance per question/session. Register it into `active_traces`
    keyed by session_id before starting that session's run_async(), and
    remove it once the question is done (traces aren't bounded in size
    otherwise, across a long batch run)."""
    session_id: str
    fulltrace_path: Path
    conversation: list = field(default_factory=list)   # feeds the .log — flattened, real tools only
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _call_counts: dict = field(default_factory=dict)   # (agent, tool, args_json) -> count

    def _write_fulltrace_line(self, line: str) -> None:
        # Opened + flushed per line rather than kept open across the whole
        # run. Small perf cost, but guarantees the file reflects progress
        # in real time (readable with `tail -f` mid-run) and survives a
        # hard kill mid-question with everything up to that point intact.
        with self._lock:
            with open(self.fulltrace_path, 'a', encoding='utf-8') as f:
                f.write(line + "\n")

    def write_header(self, query: str) -> None:
        self._write_fulltrace_line(f"QUERY: {query}")

    def record_agent_switch(self, agent_name: str) -> None:
        self._write_fulltrace_line(f"\n>> agent: {agent_name}")

    def record_reasoning(self, agent_name: str, text: str) -> None:
        self._write_fulltrace_line(f"   [reasoning] ({agent_name}) {text}")

    def record_handoff(self, agent_name: str, target: str) -> None:
        self._write_fulltrace_line(f"   [handoff] {agent_name} -> {target}")

    def check_and_count_call(self, agent_name: str, tool_name: str, args: dict) -> Optional[dict]:
        """Returns a canned error dict to short-circuit the call if this
        exact (agent, tool, args) signature has already been seen more
        than MAX_IDENTICAL_CALLS times; otherwise returns None (proceed
        normally) and increments the count regardless."""
        key = (agent_name, tool_name, json.dumps(args, sort_keys=True, default=str))
        with self._lock:
            count = self._call_counts.get(key, 0) + 1
            self._call_counts[key] = count
        if count > MAX_IDENTICAL_CALLS:
            msg = (f"LOOP DETECTED: {tool_name} has already been called with these exact "
                   f"arguments {count - 1} times by {agent_name}. This call was blocked — "
                   f"do not repeat it again. Check whether a previous tool result already "
                   f"answered this, or report the failure and stop instead of retrying.")
            self._write_fulltrace_line(f"   [LOOP BLOCKED] {msg}")
            return {"error": msg}
        return None

    def record_call(self, agent_name: str, tool_name: str, args: dict) -> None:
        self._write_fulltrace_line(
            f"   [tool call] ({agent_name}) {tool_name}({json.dumps(args, default=str)[:200]})"
        )
        if tool_name not in _SUBAGENT_TOOL_NAMES:
            with self._lock:
                self.conversation.append({
                    "role": "assistant",
                    "content": [{"name": tool_name, "input": args}],
                })

    def record_result(self, agent_name: str, tool_name: str, response: Any) -> None:
        resp_str = json.dumps(response, default=str)
        self._write_fulltrace_line(
            f"   [tool result] ({agent_name}) {tool_name} -> {resp_str[:400]}"
        )
        if tool_name not in _SUBAGENT_TOOL_NAMES:
            with self._lock:
                self.conversation.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": [{"output": [{"text": resp_str}]}],
                })

    def record_final(self, text: str) -> None:
        self._write_fulltrace_line(f"\nFINAL ANSWER: {text}")
        with self._lock:
            self.conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "content": text}],
            })


# session_id -> QuestionTrace, populated by the benchmark/CLI driver right
# before each session starts, removed once that question finishes.
active_traces: dict[str, QuestionTrace] = {}


def _extract_session_id(tool_context) -> Optional[str]:
    """Best-effort extraction across a few plausible ADK attribute paths.
    Returns None (never raises) if none match — callers must treat that
    as 'skip this trace event', not an error."""
    for accessor in (
        lambda c: c.session.id,
        lambda c: c._invocation_context.session.id,
        lambda c: c.invocation_context.session.id,
    ):
        try:
            sid = accessor(tool_context)
            if sid:
                return sid
        except Exception:
            continue
    return None


def _extract_agent_name(tool_context) -> str:
    for accessor in (
        lambda c: c.agent_name,
        lambda c: c._invocation_context.agent.name,
        lambda c: c.invocation_context.agent.name,
    ):
        try:
            name = accessor(tool_context)
            if name:
                return name
        except Exception:
            continue
    return "?"


def _tool_name(tool) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", None) or str(tool)


def before_tool_callback(tool, args, tool_context):
    """Register on EVERY agent (orchestrator + every sub-agent). Returning
    None means 'proceed with the tool call normally'; returning a dict
    means 'use this as the tool's result instead of actually running it'
    — used here to block a call that's already repeated past
    MAX_IDENTICAL_CALLS."""
    try:
        session_id = _extract_session_id(tool_context)
        trace = active_traces.get(session_id) if session_id else None
        if trace is None:
            return None

        agent_name = _extract_agent_name(tool_context)
        tool_name = _tool_name(tool)
        args_dict = dict(args) if args else {}

        block = trace.check_and_count_call(agent_name, tool_name, args_dict)
        if block is not None:
            return block  # short-circuits the actual tool execution

        trace.record_call(agent_name, tool_name, args_dict)
    except Exception:
        logger.debug("tracing.before_tool_callback failed", exc_info=True)
    return None


def after_tool_callback(tool, args, tool_context, tool_response):
    """Register on EVERY agent (orchestrator + every sub-agent). Returning
    None means 'use the tool's actual response unmodified' — this
    callback only observes."""
    try:
        session_id = _extract_session_id(tool_context)
        trace = active_traces.get(session_id) if session_id else None
        if trace is None:
            return None
        trace.record_result(_extract_agent_name(tool_context), _tool_name(tool), tool_response)
    except Exception:
        logger.debug("tracing.after_tool_callback failed", exc_info=True)
    return None