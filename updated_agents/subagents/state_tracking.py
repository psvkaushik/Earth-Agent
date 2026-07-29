"""
Shared session-state tracking for the multi-agent system.

WHY THIS EXISTS:
The original agents/ package asks the orchestrator (and every sub-agent) to
remember and retype file paths correctly across a long, noisy multi-agent
transcript — e.g. "Result saved at /abs/path/X.tif" produced by index_agent
in turn 3, needed verbatim by statistics_agent in turn 9. That's an LLM
*memory* problem, and it fails in practice: one traced run showed
statistics_agent guessing three different filenames for the same TVDI raster
(`tvdi_raster.tif`, `tvdi_2023-05-15.tif`, `question4_tvdi.tif`) — none of
them the file that was actually written — before a loop guard cut it off.

FIX: track known paths in ADK's per-session `state` dict instead, populated
deterministically by an `after_tool_callback` (regex over the literal tool
output, not model memory), and surface it to every agent via a dynamic
`instruction` callable so it's re-injected fresh on every single turn. This
can't be forgotten or mistyped because the model never has to retype it —
it's just always there, code-populated, in the current prompt.

USAGE:
    from .state_tracking import combined_after_tool_callback, with_known_paths, seed_question_dir

    instruction=with_known_paths(_SOME_AGENT_PROMPT)     # instead of a plain string
    after_tool_callback=combined_after_tool_callback     # on every sub-agent
    before_agent_callback=seed_question_dir              # on the orchestrator only
"""
import json
import re

from .. import tracing

# Excludes quotes/brackets/commas, not just whitespace: this pattern is also
# matched against raw (un-parsed) JSON-encoded text — e.g. the response
# arrives as `"...Result saved at /a/b.tif\"}], ..."` — where the path is
# followed immediately by a closing quote with no whitespace before it. A
# bare \S+ would swallow that trailing JSON syntax into the "path".
_SAVE_PATTERN = re.compile(r'Result saved at ([^\s",\]}]+)')
_QUESTION_DIR_PATTERN = re.compile(r'(benchmark/data/question\d+)')

_MAX_KNOWN_PATHS_SHOWN = 30


def _iter_strings(obj):
    """Yield every string leaf in an arbitrarily nested dict/list/str
    structure, additionally descending into any leaf that itself parses as
    JSON (MCP tool responses sometimes arrive as a dict, sometimes as a
    JSON-encoded string blob — this handles either shape without needing to
    know which one a given tool/transport produces)."""
    if isinstance(obj, str):
        yield obj
        stripped = obj.strip()
        if stripped[:1] in "{[":
            try:
                yield from _iter_strings(json.loads(stripped))
            except (json.JSONDecodeError, ValueError):
                pass
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def extract_saved_paths(obj) -> list[str]:
    """Pull every "Result saved at <path>" path out of a tool response,
    de-duplicated and in first-seen order."""
    found = []
    for s in _iter_strings(obj):
        found.extend(_SAVE_PATTERN.findall(s))
    seen = set()
    out = []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def after_tool_callback(tool, args, tool_context, tool_response):
    """Records every newly produced file path from a tool result into
    shared session state (`state["known_paths"]`), and — as a fallback —
    the directory argument of a `get_filelist` call into
    `state["question_dir"]` if nothing has set it yet.

    Attach to every sub-agent (`after_tool_callback=after_tool_callback`).
    Always returns None: this only observes tool traffic, it never rewrites
    the actual result the agent sees.
    """
    agent_name = getattr(tool_context, "agent_name", "?")
    tool_name = getattr(tool, "name", str(tool))

    new_paths = extract_saved_paths(tool_response)
    if new_paths:
        known = list(tool_context.state.get("known_paths", []))
        existing = {entry["path"] for entry in known}
        for path in new_paths:
            if path not in existing:
                known.append({"path": path, "agent": agent_name, "tool": tool_name})
                existing.add(path)
        tool_context.state["known_paths"] = known

    if tool_name == "get_filelist" and not tool_context.state.get("question_dir"):
        dir_path = args.get("dir_path") if isinstance(args, dict) else None
        if dir_path:
            tool_context.state["question_dir"] = dir_path

    return None


def combined_after_tool_callback(tool, args, tool_context, tool_response):
    """Runs both `tracing.after_tool_callback` (real tool-call capture for
    the benchmark's .log/fulltrace — see updated_agents/tracing.py) and this
    module's own `after_tool_callback` (path-registry state), as a single
    explicit function.

    Deliberately NOT relying on passing `after_tool_callback=[fn1, fn2]` as
    a list to ADK's LlmAgent — that's accepted by the field's type
    signature, but whether the runtime actually invokes every callback in
    the list (vs. e.g. stopping at the first non-None return) wasn't worth
    the time to verify against this ADK version when a single explicit
    function removes the ambiguity entirely. Attach this to every
    sub-agent: `after_tool_callback=combined_after_tool_callback`.
    """
    tracing.after_tool_callback(tool, args, tool_context, tool_response)
    after_tool_callback(tool, args, tool_context, tool_response)
    return None


def seed_question_dir(callback_context):
    """Orchestrator-only `before_agent_callback`: parses the *original* user
    query text for a `benchmark/data/questionN` directory once, on the first
    turn, and seeds it into shared session state — so every sub-agent can be
    told the exact question directory directly via `with_known_paths`
    instead of depending on the orchestrator LLM noticing and relaying it
    correctly on every handoff. Idempotent: a no-op once state already has
    a question_dir. Never blocks or alters agent execution (always returns
    None).
    """
    if callback_context.state.get("question_dir"):
        return None

    content = getattr(callback_context, "user_content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return None

    text = "\n".join(p.text for p in parts if getattr(p, "text", None))
    match = _QUESTION_DIR_PATTERN.search(text)
    if match:
        callback_context.state["question_dir"] = match.group(1)
    return None


def _known_state_block(state) -> str:
    lines = []

    question_dir = state.get("question_dir")
    if question_dir:
        lines.append(f"Question data directory: {question_dir}")

    known = state.get("known_paths", [])
    if known:
        if lines:
            lines.append("")
        lines.append(
            "Files already produced so far this run — reuse these exact "
            "paths verbatim. Never retype, shorten, reconstruct, or guess a "
            "filename; if the file you need isn't listed here, call "
            "get_filelist to check rather than assuming a name:"
        )
        for entry in known[-_MAX_KNOWN_PATHS_SHOWN:]:
            lines.append(f"- [{entry['agent']}/{entry['tool']}] {entry['path']}")

    if not lines:
        return ""
    return "\n\n## Known state (auto-populated from actual tool results — always trust this over your own memory of earlier turns)\n" + "\n".join(lines)


def with_known_paths(base_prompt: str):
    """Wrap a static prompt string into an ADK dynamic-instruction callable
    (`Callable[[ReadonlyContext], str]`) that appends a live "known state"
    block — question directory + every file path produced so far — built
    fresh from shared session state on *every* turn. Use in place of a plain
    string `instruction=`:

        instruction=with_known_paths(_SOME_AGENT_PROMPT)
    """
    def _instruction(ctx) -> str:
        return base_prompt + _known_state_block(ctx.state)
    return _instruction
