"""
updated_agents/multi_agent.py

A parallel copy of agents/multi_agent.py that fixes the file-path relay
problem the original architecture has: the orchestrator prompt already asked
the model to "extract the correct directory path... and pass it to the
sub-agents" and to "reuse that exact full path... do not shorten, guess, or
reconstruct" — and it still breaks under a long multi-agent transcript (see
updated_agents/subagents/state_tracking.py's docstring for a concrete traced
example). This package replaces that prompt-only relay with a code-enforced
one: a shared session-state path registry (state_tracking.py), populated by
an after_tool_callback on every sub-agent and surfaced to every agent via a
dynamic instruction that's re-injected fresh on every turn. agents/ is left
untouched so the two can be run side by side for comparison.

Run from the project root:
    python -m updated_agents.multi_agent
or directly (bootstrap below handles sys.path either way):
    python updated_agents/multi_agent.py
"""
import asyncio
import json
import logging
import os
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# --- sys.path bootstrap -----------------------------------------------------
# This file lives at Earth-Agent/updated_agents/multi_agent.py. Running it
# directly only adds updated_agents/ to sys.path, not its parent — insert the
# project root explicitly so `updated_agents.subagents` resolves regardless
# of invocation style.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from updated_agents.subagents import ALL_SUB_AGENTS, llm
from updated_agents.subagents.eve_patch import apply as apply_eve_patch
from updated_agents.subagents.state_tracking import seed_question_dir, with_known_paths
from updated_agents import tracing

load_dotenv()
apply_eve_patch()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "EARTH-MULTI-AGENT-UPDATED"
USER_ID = "user_1"


# --- Orchestrator ------------------------------------------------------------

if not ALL_SUB_AGENTS:
    raise SystemExit("No sub-agents loaded — check the warnings/errors above.")

# Tells tracing.py which "tool" names are actually other sub-agents wrapped
# as AgentTool, not real MCP tools — so a sub-agent handoff (e.g. the
# orchestrator invoking "index_agent") is written to the human-readable
# fulltrace but excluded from the flattened, metrics-compatible conversation
# log, matching how the original single-agent traces (and agents/'s intended
# design, per tracing.py's own docstring) never logged an agent call as if
# it were a tool call. Must run before any question is processed.
tracing.register_subagent_names(agent.name for agent in ALL_SUB_AGENTS)

print("\nSub-agents ready:")
for agent in ALL_SUB_AGENTS:
    print(f"  - {agent.name}: {agent.description}")

_SUPERVISOR_SYSTEM_PROMPT = """\
You are the orchestrator_agent, and the orchestrator for a multi-agent Earth-\
observation system. You are given multiple-choice questions about Earth \
observation data analysis. You do not have perception/raster-analysis \
tools yourself — route each question to the sub-agent(s) best suited to \
it based on its content, pass along whatever context each sub-agent \
needs, and combine their results to determine the correct choice.

A "## Known state" section is appended below this prompt automatically on \
every turn, showing the question's data directory (auto-extracted from the \
original question, you don't need to re-derive it) and every file path any \
sub-agent has produced so far this run (auto-recorded from actual tool \
results, not from your memory of earlier turns). Sub-agents get this same \
block injected into their own prompts, so you do NOT need to retype exact \
paths into your hand-off instructions for them to have access to them — but \
still tell a sub-agent which directory/prior result its subtask concerns, \
since the block can grow long and it shouldn't have to guess which entry is \
relevant. If you ever need to quote a path yourself (e.g. summarizing what \
was produced), copy it from the "## Known state" block, never from your own \
recollection of an earlier tool result.

Division of labor: a sub-agent's job is to compute the underlying fact \
(a count, a class, a measurement) and state it directly. Your job is to \
take that already-computed fact and match it against the answer choices \
in the original question to pick the letter/option. Do not re-invoke a \
sub-agent to "get the final answer" — once a sub-agent has given you a \
result, that value IS the fact; your remaining work is to match it \
against the answer choices you were given, not to ask for more \
computation.

ATTENTION:
1. If a sub-agent or tool call returns an error, you may retry that call \
   at most once. If it errors again, stop retrying it and work with \
   whatever information you already have.
2. If a sub-agent hands control back with an empty or null result (no \
   text findings), that is not the same as an error — it means the \
   sub-agent finished its tool calls but never wrote up its answer. \
   Re-invoke it once with an explicit instruction to state the findings \
   it already gathered as its final text answer, rather than repeating \
   the original request verbatim.
3. Four sub-agents are available: perception_agent (object/scene detection, \
   segmentation), index_agent (spectral/biophysical index rasters — NDVI, \
   EVI, NBR, NDWI, TVDI, etc. — from raw band inputs), inversion_agent \
   (geophysical-parameter inversion — LST, PWV, soil moisture, thermal \
   inertia, etc. — from raw band inputs), analysis_agent (time-series trend/\
   change-point/seasonality analysis and spatial statistics), and \
   statistics_agent (numeric/statistical computation on any raster, \
   including rasters that index_agent or inversion_agent produced). \
   index_agent and inversion_agent only return computed raster paths — they \
   cannot answer numeric questions themselves. If a question asks for a \
   statistic *about* an index or inverted parameter (a mean NDVI, a \
   percentage of high-EVI area, a mean LST, etc.), this requires two \
   sequential sub-agent calls: first index_agent/inversion_agent to produce \
   the raster, then statistics_agent with that exact path (from the \
   "## Known state" block, not retyped) to compute the requested statistic. \
   Do not call statistics_agent directly on raw band files expecting it to \
   compute an index or inverted parameter — it has neither capability.
4. Every sub-agent carries its own `get_filelist` tool and can resolve a \
   directory path itself. Pass sub-agents a directory path, not an \
   enumerated file list, unless you already need to reference specific \
   filenames individually (e.g. matching a pre/post pair by name).
5. For every question you must commit to the single choice you think is \
   most appropriate. Do not hedge or list multiple candidates.
6. Your response must contain exactly two lines and nothing else:
Line 1: computed value: Y coincides with choice X. <value Y>
Line 2: <Answer>X</Answer>
Do not add any other explanation, caveats, or commentary beyond these \
two lines. In case of ambiguity, pick the choice most consistent with \
the computed value and the question's context — still output only the \
two lines above. In case of error, output only: <Answer>NO Answer</Answer>
7. If a tool call fails, do not immediately retry the exact same call with \
   the exact same arguments — this wastes calls and will fail identically. \
   Check the "## Known state" block for a path you may have failed to use \
   verbatim in the failing call. If a retry with corrected arguments also \
   fails, stop and report the failure rather than repeating the sequence \
   again.
"""

orchestrator = Agent(
    model=llm,
    name="orchestrator_agent",
    sub_agents=ALL_SUB_AGENTS,
    description="An orchestrator agent that routes user queries to its sub-agents based on the query content.",
    instruction=with_known_paths(_SUPERVISOR_SYSTEM_PROMPT),
    before_agent_callback=seed_question_dir,
    # Registered on every agent (orchestrator + all sub-agents, see each
    # sub_agent_*.py) — see tracing.py's own docstring for why: ADK's event
    # stream only ever shows a sub-agent's single collapsed AgentTool
    # call/response at the orchestrator level, never the real tool calls it
    # made internally. before_tool_callback also enforces the identical-call
    # loop guard (tracing.MAX_IDENTICAL_CALLS) at the orchestrator's own
    # level (e.g. repeatedly re-invoking the same sub-agent with identical
    # args).
    before_tool_callback=tracing.before_tool_callback,
    after_tool_callback=tracing.after_tool_callback,
)

session_service = InMemorySessionService()
runner = Runner(agent=orchestrator, app_name=APP_NAME, session_service=session_service)


# --- Tracing ------------------------------------------------------------------

@dataclass
class RunTrace:
    """Structured record of one runner.run_async() pass, built up as events
    stream in. Kept separate from the live printing so callers that don't
    want console noise can still get the full trace back programmatically."""
    query: str
    final_answer: str = "Agent did not produce a final response."
    agent_sequence: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)     # every tool/agent-transfer call
    reasoning: list[dict] = field(default_factory=list)  # non-final text per agent
    input_tokens: int = 0
    output_tokens: int = 0

    def as_summary(self) -> dict:
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "agent_sequence": self.agent_sequence,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def _all_text(event) -> str | None:
    if not (event.content and event.content.parts):
        return None
    texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
    return "\n".join(texts) if texts else None

def _truncate(obj, limit=400) -> str:
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s if len(s) <= limit else s[:limit] + f"... [+{len(s) - limit} chars truncated]"


async def trace_query(query: str, session_id: str | None = None) -> RunTrace:
    """Streams a query through the orchestrator, printing every observable
    step live, and returns a RunTrace with the full structured record.

    NOTE: this reads function_call/function_response straight off the
    top-level event stream, which — same limitation as the original
    agents/multi_agent.py demo runner — only ever shows each sub-agent's own
    single collapsed AgentTool call/response, not the real tool calls made
    INSIDE that sub-agent (get_filelist, compute_tvdi, etc.). It's fine for
    a quick manual/interactive query where you mainly want to see agent
    hand-offs and reasoning. For a real full trace with every underlying
    tool call captured (what the benchmark run needs), use
    updated_agents/benchmark_ma.py's run_one_question, which drives the same
    orchestrator/runner but records via tracing.QuestionTrace instead —
    registered on every agent, tracing.before/after_tool_callback see inside
    a sub-agent's own execution regardless of who's "on top" of the call
    stack, which no amount of top-level event-stream reading here ever
    could. This function's own tracing.* callbacks (registered on the
    orchestrator/sub-agents in this same module) simply no-op for a
    trace_query() session, since no tracing.QuestionTrace is registered for
    it in tracing.active_traces.
    """
    session_id = session_id or f"session_{uuid.uuid4().hex}"
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    trace = RunTrace(query=query)
    content = types.Content(role="user", parts=[types.Part(text=query)])

    print(f"\n{'#' * 90}\nQUERY: {query}\n{'#' * 90}")

    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
        author = event.author or "?"
        if not trace.agent_sequence or trace.agent_sequence[-1] != author:
            trace.agent_sequence.append(author)
            print(f"\n>> agent: {author}")

        if event.usage_metadata:
            trace.input_tokens += event.usage_metadata.prompt_token_count or 0
            trace.output_tokens += event.usage_metadata.candidates_token_count or 0

        if event.content and event.content.parts:
            for part in event.content.parts:
                text = getattr(part, "text", None)
                fc = getattr(part, "function_call", None)
                fr = getattr(part, "function_response", None)

                if text and not event.is_final_response():
                    # Intermediate text = the model "thinking out loud" before
                    # acting — the closest thing to visible reasoning we get.
                    print(f"   [reasoning] {text}")
                    trace.reasoning.append({"agent": author, "text": text})

                if fc:
                    args = dict(fc.args) if fc.args else {}
                    print(f"   [tool call] {fc.name}({_truncate(args, 200)})")
                    trace.calls.append({"agent": author, "type": "call", "name": fc.name, "args": args})

                if fr:
                    print(f"   [tool result] {fr.name} -> {_truncate(fr.response)}")
                    trace.calls.append({"agent": author, "type": "result", "name": fr.name, "response": fr.response})

        if event.actions:
            handoff = getattr(event.actions, "transfer_to_agent", None)
            if handoff:
                print(f"   [handoff] -> {handoff}")
                trace.calls.append({"agent": author, "type": "handoff", "target": handoff})
            if getattr(event.actions, "escalate", None):
                print(f"   [escalate] {event.error_message}")

        if event.is_final_response():
            text = _all_text(event)
            if text is not None:
                trace.final_answer = text
            elif event.actions and event.actions.escalate:
                trace.final_answer = f"Agent escalated: {event.error_message or 'No specific message.'}"

    print(f"\n{'-' * 90}\nFINAL ANSWER: {trace.final_answer}\n{'-' * 90}")

    counts = Counter((c["agent"], c["type"], c["name"]) for c in trace.calls if "name" in c)
    if counts:
        print("\nCALL SUMMARY:")
        for (agent, kind, name), n in counts.items():
            print(f"  {agent:>20} {kind:<8} {name:<25} x{n}")
    print(f"\nTOKENS: in={trace.input_tokens} out={trace.output_tokens}\n")

    return trace


async def main():
    trace = await trace_query(
        """Based on temperature and vegetation data (NDVI and LST) from the agricultural region near Urumqi, Xinjiang between 2019 and 2023,  first apply the Temperature-Vegetation Dryness Index (TVDI) method by constructing a scatter plot of NDVI versus LST for each day, and calculate the TVDI value for each pixel to reflect the dryness condition and then calculate the annual average of TVDI and perform linear analysis on the annual average value data to best describes the annual trend. benchmark/data/question1
        A. Increasing dryness at 0.015 per year,
        B. Decreasing dryness at 0.037 per year,
        C. Decreasing dryness at 0.006 per year",
        D. No significant trend observed""")
    # print(json.dumps(trace.as_summary(), indent=2, default=str))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("Run failed")
        print(f"An error occurred: {e}")
