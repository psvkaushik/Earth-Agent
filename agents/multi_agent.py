"""
agents/multi_agent.py

Boots all sub-agents, wires them under a single orchestrator, and runs a
query through it while streaming out everything observable from the event
stream: which agent is active, any reasoning/planning text it emits, every
tool call + its result, sub-agent handoffs, escalations, and a final
token-usage / call-count summary.

Run from the project root:
    python -m agents.multi_agent
or directly (bootstrap below handles sys.path either way):
    python agents/multi_agent.py
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
# This file lives at Earth-Agent/agents/multi_agent.py. Running it directly
# only adds agents/ to sys.path, not its parent — insert the project root
# explicitly so `agents.subagents` resolves regardless of invocation style.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.subagents import ALL_SUB_AGENTS, llm
from agents.subagents.eve_patch import apply as apply_eve_patch

load_dotenv()
apply_eve_patch()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "EARTH-MULTI-AGENT"
USER_ID = "user_1"


# --- Orchestrator ------------------------------------------------------------

if not ALL_SUB_AGENTS:
    raise SystemExit("No sub-agents loaded — check the warnings/errors above.")

print("\nSub-agents ready:")
for agent in ALL_SUB_AGENTS:
    print(f"  - {agent.name}: {agent.description}")

_SUPERVISOR_SYSTEM_PROMPT = """\
You are the orchestrator_agent, and the orchestrator for a multi-agent Earth-\
observation system. You are given multiple-choice questions about Earth \
observation data analysis. You do not have perception/raster-analysis \
tools yourself — route each question to the sub-agent(s) best suited to \
it based on its content, pass along whatever context each sub-agent \
needs, and combine their results to determine the correct choice. Make sure to also extract the correct directory path from the question,and sub-agent responses and pass it to the sub-agents, rather than asking them to enumerate files themselves.

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
3. When a sub-agent reports a result like "Result saved at /path/to/file", \
   you must reuse that exact full path "/path/to/file" in any subsequent \
   tool or sub-agent calls that need it — do not shorten, guess, or \
   reconstruct the path yourself.
4. Three sub-agents are available: perception_agent (object/scene detection, \
   segmentation), index_agent (spectral/biophysical index rasters — NDVI, \
   EVI, NBR, NDWI, TVDI, etc. — from raw band inputs), and statistics_agent \
   (numeric/statistical computation on any raster, including rasters that \
   index_agent produced). index_agent only returns computed raster paths — \
   it cannot answer numeric questions itself. If a question asks for a \
   statistic *about* an index (a mean NDVI, a percentage of high-EVI area, \
   an index threshold count, etc.), this requires two sequential sub-agent \
   calls: first index_agent to produce the index raster, then \
   statistics_agent with that exact returned path to compute the requested \
   statistic. Do not call statistics_agent directly on raw band files \
   expecting it to compute an index — it has no index-calculation tools.
5. Every sub-agent carries its own `get_filelist` tool and can resolve a \
   directory path itself. Pass sub-agents a directory path, not an \
   enumerated file list, unless you already need to reference specific \
   filenames individually (e.g. matching a pre/post pair by name).
6. For every question you must commit to the single choice you think is \
   most appropriate. Do not hedge or list multiple candidates.
7. Your response must contain exactly two lines and nothing else:
Line 1: computed value: Y coincides with choice X. <value Y>
Line 2: <Answer>X</Answer>
Do not add any other explanation, caveats, or commentary beyond these \
two lines. In case of ambiguity, pick the choice most consistent with \
the computed value and the question's context — still output only the \
two lines above. In case of error, output only: <Answer>NO Answer</Answer>
8. If a tool call fails, do not immediately retry the exact same call with \
   the exact same arguments — this wastes calls and will fail identically. \
   First check whether a *previous* tool result already gave you the \
   correct input (e.g. a full saved path) that you failed to use verbatim \
   in the failing call. If a retry with corrected arguments also fails, \
   stop and report the failure rather than repeating the sequence again.
"""

orchestrator = Agent(
    model=llm,
    name="orchestrator_agent",
    sub_agents=ALL_SUB_AGENTS,
    description="An orchestrator agent that routes user queries to its sub-agents based on the query content.",
    instruction=_SUPERVISOR_SYSTEM_PROMPT,
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


# def _first_text(event) -> str | None:
#     if not (event.content and event.content.parts):
#         return None
#     for part in event.content.parts:
#         if getattr(part, "text", None):
#             return part.text
#     return None
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
    step live, and returns a RunTrace with the full structured record."""
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
    # trace = await trace_query(
    #     "Based on the following images, every image belongs to {Airport, BareLand, BaseballField, Beach, Bridge, Center, Church, Commercial, DenseResidential, Desert, Farmland, Forest, Industrial, Meadow, MediumResidential, Mountain, Park, Parking, Playground, Pond, Port, RailwayStation, Resort, River, School, SparseResidential, Square, Stadium, StorageTanks, Viaduct}, determine the number of images captured in park areas. benchmark/data/question190, A.2, B.3, C.5, D.6"
    # )
    # trace = await trace_query(
    #     "Define a threshold of significant increase as 20 MW. Based on fire MaxFRP in Thailand from 2018-03-01 to 2018-03-30 and from 2018-08-01 to 2018-08-30, identify and map regions where fire intensity significantly increased and visulize these areas in the map, data_dir: benchmark/data/question181," \
    #     "A.The northern highlands exhibited a significant increase in fire intensity, with 23 pixels surpassing the +20 MW threshold."
    #         "B. The central plains showed no areas with a fire intensity increase greater than 20 MW."
    #         "C. The southern peninsula had more than 100 pixels with a MaxFRP increase above 20 MW."
    #         "D. The eastern coastal region saw 5 pixels exceed the +20 MW increase threshold."
    #         "E. The entire country showed no regions with a MaxFRP increase greater than 20 MW.")

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
