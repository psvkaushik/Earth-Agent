"""
updated_agents/benchmark_ma.py

Batch-evaluation runner for the updated_agents orchestrator — a straight
mirror of agents/benchmark_ma.py, pointed at updated_agents.multi_agent
instead of agents.multi_agent, so its output is directly comparable to the
original agents/ benchmark run under the same evaluate/*.py tooling.

Two logs are written per run, same shape as agents/benchmark_ma.py:
  - <model>_<AP|IF>_multiagent.log
        One JSON record per question (question_index, timestamp,
        conversations, final_answer). "conversations" is the flat sequence
        of REAL underlying tool calls (get_filelist, compute_tvdi,
        calculate_threshold_ratio, etc.) made by ANY agent — orchestrator or
        sub-agent — captured via updated_agents/tracing.py's before/after
        tool callbacks (registered on every agent in multi_agent.py), since
        sub-agents are wired in as AgentTools and their internal tool calls
        never appear on the top-level Runner event stream otherwise. The
        opaque per-sub-agent wrapper call itself (e.g. a single
        "index_agent" call/response) is excluded, same as agents/'s version.
  - <model>_<AP|IF>_multiagent_fulltrace.txt
        The COMPLETE run: every agent's reasoning, every real tool call and
        result, every handoff — written incrementally, per call.

results_summary.json: a flat list of {"question_id": ..., "answer": ...}.

Run from the project root:
    python -m updated_agents.benchmark_ma
"""
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from tqdm import tqdm

# --- sys.path bootstrap, same convention as updated_agents/multi_agent.py ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from google.genai import types

# Reuse the fully-wired orchestrator/runner/session_service from
# multi_agent.py rather than re-constructing them here — importing this
# module runs multi_agent.py's top-level setup (sub-agent bootstrap,
# tracing.register_subagent_names, Runner/InMemorySessionService
# construction, the "Sub-agents ready:" printout) but does NOT execute its
# `if __name__ == "__main__"` demo query, so this is safe to import as a
# library.
from updated_agents.multi_agent import runner, session_service, APP_NAME, USER_ID
from updated_agents import tracing

logger = logging.getLogger(__name__)

# --- Configuration ------------------------------------------------------------
MODEL_NAME = "multiagent_updated"    # distinct from agents/benchmark_ma.py's
                                      # "multiagent_mistral" so a comparison
                                      # run's output never collides with (or
                                      # gets confused for) the baseline run's
AUTOPLANNING = True                  # mirrors the original script's `autoplanning` flag
QUESTION_SLICE = slice(0, None)      # e.g. slice(34, 188) to match the original run's range
PER_QUESTION_TIMEOUT_S = 600         # mirrors the original's max_execution_time — the
                                      # outer safety net for any kind of hang
INTER_QUESTION_DELAY_S = 1           # mirrors the original's rate-limit delay

# NOTE: identical-repeated-tool-call loop breaking lives in
# updated_agents/tracing.py (QuestionTrace.check_and_count_call +
# before_tool_callback's ability to short-circuit a call), not here — that
# catches a stuck loop at the actual point of tool invocation, inside
# whichever agent is looping, rather than needing to observe it after the
# fact from the top-level event stream.


temp_dir_path: Path | None = None
json_logger: logging.Logger | None = None
fulltrace_path: Path | None = None


def init_output_paths() -> Path:
    """Set up the output directory and both log files, mirroring
    agents/benchmark_ma.py's init_output_paths."""
    global temp_dir_path, json_logger, fulltrace_path

    temp_dir_path = Path('./evaluate_multiagent/{}_{}_{}'.format(
        MODEL_NAME,
        'AP' if AUTOPLANNING else 'IF',
        datetime.now().strftime('%y-%m-%d_%H-%M')
    )).absolute()
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            # Same field mapping as agents/benchmark_ma.py — this is what
            # makes the file eval-tool-compatible.
            log_record = {
                "question_index": record.args[0] if record.args else "unknown",
                "timestamp": self.formatTime(record, self.datefmt),
                "conversations": record.args[1] if len(record.args) > 1 else [],
                "final_answer": record.args[2] if len(record.args) > 2 else None,
            }
            return json.dumps(log_record, ensure_ascii=False, indent=4)

    json_logger = logging.getLogger("multiagent_updated_json_logger")
    json_logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        temp_dir_path / "{}_{}_multiagent.log".format(MODEL_NAME, 'AP' if AUTOPLANNING else 'IF')
    )
    handler.setFormatter(JsonFormatter())
    json_logger.addHandler(handler)
    json_logger.propagate = False  # don't also dump these JSON blobs through the root/console logger

    fulltrace_path = temp_dir_path / "{}_{}_multiagent_fulltrace.txt".format(
        MODEL_NAME, 'AP' if AUTOPLANNING else 'IF'
    )
    fulltrace_path.write_text("", encoding="utf-8")  # touch it now so a run that errors on
                                                       # question 1 still leaves a file behind

    return temp_dir_path


def load_questions(test_json_path: str = 'benchmark/question.json'):
    """Identical to agents/benchmark_ma.py's loader — same field extraction,
    same choices handling, unchanged so results stay comparable."""
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


def extract_answer(final_text: str) -> str:
    """Same extraction logic as agents/benchmark_ma.py's extract_answer,
    applied to the orchestrator's raw final text."""
    if not final_text:
        return "No answer found"
    if '<Answer>' in final_text and '</Answer>' in final_text:
        start = final_text.find('<Answer>') + len('<Answer>')
        end = final_text.find('</Answer>')
        return final_text[start:end].strip()
    return final_text


def _append_fulltrace_error(question_id, query: str, error_msg: str) -> None:
    """Only used on a hard exception before/outside a QuestionTrace's own
    incremental writes (e.g. session creation itself failing) — normal
    per-call tracing happens inside QuestionTrace, not here."""
    with open(fulltrace_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'#' * 90}\nQUESTION {question_id}\n{'#' * 90}\n")
        f.write(f"QUERY: {query}\n[EXCEPTION] {error_msg}\n")


async def run_one_question(query: str, question_id) -> tuple[str, list[dict]]:
    """Streams `query` through the shared orchestrator/runner, returning:
      - final_answer_text: the orchestrator's raw final response text
      - conversation: list of {"role", "content"} entries for the
        metrics-compatible .log — built from the REAL underlying tool
        calls via tracing.QuestionTrace, not from the top-level event
        stream (which only ever shows each sub-agent's single collapsed
        AgentTool call/response).

    Bounded by PER_QUESTION_TIMEOUT_S so a stuck sub-agent can't hang the
    whole batch run indefinitely. Additionally, tracing.py's
    before_tool_callback blocks any single (agent, tool, args) signature
    once it repeats past tracing.MAX_IDENTICAL_CALLS, independent of this
    timeout — a stuck loop gets cut off in seconds, not minutes.
    """
    session_id = f"q{question_id}_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    trace = tracing.QuestionTrace(session_id=session_id, fulltrace_path=fulltrace_path)
    trace.conversation.append({"role": "user", "content": query})
    tracing.active_traces[session_id] = trace
    trace._write_fulltrace_line(f"\n{'#' * 90}\nQUESTION {question_id}\n{'#' * 90}")
    trace.write_header(query)

    content = types.Content(role="user", parts=[types.Part(text=query)])

    final_answer_text = ""
    current_author = None

    async def _consume():
        nonlocal final_answer_text, current_author
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
            author = event.author or "?"

            if author != current_author:
                current_author = author
                trace.record_agent_switch(author)

            if event.content and event.content.parts:
                for part in event.content.parts:
                    text = getattr(part, "text", None)
                    if text and not event.is_final_response():
                        trace.record_reasoning(author, text)
                    # NOTE: function_call/function_response are deliberately
                    # NOT read here — before/after_tool_callback in
                    # tracing.py are the source of truth for tool calls,
                    # since they're the only thing that sees INSIDE an
                    # AgentTool-wrapped sub-agent's own tool calls.

            if event.actions:
                handoff = getattr(event.actions, "transfer_to_agent", None)
                if handoff:
                    trace.record_handoff(author, handoff)

            if event.is_final_response():
                if event.content and event.content.parts:
                    texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
                    if texts:
                        final_answer_text = "\n".join(texts)
                elif event.actions and event.actions.escalate:
                    final_answer_text = f"Agent escalated: {event.error_message or 'No specific message.'}"

    try:
        await asyncio.wait_for(_consume(), timeout=PER_QUESTION_TIMEOUT_S)
    except asyncio.TimeoutError:
        trace._write_fulltrace_line(f"\n   [TIMEOUT] exceeded {PER_QUESTION_TIMEOUT_S}s — aborting this question")
        if not final_answer_text:
            final_answer_text = f"Error: timed out after {PER_QUESTION_TIMEOUT_S}s"
    finally:
        tracing.active_traces.pop(session_id, None)  # don't let traces accumulate across a long batch run

    trace.record_final(final_answer_text)
    return final_answer_text, trace.conversation


async def handle_question(question) -> str:
    """Builds the query the same way agents/benchmark_ma.py did, runs it
    through the orchestrator, writes both logs, and returns the extracted
    answer letter."""
    query = question['auto'] + question['data'] if AUTOPLANNING else \
        question['instruct'] + question['data']

    if question['choices']:
        query += '\n'.join([''] + [
            '{}.{}'.format(chr(ord('A') + i), choice)
            for i, choice in enumerate(question['choices'])
        ])

    try:
        final_answer_text, conversation = await run_one_question(query, question['question_id'])
        extracted = extract_answer(final_answer_text)

        json_logger.info("Chat Content", question['question_id'], conversation, extracted)
        # RotatingFileHandler flushes on every emit by default, but that's
        # an implementation detail worth not depending on silently for a
        # long batch run — flush explicitly so the .log is guaranteed
        # current on disk after every single question.
        for h in json_logger.handlers:
            h.flush()

        return extracted

    except Exception as e:
        error_msg = f"Error processing question {question['question_id']}: {e}"
        logger.exception(error_msg)
        json_logger.info("Chat Content", question['question_id'], [], error_msg)
        for h in json_logger.handlers:
            h.flush()
        _append_fulltrace_error(question['question_id'], query, error_msg)
        return f"Error: {e}"


async def main():
    print("Initializing updated_agents multi-agent Earth Science benchmark run...")
    init_output_paths()
    print(f"Outputs will be written under: {temp_dir_path}")

    questions = load_questions()[QUESTION_SLICE]
    print(f"Loaded {len(questions)} questions for evaluation")

    results = []
    for question in tqdm(questions, desc="Processing questions"):
        answer = await handle_question(question)
        results.append({"question_id": question['question_id'], "answer": answer})
        await asyncio.sleep(INTER_QUESTION_DELAY_S)

    results_path = temp_dir_path / "results_summary.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nEvaluation completed! Results saved to {results_path}")
    print(f"Metrics-compatible log (real tool calls, agent-wrapper calls excluded): "
          f"{temp_dir_path}/{MODEL_NAME}_{'AP' if AUTOPLANNING else 'IF'}_multiagent.log")
    print(f"Full multi-agent trace (all sub-agents, written incrementally): {fulltrace_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("Benchmark run failed")
        print(f"An error occurred: {e}")
