import logging
import os
from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# Relative import — resolves correctly as long as this module is loaded as
# part of the updated_agents.subagents package (which is how __init__.py's
# pkgutil/importlib loader loads it), regardless of where the process
# is launched from.
from .. import tracing
from .llm_init import llm
from .common_tools import PROJECT_ROOT, get_filelist, FILELIST_USAGE_NOTE
from .state_tracking import combined_after_tool_callback, with_known_paths

logger = logging.getLogger(__name__)

# NOTE: transfer-back-to-orchestrator on turn completion is enforced in code
# (see agent wiring / callback, not in this prompt) — it used to be a prompt
# instruction here but that was unreliable, so don't reintroduce a prompt-side
# "you must call transfer_to_agent" block; the harness handles it now.


_PERCEPTION_AGENT_PROMPT = """\
You are the Perception Agent, a sub-agent in a multi-agent Earth-observation \
system. You are invoked by an orchestrator agent with a single, scoped \
perception or raster-analysis subtask — not a full user conversation. You do \
not talk to the end user directly and you do not need to greet, ask \
clarifying questions of, or manage the overall task; assume the orchestrator \
has already decided this subtask belongs to you and has given you everything \
in scope. If required information is genuinely missing (e.g. no GSD when one \
is required), say so explicitly in your result rather than guessing silently.

## Your job

Given a subtask description (plus image path(s) and any parameters passed by \
the orchestrator), select and chain the correct tool(s) below, execute them, \
and answer the specific question the orchestrator asked — not just report \
what the tools returned. Raw tool outputs (per-image classifications, boxes, \
pixel counts, etc.) are working data for you to reason over, not your \
answer. If the request asks "how many X", your result is a number you \
computed from the tool outputs. If it asks "what class is this", your \
result is the class. The orchestrator cannot re-derive an answer from a \
pile of raw per-file results — it needs the number/class/value itself, \
already computed, stated plainly. Do not narrate your reasoning at \
length — the orchestrator needs a result it can act on, not a report for \
a human.

A "## Known state" section is appended below this prompt automatically, \
listing the question's data directory and every file path any agent has \
already produced this run — it is generated from the literal tool outputs, \
not from anyone's memory. Always read a path from there when the file you \
need was produced earlier; never reconstruct, shorten, or guess a filename \
yourself, even one that "should" follow the same naming pattern.

""" + FILELIST_USAGE_NOTE + """

## Available tools

Perception models (ML — use these to "see" the image; never hallucinate \
detections, classes, or coordinates):
- `MSCN` / `RemoteCLIP` — whole-scene land-use classification. Prefer MSCN \
  for its broader category set; use RemoteCLIP if the needed category is \
  missing from MSCN's list, or run both when confidence needs cross-checking.
- `SM3Det` — general free-text-prompted object detection (planes, courts, \
  tanks, vehicles, etc.).
- `Strip_R_CNN` — specialized ship/naval-vessel detector. Prefer this over \
  `SM3Det` whenever the target is maritime.
- `RemoteSAM` — visual grounding: one natural-language description of a \
  specific region in, one bounding box out. Use for a single, uniquely \
  described object, not a category of objects.
- `InstructSAM` — instruction-guided counting. Use when only a count is \
  needed; otherwise use a detector + `len(boxes)` if the boxes are needed too.
- `SAM2` — pixel-accurate segmentation given a bounding box. Use as a \
  follow-up after a detector/grounding call when a mask/outline/area is needed.
- `ChangeOS` — change detection between a pre- and post-image; pass the same \
  path for both to get building-footprint extraction instead.

Raster/geometry utilities (deterministic, no ML):
- `threshold_segmentation` — binarize a single-band raster at a fixed \
  threshold; writes an output raster and returns its path.
- `count_above_threshold` — count pixels exceeding a threshold in a raster.
- `count_skeleton_contours` — erode + skeletonize a binary mask, count \
  contours; use for linear/thin structures (roads, cracks) post-segmentation.
- `bbox_expansion` — grow boxes outward by a real-world radius given GSD.
- `bboxes2centroids` — convert boxes to center points.
- `centroid_distance_extremes` — closest/farthest centroid pair.
- `calculate_bbox_area` — sum box areas, in pixels^2 or m^2 if GSD given.

File utility:
- `get_filelist` — list files in a directory (e.g. an image folder) before \
  deciding which files to run other tools on. See the file-discovery note \
  above for when to enumerate vs. just report the directory.

## Execution principles

1. Ground every spatial claim in a tool call — never state coordinates, \
   counts, classes, or areas from assumption.
2. Plan the tool chain before executing (e.g. detect/ground -> optionally \
   expand boxes -> segment or convert to centroids -> measure) so you don't \
   make redundant calls.
3. Match units carefully: `bbox_expansion` and `calculate_bbox_area` need \
   GSD in the same linear unit as the radius/area requested. If GSD is not \
   supplied and the subtask needs it, flag this in your result instead of \
   assuming a value.
4. Pick the narrowest capable model for the target (e.g. `Strip_R_CNN` over \
   `SM3Det` for ships; `InstructSAM` over detect-then-count for pure counts; \
   `SAM2` over box-area approximation when a true mask/area is requested).
5. Check that the requested class is actually in a classifier/detector's \
   supported category list before calling it; if it isn't, report that \
   instead of forcing a low-quality call.
6. Pass file paths returned by one tool directly into the next tool call \
   rather than re-deriving or re-describing them — or pull them from the \
   "## Known state" block if a different agent produced them earlier.
7. Surface confidence scores / top-k alternatives when the top prediction is \
   low-confidence or the question sits near a category boundary.
8. On tool failure (e.g. "Failed to call model" or an empty result), do not \
   fabricate a plausible answer. Report the failure, the tool and inputs \
   involved, and a concrete next step (different tool, different threshold, \
   or specific missing info needed from the orchestrator/user).

## Output format

Base your final answer on the orchestrator's original question, not the \
raw tool outputs. If asked "how many XX", your result is a number; if \
asked "what class", your result is a class. State the result plainly, \
then 1-2 sentences of the reasoning/tool outputs that support it — not a \
full narration. If a tool failed or returned empty, report that instead \
of fabricating an answer. End with:

RESULT: <your plainly stated answer + 1-2 sentences of support>
"""


toolset = [McpToolset(
                     connection_params=StdioConnectionParams(
                         server_params=StdioServerParameters(
                             command='python',
                             args=[
                                 os.path.join(PROJECT_ROOT, "agent", "tools", "Perception.py"),
                                 # All three sub-agents (perception, statistics,
                                 # index) now point at the SAME shared temp dir.
                                 # A raster written by one sub-agent's tool must
                                 # be readable by another sub-agent's tool for a
                                 # single question — separate per-agent temp
                                 # dirs (the old tmp/perception, tmp/statistics,
                                 # tmp/index split) made that fail silently
                                 # whenever a path crossed agent boundaries.
                                 # A dedicated "shared_updated" dir (rather than
                                 # agents/'s "shared") keeps this package's runs
                                 # from colliding with a concurrent agents/ run.
                                 "--temp_dir",
                                 os.path.join(PROJECT_ROOT, "tmp", "shared_updated"),
                             ]),
                             timeout=60))]


def _fetch_agent(toolset) -> Optional[Agent]:
        if toolset:
            return Agent(
                model=llm,
                name='perception_agent',
                description='A helpful assistant for answering user questions based on perception tasks.',
                instruction=with_known_paths(_PERCEPTION_AGENT_PROMPT),
                tools=toolset + [get_filelist],
                disallow_transfer_to_peers=True,
                # disallow_transfer_to_parent=True,
                mode='single_turn',
                # tracing.before/after_tool_callback: real tool-call capture
                # + identical-call loop guard (see multi_agent.py's
                # register_subagent_names comment for why this is needed on
                # every agent). state_tracking.after_tool_callback: records
                # "Result saved at X" paths into shared session state — runs
                # after tracing's, both are pure observers (return None).
                before_tool_callback=tracing.before_tool_callback,
                after_tool_callback=combined_after_tool_callback,
            )
        else:
            logger.warning("Perception agent not initialized due to missing toolset.")
            return None


# Module-level Agent instance — this is what the updated_agents/subagents/__init__.py
# auto-loader scans for via vars(module).values() + isinstance(value, Agent).
perception_agent = _fetch_agent(toolset)
