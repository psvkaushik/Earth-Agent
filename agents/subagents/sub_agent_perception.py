import logging
import os
from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# Relative import — resolves correctly as long as this module is loaded as
# part of the agents.subagents package (which is how __init__.py's
# pkgutil/importlib loader loads it), regardless of where the process
# is launched from.
from .llm_init import llm

logger = logging.getLogger(__name__)

# CHANGE: anchor a fixed PROJECT_ROOT, independent of process cwd. This file
# lives at Earth-Agent/agents/subagents/sub_agent_perception.py, so the
# project root is two levels up. Any tool here that touches the real
# filesystem (get_filelist, and anything similar you add later) should
# resolve relative paths against this, not against whatever directory the
# process happened to be launched from — otherwise a path like
# "benchmark/data/question189" only works when you run from Earth-Agent/
# itself, and silently breaks (or worse, crashes the sub-agent node) when
# run from agents/ or anywhere else.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
- `get_filelist` — list files in a directory (e.g. an image folder) before \
  deciding which files to run other tools on.

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
   rather than re-deriving or re-describing them.
7. Surface confidence scores / top-k alternatives when the top prediction is \
   low-confidence or the question sits near a category boundary.
8. On tool failure (e.g. "Failed to call model" or an empty result), do not \
   fabricate a plausible answer. Report the failure, the tool and inputs \
   involved, and a concrete next step (different tool, different threshold, \
   or specific missing info needed from the orchestrator/user).

## Output format

Base your final answer on the orchestrator's original question, not on the raw tool outputs. If the orchestrator asked "how many X", your result is a number; if it asked "what class is this", your result is a class. Give your final answer and along with it the reasoning you used to derive it from the tool outputs. If a tool failed or returned an empty result, report that instead of fabricating an answer.
"""


def get_filelist(dir_path: str):
    """
    Returns a list of files in the specified directory.

    Parameters:
        dir_path (str): Path to the directory. May be given relative to the
            project root (e.g. "benchmark/data/question189") — it will be
            resolved against PROJECT_ROOT regardless of the process's cwd,
            so it works the same whether you run from Earth-Agent/,
            Earth-Agent/agents/, or anywhere else.

    Returns:
        list: List of file names in the directory (dotfiles excluded), or
        a dict with an "error" key describing what went wrong. Returning a
        dict on failure rather than raising is deliberate — an uncaught
        exception here crashes the whole sub-agent node ("Dynamic node
        perception_agent failed") with no detail the orchestrator can act
        on; a returned error string at least gives it something to retry
        or report against.
    """
    resolved = dir_path if os.path.isabs(dir_path) else os.path.join(PROJECT_ROOT, dir_path)

    try:
        return sorted([f for f in os.listdir(resolved) if not f.startswith('.')])
    except FileNotFoundError:
        return {"error": f"Directory not found: {resolved} (from input '{dir_path}')"}
    except NotADirectoryError:
        return {"error": f"Not a directory: {resolved}"}
    except OSError as e:
        return {"error": f"Could not list {resolved}: {e}"}


toolset = [McpToolset(
                     connection_params=StdioConnectionParams(
                         server_params=StdioServerParameters(
                             command='python',
                             # NOTE: singular "agent/" here — double check this
                             # matches your real directory name ("agent" vs
                             # "agents"); a mismatch fails silently at
                             # McpToolset init rather than a loud crash.
                             args=[
                                 "/home/egm/Desktop/Earth-Agent/agent/tools/Perception.py",
                                 # CHANGE: Perception.py does `TEMP_DIR =
                                 # Path(args.temp_dir)` with no default —
                                 # without this flag the subprocess crashes
                                 # on import, the MCP session fails to
                                 # connect, and the agent silently falls
                                 # back to only its Python-side tools
                                 # (get_filelist) with no real analysis.
                                 "--temp_dir",
                                 os.path.join(PROJECT_ROOT, "tmp", "perception"),
                             ]),
                             timeout=300))]


def _fetch_agent(toolset) -> Optional[Agent]:
        if toolset:
            return Agent(
                model=llm,
                name='perception_agent',
                description='A helpful assistant for answering user questions based on perception tasks.',
                instruction=_PERCEPTION_AGENT_PROMPT,
                tools=toolset + [get_filelist],
                disallow_transfer_to_peers=True,
                # mode='single_turn',
            )
        else:
            logger.warning("Perception agent not initialized due to missing toolset.")
            return None


# Module-level Agent instance — this is what the agents/subagents/__init__.py
# auto-loader scans for via vars(module).values() + isinstance(value, Agent).
perception_agent = _fetch_agent(toolset)