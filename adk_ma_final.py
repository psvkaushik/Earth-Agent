"""
ADK (Agent Development Kit) port of the LangChain/LangGraph multi-agent
Earth Science evaluation script.

Architecture, preserved 1:1 from the original:
  - One ORCHESTRATOR LlmAgent that never touches raw MCP tools directly.
  - Five KIT SPECIALIST LlmAgents (index / inversion / perception /
    analysis / statistics), each scoped to its own MCP tools + the shared
    get_filelist tool.
  - The orchestrator "calls" a specialist through a delegation tool
    (call_index_kit, call_inversion_kit, ...). Each specialist keeps its
    OWN conversation memory across delegations *within the same question*
    (matching the original `_kit_message_history` dict), but does not see
    the orchestrator's reasoning or other specialists' work unless the
    orchestrator explicitly includes it in the instructions it passes.

IMPORTANT DESIGN NOTE ON "REMEMBERING OUTPUT PATHS":
  The original script's rule ("once a specialist reports
  'Result saved at /some/path', remember that output directory and pass
  the exact path to the next specialist that needs it") is NOT reimplemented
  as string-parsing / path-tracking code here. It never was: in the
  original it was a PROMPT instruction (ORCHESTRATOR_SYS_PROMPT rule #3),
  enforced by the model reading its own conversation history. That is
  preserved as-is below -- the orchestrator's ADK session naturally
  accumulates every tool (specialist) result, including any "Result saved
  at ..." text, so the model can copy paths forward itself. Do not add
  code that regexes for "Result saved at" and auto-injects paths into the
  next delegation call -- that would silently paper over cases where the
  model is supposed to be doing this reasoning itself, and it would break
  the moment a tool's wording changes.

VERSION NOTE:
  Written against `google-adk` (the `google.adk` package) as of early/mid
  2026 APIs (LlmAgent, MCPToolset w/ tool_filter, FunctionTool, Runner,
  InMemorySessionService, LiteLlm). MCP toolset construction in particular
  has churned across ADK releases -- if `MCPToolset(connection_params=...)`
  doesn't match your installed version, check
  `google.adk.tools.mcp_tool.mcp_toolset` for the current signature; the
  rest of the script (kit assignment, tool distribution, delegation tools,
  tracing, logging) does not depend on that detail.
"""

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"

import json
import logging
import asyncio
import base64
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

from mcp import StdioServerParameters

from google.genai import types as genai_types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.models.lite_llm import LiteLlm

# Change to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Global variables
# ============================================================================
logger = None
temp_dir_path = None

model_name = 'eve'
autoplanning = True
RETRY_IDS = None
BATCH_TOTAL = 1
BATCH_INDEX = 0

username = os.getenv("EVE_USERNAME")
password = os.getenv("EVE_PASSWORD")
token = base64.b64encode(f"{username}:{password}".encode()).decode()
EVE_HEADERS = {"Authorization": f"Basic {token}"}

APP_NAME = "earth_science_agent"

KIT_SPECS = {
    "index": {
        "match_keywords": ["index"],
        "description": (
            "computing per-pixel spectral/thermal indices from raster bands -- one or more raster files in, a new raster "
            "(or, for one tool, a single percentage) out. Covers: NDVI, NDWI, NDBI, EVI, NBR (burn ratio), FVC (fractional "
            "vegetation cover), WRI (water ratio index), NDTI (turbidity index), NDSI (snow index) and the percentage of "
            "extreme snow/ice loss from a binary snow-loss map, FRP (fire radiative power) from thermal-anomaly rasters, and "
            "TVDI (Temperature Vegetation Dryness Index) computed directly from a paired NDVI raster and LST raster. Batch "
            "versions exist for all of these. This kit does NOT retrieve LST or other geophysical parameters from raw bands "
            "(that's the inversion kit) and does NOT do time-series/trend analysis (that's the analysis kit)."
        ),
    },
    "inversion": {
        "match_keywords": ["inversion"],
        "description": (
            "retrieving geophysical parameters from raw satellite bands via physical/statistical inversion models. Covers: "
            "Land Surface Temperature (LST) by several methods (single-channel, multi-channel, split-window, "
            "temperature-emissivity separation, MODIS day/night, thermal time model), plus mean/max LST conditioned on an "
            "NDVI threshold; Precipitable Water Vapor (PWV) via band ratio; soil moisture proxy via Apparent Thermal Inertia "
            "(ATI, from day/night brightness temperature + albedo); sea ice concentration (NASA Team algorithm) and other "
            "SAR/passive-microwave dual-polarization, dual-frequency, or multi-frequency brightness-temperature retrievals; "
            "and water turbidity (NTU). These tools take raw band/brightness-temperature/albedo rasters in, not "
            "already-computed indices -- except the LST-by-NDVI-threshold tools, which do take an ndvi_threshold alongside "
            "raw LST/NDVI bands."
        ),
    },
    "perception": {
        "match_keywords": ["perception"],
        "description": (
            "image perception on individual raster/image files, one image (or one before/after pair) at a time. Covers: "
            "object detection and segmentation (RemoteSAM, SAM2, Strip_R_CNN, SM3Det, InstructSAM), change detection between "
            "a pre/post image pair (ChangeOS), zero-shot classification/embedding (RemoteCLIP), no-reference image quality "
            "scoring (MSCN), thresholding a single image into a binary mask and counting pixels/contours above a threshold, "
            "and bounding-box geometry utilities (expand a bbox by a radius, get bbox centroids, distance between centroid "
            "extremes, bbox area). This kit does not do time-series or multi-date statistics."
        ),
    },
    "analysis": {
        "match_keywords": ["analysis", "analytic"],
        "description": (
            "spatiotemporal analysis over a NUMERIC TIME SERIES, spatial array, or single raster you already have -- not raw "
            "satellite bands. Covers: linear trend fitting, Mann-Kendall trend test, Sen's slope, STL seasonal decomposition, "
            "change-point detection, autocorrelation function, ACF-based seasonality detection, counting spikes in a "
            "sequence of values, and spatial hotspot analysis (Getis-Ord Gi*) plus hotspot-direction analysis on a raster. "
            "This kit does NOT compute indices (NDVI, TVDI, etc.), does NOT retrieve LST or other geophysical parameters "
            "from raw bands, and does NOT extract per-image statistics (mean/std/threshold ratio) from a list of rasters --  "
            "get those numbers from the index, inversion, or statistics kit first, then hand the resulting numbers (or a "
            "single raster, for the hotspot tools) to this kit."
        ),
    },
    "statistics": {
        "match_keywords": ["statistic", "stats"],
        "description": (
            "general-purpose numeric/image statistics, thresholding, and image algebra -- turning one or many raster files "
            "into plain numbers (or doing simple math on numbers you already have). Covers: single/batch image mean, std, "
            "median, min, max, sum, skewness, kurtosis, and coefficient of variation over one or many raster files; "
            "threshold-based ratios/counts/masks over one or many images (including multi-band and multi-condition "
            "thresholds); fire-pixel counts and fire-prone-area identification; percentile lookups; simple two-number "
            "arithmetic (difference, division, percentage_change, multiply) and Kelvin<->Celsius conversion; image-to-image "
            "algebra (subtract two rasters, divide two rasters and get the mean, compute the area an image covers, generate "
            "a colormap visualization); and radiometric correction / cloud masking preprocessing. This is the right kit to "
            "turn raster(s) into number(s) for the analysis kit to then analyze."
        ),
    },
}

# NOTE ON get_filelist: every specialist's system prompt below claims access
# to get_filelist "plus" its own tools -- this is deliberate (see
# SHARED_TOOL_NAMES / load_tools_per_kit): the tool itself is defined once,
# in the statistics server, but distributed to every kit at startup so any
# specialist can list files without needing to ask the orchestrator first.

SPECIALIST_PROMPT_TEMPLATE = '''
You are a geoscientist, and you need to use tools to complete a sub-task about Earth observation data analysis given to you by a lead agent. You have access to tools for {tools_desc}, plus get_filelist. Note that if a tool returns an error, you can only try again once. Report your result clearly, including any output file paths, so the lead agent can use it.
ATTENTION:
1. When a tool returns "Result saved at /path/to/file", you must use the full returned path "/path/to/file" in all subsequent tool calls.
2. The lead agent's instructions will always include the data directory path for this question, and, whenever relevant, the exact file paths that have already been retrieved or produced -- either filenames the lead agent already discovered, or paths a specialist previously reported back to it (e.g. "Result saved at ..."). Treat those paths as given: use them exactly as written, never re-verify them, and never assume, guess, or slightly alter a path you were not actually given.
3. Only call get_filelist yourself if the instructions you received contain NO usable file paths at all -- i.e. only a bare directory, with no filenames and no previously retrieved paths to work from. If any paths are already present in what you were given, use those directly instead of calling get_filelist.
4. Before processing multiple files, check whether a batch version of the tool exists (e.g. a tool literally named calculate_batch_X, or a parameter typed as a list/array) and use it to handle all the files in one call. Only call a tool once per file, one by one, if no batch option exists for that specific operation -- don't default to looping if a batch tool is available. Also check whether a tool's parameters are file paths or require the data loaded into memory first. If you are unsure what a tool accepts, check its documentation rather than guessing. After using a batch tool, verify the number of outputs matches the number of relevant input files (from get_filelist, or the date range/paths you were given). If some are missing, either process the rest or explicitly tell the lead agent which dates/files are missing -- do not silently report partial results as if they were complete.
5. If you're unable to answer a question with the available tools, report the limitation clearly and specifically: say what kind of capability you would need (e.g. "I don't have a tool to compute indices or read raw satellite bands -- I only analyze numeric sequences or rasters that already exist") so the lead agent knows this needs a different kit. Do not offer generic external-software instructions (e.g. "use QGIS/GDAL/ArcGIS") as a substitute for actually doing the task, and do not guess any answer or hallucinate.
6. IMPORTANT: your memory of this question persists across every delegation call you receive from the lead agent within this same question. If you produced an output earlier ("Result saved at ..."), you already have that path in your own history -- reuse it directly if the lead agent's new instructions refer back to "the file you produced" or similar, without asking the lead agent to repeat it. Never re-derive or re-guess a path you already produced.
7. Never invent, estimate, or fill in placeholder numbers for a tool call. If a tool needs image/pixel statistics (means, thresholds, trends, etc.), first call the appropriate statistics tool (e.g. calculate_tif_average, calc_batch_image_mean) to obtain the real values from the actual files -- do not guess plausible-looking numbers yourself.
8. If a tool's required parameters do not genuinely match the data you have (e.g. it asks for day/night or emissivity inputs but you only have a single LST/NDVI pair), do not force mismatched files into those parameters just to make the call succeed. Look for a simpler/more direct tool that matches your actual inputs, or report that no available tool fits.
9. If the same tool call (or a trivial variant, like guessing a slightly different file path) has already failed once, do not try it or a near-identical variant again. Either switch to a genuinely different tool/approach, or immediately report the limitation back rather than continuing to search.
10. Complete the full sub-task before reporting back to the lead agent. If it involves processing many files/dates, keep making tool calls until all of them are done, in this same turn -- do not stop partway and ask the lead agent whether you should continue. The lead agent cannot see your file-by-file progress and only receives your final report, so an unfinished job looks identical to a wrong one. Only stop early for a genuine hard blocker (see rule 5).
11. If a tool call errors on a specific input, investigate why that specific input is a problem (e.g. check whether that one file has too few valid pixels, is all-NoData, etc.) rather than changing an optional parameter (e.g. enabling a uint8/normalization flag) on the whole batch just to make the error disappear. Changing how data is represented (e.g. rescaling continuous physical values into 0-255 before averaging) changes what your result means, even if the call now succeeds.
12. Before reporting a numeric result, sanity-check it: if every value in a computed batch comes back identical (e.g. all exactly 0.0) or otherwise looks degenerate, treat that as a likely tool/data problem, not a real finding -- investigate further (e.g. check a single file individually) or report the anomaly clearly instead of presenting it as a normal result.
13. When you produce many output files in one sub-task, state the common output directory ONCE, clearly and explicitly (e.g. "All outputs saved under: /abs/path/to/dir"), in addition to any individual paths you list. The lead agent has to extract this directory from your report to pass it to the next specialist -- if it's only implied by a long list of individual full paths (especially a truncated one), it's easy to lose. Note if the output directory differs from the input directory you were given, since it usually will.
14. If a directory you were given (or its get_filelist listing) does not contain files matching what your instructions describe -- e.g. you were asked to use files with a certain name pattern, but the directory only contains files with a different name or extension -- do NOT invent a plausible-looking filename and try it anyway. Report back exactly what that directory actually contains and state clearly that the expected files are missing, so the lead agent knows to check whether it gave you the correct path (per rule 2, only ever reuse a path you were actually given verbatim -- this includes not "correcting" a given path into a guessed one).
15. If one or more points in a data series could not be computed (missing/invalid/no-data), never place a placeholder word or phrase (e.g. "no valid data", "N/A", "error") inside what is otherwise a numeric array you pass into a tool call -- a tool expecting a list of numbers will fail on that, and can break the whole run rather than just that one data point. When calling a tool, pass only the clean numeric values (dropping the missing point(s) from the array), and separately state in your own report -- in prose, not inside the array -- which date/index was dropped and why, so the lead agent (and any specialist it relays this to) has that context without it corrupting a numeric argument.
16. Never pass more than 15 file paths in a single tool call argument, even to a batch-capable tool. If you have more than 15 relevant files, split the work into multiple sequential calls of at most 15 paths each. This keeps each tool call's arguments short enough to generate reliably -- very long, repetitive argument lists are more likely to come out malformed and cause the whole call to fail.
'''

for _kit, _spec in KIT_SPECS.items():
    _spec["system_prompt"] = SPECIALIST_PROMPT_TEMPLATE.format(tools_desc=_spec["description"])

ORCHESTRATOR_SYS_PROMPT = '''
You are a geoscientist, and you need to use tools to answer multiple-choice questions about Earth observation data analysis. Note that if a tool returns an error, you can only try again once. Ultimately, you only need to explicitly tell me the correct choice.

You have access to five tools, each delegating a sub-task to a specialist agent with its own set of underlying tools:
- call_index_kit: computes per-pixel spectral/thermal indices from raster bands -- raster(s) in, a new raster out (NDVI, NDWI, NDBI, EVI, NBR, FVC, WRI, NDTI, NDSI, extreme snow-loss %, FRP, and TVDI dryness index computed directly from NDVI+LST rasters). Does NOT do LST/geophysical retrieval or time-series analysis.
- call_inversion_kit: retrieves geophysical parameters from raw bands -- LST (several methods), PWV, soil moisture proxy (ATI), sea ice concentration, water turbidity, SAR polarization-based retrievals. Takes raw bands/brightness-temperature in, not already-computed indices.
- call_perception_kit: object detection, segmentation, change detection, classification, image-quality scoring, and bbox geometry on individual images. One image (or one before/after pair) at a time.
- call_statistics_kit: turns one or many raster files into plain numbers (mean/std/median/percentiles/threshold ratios/fire-pixel counts), simple image algebra and arithmetic, unit conversion, and get_filelist.
- call_analysis_kit: analyzes a NUMERIC TIME SERIES or spatial array you already have -- trend/Mann-Kendall/Sen's slope/seasonality/change-points/spikes/hotspots. Does NOT compute indices, retrieve LST, or read raw rasters -- it only works on numbers (or a single raster, for the hotspot tools) that another kit already produced.
Each specialist remembers its own earlier calls within this question (so you can tell it "as before" or point out a prior error), but it does not see your reasoning or other specialists' results unless you include them in the instructions you give it.

ATTENTION:
1. When a tool returns "Result saved at /path/to/file", you must use the full returned path "/path/to/file" in all subsequent tool calls.
2. For each question, you must provide the choice you think is most appropriate. Don't give me another format. Your final answer must be the LETTER of the option (A, B, C, D, ...) exactly as labeled in the question -- never the option's text/value itself, even if that feels more precise or informative. Your final answer format must be:
<Answer>X<Answer>
where X is that single letter.
3. There are two directories/paths you must keep track of YOURSELF throughout the question, purely by reading your own conversation history -- nothing tracks this for you automatically:
   (a) The INPUT data directory for this question (given in the question text) -- pass this to any specialist that needs to read the original source files.
   (b) Every output path any specialist reports back to you ("Result saved at /some/path/file.tif"). Whenever a later specialist needs a file another specialist produced, look back through what you were told in this conversation and copy that exact full path into your instructions -- specialists cannot see each other's results, so YOU are the only place that path is remembered and carried forward. Never invent, guess, re-derive, or slightly alter a path; only ever reuse one verbatim from something a specialist actually told you. In particular, a specialist's OUTPUT directory is very often different from the INPUT directory you originally gave it (e.g. it may write into its own working/output folder) -- when you next need those outputs, pass the output directory/paths the specialist actually reported, never the original input directory, even if that seems simpler or "should" be the same place.
4. If a specialist is unable to attend to your query and states its limitation, try a different specialist instead. If none of the specialists can answer, report the limitation clearly, do not get stuck in a loop calling the same specialist, do not guess any answer or hallucinate.
5. Tell each specialist WHAT you need, not HOW to do it. Describe the goal (e.g. "compute the TVDI dryness index for this NDVI/LST data and give me its annual trend") and give it the context it needs (paths, dates, region), but do not specify formulas, algorithms, step-by-step methods, or which exact tool it should call. The specialist knows its own tools and how to use them correctly -- if you specify the method yourself, you risk describing the wrong formula and the specialist following it instead of using the correct tool for the job.
6. Match the task to the right specialist's actual capability, not just its general theme -- use the tool descriptions above, not just the kit name. A useful rule of thumb: if the input is raw satellite bands and you need a derived raster or geophysical value, that's index or inversion; if the input is one image and you need detection/segmentation/classification, that's perception; if the input is one or more raster files and you need a plain number (mean, threshold ratio, etc.), that's statistics; if the input is a numeric sequence or spatial array you ALREADY HAVE and you need trend/seasonality/spike/hotspot analysis, that's analysis. The analysis kit in particular cannot read raster files, compute indices, or retrieve LST itself -- get those numbers or that index raster from another kit first, then pass the results to analysis. If a specialist says it lacks the tools to do something, do not re-send it the same or similar instructions again -- move to a different specialist (per rule 4) or reconsider which kit actually owns that computation.
7. Never delegate the same sub-task, or a lightly reworded version of it, to the same specialist more than twice in a row. If it fails or reports a limitation twice, stop and either try a different specialist or report the overall limitation to me -- do not keep looping.
8. Before writing your final <Answer>, check that the choice you're selecting is actually consistent with the numeric result(s) reported to you in this conversation. If they don't match, recheck your reasoning rather than picking a letter that contradicts your own computed numbers.
9. If a specialist's response is incomplete -- it only processed some of the files/dates you asked about, asks whether it should continue, or otherwise flags that it stopped partway -- do not proceed to another kit with that partial result. Immediately re-delegate to the same specialist instructing it to complete the remaining files/dates, and only move on once you have a complete result (or the specialist reports a genuine hard limitation, per rule 4).
10. Be suspicious of results that look degenerate across an entire batch (e.g. every value is exactly 0.0, or all values are identical). Ask the specialist to double-check rather than passing such results straight into your final answer -- a flat/constant result across many different files or dates is more likely a tool or data problem than a real finding.
11. If a specialist reports that one or more points in a series could not be computed (missing/invalid/no-data), never embed a placeholder phrase (e.g. "no valid data", "N/A") inline among numeric values when relaying that series to another specialist -- a specialist that turns around and passes that series into a tool call expecting pure numbers can crash the entire run, not just that one data point. Instead, pass forward only the clean numeric values, and separately state in prose which date/index was dropped and why, so the next specialist can decide how to handle the gap without a stray non-numeric token corrupting a numeric argument.
'''

# ============================================================================
# Logging
# ============================================================================

def init_global_params():
    global temp_dir_path, logger
    if temp_dir_path is None:
        batch_suffix = f'_b{BATCH_INDEX}of{BATCH_TOTAL}' if BATCH_TOTAL > 1 else ''
        temp_dir_path = Path('./evaluate_adk/{}_{}_{}{}'.format(
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
        temp_dir_path / "{}_{}_adk.log".format(model_name, 'AP' if autoplanning else "IF")
    )
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return temp_dir_path, logger


def init_chat_logger():
    global temp_dir_path
    return temp_dir_path / "{}_{}_adk.chat".format(model_name, 'AP' if autoplanning else "IF")


def init_trace_logger():
    global temp_dir_path
    return temp_dir_path / "{}_{}_trace.txt".format(model_name, 'AP' if autoplanning else "IF")


def write_trace(trace_log_path, text: str):
    with open(trace_log_path, 'a', encoding='utf-8') as f:
        f.write(text + "\n")


def save_chat_message(chat_log_path, message_data):
    import uuid
    chat_record = {
        "__module__": "google.adk",
        "__name__": "ChatMessage",
        "id": str(uuid.uuid4()).replace('-', ''),
        "name": message_data.get('name', 'adk_agent'),
        "role": message_data.get('role', 'assistant'),
        "content": message_data.get('content', []),
        "metadata": message_data.get('metadata', None),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(chat_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(chat_record, ensure_ascii=False) + '\n')


# ============================================================================
# Event helpers (ADK's equivalent of the LangChain BaseMessage stream)
# ============================================================================
#
# An ADK `Event` (yielded by `runner.run_async`) has:
#   event.author              -> agent/tool name that produced it
#   event.content.parts       -> list of Part; each Part is one of:
#         part.text                       (assistant text)
#         part.function_call.name/args    (a tool call)
#         part.function_response.name/response  (a tool result)
#   event.is_final_response()

def event_to_log_entries(event):
    """Turn one ADK Event into the same lightweight {"role":...} shape the
    original script used for its .log / .trace rendering, so the rest of
    the pipeline (rendering, flattening) barely has to change."""
    entries = []
    if event.content is None or not event.content.parts:
        return entries

    assistant_content = []
    for part in event.content.parts:
        if getattr(part, "text", None):
            assistant_content.append({"type": "text", "content": part.text})
        elif getattr(part, "function_call", None):
            assistant_content.append({
                "name": part.function_call.name,
                "input": dict(part.function_call.args or {}),
            })
        elif getattr(part, "function_response", None):
            entries.append({
                "role": "tool",
                "name": part.function_response.name,
                "content": [{"output": [{"text": str(part.function_response.response)}]}]
            })

    if assistant_content:
        entries.insert(0, {"role": "assistant", "content": assistant_content})
    return entries


def format_specialist_return(events) -> str:
    """Format the events generated during ONE delegation call into a single
    string for the orchestrator -- kept close to a raw ReAct trace (not a
    paraphrase) so a failed tool call's actual error text reaches the
    orchestrator directly, same rationale as the original script."""
    lines = []
    for event in events:
        for entry in event_to_log_entries(event):
            if entry["role"] == "assistant":
                for item in entry["content"]:
                    if "content" in item:
                        text = item["content"].strip()
                        if text:
                            lines.append(text)
                    elif "name" in item:
                        lines.append(f"[called {item['name']}({json.dumps(item['input'], ensure_ascii=False)})]")
            elif entry["role"] == "tool":
                text = entry["content"][0]["output"][0]["text"]
                lines.append(f"[{entry['name']} returned] {text}")
    return "\n".join(lines) if lines else "(specialist made no tool calls or produced no output)"


def render_log_entries(entries, indent=0):
    pad = "    " * indent
    lines = []
    for entry in entries:
        role = entry.get("role")
        if role == "user":
            lines.append(f"{pad}[USER] {entry.get('content')}")
        elif role == "assistant":
            for item in entry.get("content", []):
                if "content" in item:
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
    lines.append("")
    return "\n".join(lines)


DELEGATION_TOOL_PREFIX = "call_"
DELEGATION_TOOL_SUFFIX = "_kit"


def _is_delegation_wrapper(tool_name: str) -> bool:
    return tool_name.startswith(DELEGATION_TOOL_PREFIX) and tool_name.endswith(DELEGATION_TOOL_SUFFIX)


def flatten_tool_calls(conversation_log):
    """Same purpose as the original: produce a single chronological list of
    only real tool calls/results (specialist sub-traces spliced in,
    delegation-wrapper calls/results dropped) for the machine-parseable
    .log file."""
    flat = []
    for entry in conversation_log:
        role = entry.get("role")
        if role == "specialist":
            flat.extend(flatten_tool_calls(entry.get("content", [])))
        elif role == "assistant":
            kept = []
            for item in entry.get("content", []):
                if "content" in item:
                    kept.append(item)
                elif "name" in item:
                    if _is_delegation_wrapper(item["name"]):
                        continue
                    kept.append(item)
            if kept:
                flat.append({"role": "assistant", "content": kept})
        elif role == "tool":
            if _is_delegation_wrapper(entry.get("name", "")):
                continue
            flat.append(entry)
        else:
            flat.append(entry)
    return flat


class TraceWriter:
    """Lightweight stand-in for the old TraceCallbackHandler: writes each
    tool call/result to the trace .txt file the moment it happens. Used as
    an ADK `before_tool_callback` / `after_tool_callback` pair so specialist
    internal tool use is visible in real time, same as before."""

    def __init__(self, trace_log_path, label: str):
        self.trace_log_path = trace_log_path
        self.label = label

    def before_tool(self, tool, args, tool_context):
        ts = datetime.now().strftime('%H:%M:%S')
        write_trace(self.trace_log_path, f"[{ts}] [{self.label}] -> CALL {tool.name}({json.dumps(args, ensure_ascii=False)})")
        return None  # don't short-circuit the actual tool call

    def after_tool(self, tool, args, tool_context, tool_response):
        ts = datetime.now().strftime('%H:%M:%S')
        write_trace(self.trace_log_path, f"[{ts}] [{self.label}] <- RESULT {tool_response}")
        return None  # don't override the actual tool response


# ============================================================================
# Config / MCP loading
# ============================================================================

def load_adk_config(config_path='./agent/config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_config = config['models'][0]
    llm_kwargs = dict(model_config.get('generate_args', {}))
    llm = LiteLlm(
        model="openai/EVE-Instruct",
        api_base=os.getenv("EVE_ENDPOINT"),
        api_key="EMPTY",
        temperature=0,
        extra_headers=EVE_HEADERS,
        **llm_kwargs,
    )

    mcp_server_params = {}
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

        mcp_server_params[server_name] = StdioServerParameters(
            command=server_config['command'],
            args=updated_args,
            env=server_env,
        )

    return llm, mcp_server_params


def assign_servers_to_kits(mcp_server_params: dict):
    kit_to_servers = {kit: [] for kit in KIT_SPECS}
    unmatched = []

    for server_name in mcp_server_params:
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
        print(f"  (unmatched servers, available to ALL kits as shared utilities): {unmatched}")
    print("===================================\n")

    for kit, servers in kit_to_servers.items():
        if not servers:
            print(f"WARNING: kit '{kit}' matched no MCP server. Check KIT_SPECS['{kit}']"
                  f"['match_keywords'] against your agent/config.json server names.")

    return kit_to_servers, unmatched


SHARED_TOOL_NAMES = {"get_filelist"}

# Hard backstop against the orchestrator looping on one kit within a single
# question (prompt rule ORCHESTRATOR_SYS_PROMPT #7 asks it not to, but this
# guarantees it even if the model ignores that). Once a kit has been
# delegated to this many times for the same question, further delegation
# calls to it short-circuit with a canned message instead of running the
# specialist again.
MAX_DELEGATIONS_PER_KIT_PER_QUESTION = 4


async def load_tools_per_kit(mcp_server_params: dict):
    """Open one MCPToolset per configured server (one process per server,
    matching the original's single `client.get_tools(server_name=...)`
    call per server), fetch its tool objects once, then redistribute the
    SAME tool objects across every kit that needs them -- including
    get_filelist, which every kit's system prompt requires."""
    kit_to_servers, unmatched_servers = assign_servers_to_kits(mcp_server_params)

    toolsets = {}
    tools_by_server = {}
    for server_name, params in mcp_server_params.items():
        toolset = MCPToolset(connection_params=StdioConnectionParams(server_params=params, timeout=60))
        toolsets[server_name] = toolset
        tools_by_server[server_name] = await toolset.get_tools()

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

    for server_name in unmatched_servers:
        shared_tools.extend(tools_by_server[server_name])

    kit_tools = {}
    for kit, servers in kit_to_servers.items():
        tools = list(shared_tools)
        for server_name in servers:
            tools.extend(tools_by_server[server_name])
        seen = set()
        deduped = []
        for tool in tools:
            if tool.name not in seen:
                seen.add(tool.name)
                deduped.append(tool)
        kit_tools[kit] = deduped

    return kit_tools, toolsets


# ============================================================================
# Kit specialist agents + delegation tools
# ============================================================================

def build_kit_subagents(llm, kit_tools: dict, trace_log_path):
    """One LlmAgent per kit, scoped to that kit's tools, with a
    before/after-tool callback pair so its internal tool use streams to the
    trace file in real time (mirrors the old TraceCallbackHandler)."""
    subagents = {}
    for kit, tools in kit_tools.items():
        spec = KIT_SPECS[kit]
        tracer = TraceWriter(trace_log_path, label=kit.upper())
        subagents[kit] = LlmAgent(
            name=f"{kit}_specialist",
            model=llm,
            instruction=spec["system_prompt"],
            tools=tools,
            before_tool_callback=tracer.before_tool,
            after_tool_callback=tracer.after_tool,
        )
        print(f"  Built '{kit}' sub-agent with {len(tools)} tools")
    return subagents


def build_kit_delegation_tool(kit_name, kit_agent, session_service, kit_sessions, kit_counts, trace_log_path):
    """Build the `call_<kit>_kit` FunctionTool the orchestrator uses to
    delegate. `kit_sessions` is a dict (reset once per question by the
    caller) mapping kit_name -> session_id, so repeated delegations to the
    same kit within one question reuse that kit's own ADK session and
    therefore its own conversation memory -- the direct analogue of the
    original `_kit_message_history` dict. `kit_counts` is a parallel dict
    (also reset once per question by the caller) mapping kit_name -> number
    of delegation calls made to it so far this question; it enforces
    MAX_DELEGATIONS_PER_KIT_PER_QUESTION as a hard backstop against looping,
    independent of whether the orchestrator's prompt instructions are
    followed."""
    spec = KIT_SPECS[kit_name]
    kit_runner = Runner(agent=kit_agent, app_name=APP_NAME, session_service=session_service)

    async def _run(instructions: str) -> str:
        """Delegate a sub-task to this kit's specialist. `instructions`
        must be full, self-contained instructions: describe the goal
        (WHAT you need), not the method (HOW to do it) -- no formulas,
        algorithms, or tool names, since the specialist knows its own
        tools. Always include the data directory path for this question,
        plus the exact file paths relevant to this sub-task (either ones
        you discovered yourself, or ones a specialist previously reported
        back to you, e.g. "Result saved at ..."). Never invent, guess, or
        alter a path -- copy it in verbatim."""
        kit_counts[kit_name] = kit_counts.get(kit_name, 0) + 1
        if kit_counts[kit_name] > MAX_DELEGATIONS_PER_KIT_PER_QUESTION:
            msg = (f"[delegation blocked] You have already delegated to the {kit_name.upper()} "
                   f"specialist {MAX_DELEGATIONS_PER_KIT_PER_QUESTION} times for this question "
                   f"without a usable result. Do not call {kit_name}_kit again for this question "
                   f"-- switch to a different specialist if one might help, or report the "
                   f"limitation clearly in your final answer instead of guessing.")
            write_trace(trace_log_path, f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"[ORCHESTRATOR] delegation to {kit_name.upper()} BLOCKED (limit reached): {instructions}")
            return msg

        session_id = kit_sessions.get(kit_name)
        if session_id is None:
            session = await session_service.create_session(app_name=APP_NAME, user_id=f"{kit_name}_specialist")
            session_id = session.id
            kit_sessions[kit_name] = session_id

        write_trace(trace_log_path, f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"[ORCHESTRATOR] delegating to {kit_name.upper()}: {instructions}")

        events = []
        async for event in kit_runner.run_async(
            user_id=f"{kit_name}_specialist",
            session_id=session_id,
            new_message=genai_types.Content(role="user", parts=[genai_types.Part(text=instructions)]),
        ):
            events.append(event)

        _current_subagent_traces.append({
            "kit": kit_name,
            "events": events,
        })

        return format_specialist_return(events)

    _run.__name__ = f"call_{kit_name}_kit"
    _run.__doc__ = (
        f"Delegate a sub-task to the {kit_name.upper()} specialist. "
        f"Covers {spec['description']}. Pass complete, self-contained "
        f"instructions -- this specialist remembers its OWN earlier "
        f"delegations within this question, but it does NOT see your "
        f"reasoning or other specialists' work unless you include it."
    )
    return FunctionTool(func=_run)


_current_subagent_traces = []


async def create_multi_agent_system(llm, mcp_server_params, trace_log_path):
    kit_tools, toolsets = await load_tools_per_kit(mcp_server_params)

    print("Building kit specialist sub-agents...")
    subagents = build_kit_subagents(llm, kit_tools, trace_log_path)

    orchestrator_session_service = InMemorySessionService()
    # kit_sessions_holder["sessions"] / ["counts"] are intentionally created
    # fresh (cleared in place, not rebound) once per question by the caller
    # (see handle_question) -- passed in here as the actual mutable dict
    # objects the closures below capture and read from directly on every
    # delegation call, for the lifetime of the whole run.
    kit_sessions_holder = {"sessions": {}, "counts": {}}

    orch_tracer = TraceWriter(trace_log_path, label="ORCHESTRATOR")
    delegation_tools = [
        build_kit_delegation_tool(kit, agent, orchestrator_session_service,
                                   kit_sessions_holder["sessions"], kit_sessions_holder["counts"], trace_log_path)
        for kit, agent in subagents.items()
    ]

    orchestrator = LlmAgent(
        name="orchestrator",
        model=llm,
        instruction=ORCHESTRATOR_SYS_PROMPT,
        tools=delegation_tools,
        before_tool_callback=orch_tracer.before_tool,
        after_tool_callback=orch_tracer.after_tool,
    )

    total_tools = sum(len(t) for t in kit_tools.values())
    print(f"Successfully loaded {total_tools} underlying MCP tools across "
          f"{len(subagents)} specialists + orchestrator "
          f"({len(delegation_tools)} orchestrator-level delegation tools)")

    return orchestrator, orchestrator_session_service, kit_sessions_holder, toolsets


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


def extract_answer_from_text(text):
    if text and '<Answer>' in text and '</Answer>' in text:
        start = text.find('<Answer>') + len('<Answer>')
        end = text.find('</Answer>')
        return text[start:end].strip()
    return text or "No answer found"


# ============================================================================
# Question handling
# ============================================================================

async def handle_question(orchestrator, session_service, kit_sessions_holder, question, chat_log_path, trace_log_path):
    global _current_subagent_traces
    _current_subagent_traces = []
    kit_sessions_holder["sessions"].clear()  # fresh specialist memory each question, same as the old _kit_message_history reset
    kit_sessions_holder["counts"].clear()  # fresh per-kit delegation-count budget each question (see MAX_DELEGATIONS_PER_KIT_PER_QUESTION)

    runner = Runner(agent=orchestrator, app_name=APP_NAME, session_service=session_service)

    try:
        query = question['auto'] + question['data'] if autoplanning else \
            question['instruct'] + question['data']

        if question['choices']:
            query += '\n'.join([''] + [
                '{}.{}'.format(chr(ord('A') + i), choice)
                for i, choice in enumerate(question['choices'])
            ])

        full_query = query  # ORCHESTRATOR_SYS_PROMPT is bound via `instruction=`

        print(f"\n--- Processing Question {question['question_id']} ---")
        print(f"Query: {query[:200]}...")

        write_trace(trace_log_path, "\n" + "=" * 100 +
                    f"\nQuestion ID: {question['question_id']}    "
                    f"START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 100 +
                    f"\n[USER] {full_query}")

        save_chat_message(chat_log_path, {
            "name": "user", "role": "user", "content": full_query,
            "metadata": {"question_id": question['question_id']}
        })

        session = await session_service.create_session(
            app_name=APP_NAME, user_id="orchestrator",
            session_id=f"q_{question['question_id']}",
        )

        orchestrator_events = []
        final_text = None
        async for event in runner.run_async(
            user_id="orchestrator",
            session_id=session.id,
            new_message=genai_types.Content(role="user", parts=[genai_types.Part(text=full_query)]),
        ):
            orchestrator_events.append(event)
            if event.is_final_response() and event.content and event.content.parts:
                texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
                if texts:
                    final_text = "\n".join(texts)

        final_answer = extract_answer_from_text(final_text)

        # ---- Build conversation_log: orchestrator turns, with each
        # delegation call's specialist trajectory inlined right after it. ----
        conversation_log = [{"role": "user", "content": full_query}]
        trace_iter = iter(_current_subagent_traces)

        for event in orchestrator_events:
            for entry in event_to_log_entries(event):
                conversation_log.append(entry)
                if entry.get("role") == "assistant":
                    delegated = any(_is_delegation_wrapper(item.get("name", ""))
                                     for item in entry["content"] if "name" in item)
                    if delegated:
                        sub = next(trace_iter, None)
                        if sub is not None:
                            sub_trace = []
                            for sub_event in sub["events"]:
                                sub_trace.extend(event_to_log_entries(sub_event))
                            conversation_log.append({
                                "role": "specialist",
                                "kit": sub["kit"],
                                "content": sub_trace,
                            })

        for sub in trace_iter:
            sub_trace = []
            for sub_event in sub["events"]:
                sub_trace.extend(event_to_log_entries(sub_event))
            conversation_log.append({"role": "specialist", "kit": sub["kit"], "content": sub_trace})

        # ---- .chat file ----
        for entry in conversation_log:
            if entry.get("role") == "assistant":
                chat_content = []
                for item in entry["content"]:
                    if "content" in item:
                        chat_content.append({"type": "text", "text": item["content"]})
                    elif "name" in item:
                        chat_content.append({"type": "tool_use", "name": item["name"], "input": item.get("input", {})})
                if chat_content:
                    save_chat_message(chat_log_path, {
                        "name": question['question_id'], "role": "assistant",
                        "content": chat_content, "metadata": None
                    })
            elif entry.get("role") == "tool":
                save_chat_message(chat_log_path, {
                    "name": "system", "role": "system",
                    "content": [{
                        "type": "tool_result",
                        "output": [{"type": "text", "text": entry["content"][0]["output"][0]["text"], "annotations": None, "meta": None}],
                        "name": entry.get("name")
                    }],
                    "metadata": None
                })
            elif entry.get("role") == "specialist":
                save_chat_message(chat_log_path, {
                    "name": f"{entry['kit']}_specialist", "role": "specialist",
                    "content": entry["content"],
                    "metadata": {"question_id": question['question_id'], "kit": entry['kit']}
                })

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
            "name": "system", "role": "system",
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
    print("Initializing multi-agent (supervisor + 5 kit specialists) Earth Science Agent [ADK]...")

    init_global_params()
    chat_log_path = init_chat_logger()
    trace_log_path = init_trace_logger()
    print(f"Chat log will be saved to: {chat_log_path}")
    print(f"Human-readable trace will be saved to: {trace_log_path}")

    llm, mcp_server_params = load_adk_config()
    orchestrator, session_service, kit_sessions_holder, toolsets = await create_multi_agent_system(
        llm, mcp_server_params, trace_log_path
    )

    try:
        questions = load_questions()[:6]
        if RETRY_IDS is not None:
            retry_set = set(RETRY_IDS)
            questions = [q for q in questions if q['question_id'] in retry_set]
        if BATCH_TOTAL > 1:
            questions = [q for i, q in enumerate(questions) if i % BATCH_TOTAL == BATCH_INDEX]
        print(f"Loaded {len(questions)} questions for evaluation"
              + (f" (batch {BATCH_INDEX+1}/{BATCH_TOTAL})" if BATCH_TOTAL > 1 else ""))

        results = []
        for question in tqdm(questions, desc="Processing questions"):
            answer = await handle_question(orchestrator, session_service, kit_sessions_holder,
                                             question, chat_log_path, trace_log_path)
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
        for toolset in toolsets.values():
            try:
                await toolset.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())