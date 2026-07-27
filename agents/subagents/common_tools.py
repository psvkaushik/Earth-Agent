"""
Tools shared across every sub-agent. Kept in one place so behavior (path
resolution, failure mode) can't silently diverge between agents the way it
did when get_filelist was defined once in sub_agent_perception.py (dict-on-
error, PROJECT_ROOT-anchored) and again inside the MCP tools file (raises
on error, resolved against whatever the subprocess's cwd happens to be).

Every sub-agent should import get_filelist from here and add it to its own
`tools` list alongside its McpToolset. This keeps it a local, in-process
Python call for every agent rather than a round-trip through an MCP
subprocess, and — more importantly — means an agent can enumerate a
directory's contents itself instead of asking the orchestrator to pass a
file list in the message. Large file lists passed agent-to-agent in text
are pure token waste; a directory path is enough.
"""
import os

# agents/subagents/common_tools.py -> two levels up is the project root
# (Earth-Agent/). Anchoring here means a relative path like
# "benchmark/data/question189" resolves the same way no matter what
# directory the process was launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_filelist(dir_path: str):
    """
    Returns a list of files in the specified directory.

    Parameters:
        dir_path (str): Path to the directory. May be given relative to the
            project root (e.g. "benchmark/data/questionX") — it will be
            resolved against PROJECT_ROOT regardless of the process's cwd,
            so it works the same whether you run from Earth-Agent/,
            Earth-Agent/agents/, or anywhere else. Absolute paths are used
            as-is.

    Returns:
        list: List of file names in the directory (dotfiles excluded), or
        a dict with an "error" key describing what went wrong. Returning a
        dict on failure rather than raising is deliberate — an uncaught
        exception here crashes the whole sub-agent node with no detail the
        orchestrator can act on; a returned error string at least gives it
        something to retry or report against.
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


# Shared instruction fragment every sub-agent prompt should include verbatim
# so behavior is consistent across agents rather than each prompt describing
# its own ad hoc convention for handling file paths.
FILELIST_USAGE_NOTE = """\
## File discovery — do this yourself, don't wait to be told

You have `get_filelist` available directly as a tool, independent of any \
perception/statistics MCP tools. Use it yourself to enumerate a directory's \
contents rather than asking the orchestrator to list files for you or \
waiting for it to paste a file list into your subtask — the orchestrator \
will typically just hand you a directory path and expect you to resolve it.

When you need to report file paths back to the orchestrator in your \
RESULT:
- If there are a handful (roughly 10 or fewer), name them directly in your \
  result text.
- If there are many, do NOT enumerate them all in your RESULT — that wastes \
  tokens on both sides. Instead state the directory path and let the \
  orchestrator (or whichever agent needs them next) call `get_filelist` \
  itself. Only call out specific filenames if they're individually \
  significant to the answer (e.g. "the outlier was file X").
"""