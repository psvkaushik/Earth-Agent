"""
Shared path-resolution helpers for every MCP tool server in agent/tools/
(Perception.py, Statistics.py, Index.py). Centralized here so all three
tool subprocesses resolve paths identically — a raster written by one
sub-agent's tool must be readable by whichever sub-agent's tool is asked
to read it next, regardless of which one originally wrote it. Before this
existed, each tool file had its own --temp_dir (tmp/perception,
tmp/statistics, tmp/index) and its own ad hoc resolution logic, which
produced exactly the failure this module fixes: a raster written under
one TEMP_DIR silently unreadable via a different TEMP_DIR, or — worse —
an absolute output_path silently escaping TEMP_DIR entirely because
`Path(TEMP_DIR) / Path(absolute)` discards the left side in pathlib.
"""
import argparse
from pathlib import Path

# agent/tools/common.py -> two levels up is the project root (Earth-Agent/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Single shared directory for all tool-written outputs across every
# sub-agent (perception, statistics, index) for the current run. One
# directory means a raster path returned by one sub-agent's tool is
# directly usable, unmodified, by another sub-agent's tool — there is no
# longer a "which agent's TEMP_DIR is this relative to" question.
DEFAULT_TEMP_DIR = PROJECT_ROOT / "tmp" / "shared"


def parse_temp_dir(default: Path = DEFAULT_TEMP_DIR) -> Path:
    """Parse --temp_dir from argv, falling back to `default` instead of
    crashing at import time if the launcher forgot the flag. (Previously
    `Path(args.temp_dir)` with args.temp_dir == None raised TypeError
    before FastMCP ever registered a single tool, which killed the whole
    MCP session silently from the client's point of view — just
    "Connection closed", no indication why.)"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp_dir', type=str, default=None)
    args, _unknown = parser.parse_known_args()
    temp_dir = Path(args.temp_dir) if args.temp_dir else default
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def make_resolvers(temp_dir: Path):
    """Build a (_resolve_path, _resolve_output_path) pair bound to the
    given temp_dir. Every tool file calls this once at import time with
    its own parsed temp_dir (normally the same shared directory across
    all three tool files, but each stays independently overridable)."""

    def _resolve_path(path: str) -> str:
        """Resolve a possibly-relative input path for READING. Checked in
        order:
          1) already absolute -> used as-is
          2) PROJECT_ROOT-relative -> source/benchmark data
          3) temp_dir-relative -> an output some other tool already wrote
        Falls back to the PROJECT_ROOT-relative guess if neither exists,
        so a resulting FileNotFoundError still shows a sensible path
        rather than silently resolving to nothing."""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        project_relative = PROJECT_ROOT / p
        if project_relative.exists():
            return str(project_relative)
        temp_relative = temp_dir / p
        if temp_relative.exists():
            return str(temp_relative)
        return str(project_relative)

    def _resolve_output_path(output_path: str) -> Path:
        """Force a WRITE target under temp_dir, even if output_path is
        absolute (e.g. a caller reused a previous tool's full returned
        path as this call's output_path argument). Without this,
        `temp_dir / Path(output_path)` would silently DISCARD temp_dir
        when output_path is absolute, since in pathlib
        Path('/a') / '/b' == Path('/b') — writing outside temp_dir
        entirely with no error. An absolute path is rebased under
        temp_dir instead of being allowed to escape it."""
        p = Path(output_path)
        if p.is_absolute():
            try:
                p = p.relative_to(temp_dir)
            except ValueError:
                p = Path(*p.parts[1:]) if p.parts else p
        return temp_dir / p

    return _resolve_path, _resolve_output_path