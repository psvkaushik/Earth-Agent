import importlib
import logging
import pkgutil
import os
import sys

from google.adk.agents.llm_agent import Agent

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Earth-Agent/agents/subagents
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))     # .../Earth-Agent

# CHANGED (the actual fix): the traceback fails on `No module named 'agents'`,
# meaning Earth-Agent/ isn't on sys.path — this happens whenever this file is
# run directly (python agents/subagents/__init__.py) or launched from a cwd
# other than Earth-Agent/, since Python only auto-adds the *script's own*
# directory, never its parent. Insert the project root explicitly so
# `agents` (and therefore `agents.subagents.<module>`) is always importable,
# regardless of how/where this gets invoked.
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .llm_init import llm  # noqa: F401  (re-exported for `from agents.subagents import llm`)

ALL_SUB_AGENTS: list[Agent] = []

# __name__ is "agents.subagents" for a normal package import, or "__main__"
# if this file is executed directly — fall back to the real dotted path
# either way so importlib.import_module below always gets a valid package.
_PACKAGE = __name__ if __name__ != "__main__" else "agents.subagents"

for _, module_name, _ in pkgutil.iter_modules([_PKG_DIR]):
    if not module_name.startswith("sub_agent_"):
        continue
    module = importlib.import_module(f"{_PACKAGE}.{module_name}")
    for value in vars(module).values():
        if isinstance(value, Agent) and value not in ALL_SUB_AGENTS:
            ALL_SUB_AGENTS.append(value)
            logger.info("Loaded sub-agent '%s' from %s", value.name, module_name)

if not ALL_SUB_AGENTS:
    logger.warning("No sub-agents were initialized — check MCP toolset/env configs.")

print(f"Loaded {len(ALL_SUB_AGENTS)} sub-agents: {[agent.name for agent in ALL_SUB_AGENTS]}")