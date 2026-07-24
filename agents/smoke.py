"""
Smoke test: confirms every sub_agent_*.py under agents/subagents/ loads and
exposes at least one google.adk Agent instance.

Run from anywhere, either:
    cd ~/Desktop/Earth-Agent && python -m agents.multi_agent
or directly:
    python agents/multi_agent.py
    python multi_agent.py   (from inside agents/)
"""
import os
import sys

# Bootstrap: this file lives at Earth-Agent/agents/multi_agent.py. When run
# directly, Python only adds THIS file's directory (agents/) to sys.path,
# not its parent — so `agents` itself isn't importable yet. Insert the
# project root explicitly so `from agents.subagents import ...` below
# always resolves, regardless of how/where this script is launched.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.subagents import ALL_SUB_AGENTS

if not ALL_SUB_AGENTS:
    raise SystemExit("No sub-agents loaded — check the warnings/errors above.")

print("\nSub-agents ready:")
for agent in ALL_SUB_AGENTS:
    print(f"  - {agent.name}: {agent.description}")