"""
Patches litellm's acompletion/completion to fix a role-ordering violation
that occurs when google-adk sub-agents are exposed to a parent orchestrator
as callable tools (AgentTool).

ROOT CAUSE:
When a sub-agent finishes running as a tool call, google-adk replays that
sub-agent's full internal transcript back into the *parent* orchestrator's
message history as a sequence of `user`-role "For context:" messages,
positioned immediately after the single `tool`-role summary message, e.g.:

    {"role": "assistant", "tool_calls": [...]}
    {"role": "tool", "content": "..."}
    {"role": "user", "content": "..."}      <-- ADK's internal trace, not a
    {"role": "user", "content": [...]}          real user turn
    ...

This is harmless against Gemini's native API, which doesn't enforce strict
role alternation. It is a hard violation of OpenAI's chat completions schema,
which requires every `tool` message to be immediately followed by an
`assistant` message. Routing an ADK sub-agent ecosystem through LiteLlm to
an OpenAI-compatible endpoint (as opposed to native Gemini) will hit this on
essentially every sub-agent round trip:

    openai.BadRequestError: Error code: 400 - {'error': {'message':
    "Unexpected role 'user' after role 'tool'", ...}}

FIX:
Monkeypatch litellm.acompletion / litellm.completion so that, right before
the request is sent, we walk the outgoing `messages` list and insert a
minimal placeholder `assistant` message wherever a `tool` message is
immediately followed by a `user` message. This satisfies OpenAI's alternation
requirement without dropping or rewriting any real content — the offending
`user` messages are ADK's own internal tracing, so nothing semantically
meaningful is lost by the insertion.

USAGE:
    from agenticeoecosystem.agents.subagents.eve_patch import apply as apply_eve_patch
    apply_eve_patch()

Call this once, before constructing any Runner/Agent, ideally at the top of
your entrypoint module (multi_agent.py).
"""

from __future__ import annotations

import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)

_PLACEHOLDER_ASSISTANT_CONTENT = "(continuing)"

_original_acompletion = litellm.acompletion
_original_completion = litellm.completion

_applied = False


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Insert a placeholder assistant message wherever `tool` is immediately
    followed by `user`, to satisfy OpenAI-compatible strict role alternation.
    Returns a new list; does not mutate the input.
    """
    if not messages:
        return messages

    fixed: list[dict[str, Any]] = []
    inserted = 0
    for msg in messages:
        role = msg.get("role")
        if fixed and fixed[-1].get("role") == "tool" and role == "user":
            fixed.append(
                {"role": "assistant", "content": _PLACEHOLDER_ASSISTANT_CONTENT}
            )
            inserted += 1
        fixed.append(msg)

    if inserted:
        logger.debug(
            "eve_patch: inserted %d placeholder assistant message(s) to fix "
            "tool->user role ordering",
            inserted,
        )
    return fixed


async def _patched_acompletion(*args: Any, **kwargs: Any):
    if "messages" in kwargs:
        kwargs["messages"] = _sanitize_messages(kwargs["messages"])
    elif args:
        # messages is sometimes passed positionally; litellm's public
        # signature is acompletion(model, messages=None, ...), so guard
        # defensively rather than assume position 0/1.
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, list) and a and isinstance(a[0], dict) and "role" in a[0]:
                args[i] = _sanitize_messages(a)
                break
        args = tuple(args)
    return await _original_acompletion(*args, **kwargs)


def _patched_completion(*args: Any, **kwargs: Any):
    if "messages" in kwargs:
        kwargs["messages"] = _sanitize_messages(kwargs["messages"])
    elif args:
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, list) and a and isinstance(a[0], dict) and "role" in a[0]:
                args[i] = _sanitize_messages(a)
                break
        args = tuple(args)
    return _original_completion(*args, **kwargs)


def apply() -> None:
    """Idempotently monkeypatch litellm.acompletion/completion with the
    role-sanitizing wrappers. Safe to call multiple times."""
    global _applied
    if _applied:
        logger.debug("eve_patch: already applied, skipping")
        return

    litellm.acompletion = _patched_acompletion
    litellm.completion = _patched_completion
    _applied = True
    logger.info("eve_patch: patched litellm.acompletion/completion for tool->user role fix")


def remove() -> None:
    """Restore the original litellm.acompletion/completion functions."""
    global _applied
    litellm.acompletion = _original_acompletion
    litellm.completion = _original_completion
    _applied = False
    logger.info("eve_patch: removed litellm patch, originals restored")