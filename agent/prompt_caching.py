"""Helpers for Anthropic/OpenRouter prompt caching.

Hermes uses explicit Anthropic cache breakpoints on the final message in each
request. This keeps the cached prefix incremental across tool-loop turns while
remaining compatible with routed providers such as Google Vertex and Bedrock.
"""

import copy
from typing import Any, Dict, List


def _normalize_text_content(msg: dict) -> None:
    """Canonicalize string content into Anthropic-style text blocks."""
    if msg.get("role") == "tool":
        return

    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content}]


def _apply_cache_marker(msg: dict, cache_marker: dict) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        msg["cache_control"] = cache_marker
        return

    if content is None:
        msg["cache_control"] = cache_marker
        return

    _normalize_text_content(msg)
    content = msg.get("content")

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def build_anthropic_cache_control(cache_ttl: str = "5m") -> Dict[str, str]:
    """Build a top-level Anthropic/OpenRouter cache_control payload."""
    marker: Dict[str, str] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"
    return marker


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
) -> List[Dict[str, Any]]:
    """Apply a single incremental Anthropic cache breakpoint.

    Marks only the final non-system message (or the sole system message) so the
    provider can reuse the longest previously cached prefix while we avoid
    rotating multiple breakpoint positions across turns.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    for msg in messages:
        _normalize_text_content(msg)

    marker = build_anthropic_cache_control(cache_ttl)

    target_idx = 0
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") != "system":
            target_idx = idx
            break

    _apply_cache_marker(messages[target_idx], marker)

    return messages
