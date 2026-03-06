#!/usr/bin/env python3
"""
Discord thread fork request tool.

This tool does not create the thread directly. It records a structured request
that the gateway runtime interprets after the model finishes its current turn.
That keeps Discord API calls inside the live gateway event loop instead of the
agent's synchronous tool-execution path.
"""

import json
import os
from typing import Any, Callable, Dict, Iterable, Optional, Set


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().strip("\"'").lower()
    if not text:
        return default
    if text in ("true", "1", "yes", "y", "on"):
        return True
    if text in ("false", "0", "no", "n", "off"):
        return False
    return default


def _parse_id_set(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip() for v in value if str(v).strip()}
    text = str(value).strip()
    if not text:
        return set()
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
        except Exception:
            decoded = None
        if isinstance(decoded, list):
            return {str(v).strip() for v in decoded if str(v).strip()}
    return {part.strip() for part in text.split(",") if part.strip()}


def _current_discord_auto_fork_settings() -> Dict[str, Any]:
    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    extra = {}
    if Platform.DISCORD in config.platforms:
        discord_extra = config.platforms[Platform.DISCORD].extra
        if isinstance(discord_extra, dict):
            extra = discord_extra

    enabled_env = (os.getenv("DISCORD_AUTO_FORK_ENABLED", "") or "").strip()
    enabled = _parse_bool(
        enabled_env if enabled_env else extra.get("auto_fork_enabled"),
        default=False,
    )

    allowed_env = (os.getenv("DISCORD_AUTO_FORK_ALLOWED_CHANNEL_IDS", "") or "").strip()
    allowed_ids = _parse_id_set(
        allowed_env if allowed_env else extra.get("auto_fork_allowed_channel_ids")
    )

    notice_env = (os.getenv("DISCORD_AUTO_FORK_MAIN_CHANNEL_NOTICE", "") or "").strip()
    main_channel_notice = _parse_bool(
        notice_env if notice_env else extra.get("auto_fork_main_channel_notice"),
        default=True,
    )

    return {
        "enabled": enabled,
        "allowed_channel_ids": allowed_ids,
        "main_channel_notice": main_channel_notice,
    }


def check_fork_thread_requirements() -> bool:
    platform = (os.getenv("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
    chat_type = (os.getenv("HERMES_SESSION_CHAT_TYPE", "") or "").strip().lower()
    chat_id = (os.getenv("HERMES_SESSION_CHAT_ID", "") or "").strip()
    live_available_env = (os.getenv("HERMES_DISCORD_FORK_THREAD_AVAILABLE", "") or "").strip()

    if platform != "discord":
        return False
    if chat_type not in {"group", "channel"}:
        return False
    if live_available_env and not _parse_bool(live_available_env, default=False):
        return False

    settings = _current_discord_auto_fork_settings()
    if not settings["enabled"]:
        return False

    allowed_ids: Iterable[str] = settings["allowed_channel_ids"]
    return not allowed_ids or chat_id in allowed_ids


def fork_thread_tool(
    args: Dict[str, Any],
    callback: Optional[Callable[..., Dict[str, Any]]] = None,
    **_kwargs,
) -> str:
    platform = (os.getenv("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
    chat_type = (os.getenv("HERMES_SESSION_CHAT_TYPE", "") or "").strip().lower()
    chat_id = (os.getenv("HERMES_SESSION_CHAT_ID", "") or "").strip()
    live_available_env = (os.getenv("HERMES_DISCORD_FORK_THREAD_AVAILABLE", "") or "").strip()
    if platform != "discord":
        return json.dumps(
            {"success": False, "error": "fork_thread is only available in Discord gateway sessions."},
            ensure_ascii=False,
        )
    if chat_type not in {"group", "channel"}:
        return json.dumps(
            {"success": False, "error": "fork_thread can only be used from Discord server channels."},
            ensure_ascii=False,
        )
    if live_available_env and not _parse_bool(live_available_env, default=False):
        return json.dumps(
            {"success": False, "error": "Discord live thread forking is not available in this session."},
            ensure_ascii=False,
        )

    settings = _current_discord_auto_fork_settings()
    if not settings["enabled"]:
        return json.dumps(
            {"success": False, "error": "Automatic Discord thread forking is disabled in config."},
            ensure_ascii=False,
        )
    if settings["allowed_channel_ids"] and chat_id not in settings["allowed_channel_ids"]:
        return json.dumps(
            {"success": False, "error": "This channel is not in the auto-fork allowlist."},
            ensure_ascii=False,
        )

    title = str(args.get("title", "") or "").strip()
    visibility = str(args.get("visibility", "auto") or "auto").strip().lower()
    reason = str(args.get("reason", "") or "").strip()
    if visibility not in {"auto", "public", "private"}:
        return json.dumps(
            {"success": False, "error": "visibility must be one of: auto, public, private."},
            ensure_ascii=False,
        )

    if callback is None:
        return json.dumps(
            {
                "success": False,
                "error": "fork_thread is not available in this execution context.",
            },
            ensure_ascii=False,
        )

    try:
        payload = callback(title=title, visibility=visibility, reason=reason)
    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to fork thread: {exc}",
            },
            ensure_ascii=False,
        )

    if not isinstance(payload, dict):
        payload = {"success": False, "error": "fork_thread callback returned an invalid response."}
    payload.setdefault("title", title)
    payload.setdefault("visibility", visibility)
    payload.setdefault("reason", reason)
    payload.setdefault("redirect_final_response", True)
    payload.setdefault("main_channel_notice", settings["main_channel_notice"])
    return json.dumps(payload, ensure_ascii=False)


FORK_THREAD_SCHEMA = {
    "name": "fork_thread",
    "description": """Fork aggressively. Default to forking when ANY of these apply:

Repeated back-and-forth (3+ exchanges) on a single question or argument
The response will exceed ~200 words and is not the current main topic/conversation in the channel
The topic only benefits the person asking
Trivially wrong claims being debated (don't dignify with main-channel real estate)
Debugging, troubleshooting, or "how do I" walkthroughs
Off-topic tangents relative to what the channel is actually discussing

Stay in-channel only for: brief factual answers, responses relevant to an active group discussion, or when multiple people are engaged in the same thread of conversation. You should fork by default, if you don't think you can improve the signal of the conversation for other people that are not the user calling you, as your FIRST tool call.
""",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Optional thread title. Keep it short and descriptive.",
            },
            "visibility": {
                "type": "string",
                "enum": ["auto", "public", "private"],
                "description": "Thread visibility. Use auto unless the user explicitly requests otherwise.",
            },
            "reason": {
                "type": "string",
                "description": "Short private rationale for the fork decision; used only for runtime metadata.",
            },
        },
        "required": [],
    },
}


from tools.registry import registry

registry.register(
    name="fork_thread",
    toolset="discord_routing",
    schema=FORK_THREAD_SCHEMA,
    handler=fork_thread_tool,
    check_fn=check_fork_thread_requirements,
)
