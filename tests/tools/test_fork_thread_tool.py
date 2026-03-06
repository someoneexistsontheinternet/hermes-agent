"""Tests for tools/fork_thread_tool.py."""

import json

from gateway.config import GatewayConfig, Platform, PlatformConfig
from tools.fork_thread_tool import check_fork_thread_requirements, fork_thread_tool


def _gateway_config(enabled=True, allowed_channel_ids=None, main_notice=True):
    return GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={
                    "auto_fork_enabled": enabled,
                    "auto_fork_allowed_channel_ids": allowed_channel_ids or [],
                    "auto_fork_main_channel_notice": main_notice,
                },
            )
        }
    )


def test_check_fork_thread_requirements_requires_discord_group(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "ch123")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: _gateway_config(enabled=True))

    assert check_fork_thread_requirements() is True

    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "dm")
    assert check_fork_thread_requirements() is False


def test_check_fork_thread_requirements_honors_channel_allowlist(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "ch999")
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: _gateway_config(enabled=True, allowed_channel_ids=["ch123"]),
    )

    assert check_fork_thread_requirements() is False


def test_fork_thread_tool_invokes_callback(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "ch123")
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: _gateway_config(enabled=True, allowed_channel_ids=["ch123"], main_notice=False),
    )
    captured = {}

    result = json.loads(
        fork_thread_tool(
            {"title": "deep-dive", "visibility": "private", "reason": "One-person follow-up"},
            callback=lambda **kwargs: captured.update(kwargs) or {"success": True, "thread_id": "th1"},
        )
    )

    assert result["success"] is True
    assert captured == {
        "title": "deep-dive",
        "visibility": "private",
        "reason": "One-person follow-up",
    }
    assert result["title"] == "deep-dive"
    assert result["visibility"] == "private"
    assert result["reason"] == "One-person follow-up"
    assert result["redirect_final_response"] is True
    assert result["main_channel_notice"] is False


def test_fork_thread_tool_without_callback_errors(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "ch123")
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: _gateway_config(enabled=True, allowed_channel_ids=["ch123"]),
    )

    result = json.loads(fork_thread_tool({"title": "deep-dive"}))

    assert result["success"] is False
    assert "execution context" in result["error"]
