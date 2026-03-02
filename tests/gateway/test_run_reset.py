"""Tests for gateway reset command behavior."""

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def test_reset_command_clears_discord_channel_context(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="ch123",
        chat_name="Test place / #bot-channel",
        chat_type="group",
        user_id="u123",
        user_name="giftedgummybee",
    )
    # Ensure an existing session so /new executes the reset path.
    runner.session_store.get_or_create_session(source)

    called = []
    runner.adapters[Platform.DISCORD] = SimpleNamespace(
        reset_channel_context=lambda channel_id: called.append(channel_id)
    )

    event = MessageEvent(
        text="/new",
        message_type=MessageType.COMMAND,
        source=source,
    )
    response = asyncio.run(runner._handle_reset_command(event))

    assert called == ["ch123"]
    assert "Session reset" in response


def test_discord_plain_text_turn_uses_context_without_duplicate_tail(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="ch123",
        chat_name="Test place / #bot-channel",
        chat_type="group",
        user_id="u123",
        user_name="giftedgummybee",
    )

    extra_context = "[Discord context]\n00:11 giftedgummybee: previous message"
    captured = {}

    async def _fake_run_agent(*, message, **kwargs):
        captured["message"] = message
        return {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "ok"},
            ],
            "history_input_len": 0,
            "tools": [],
        }

    runner._run_agent = _fake_run_agent

    event = MessageEvent(
        text="what is 3+4?",
        message_type=MessageType.TEXT,
        source=source,
        extra_context=extra_context,
    )

    response = asyncio.run(runner._handle_message(event))

    assert response == "ok"
    assert captured["message"] == extra_context


def test_multimodal_image_payload_prefers_local_base64_data_url(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z6xkAAAAASUVORK5CYII="
    )
    local_image = tmp_path / "tiny.png"
    local_image.write_bytes(png_bytes)

    payload_url = runner._resolve_image_payload_url(
        {
            "path": str(local_image),
            "media_type": "image/png;source_url=https://cdn.discordapp.com/example.png",
            "source_url": "https://cdn.discordapp.com/example.png",
        }
    )

    assert payload_url is not None
    assert payload_url.startswith("data:image/png;base64,")
    encoded = payload_url.split(",", 1)[1]
    assert base64.b64decode(encoded) == png_bytes


def test_multimodal_image_payload_returns_none_when_no_data_url_possible(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    with patch.object(runner, "_http_image_to_data_url", return_value=None):
        payload_url = runner._resolve_image_payload_url(
            {
                "path": str(tmp_path / "missing.png"),
                "media_type": "image/png",
                "source_url": "https://cdn.discordapp.com/example.png",
            }
        )

    assert payload_url is None


def test_sanitize_multimodal_history_content_rewrites_image_urls(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    with patch.object(
        runner,
        "_resolve_image_payload_url",
        return_value="data:image/png;base64,ZmFrZQ==",
    ):
        sanitized = runner._sanitize_multimodal_history_content(
            [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "https://cdn.discordapp.com/example.png"}},
            ]
        )

    assert sanitized[0] == {"type": "text", "text": "hello"}
    assert sanitized[1]["type"] == "image_url"
    assert sanitized[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_sanitize_multimodal_history_content_replaces_unavailable_images(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    with patch.object(runner, "_resolve_image_payload_url", return_value=None):
        sanitized = runner._sanitize_multimodal_history_content(
            [{"type": "image_url", "image_url": {"url": "https://cdn.discordapp.com/example.png"}}]
        )

    assert sanitized == [{"type": "text", "text": "[image omitted: unavailable]"}]


def test_resolve_image_payload_url_preserves_existing_data_url(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    data_url = "data:image/png;base64,ZmFrZQ=="
    payload_url = runner._resolve_image_payload_url(
        {
            "path": data_url,
            "media_type": "image/png",
            "source_url": "",
        }
    )

    assert payload_url == data_url


def test_sanitize_multimodal_history_content_keeps_data_url(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    data_url = "data:image/png;base64,ZmFrZQ=="
    sanitized = runner._sanitize_multimodal_history_content(
        [{"type": "image_url", "image_url": {"url": data_url}}]
    )

    assert sanitized == [{"type": "image_url", "image_url": {"url": data_url}}]


def test_run_agent_accepts_multimodal_message_without_scope_error(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_db = None
    runner.adapters = {}

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="ch123",
        chat_name="Test place / #bot-channel",
        chat_type="group",
        user_id="u123",
        user_name="giftedgummybee",
    )

    class FakeAIAgent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, user_message, conversation_history=None):
            assert isinstance(user_message, list)
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
                "api_calls": 1,
                "interrupted": False,
            }

    with patch("run_agent.AIAgent", FakeAIAgent):
        result = asyncio.run(
            runner._run_agent(
                message=[{"type": "text", "text": "hello"}],
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="session-key-1",
            )
        )

    assert result["final_response"] == "ok"
