"""Tests for gateway reset command behavior."""

import asyncio
import base64
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_context


def test_gateway_prefers_hermes_home_for_context_files(monkeypatch, tmp_path):
    hermes_home = tmp_path / "workspace"
    hermes_home.mkdir()
    (hermes_home / "AGENTS.md").write_text("workspace rules")

    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    monkeypatch.setattr("gateway.run._hermes_home", hermes_home)

    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    assert runner._context_cwd == str(hermes_home)


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


def test_discord_plain_text_turn_appends_current_user_and_request(tmp_path):
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
    assert captured["message"] == (
        f"{extra_context}\n\n"
        "**Current user**: giftedgummybee\n"
        "**Request**: what is 3+4?"
    )


def test_discord_plain_text_turn_appends_reply_context_for_current_turn(tmp_path):
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

    class _FakeDiscordAdapter:
        is_connected = True
        _archive_db = SimpleNamespace(
            get_reply_context=lambda **kwargs: {
                "author_display": "Hermes-Bot",
                "preview": "2 plus 2 equals 4",
            }
        )

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SimpleNamespace(success=True)

    runner.adapters[Platform.DISCORD] = _FakeDiscordAdapter()

    event = MessageEvent(
        text="what is 3+4?",
        message_type=MessageType.TEXT,
        source=source,
        extra_context=extra_context,
        reply_to_message_id="m42",
        raw_message=SimpleNamespace(reference=SimpleNamespace(channel_id="ch123")),
    )

    response = asyncio.run(runner._handle_message(event))

    assert response == "ok"
    assert captured["message"] == (
        f"{extra_context}\n\n"
        '**Replying to Hermes-Bot**: "2 plus 2 equals 4"\n'
        "**Current user**: giftedgummybee\n"
        "**Request**: what is 3+4?"
    )


def test_handle_message_applies_delivery_state_reply_target(tmp_path):
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

    async def _fake_run_agent(*, message, delivery_state, **kwargs):
        delivery_state["reply_to"] = "m999"
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
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m111",
    )

    response = asyncio.run(runner._handle_message(event))

    assert response == "ok"
    assert getattr(event, "_response_reply_to_message_id") == "m999"


def test_handle_message_records_parent_handoff_when_thread_resume_is_deferred(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    class _FakeDiscordAdapter:
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SimpleNamespace(success=True)

    runner.adapters[Platform.DISCORD] = _FakeDiscordAdapter()

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="ch123",
        chat_name="Test place / #bot-channel",
        chat_type="group",
        user_id="u123",
        user_name="giftedgummybee",
    )
    thread_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="th789",
        chat_name="deep-dive",
        chat_type="thread",
        user_id="u123",
        user_name="giftedgummybee",
        thread_id="th789",
    )

    async def _fake_run_agent(*, message, delivery_state, **kwargs):
        delivery_state.update(
            {
                "chat_id": "th789",
                "reply_to": None,
                "thread_result": {
                    "thread_id": "th789",
                    "thread_mention": "<#th789>",
                    "thread_name": "deep-dive",
                },
                "thread_source": thread_source,
                "thread_session_id": "session-thread",
                "thread_session_key": "agent:main:discord:thread:th789",
                "transcript_notice": "[Continued in thread <#th789>]",
                "main_notice": "Taking this to a thread: <#th789>",
            }
        )
        return {
            "final_response": "",
            "messages": [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "fork_thread", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_a",
                    "content": json.dumps(
                        {
                            "success": True,
                            "requested": True,
                            "thread_id": "th789",
                            "thread_mention": "<#th789>",
                            "title": "deep-dive",
                        }
                    ),
                },
            ],
            "history_input_len": 0,
            "tools": [],
            "deferred_pending_event": True,
        }

    runner._run_agent = _fake_run_agent

    event = MessageEvent(
        text="Can you go deeper?",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m123",
    )

    response = asyncio.run(runner._handle_message(event))

    assert response == ""
    assert getattr(event, "_response_handled") is True

    session_entry = runner.session_store.get_or_create_session(source)
    history = runner.session_store.load_transcript(session_entry.session_id)
    assert any(
        msg.get("role") == "assistant"
        and msg.get("content") == "[Continued in thread <#th789>]"
        for msg in history
    )


def test_set_session_env_tracks_live_discord_fork_availability(tmp_path):
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
    context = build_session_context(source, runner.config, connected_platforms=[])

    runner.adapters[Platform.DISCORD] = SimpleNamespace(
        is_connected=True,
        is_auto_fork_available=lambda _source: True,
    )
    runner._set_session_env(context)
    assert os.environ["HERMES_DISCORD_FORK_THREAD_AVAILABLE"] == "1"

    runner.adapters[Platform.DISCORD] = SimpleNamespace(
        is_connected=False,
        is_auto_fork_available=lambda _source: True,
    )
    runner._set_session_env(context)
    assert os.environ["HERMES_DISCORD_FORK_THREAD_AVAILABLE"] == "0"

    runner._clear_session_env()
    assert "HERMES_DISCORD_FORK_THREAD_AVAILABLE" not in os.environ


def test_handle_message_appends_discord_cost_summary_and_updates_tokens(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )
    runner.config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True, token="fake-token")

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="ch123",
        chat_name="Test place / #bot-channel",
        chat_type="group",
        user_id="u123",
        user_name="giftedgummybee",
    )

    async def _fake_run_agent(*, message, **kwargs):
        return {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "ok"},
            ],
            "history_input_len": 0,
            "tools": [],
            "request_usage": {
                "prompt_tokens": 30,
                "completion_tokens": 12,
                "total_tokens": 42,
                "cost_usd": 0.034,
            },
        }

    runner._run_agent = _fake_run_agent

    event = MessageEvent(
        text="what is 3+4?",
        message_type=MessageType.TEXT,
        source=source,
    )

    response = asyncio.run(runner._handle_message(event))

    assert response == "ok\n\n-# $0.03 USD spent"
    session_entry = runner.session_store.get_or_create_session(source)
    assert session_entry.input_tokens == 30
    assert session_entry.output_tokens == 12
    assert session_entry.total_tokens == 42


def test_handle_message_omits_discord_cost_summary_when_disabled(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )
    runner.config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={"cost_summary_enabled": False},
    )

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="ch123",
        chat_name="Test place / #bot-channel",
        chat_type="group",
        user_id="u123",
        user_name="giftedgummybee",
    )

    async def _fake_run_agent(*, message, **kwargs):
        return {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "ok"},
            ],
            "history_input_len": 0,
            "tools": [],
            "request_usage": {
                "prompt_tokens": 30,
                "completion_tokens": 12,
                "total_tokens": 42,
                "cost_usd": 0.034,
            },
        }

    runner._run_agent = _fake_run_agent

    event = MessageEvent(
        text="what is 3+4?",
        message_type=MessageType.TEXT,
        source=source,
    )

    response = asyncio.run(runner._handle_message(event))

    assert response == "ok"


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


def test_multimodal_image_payload_uses_sniffed_mime_on_mismatch(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z6xkAAAAASUVORK5CYII="
    )
    local_image = tmp_path / "tiny.webp"
    local_image.write_bytes(png_bytes)

    payload_url = runner._resolve_image_payload_url(
        {
            "path": str(local_image),
            "media_type": "image/webp;source_url=https://cdn.discordapp.com/example.webp",
            "source_url": "https://cdn.discordapp.com/example.webp",
        }
    )

    assert payload_url is not None
    assert payload_url.startswith("data:image/png;base64,")
    encoded = payload_url.split(",", 1)[1]
    assert base64.b64decode(encoded) == png_bytes


def test_http_image_to_data_url_uses_sniffed_mime_on_header_mismatch(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z6xkAAAAASUVORK5CYII="
    )

    class _FakeHTTPResponse:
        def __init__(self, body: bytes):
            self._body = body
            self.headers = {"Content-Type": "image/webp"}

        def read(self, _limit: int) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    with patch("gateway.run.urllib.request.urlopen", return_value=_FakeHTTPResponse(png_bytes)):
        payload_url = runner._http_image_to_data_url(
            "https://cdn.discordapp.com/example.webp",
            media_type="image/webp",
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


def test_resolve_image_payload_url_normalizes_mismatched_data_url(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z6xkAAAAASUVORK5CYII="
    )
    bad_data_url = "data:image/webp;base64," + base64.b64encode(png_bytes).decode("ascii")

    payload_url = runner._resolve_image_payload_url(
        {
            "path": bad_data_url,
            "media_type": "image/webp",
            "source_url": "",
        }
    )

    assert payload_url is not None
    assert payload_url.startswith("data:image/png;base64,")
    encoded = payload_url.split(",", 1)[1]
    assert base64.b64decode(encoded) == png_bytes


def test_sanitize_multimodal_history_content_keeps_data_url(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    data_url = "data:image/png;base64,ZmFrZQ=="
    sanitized = runner._sanitize_multimodal_history_content(
        [{"type": "image_url", "image_url": {"url": data_url}}]
    )

    assert sanitized == [{"type": "image_url", "image_url": {"url": data_url}}]


def test_sanitize_multimodal_history_content_normalizes_data_url_mime(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z6xkAAAAASUVORK5CYII="
    )
    bad_data_url = "data:image/webp;base64," + base64.b64encode(png_bytes).decode("ascii")

    sanitized = runner._sanitize_multimodal_history_content(
        [{"type": "image_url", "image_url": {"url": bad_data_url}}]
    )

    assert sanitized[0]["type"] == "image_url"
    normalized = sanitized[0]["image_url"]["url"]
    assert normalized.startswith("data:image/png;base64,")
    encoded = normalized.split(",", 1)[1]
    assert base64.b64decode(encoded) == png_bytes


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

        def run_conversation(self, user_message, conversation_history=None, task_id=None):
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


def test_run_agent_cross_user_interrupt_rebuilds_pending_discord_turn(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_db = None

    source_a = SessionSource(
        platform=Platform.DISCORD,
        chat_id="ch123",
        chat_name="Test place / #bot-channel",
        chat_type="group",
        user_id="u123",
        user_name="giftedgummybee",
    )
    source_b = SessionSource(
        platform=Platform.DISCORD,
        chat_id="ch123",
        chat_name="Test place / #bot-channel",
        chat_type="group",
        user_id="u456",
        user_name="neggles",
    )

    pending_event = MessageEvent(
        text="@Hermes-Bot what is 1+1",
        message_type=MessageType.TEXT,
        source=source_b,
        message_id="m222",
        extra_context="[Discord context]\n00:11 giftedgummybee: previous message",
        reply_to_message_id="m_bot",
        raw_message=SimpleNamespace(reference=SimpleNamespace(channel_id="ch123")),
    )

    class _FakeDiscordAdapter:
        def __init__(self):
            self.is_connected = True
            self._active_sessions = {"ch123": asyncio.Event()}
            self._archive_db = SimpleNamespace(
                get_reply_context=lambda **kwargs: {
                    "author_display": "Hermes-Bot",
                    "preview": "previous partial answer from Hermes",
                }
            )
            self._pending = [pending_event]

        def get_pending_message(self, chat_id):
            if self._pending:
                return self._pending.pop(0)
            return None

    runner.adapters[Platform.DISCORD] = _FakeDiscordAdapter()

    observed_user_messages = []
    observed_histories = []

    class FakeAIAgent:
        call_count = 0

        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, user_message, conversation_history=None, task_id=None):
            FakeAIAgent.call_count += 1
            observed_user_messages.append(user_message)
            observed_histories.append(list(conversation_history or []))
            if FakeAIAgent.call_count == 1:
                return {
                    "final_response": "Operation interrupted.",
                    "messages": [
                        {"role": "assistant", "content": "stable history"},
                        {"role": "user", "content": "first question"},
                        {"role": "assistant", "content": "partial reply"},
                    ],
                    "api_calls": 1,
                    "interrupted": True,
                }
            return {
                "final_response": "2",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": "2"},
                ],
                "api_calls": 1,
                "interrupted": False,
            }

    delivery_state = runner._new_delivery_state(
        source_a,
        MessageEvent(
            text="first question",
            message_type=MessageType.TEXT,
            source=source_a,
            message_id="m111",
        ),
    )

    with patch("run_agent.AIAgent", FakeAIAgent):
        result = asyncio.run(
            runner._run_agent(
                message="first question",
                context_prompt="",
                history=[{"role": "assistant", "content": "stable history"}],
                source=source_a,
                session_id="session-1",
                session_key="agent:main:discord:group:ch123",
                event=MessageEvent(
                    text="first question",
                    message_type=MessageType.TEXT,
                    source=source_a,
                    message_id="m111",
                ),
                delivery_state=delivery_state,
            )
        )

    assert result["final_response"] == "2"
    assert observed_user_messages[0] == "first question"
    resumed_payload = observed_user_messages[1]
    assert "interrupted by a new message from neggles" in resumed_payload
    assert "[Discord context]" in resumed_payload
    assert '**Replying to Hermes-Bot**: "previous partial answer from Hermes"' in resumed_payload
    assert "**Current user**: neggles" in resumed_payload
    assert "**Request**: @Hermes-Bot what is 1+1" in resumed_payload
    assert observed_histories[1] == [{"role": "assistant", "content": "stable history"}]
    assert delivery_state["reply_to"] == "m222"


def test_extract_fork_thread_request_uses_latest_successful_tool_result(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "fork_thread", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_a",
            "content": json.dumps(
                {
                    "success": True,
                    "requested": True,
                    "title": "deep-dive",
                    "visibility": "private",
                }
            ),
        },
    ]

    request = runner._extract_fork_thread_request(messages)

    assert request == {"title": "deep-dive", "visibility": "private", "reason": ""}


def test_run_agent_interrupt_in_live_fork_thread_defers_to_thread_session(tmp_path):
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
    thread_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="th789",
        chat_name="deep-dive",
        chat_type="thread",
        user_id="u123",
        user_name="giftedgummybee",
        thread_id="th789",
    )
    pending_event = MessageEvent(
        text="actualy no look up current LITE prices",
        message_type=MessageType.TEXT,
        source=thread_source,
        message_id="m333",
        extra_context="[Thread seed | fork:public]\n09/03/2026 11\n16:45 <giftedgummybee>: @Hermes-Bot actualy no look up current LITE prices",
    )

    class _FakeDiscordAdapter:
        def __init__(self):
            self.is_connected = True
            self._active_sessions = {"ch123": asyncio.Event(), "th789": asyncio.Event()}
            self._pending_messages = {"th789": pending_event}

        def get_pending_message(self, chat_id):
            return self._pending_messages.pop(chat_id, None)

    runner.adapters[Platform.DISCORD] = _FakeDiscordAdapter()

    observed_user_messages = []

    class FakeAIAgent:
        call_count = 0

        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, user_message, conversation_history=None, task_id=None):
            FakeAIAgent.call_count += 1
            observed_user_messages.append(user_message)
            return {
                "final_response": "Operation interrupted.",
                "messages": [
                    {"role": "assistant", "content": "stable history"},
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "partial reply"},
                ],
                "api_calls": 1,
                "interrupted": True,
            }

    delivery_state = runner._new_delivery_state(
        source,
        MessageEvent(
            text="first question",
            message_type=MessageType.TEXT,
            source=source,
            message_id="m111",
        ),
    )
    delivery_state.update(
        {
            "chat_id": "th789",
            "reply_to": None,
            "thread_result": {"thread_id": "th789", "thread_mention": "<#th789>"},
            "thread_source": thread_source,
            "thread_session_id": "session-thread",
            "thread_session_key": "agent:main:discord:thread:th789",
            "transcript_notice": "[Continued in thread <#th789>]",
        }
    )

    with patch("run_agent.AIAgent", FakeAIAgent):
        result = asyncio.run(
            runner._run_agent(
                message="first question",
                context_prompt="",
                history=[{"role": "assistant", "content": "stable history"}],
                source=source,
                session_id="session-parent",
                session_key="agent:main:discord:group:ch123",
                event=MessageEvent(
                    text="first question",
                    message_type=MessageType.TEXT,
                    source=source,
                    message_id="m111",
                ),
                delivery_state=delivery_state,
            )
        )

    assert result["final_response"] == ""
    assert result["deferred_pending_event"] is True
    assert FakeAIAgent.call_count == 1
    assert observed_user_messages == ["first question"]
    assert runner.adapters[Platform.DISCORD]._pending_messages["th789"] is pending_event


def test_handle_message_collapses_trailing_main_channel_fork_handoff_history(tmp_path):
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

    session_entry = runner.session_store.get_or_create_session(source)
    runner.session_store.append_to_transcript(
        session_entry.session_id,
        {"role": "user", "content": "@Hermes-Bot How do I do mergesort with a stack?"},
    )
    runner.session_store.append_to_transcript(
        session_entry.session_id,
        {
            "role": "assistant",
            "content": "<REASONING_SCRATCHPAD>fork it</REASONING_SCRATCHPAD>",
            "tool_calls": [
                {
                    "id": "call_fork",
                    "type": "function",
                    "function": {"name": "fork_thread", "arguments": "{}"},
                }
            ],
        },
    )
    runner.session_store.append_to_transcript(
        session_entry.session_id,
        {
            "role": "tool",
            "tool_call_id": "call_fork",
            "content": json.dumps(
                {
                    "success": True,
                    "requested": True,
                    "thread_id": "th456",
                    "thread_mention": "<#th456>",
                }
            ),
        },
    )
    runner.session_store.append_to_transcript(
        session_entry.session_id,
        {"role": "assistant", "content": "[Continued in thread <#th456>]"},
    )

    captured = {}

    async def _fake_run_agent(*, history, message, **kwargs):
        captured["history"] = history
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
        text="@Hermes-Bot follow-up in the main channel",
        message_type=MessageType.TEXT,
        source=source,
    )

    response = asyncio.run(runner._handle_message(event))

    assert response == "ok"
    assert len(captured["history"]) == 1
    assert captured["history"][0]["role"] == "assistant"
    assert "<#th456>" in captured["history"][0]["content"]
    assert "continue" in captured["history"][0]["content"].lower()
    assert "mergesort" in captured["history"][0]["content"].lower()


def test_handle_message_routes_auto_fork_response_to_thread(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )
    runner.config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True,
        token="fake-token",
        home_channel=HomeChannel(platform=Platform.DISCORD, chat_id="ch123", name="Home"),
    )

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="ch123",
        chat_name="Test place / #bot-channel",
        chat_type="group",
        user_id="u123",
        user_name="giftedgummybee",
    )

    sent = []

    class _FakeDiscordAdapter:
        def is_auto_fork_available(self, source):
            return True

        def auto_fork_main_channel_notice_enabled(self):
            return True

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SimpleNamespace(success=True)

        async def create_fork_thread_result(self, event, requested_name="", visibility="auto"):
            return {
                "success": True,
                "thread_id": "th456",
                "thread_name": "deep-dive",
                "thread_mention": "<#th456>",
                "visibility": "private",
            }

        async def deliver_response(self, chat_id, response, reply_to=None):
            sent.append((chat_id, response, reply_to))
            return True

    runner.adapters[Platform.DISCORD] = _FakeDiscordAdapter()

    async def _fake_run_agent(*, message, **kwargs):
        return {
            "final_response": "Long answer for the thread",
            "messages": [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "fork_thread", "arguments": "{\"title\": \"deep-dive\"}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": json.dumps(
                        {
                            "success": True,
                            "requested": True,
                            "title": "deep-dive",
                            "visibility": "auto",
                            "reason": "One-person deep dive",
                        }
                    ),
                },
                {"role": "assistant", "content": "Long answer for the thread"},
            ],
            "history_input_len": 0,
            "tools": [{"type": "function", "function": {"name": "fork_thread"}}],
            "request_usage": {
                "prompt_tokens": 30,
                "completion_tokens": 12,
                "total_tokens": 42,
                "cost_usd": 0.034,
            },
        }

    runner._run_agent = _fake_run_agent

    event = MessageEvent(
        text="Can you go deep on this?",
        message_type=MessageType.TEXT,
        source=source,
    )

    response = asyncio.run(runner._handle_message(event))

    assert sent == [("th456", "Long answer for the thread\n\n-# $0.03 USD spent", None)]
    assert response == "Taking this to a thread: <#th456>"
    assert getattr(event, "_response_handled", False) is True

    main_entry = runner.session_store.get_or_create_session(source)
    main_history = runner.session_store.load_transcript(main_entry.session_id)
    assert any(
        msg.get("content") == "[Continued in thread <#th456>]"
        for msg in main_history
        if msg.get("role") == "assistant"
    )

    thread_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="th456",
        chat_name="deep-dive",
        chat_type="thread",
        user_id="u123",
        user_name="giftedgummybee",
        thread_id="th456",
    )
    thread_entry = runner.session_store.get_or_create_session(thread_source)
    thread_history = runner.session_store.load_transcript(thread_entry.session_id)
    assert any(
        msg.get("content") == "Long answer for the thread"
        for msg in thread_history
        if msg.get("role") == "assistant"
    )


def test_activate_live_fork_thread_switches_delivery_target_immediately(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    sent = []
    active_event = asyncio.Event()

    class _FakeDiscordAdapter:
        _active_sessions = {"ch123": active_event}

        def is_auto_fork_available(self, source):
            return True

        def auto_fork_main_channel_notice_enabled(self):
            return True

        async def create_fork_thread_result(self, event, requested_name="", visibility="auto"):
            return {
                "success": True,
                "thread_id": "th789",
                "thread_name": "deep-dive",
                "thread_mention": "<#th789>",
                "visibility": "private",
            }

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            sent.append((chat_id, content, reply_to))
            return SimpleNamespace(success=True)

    runner.adapters[Platform.DISCORD] = _FakeDiscordAdapter()

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="ch123",
        chat_name="Test place / #bot-channel",
        chat_type="group",
        user_id="u123",
        user_name="giftedgummybee",
    )
    event = MessageEvent(
        text="Can you go deeper?",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m123",
    )
    delivery_state = {
        "chat_id": source.chat_id,
        "reply_to": event.message_id,
        "thread_result": None,
        "thread_source": None,
        "thread_session_id": None,
        "thread_session_key": None,
        "transcript_notice": None,
        "main_notice": None,
        "main_notice_sent": False,
        "thread_transcript_recorded": False,
    }

    result = asyncio.run(
        runner._activate_live_fork_thread(
            event=event,
            source=source,
            delivery_state=delivery_state,
            title="deep-dive",
            visibility="private",
            reason="One-person follow-up",
        )
    )

    assert result["success"] is True
    assert delivery_state["chat_id"] == "th789"
    assert delivery_state["reply_to"] is None
    assert getattr(event, "_response_chat_id") == "th789"
    assert getattr(event, "_response_reply_to_message_id") is None
    assert getattr(event, "_active_session_aliases") == ["th789"]
    assert runner.adapters[Platform.DISCORD]._active_sessions["th789"] is active_event
    assert sent == [("ch123", "Taking this to a thread: <#th789>", "m123")]
    assert "<#th789>" in result["parent_session_note"]

    thread_history = runner.session_store.load_transcript(delivery_state["thread_session_id"])
    assert thread_history[0]["role"] == "session_meta"
    assert thread_history[1]["role"] == "user"
    assert thread_history[1]["content"] == (
        "[Forked from Test place / #bot-channel] Can you go deeper?"
    )


def test_append_thread_fork_transcript_is_idempotent_after_live_seed(tmp_path):
    runner = GatewayRunner(
        GatewayConfig(sessions_dir=tmp_path / "sessions")
    )

    original_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="ch123",
        chat_name="Test place / #bot-channel",
        chat_type="group",
        user_id="u123",
        user_name="giftedgummybee",
    )
    thread_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="th789",
        chat_name="deep-dive",
        chat_type="thread",
        user_id="u123",
        user_name="giftedgummybee",
        thread_id="th789",
    )

    runner._ensure_thread_fork_transcript_seeded(
        thread_source=thread_source,
        original_source=original_source,
        request_text="Can you go deeper?",
        tool_defs=[],
    )
    runner._append_thread_fork_transcript(
        thread_source=thread_source,
        original_source=original_source,
        request_text="Can you go deeper?",
        response="Long answer for the thread",
        tool_defs=[],
    )
    runner._append_thread_fork_transcript(
        thread_source=thread_source,
        original_source=original_source,
        request_text="Can you go deeper?",
        response="Long answer for the thread",
        tool_defs=[],
    )

    thread_entry = runner.session_store.get_or_create_session(thread_source)
    thread_history = runner.session_store.load_transcript(thread_entry.session_id)
    assistant_messages = [
        msg.get("content")
        for msg in thread_history
        if msg.get("role") == "assistant"
    ]
    user_messages = [
        msg.get("content")
        for msg in thread_history
        if msg.get("role") == "user"
    ]

    assert assistant_messages == ["Long answer for the thread"]
    assert user_messages == [
        "[Forked from Test place / #bot-channel] Can you go deeper?"
    ]
