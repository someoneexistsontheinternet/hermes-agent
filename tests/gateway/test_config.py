"""Tests for gateway configuration management."""

import asyncio
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    SessionResetPolicy,
    load_gateway_config,
)
from gateway.platforms.discord import DiscordAdapter


class TestHomeChannelRoundtrip:
    def test_to_dict_from_dict(self):
        hc = HomeChannel(platform=Platform.DISCORD, chat_id="999", name="general")
        d = hc.to_dict()
        restored = HomeChannel.from_dict(d)

        assert restored.platform == Platform.DISCORD
        assert restored.chat_id == "999"
        assert restored.name == "general"


class TestPlatformConfigRoundtrip:
    def test_to_dict_from_dict(self):
        pc = PlatformConfig(
            enabled=True,
            token="tok_123",
            home_channel=HomeChannel(
                platform=Platform.TELEGRAM,
                chat_id="555",
                name="Home",
            ),
            extra={"foo": "bar"},
        )
        d = pc.to_dict()
        restored = PlatformConfig.from_dict(d)

        assert restored.enabled is True
        assert restored.token == "tok_123"
        assert restored.home_channel.chat_id == "555"
        assert restored.extra == {"foo": "bar"}

    def test_disabled_no_token(self):
        pc = PlatformConfig()
        d = pc.to_dict()
        restored = PlatformConfig.from_dict(d)
        assert restored.enabled is False
        assert restored.token is None


class TestGetConnectedPlatforms:
    def test_returns_enabled_with_token(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
                Platform.DISCORD: PlatformConfig(enabled=False, token="d"),
                Platform.SLACK: PlatformConfig(enabled=True),  # no token
            },
        )
        connected = config.get_connected_platforms()
        assert Platform.TELEGRAM in connected
        assert Platform.DISCORD not in connected
        assert Platform.SLACK not in connected

    def test_empty_platforms(self):
        config = GatewayConfig()
        assert config.get_connected_platforms() == []


class TestSessionResetPolicy:
    def test_roundtrip(self):
        policy = SessionResetPolicy(mode="idle", at_hour=6, idle_minutes=120)
        d = policy.to_dict()
        restored = SessionResetPolicy.from_dict(d)
        assert restored.mode == "idle"
        assert restored.at_hour == 6
        assert restored.idle_minutes == 120

    def test_defaults(self):
        policy = SessionResetPolicy()
        assert policy.mode == "both"
        assert policy.at_hour == 4
        assert policy.idle_minutes == 1440


class TestGatewayConfigRoundtrip:
    def test_full_roundtrip(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="tok",
                    home_channel=HomeChannel(Platform.TELEGRAM, "123", "Home"),
                ),
            },
            reset_triggers=["/new"],
        )
        d = config.to_dict()
        restored = GatewayConfig.from_dict(d)

        assert Platform.TELEGRAM in restored.platforms
        assert restored.platforms[Platform.TELEGRAM].token == "tok"
        assert restored.reset_triggers == ["/new"]


class TestLoadGatewayConfigFromYaml:
    def test_bridges_discord_context_settings(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "config.yaml").write_text(
            "gateway:\n"
            "  discord:\n"
            "    context_max_chars: 9000\n"
            "    fresh_context_limit: 20\n"
            "    delta_reset_threshold: 50\n"
            "    context_empty_delta_fallback: true\n"
            "    archive_enabled: true\n"
            "    archive_db_path: /tmp/discord_data.sqlite\n"
            "    allowed_guild_ids: [\"111\"]\n"
            "    allowed_channel_ids: [\"222\"]\n"
            "    full_scrape_enabled: true\n"
            "    full_scrape_interval_sec: 70\n"
            "    full_scrape_max_channels_per_tick: 9\n"
            "    full_scrape_max_pages_per_channel: 4\n"
            "    full_scrape_seed_limit: 333\n"
            "    full_scrape_include_threads: true\n"
            "    backfill_enabled: true\n"
            "    backfill_interval_sec: 420\n"
            "    backfill_max_channels_per_tick: 3\n"
            "    backfill_max_pages_per_channel: 2\n"
            "    scrape_progress_every_pages: 5\n",
            encoding="utf-8",
        )

        with patch("gateway.config.Path.home", return_value=tmp_path):
            with patch.dict(os.environ, {}, clear=False):
                config = load_gateway_config()

        assert Platform.DISCORD in config.platforms
        extra = config.platforms[Platform.DISCORD].extra
        assert extra["context_max_chars"] == 9000
        assert extra["fresh_context_limit"] == 20
        assert extra["delta_reset_threshold"] == 50
        assert extra["context_empty_delta_fallback"] is True
        assert extra["archive_enabled"] is True
        assert extra["archive_db_path"] == "/tmp/discord_data.sqlite"
        assert extra["allowed_guild_ids"] == ["111"]
        assert extra["allowed_channel_ids"] == ["222"]
        assert extra["full_scrape_enabled"] is True
        assert extra["full_scrape_interval_sec"] == 70
        assert extra["full_scrape_max_channels_per_tick"] == 9
        assert extra["full_scrape_max_pages_per_channel"] == 4
        assert extra["full_scrape_seed_limit"] == 333
        assert extra["full_scrape_include_threads"] is True
        assert extra["backfill_enabled"] is True
        assert extra["backfill_interval_sec"] == 420
        assert extra["backfill_max_channels_per_tick"] == 3
        assert extra["backfill_max_pages_per_channel"] == 2
        assert extra["scrape_progress_every_pages"] == 5


class TestDiscordAdapterContextSettings:
    def test_uses_config_values_when_env_unset(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"context_max_chars": "12000"},
            )
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISCORD_CONTEXT_MAX_CHARS", None)
            assert adapter._context_max_chars() == 12000

    def test_env_values_override_config(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"context_max_chars": 12000},
            )
        )
        with patch.dict(
            os.environ,
            {
                "DISCORD_CONTEXT_MAX_CHARS": "3333",
            },
            clear=False,
        ):
            assert adapter._context_max_chars() == 3333

    def test_archive_and_delta_settings_from_config(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={
                    "archive_enabled": True,
                    "archive_db_path": "/tmp/x.sqlite",
                    "fresh_context_limit": 18,
                    "delta_reset_threshold": 70,
                    "context_empty_delta_fallback": True,
                    "allowed_guild_ids": ["1", "2"],
                    "allowed_channel_ids": "10,11",
                    "full_scrape_enabled": True,
                    "full_scrape_interval_sec": 70,
                    "full_scrape_max_channels_per_tick": 9,
                    "full_scrape_max_pages_per_channel": 4,
                    "full_scrape_seed_limit": 333,
                    "full_scrape_include_threads": True,
                    "backfill_enabled": True,
                    "backfill_interval_sec": 420,
                    "backfill_max_channels_per_tick": 3,
                    "backfill_max_pages_per_channel": 2,
                    "scrape_progress_every_pages": 5,
                },
            )
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISCORD_ARCHIVE_ENABLED", None)
            os.environ.pop("DISCORD_ARCHIVE_DB_PATH", None)
            os.environ.pop("DISCORD_FRESH_CONTEXT_LIMIT", None)
            os.environ.pop("DISCORD_DELTA_RESET_THRESHOLD", None)
            os.environ.pop("DISCORD_CONTEXT_EMPTY_DELTA_FALLBACK", None)
            os.environ.pop("DISCORD_ALLOWED_GUILD_IDS", None)
            os.environ.pop("DISCORD_ALLOWED_CHANNEL_IDS", None)
            os.environ.pop("DISCORD_FULL_SCRAPE_ENABLED", None)
            os.environ.pop("DISCORD_FULL_SCRAPE_INTERVAL_SEC", None)
            os.environ.pop("DISCORD_FULL_SCRAPE_MAX_CHANNELS_PER_TICK", None)
            os.environ.pop("DISCORD_FULL_SCRAPE_MAX_PAGES_PER_CHANNEL", None)
            os.environ.pop("DISCORD_FULL_SCRAPE_SEED_LIMIT", None)
            os.environ.pop("DISCORD_FULL_SCRAPE_INCLUDE_THREADS", None)
            os.environ.pop("DISCORD_BACKFILL_ENABLED", None)
            os.environ.pop("DISCORD_BACKFILL_INTERVAL_SEC", None)
            os.environ.pop("DISCORD_BACKFILL_MAX_CHANNELS_PER_TICK", None)
            os.environ.pop("DISCORD_BACKFILL_MAX_PAGES_PER_CHANNEL", None)
            os.environ.pop("DISCORD_SCRAPE_PROGRESS_EVERY_PAGES", None)
            assert adapter._archive_enabled() is True
            assert adapter._archive_db_path() == "/tmp/x.sqlite"
            assert adapter._fresh_context_limit() == 18
            assert adapter._delta_reset_threshold() == 70
            assert adapter._empty_delta_fallback_enabled() is True
            assert adapter._allowed_guild_ids() == {"1", "2"}
            assert adapter._allowed_channel_ids() == {"10", "11"}
            assert adapter._full_scrape_enabled() is True
            assert adapter._full_scrape_interval_sec() == 70
            assert adapter._full_scrape_max_channels_per_tick() == 9
            assert adapter._full_scrape_max_pages_per_channel() == 4
            assert adapter._full_scrape_seed_limit() == 333
            assert adapter._full_scrape_include_threads() is True
            assert adapter._backfill_enabled() is True
            assert adapter._backfill_interval_sec() == 420
            assert adapter._backfill_max_channels_per_tick() == 3
            assert adapter._backfill_max_pages_per_channel() == 2
            assert adapter._scrape_progress_every_pages() == 5

    def test_scrape_defaults_align_with_dce_style(self):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token", extra={})
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISCORD_FULL_SCRAPE_INTERVAL_SEC", None)
            os.environ.pop("DISCORD_FULL_SCRAPE_MAX_CHANNELS_PER_TICK", None)
            os.environ.pop("DISCORD_FULL_SCRAPE_MAX_PAGES_PER_CHANNEL", None)
            os.environ.pop("DISCORD_FULL_SCRAPE_SEED_LIMIT", None)
            os.environ.pop("DISCORD_FULL_SCRAPE_INCLUDE_THREADS", None)
            os.environ.pop("DISCORD_BACKFILL_INTERVAL_SEC", None)
            os.environ.pop("DISCORD_BACKFILL_MAX_CHANNELS_PER_TICK", None)
            os.environ.pop("DISCORD_BACKFILL_MAX_PAGES_PER_CHANNEL", None)
            assert adapter._full_scrape_interval_sec() == 15
            assert adapter._full_scrape_max_channels_per_tick() == 1
            assert adapter._full_scrape_max_pages_per_channel() == 100
            assert adapter._full_scrape_seed_limit() == 100
            assert adapter._full_scrape_include_threads() is False
            assert adapter._backfill_interval_sec() == 30
            assert adapter._backfill_max_channels_per_tick() == 1
            assert adapter._backfill_max_pages_per_channel() == 50

    def test_context_header_first_turn_includes_channel_label(self):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token")
        )
        rows = [
            {"created_at": 1700000000, "author_name": "alice", "content": "hello"}
        ]
        block = adapter._render_context_block(
            rows,
            "Test place / #bot-channel",
            "fresh_last_20",
            include_channel_label=True,
        )
        assert block.startswith("[Discord context | Test place / #bot-channel]\n")

    def test_context_header_followup_uses_generic_label(self):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token")
        )
        rows = [
            {"created_at": 1700000000, "author_name": "alice", "content": "hello"}
        ]
        block = adapter._render_context_block(
            rows,
            "Test place / #bot-channel",
            "delta_after_anchor_2",
            include_channel_label=False,
        )
        first_line = block.splitlines()[0]
        assert first_line == "[Discord context]"

    def test_empty_delta_falls_back_to_fresh_window(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={
                    "fresh_context_limit": 20,
                    "delta_reset_threshold": 50,
                    "context_empty_delta_fallback": True,
                },
            )
        )

        class FakeArchiveDB:
            def __init__(self):
                self.last_anchor = None

            def get_turn_anchor(self, channel_id):
                return "m_prev"

            def count_new_non_bot_messages(self, channel_id, anchor):
                return 0

            def list_messages_after(self, channel_id, after_message_id, limit=1000, include_bots=False):
                # Empty delta should trigger fallback when enabled.
                return []

            def list_recent_messages(self, channel_id, limit=20, include_bots=False):
                return [
                    {"message_id": "m_old", "created_at": 1700000000, "author_name": "alice", "content": "prior msg"},
                    {"message_id": "m_current", "created_at": 1700000010, "author_name": "alice", "content": "current"},
                ]

            def set_turn_anchor(self, channel_id, message_id):
                self.last_anchor = message_id

        adapter._archive_db = FakeArchiveDB()

        message = SimpleNamespace(
            id="m_current",
            channel=SimpleNamespace(
                id="ch1",
                name="bot-channel",
                guild=SimpleNamespace(name="Test place"),
            ),
        )

        block, force_auto_reset, reset_reason = asyncio.run(
            adapter._build_recent_channel_context(message)
        )

        assert block.startswith("[Discord context]")
        assert "prior msg" in block
        assert force_auto_reset is False
        assert reset_reason == ""

    def test_empty_delta_is_empty_without_fallback(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"fresh_context_limit": 20, "delta_reset_threshold": 50},
            )
        )

        class FakeArchiveDB:
            def __init__(self):
                self.last_anchor = None

            def get_turn_anchor(self, channel_id):
                return "m_prev"

            def count_new_non_bot_messages(self, channel_id, anchor):
                return 0

            def list_messages_after(self, channel_id, after_message_id, limit=1000, include_bots=False):
                # Empty delta should stay empty when fallback is disabled.
                return []

            def list_recent_messages(self, channel_id, limit=20, include_bots=False):
                return [
                    {"message_id": "m_old", "created_at": 1700000000, "author_name": "alice", "content": "prior msg"},
                    {"message_id": "m_current", "created_at": 1700000010, "author_name": "alice", "content": "current"},
                ]

            def set_turn_anchor(self, channel_id, message_id):
                self.last_anchor = message_id

        adapter._archive_db = FakeArchiveDB()

        message = SimpleNamespace(
            id="m_current",
            channel=SimpleNamespace(
                id="ch1",
                name="bot-channel",
                guild=SimpleNamespace(name="Test place"),
            ),
        )

        block, force_auto_reset, reset_reason = asyncio.run(
            adapter._build_recent_channel_context(message)
        )

        assert block == ""
        assert force_auto_reset is False
        assert reset_reason == ""

    def test_delta_context_keeps_current_turn_line(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"fresh_context_limit": 20, "delta_reset_threshold": 50},
            )
        )

        class FakeArchiveDB:
            def get_turn_anchor(self, channel_id):
                return "m_prev"

            def count_new_non_bot_messages(self, channel_id, anchor):
                return 1

            def list_messages_after(self, channel_id, after_message_id, limit=1000, include_bots=False):
                return [
                    {"message_id": "m_current", "created_at": 1700000010, "author_name": "alice", "content": "current"}
                ]

            def set_turn_anchor(self, channel_id, message_id):
                return None

        adapter._archive_db = FakeArchiveDB()

        message = SimpleNamespace(
            id="m_current",
            channel=SimpleNamespace(
                id="ch1",
                name="bot-channel",
                guild=SimpleNamespace(name="Test place"),
            ),
        )

        block, force_auto_reset, reset_reason = asyncio.run(
            adapter._build_recent_channel_context(message)
        )

        assert "current" in block
        assert force_auto_reset is False
        assert reset_reason == ""

    def test_delta_context_uses_threshold_as_followup_limit(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"fresh_context_limit": 20, "delta_reset_threshold": 3},
            )
        )

        class FakeArchiveDB:
            def __init__(self):
                self.last_limit = None

            def get_turn_anchor(self, channel_id):
                return "m_prev"

            def count_new_non_bot_messages(self, channel_id, anchor):
                return 3

            def list_messages_after(self, channel_id, after_message_id, limit=1000, include_bots=False):
                self.last_limit = limit
                return [{"message_id": "m6", "created_at": 1700000015, "author_name": "alice", "content": "six"}]

            def set_turn_anchor(self, channel_id, message_id):
                return None

        fake_db = FakeArchiveDB()
        adapter._archive_db = fake_db

        message = SimpleNamespace(
            id="m6",
            channel=SimpleNamespace(
                id="ch1",
                name="bot-channel",
                guild=SimpleNamespace(name="Test place"),
            ),
        )

        block, force_auto_reset, reset_reason = asyncio.run(
            adapter._build_recent_channel_context(message)
        )

        assert "six" in block
        assert fake_db.last_limit == 3
        assert force_auto_reset is False
        assert reset_reason == ""

    def test_delta_context_drops_ping_only_bot_mentions(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"fresh_context_limit": 20, "delta_reset_threshold": 50},
            )
        )
        adapter._client = SimpleNamespace(
            user=SimpleNamespace(
                id="1477229390382235791",
                name="Hermes-Bot",
                global_name=None,
                display_name="Hermes-Bot",
            )
        )

        class FakeArchiveDB:
            def get_turn_anchor(self, channel_id):
                return "m_prev"

            def count_new_non_bot_messages(self, channel_id, anchor):
                return 2

            def list_messages_after(self, channel_id, after_message_id, limit=1000, include_bots=False):
                return [
                    {
                        "message_id": "m_ping",
                        "created_at": 1700000010,
                        "author_name": "alice",
                        "content": "<@1477229390382235791>",
                    },
                    {
                        "message_id": "m_real",
                        "created_at": 1700000011,
                        "author_name": "alice",
                        "content": "<@1477229390382235791> what is 1+1?",
                    },
                ]

            def set_turn_anchor(self, channel_id, message_id):
                return None

        adapter._archive_db = FakeArchiveDB()

        message = SimpleNamespace(
            id="m_real",
            channel=SimpleNamespace(
                id="ch1",
                name="bot-channel",
                guild=SimpleNamespace(name="Test place"),
            ),
        )

        block, force_auto_reset, reset_reason = asyncio.run(
            adapter._build_recent_channel_context(message)
        )

        lines = block.splitlines()
        assert len(lines) == 3
        assert lines[0] == "[Discord context]"
        assert "what is 1+1?" in lines[2]
        assert lines[2].endswith(": <@1477229390382235791>") is False
        assert force_auto_reset is False
        assert reset_reason == ""

    def test_delta_context_appends_changes_block(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"fresh_context_limit": 20, "delta_reset_threshold": 50},
            )
        )

        class FakeArchiveDB:
            def get_turn_anchor(self, channel_id):
                return "m_prev"

            def count_new_non_bot_messages(self, channel_id, anchor):
                return 0

            def list_messages_after(self, channel_id, after_message_id, limit=1000, include_bots=False):
                return []

            def list_changes_since_anchor(
                self,
                channel_id,
                anchor_message_id,
                limit=1000,
                include_bots=False,
            ):
                return [
                    {
                        "message_id": "m1",
                        "original_created_at": datetime(2026, 3, 1, 2, 10, 4).timestamp(),
                        "author_display": "alice",
                        "change_type": "edit",
                        "before_content": "old version",
                        "after_content": "new version",
                    },
                    {
                        "message_id": "m2",
                        "original_created_at": datetime(2026, 3, 1, 2, 10, 10).timestamp(),
                        "author_display": "bob",
                        "change_type": "delete",
                        "before_content": "gone soon",
                        "after_content": "",
                    },
                ]

            def set_turn_anchor(self, channel_id, message_id):
                return None

        adapter._archive_db = FakeArchiveDB()

        message = SimpleNamespace(
            id="m_current",
            channel=SimpleNamespace(
                id="ch1",
                name="bot-channel",
                guild=SimpleNamespace(name="Test place"),
            ),
        )

        block, force_auto_reset, reset_reason = asyncio.run(
            adapter._build_recent_channel_context(message)
        )

        assert block.startswith("[Discord context]")
        assert "[Changes]" in block
        assert "old version -> new version" in block
        assert "gone soon -> [Deleted]" in block
        assert force_auto_reset is False
        assert reset_reason == ""

    def test_context_line_format_uses_minute_second_and_bracketed_username(self):
        hour_header, line = DiscordAdapter._format_archive_history_line(
            {
                "created_at": datetime(2026, 3, 1, 1, 23, 45).timestamp(),
                "author_display": "giftedgummybee",
                "content": "hello world",
            }
        )

        assert hour_header == "01/03/2026 01"
        assert line == "23:45 <giftedgummybee>: hello world"

    def test_reset_channel_context_clears_anchor_and_header_state(self):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token")
        )

        class FakeArchiveDB:
            def __init__(self):
                self.cleared = []

            def clear_turn_anchor(self, channel_id):
                self.cleared.append(channel_id)

        fake_db = FakeArchiveDB()
        adapter._archive_db = fake_db
        adapter._context_header_sent_channels.add("ch1")

        adapter.reset_channel_context("ch1")

        assert "ch1" not in adapter._context_header_sent_channels
        assert "ch1" in adapter._force_fresh_context_channels
        assert fake_db.cleared == ["ch1"]

    def test_forced_fresh_window_after_reset_uses_recent_once_then_delta(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"fresh_context_limit": 20, "delta_reset_threshold": 50},
            )
        )

        class FakeArchiveDB:
            def __init__(self):
                self.anchor = "90"
                self.calls = []

            def clear_turn_anchor(self, channel_id):
                return None

            def get_turn_anchor(self, channel_id):
                return self.anchor

            def count_new_non_bot_messages(self, channel_id, anchor):
                return 1

            def list_recent_messages(self, channel_id, limit=20, include_bots=False):
                self.calls.append(("recent", limit))
                return [
                    {
                        "message_id": "100",
                        "created_at": 1700000010,
                        "author_name": "alice",
                        "content": "fresh line",
                    }
                ]

            def list_messages_after(self, channel_id, after_message_id, limit=1000, include_bots=False):
                self.calls.append(("after", after_message_id))
                return [
                    {
                        "message_id": "101",
                        "created_at": 1700000020,
                        "author_name": "alice",
                        "content": "delta line",
                    }
                ]

            def set_turn_anchor(self, channel_id, message_id):
                self.anchor = str(message_id)

        fake_db = FakeArchiveDB()
        adapter._archive_db = fake_db
        adapter.reset_channel_context("ch1")

        first_message = SimpleNamespace(
            id="100",
            channel=SimpleNamespace(
                id="ch1",
                name="bot-channel",
                guild=SimpleNamespace(name="Test place"),
            ),
        )
        first_block, _, _ = asyncio.run(
            adapter._build_recent_channel_context(first_message)
        )

        assert fake_db.calls[0] == ("recent", 20)
        assert first_block.startswith("[Discord context | Test place / #bot-channel]")
        assert "ch1" not in adapter._force_fresh_context_channels

        second_message = SimpleNamespace(
            id="101",
            channel=SimpleNamespace(
                id="ch1",
                name="bot-channel",
                guild=SimpleNamespace(name="Test place"),
            ),
        )
        second_block, _, _ = asyncio.run(
            adapter._build_recent_channel_context(second_message)
        )

        assert fake_db.calls[1] == ("after", "100")
        assert second_block.startswith("[Discord context]")

    def test_archive_message_still_upserts_if_bootstrap_fails(self):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token")
        )
        upserted = []

        class FakeArchiveDB:
            def upsert_message(self, row):
                upserted.append(row)

        adapter._archive_db = FakeArchiveDB()

        async def _boom(_channel):
            raise RuntimeError("bootstrap failure")

        message = SimpleNamespace(channel=SimpleNamespace(id="ch1"))
        row = {"message_id": "100", "channel_id": "ch1", "created_at": 1700000000.0}

        with patch.object(adapter, "_is_channel_allowed", return_value=True):
            with patch.object(adapter, "_bootstrap_channel_archive", _boom):
                with patch.object(adapter, "_message_to_archive_row", return_value=row):
                    asyncio.run(adapter._archive_message(message))

        assert upserted == [row]

    def test_followup_image_carryover_uses_fresh_window_cap(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"fresh_context_limit": 20},
            )
        )

        class FakeArchiveDB:
            def __init__(self):
                self.last_limit = None

            def get_turn_anchor(self, channel_id):
                return None

            def list_recent_messages(self, channel_id, limit=20, include_bots=False):
                self.last_limit = limit
                base_ts = 1700000000.0
                rows = [
                    {
                        "message_id": f"m{i}",
                        "author_id": "u1",
                        "created_at": base_ts + i,
                        "attachments": [],
                    }
                    for i in range(limit)
                ]
                if limit > 20:
                    rows[0]["attachments"] = [
                        {
                            "url": "https://example.com/old.png",
                            "filename": "old.png",
                            "content_type": "image/png",
                        }
                    ]
                return rows

        fake_db = FakeArchiveDB()
        adapter._archive_db = fake_db

        message = SimpleNamespace(
            id="m_current",
            channel=SimpleNamespace(id="ch1"),
            author=SimpleNamespace(id="u1"),
            created_at=datetime.fromtimestamp(1700000200),
        )

        with patch.dict(os.environ, {"DISCORD_FOLLOWUP_IMAGE_LOOKBACK": "50"}, clear=False):
            items = adapter._collect_recent_user_image_items(message, max_images=1)

        assert fake_db.last_limit == 20
        assert items == []

    def test_followup_image_carryover_uses_channel_images_and_delta_limit(self):
        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"fresh_context_limit": 20, "delta_reset_threshold": 3},
            )
        )

        class FakeArchiveDB:
            def __init__(self):
                self.after_limit = None
                self.recent_called = False

            def get_turn_anchor(self, channel_id):
                return "m_prev"

            def count_new_non_bot_messages(self, channel_id, anchor):
                return 2

            def list_messages_after(self, channel_id, after_message_id, limit=1000, include_bots=False):
                self.after_limit = limit
                return [
                    {
                        "message_id": "m_old",
                        "author_id": "u2",
                        "created_at": 1700000001.0,
                        "attachments": [
                            {
                                "url": "https://example.com/u2.png",
                                "filename": "u2.png",
                                "content_type": "image/png",
                            }
                        ],
                    },
                    {
                        "message_id": "m_new",
                        "author_id": "u3",
                        "created_at": 1700000002.0,
                        "attachments": [
                            {
                                "url": "https://example.com/u3.png",
                                "filename": "u3.png",
                                "content_type": "image/png",
                            }
                        ],
                    },
                    {
                        "message_id": "m_current",
                        "author_id": "u1",
                        "created_at": 1700000003.0,
                        "attachments": [],
                    },
                ]

            def list_recent_messages(self, channel_id, limit=20, include_bots=False):
                self.recent_called = True
                return []

        fake_db = FakeArchiveDB()
        adapter._archive_db = fake_db

        message = SimpleNamespace(
            id="m_current",
            channel=SimpleNamespace(id="ch1"),
            author=SimpleNamespace(id="u1"),
            created_at=datetime.fromtimestamp(1700000200),
        )

        items = adapter._collect_recent_channel_image_items(message, max_images=2)

        assert fake_db.after_limit == 3
        assert fake_db.recent_called is False
        assert [item["source_url"] for item in items] == [
            "https://example.com/u3.png",
            "https://example.com/u2.png",
        ]

    def test_replace_user_mentions_uses_global_username(self):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token")
        )

        message = SimpleNamespace(
            mentions=[
                SimpleNamespace(id="123", name="alice"),
                SimpleNamespace(id="456", name="bob"),
            ]
        )
        text = "ping <@123> and <@!456> and <@999>"

        rendered = adapter._replace_user_mentions(text, message)

        assert rendered == "ping @alice and @bob and <@999>"

    def test_pick_round_robin_batch_rotates(self):
        channels = ["c1", "c2", "c3", "c4"]
        batch, next_idx = DiscordAdapter._pick_round_robin_batch(channels, 1, 2)
        assert batch == ["c2", "c3"]
        assert next_idx == 3

        batch, next_idx = DiscordAdapter._pick_round_robin_batch(channels, next_idx, 3)
        assert batch == ["c4", "c1", "c2"]
        assert next_idx == 2

    def test_sync_channel_forward_seeds_when_cursor_missing(self):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token")
        )

        class FakeArchiveDB:
            def __init__(self):
                self.rows = []

            def get_channel_cursor(self, channel_id):
                return None

            def upsert_message(self, row):
                self.rows.append(row)

        class FakeChannel:
            id = "ch1"
            guild = None

            @staticmethod
            def history(limit=0, oldest_first=False):
                msgs = [SimpleNamespace(id="1"), SimpleNamespace(id="2")]

                async def _gen():
                    for msg in msgs:
                        yield msg

                return _gen()

        fake_db = FakeArchiveDB()
        adapter._archive_db = fake_db
        channel = FakeChannel()

        with patch.object(adapter, "_is_channel_allowed", return_value=True):
            with patch.object(
                adapter,
                "_message_to_archive_row",
                side_effect=lambda msg: {
                    "message_id": str(msg.id),
                    "channel_id": "ch1",
                    "created_at": 1700000000.0,
                },
            ):
                inserted = asyncio.run(
                    adapter._sync_channel_forward(
                        channel,
                        max_pages=2,
                        seed_limit=5,
                        drain_all_pages=False,
                    )
                )

        assert inserted == 2
        assert [row["message_id"] for row in fake_db.rows] == ["1", "2"]

    def test_sync_channel_backfill_marks_complete_when_no_older_rows(self):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token")
        )

        class FakeArchiveDB:
            def __init__(self):
                self.completed = []

            @staticmethod
            def get_backfill_state(channel_id):
                return {
                    "channel_id": channel_id,
                    "oldest_message_id": "100",
                    "oldest_created_at": 1000.0,
                    "complete": False,
                }

            @staticmethod
            def get_oldest_message(channel_id):
                return None

            @staticmethod
            def upsert_message(row):
                return None

            @staticmethod
            def upsert_backfill_state(channel_id, **kwargs):
                return None

            def mark_backfill_complete(self, channel_id, complete=True):
                self.completed.append((channel_id, complete))

        class FakeChannel:
            id = "ch1"
            guild = None

            @staticmethod
            def history(limit=0, oldest_first=False, before=None):
                async def _gen():
                    if False:
                        yield None

                return _gen()

        fake_db = FakeArchiveDB()
        adapter._archive_db = fake_db
        channel = FakeChannel()

        with patch.object(adapter, "_is_channel_allowed", return_value=True):
            inserted = asyncio.run(
                adapter._sync_channel_backfill(channel, max_pages=2)
            )

        assert inserted == 0
        assert fake_db.completed == [("ch1", True)]

    def test_materialize_message_text_uses_forward_snapshot_when_content_empty(self):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token")
        )
        message = SimpleNamespace(
            content="",
            system_content="",
            embeds=[],
            message_snapshots=[
                SimpleNamespace(
                    content="Forwarded payload body",
                    embeds=[],
                )
            ],
            mentions=[],
        )

        rendered = adapter._materialize_message_text(message)

        assert rendered.startswith("[forwarded message]")
        assert "Forwarded payload body" in rendered

    def test_message_to_archive_row_includes_forward_snapshot_text(self):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token")
        )
        message = SimpleNamespace(
            id="m1",
            content="",
            system_content="",
            embeds=[],
            message_snapshots=[
                SimpleNamespace(
                    content="Forwarded text from another channel",
                    embeds=[],
                )
            ],
            mentions=[],
            attachments=[],
            channel=SimpleNamespace(
                id="ch1",
                name="general",
                guild=SimpleNamespace(id="g1", name="Guild One"),
            ),
            author=SimpleNamespace(id="u1", name="alice", bot=False),
            created_at=datetime.fromtimestamp(1700000000),
            edited_at=None,
        )

        row = adapter._message_to_archive_row(message)

        assert row["content"].startswith("[forwarded message]")
        assert "Forwarded text from another channel" in row["content"]

    def test_handle_message_uses_forward_snapshot_in_event_text(self):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token")
        )
        adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
        captured = {}

        async def _noop_archive(_message):
            return None

        async def _empty_context(_message):
            return "", False, ""

        async def _capture_event(event):
            captured["event"] = event

        message = SimpleNamespace(
            id="m2",
            content="",
            system_content="",
            embeds=[],
            message_snapshots=[
                SimpleNamespace(
                    content="Forwarded event payload",
                    embeds=[],
                )
            ],
            mentions=[],
            attachments=[],
            channel=SimpleNamespace(
                id="ch1",
                name="general",
                guild=SimpleNamespace(name="Guild One"),
            ),
            author=SimpleNamespace(id="u1", name="alice"),
            created_at=datetime.fromtimestamp(1700000000),
            reference=None,
        )

        with patch.dict(os.environ, {"DISCORD_REQUIRE_MENTION": "false"}, clear=False):
            with patch.object(adapter, "_is_channel_allowed", return_value=True):
                with patch.object(adapter, "_archive_message", _noop_archive):
                    with patch.object(adapter, "_build_recent_channel_context", _empty_context):
                        with patch.object(adapter, "handle_message", _capture_event):
                            asyncio.run(adapter._handle_message(message))

        assert "event" in captured
        assert "Forwarded event payload" in captured["event"].text
