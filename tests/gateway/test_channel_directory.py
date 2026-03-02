"""Tests for gateway/channel_directory.py."""

import sys
from types import SimpleNamespace

from gateway import channel_directory


class _FakeChannel:
    def __init__(self, channel_id: str, name: str, view_channel: bool, read_history: bool):
        self.id = channel_id
        self.name = name
        self._view_channel = view_channel
        self._read_history = read_history

    def permissions_for(self, _member):
        return SimpleNamespace(
            view_channel=self._view_channel,
            read_message_history=self._read_history,
        )


def test_build_discord_filters_channels_without_read_access(monkeypatch):
    monkeypatch.setitem(sys.modules, "discord", SimpleNamespace())
    monkeypatch.setattr(channel_directory, "_build_from_sessions", lambda _platform: [])

    guild = SimpleNamespace(
        name="Guild A",
        me=object(),
        text_channels=[
            _FakeChannel("1", "ok", True, True),
            _FakeChannel("2", "no-view", False, True),
            _FakeChannel("3", "no-history", True, False),
        ],
    )
    client = SimpleNamespace(guilds=[guild], user=SimpleNamespace(id=42))
    adapter = SimpleNamespace(_client=client)

    rows = channel_directory._build_discord(adapter)

    assert len(rows) == 1
    assert rows[0]["id"] == "1"
    assert rows[0]["name"] == "ok"
    assert rows[0]["guild"] == "Guild A"


def test_build_discord_skips_guild_when_bot_member_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "discord", SimpleNamespace())
    monkeypatch.setattr(channel_directory, "_build_from_sessions", lambda _platform: [])

    class _GuildWithoutMember:
        name = "Guild B"
        me = None
        text_channels = [_FakeChannel("10", "general", True, True)]

        @staticmethod
        def get_member(_user_id):
            return None

    client = SimpleNamespace(guilds=[_GuildWithoutMember()], user=SimpleNamespace(id=99))
    adapter = SimpleNamespace(_client=client)

    rows = channel_directory._build_discord(adapter)

    assert rows == []
