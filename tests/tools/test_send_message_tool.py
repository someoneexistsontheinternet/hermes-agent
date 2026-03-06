"""Tests for direct messaging helpers."""

import asyncio
import sys
import types

from tools.send_message_tool import _send_discord


class _FakeDiscordResponse:
    def __init__(self, message_id: str):
        self.status = 200
        self._message_id = message_id

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return ""

    async def json(self):
        return {"id": self._message_id}


class _FakeDiscordSession:
    def __init__(self, sent):
        self._sent = sent

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None, json=None):
        self._sent.append(json["content"])
        return _FakeDiscordResponse(str(len(self._sent)))


def test_send_discord_uses_newline_aware_chunking(monkeypatch):
    sent = []
    fake_aiohttp = types.SimpleNamespace(ClientSession=lambda: _FakeDiscordSession(sent))
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    line = "x" * 800
    content = f"{line}\n{line}\n{line}"

    result = asyncio.run(_send_discord("token", "123", content))

    assert result["success"] is True
    assert result["message_ids"] == ["1", "2"]
    assert all(len(chunk) <= 2000 for chunk in sent)
    assert sent[0].rsplit(" (1/2)", 1)[0] == f"{line}\n{line}"
    assert sent[1].rsplit(" (2/2)", 1)[0] == line
