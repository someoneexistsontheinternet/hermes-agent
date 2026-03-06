"""Tests for gateway message chunking helpers."""

from gateway.platforms.base import format_discord_chunk_suffix, split_message_chunks


def test_split_message_chunks_prefers_last_newline_before_limit():
    line = "x" * 800
    content = f"{line}\n{line}\n{line}"

    chunks = split_message_chunks(content, 2000)

    assert len(chunks) == 2
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert chunks[0].rsplit(" (1/2)", 1)[0] == f"{line}\n{line}"
    assert chunks[1].rsplit(" (2/2)", 1)[0] == line


def test_format_discord_chunk_suffix_uses_subtext_style():
    assert format_discord_chunk_suffix("hello (1/2)") == "hello\n-# (1/2)"
    assert format_discord_chunk_suffix("hello") == "hello"
