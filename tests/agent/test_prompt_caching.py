"""Tests for agent/prompt_caching.py — Anthropic cache control helpers."""

from agent.prompt_caching import (
    _apply_cache_marker,
    apply_anthropic_cache_control,
    build_anthropic_cache_control,
)


MARKER = {"type": "ephemeral"}


class TestBuildAnthropicCacheControl:
    def test_default_ttl(self):
        assert build_anthropic_cache_control() == {"type": "ephemeral"}

    def test_1h_ttl(self):
        assert build_anthropic_cache_control("1h") == {
            "type": "ephemeral",
            "ttl": "1h",
        }


class TestApplyCacheMarker:
    def test_tool_message_gets_top_level_marker(self):
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, MARKER)
        assert msg["cache_control"] == MARKER

    def test_none_content_gets_top_level_marker(self):
        msg = {"role": "assistant", "content": None}
        _apply_cache_marker(msg, MARKER)
        assert msg["cache_control"] == MARKER

    def test_string_content_wrapped_in_list(self):
        msg = {"role": "user", "content": "Hello"}
        _apply_cache_marker(msg, MARKER)
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Hello"
        assert msg["content"][0]["cache_control"] == MARKER

    def test_list_content_last_item_gets_marker(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"},
            ],
        }
        _apply_cache_marker(msg, MARKER)
        assert "cache_control" not in msg["content"][0]
        assert msg["content"][1]["cache_control"] == MARKER

    def test_empty_list_content_no_crash(self):
        msg = {"role": "user", "content": []}
        # Should not crash on empty list
        _apply_cache_marker(msg, MARKER)


class TestApplyAnthropicCacheControl:
    def test_empty_messages(self):
        result = apply_anthropic_cache_control([])
        assert result == []

    def test_returns_deep_copy(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = apply_anthropic_cache_control(msgs)
        assert result is not msgs
        assert result[0] is not msgs[0]
        # Original should be unmodified
        assert "cache_control" not in msgs[0].get("content", "")

    def test_system_message_gets_marker(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # System message should have cache_control
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["type"] == "ephemeral"

    def test_last_non_system_gets_marker(self):
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        result = apply_anthropic_cache_control(msgs)
        for msg in result[:-1]:
            content = msg.get("content")
            if isinstance(content, list) and content:
                assert "cache_control" not in content[-1]
            assert "cache_control" not in msg
        assert result[-1]["content"][0]["cache_control"]["type"] == "ephemeral"

    def test_1h_ttl(self):
        msgs = [{"role": "system", "content": "System prompt"}]
        result = apply_anthropic_cache_control(msgs, cache_ttl="1h")
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["ttl"] == "1h"

    def test_single_breakpoint_only(self):
        msgs = [
            {"role": "system", "content": "System"},
        ] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(10)
        ]
        result = apply_anthropic_cache_control(msgs)
        # Count how many messages have cache_control
        count = 0
        for msg in result:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "cache_control" in item:
                        count += 1
            elif "cache_control" in msg:
                count += 1
        assert count == 1

    def test_final_tool_message_gets_marker(self):
        first_turn = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "assistant", "content": "msg4"},
            {"role": "tool", "content": "tool-result-1"},
            {"role": "tool", "content": "tool-result-2"},
        ]

        result = apply_anthropic_cache_control(first_turn)
        assert result[-1]["cache_control"]["type"] == "ephemeral"
        assert "cache_control" not in result[-2]
