from types import SimpleNamespace

import pytest

from environments.agent_loop import (
    HermesAgentLoop,
    _extract_scratchpad_from_content,
    _merge_reasoning_sources,
    _parse_bare_tool_calls,
    _parse_dsml_tool_calls,
    _parse_markdown_tool_calls,
    _parse_xml_tool_calls,
)


def _mock_response(content: str, reasoning: str | None = None):
    message = SimpleNamespace(content=content, tool_calls=None)
    if reasoning is not None:
        message.reasoning = reasoning
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", id="resp_123", usage=None)


class _DummyServer:
    model_name = "test/model"
    base_url = "https://example.com/v1"

    def __init__(self, response):
        self._response = response
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _SequenceServer:
    model_name = "test/model"
    base_url = "https://example.com/v1"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_extract_scratchpad_from_content_splits_visible_text():
    visible, reasoning = _extract_scratchpad_from_content(
        "\n<REASONING_SCRATCHPAD>\ninternal note\n</REASONING_SCRATCHPAD>\n\nVisible answer"
    )
    assert visible == "Visible answer"
    assert reasoning == "internal note"


def test_merge_reasoning_sources_deduplicates():
    merged = _merge_reasoning_sources("same text", "same text", "other")
    assert merged == "same text\n\nother"


def test_parse_markdown_tool_calls_extracts_ui_style_calls():
    content, tool_calls = _parse_markdown_tool_calls(
        "I'll inspect the file.\n"
        "**Calling:** `read_file`\n"
        "```json\n"
        "{\"path\": \"/repo/example.py\", \"offset\": 10}\n"
        "```\n"
    )

    assert content == "I'll inspect the file."
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert tool_calls[0]["function"]["arguments"] == '{"path": "/repo/example.py", "offset": 10}'


def test_parse_xml_tool_calls_extracts_custom_xml_schema():
    content, tool_calls = _parse_xml_tool_calls(
        "<tool>\nread_file\n</tool>\n<args>\n{\"path\": \"/repo/example.py\", \"offset\": 10}\n</args>",
        {"read_file"},
    )

    assert content == ""
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert tool_calls[0]["function"]["arguments"] == '{"path": "/repo/example.py", "offset": 10}'


def test_parse_dsml_tool_calls_extracts_deepseek_schema():
    content, tool_calls = _parse_dsml_tool_calls(
        'Let me read it.\n'
        '<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="read_file">\n'
        '<｜DSML｜parameter name="path" string="true">/repo/example.py</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="offset" string="false">10</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜tool_calls>',
        {"read_file"},
    )

    assert content == "Let me read it."
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert tool_calls[0]["function"]["arguments"] == '{"path": "/repo/example.py", "offset": 10}'


def test_parse_bare_tool_calls_extracts_function_style_calls():
    content, tool_calls = _parse_bare_tool_calls(
        'Let me continue reading:\nread_file({"path": "/repo/example.py", "offset": 10})',
        {"read_file"},
    )

    assert content == "Let me continue reading:"
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert tool_calls[0]["function"]["arguments"] == '{"path": "/repo/example.py", "offset": 10}'


@pytest.mark.asyncio
async def test_agent_loop_preserves_raw_scratchpad_in_history():
    agent = HermesAgentLoop(
        server=_DummyServer(
            _mock_response(
                "\n<REASONING_SCRATCHPAD>\nprivate reasoning\n</REASONING_SCRATCHPAD>\n\nPublic answer"
            )
        ),
        tool_schemas=[],
        valid_tool_names=set(),
        max_turns=1,
    )

    result = await agent.run([{"role": "user", "content": "Solve it"}])

    assert result.finished_naturally is True
    assert result.messages[-1]["role"] == "assistant"
    assert result.messages[-1]["content"].startswith("\n<REASONING_SCRATCHPAD>")
    assert "Public answer" in result.messages[-1]["content"]
    assert "reasoning_content" not in result.messages[-1]


@pytest.mark.asyncio
async def test_agent_loop_sends_tool_choice_auto_with_tools():
    server = _DummyServer(_mock_response("Done"))
    agent = HermesAgentLoop(
        server=server,
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run a command",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        valid_tool_names={"terminal"},
        max_turns=1,
    )

    await agent.run([{"role": "user", "content": "Inspect the repo"}])

    assert server.calls[0]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_agent_loop_can_omit_native_tool_schema_payload():
    server = _DummyServer(_mock_response("Done"))
    agent = HermesAgentLoop(
        server=server,
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        valid_tool_names={"read_file"},
        max_turns=1,
        send_tool_schemas=False,
    )

    await agent.run([{"role": "user", "content": "Inspect the repo"}])

    assert "tools" not in server.calls[0]
    assert "tool_choice" not in server.calls[0]


@pytest.mark.asyncio
async def test_agent_loop_reprompts_intended_tool_use_without_tool_calls():
    server = _SequenceServer(
        [
            _mock_response("I'll inspect the repository now."),
            _mock_response("No change needed."),
        ]
    )
    agent = HermesAgentLoop(
        server=server,
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        valid_tool_names={"read_file"},
        max_turns=2,
    )

    result = await agent.run([{"role": "user", "content": "Fix the bug"}])

    assert len(server.calls) == 2
    assert result.messages[-2]["role"] == "user"
    assert "did not make a structured tool call" in result.messages[-2]["content"]


@pytest.mark.asyncio
async def test_agent_loop_recovers_dsml_tool_call_from_reasoning():
    server = _DummyServer(
        _mock_response(
            "",
            reasoning=(
                "I need a local tool.\n"
                '<｜DSML｜tool_calls>\n'
                '<｜DSML｜invoke name="memory">\n'
                '</｜DSML｜invoke>\n'
                '</｜DSML｜tool_calls>'
            ),
        )
    )
    agent = HermesAgentLoop(
        server=server,
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "memory",
                    "description": "Unavailable memory tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        valid_tool_names={"memory"},
        max_turns=1,
    )

    result = await agent.run([{"role": "user", "content": "Use memory"}])

    assert result.messages[-2]["role"] == "assistant"
    assert result.messages[-2]["tool_calls"][0]["function"]["name"] == "memory"
    assert result.messages[-2]["reasoning_content"] == "I need a local tool."
    assert result.messages[-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_agent_loop_can_omit_native_reasoning_from_history():
    server = _DummyServer(
        _mock_response(
            "",
            reasoning=(
                "I need a local tool.\n"
                '<｜DSML｜tool_calls>\n'
                '<｜DSML｜invoke name="memory">\n'
                '</｜DSML｜invoke>\n'
                '</｜DSML｜tool_calls>'
            ),
        )
    )
    agent = HermesAgentLoop(
        server=server,
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "memory",
                    "description": "Unavailable memory tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        valid_tool_names={"memory"},
        max_turns=1,
        preserve_reasoning_in_history=False,
    )

    result = await agent.run([{"role": "user", "content": "Use memory"}])

    assert result.messages[-2]["role"] == "assistant"
    assert result.messages[-2]["tool_calls"][0]["function"]["name"] == "memory"
    assert "reasoning_content" not in result.messages[-2]
    assert result.reasoning_per_turn[-1] == "I need a local tool."
