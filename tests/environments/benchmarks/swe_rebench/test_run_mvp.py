import json
from argparse import Namespace

import pytest

from agent.prompt_builder import DEFAULT_AGENT_IDENTITY, TOOL_USE_ENFORCEMENT_GUIDANCE
from environments.benchmarks.swe_rebench import prompts
from environments.benchmarks.swe_rebench import run_mvp


def test_default_system_prompt_appends_benchmark_prompt_to_hermes_prompt():
    prompt = prompts.DEFAULT_SYSTEM_PROMPT

    assert DEFAULT_AGENT_IDENTITY in prompt
    assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt
    assert prompts.BENCHMARK_SYSTEM_PROMPT in prompt
    assert prompt.index(DEFAULT_AGENT_IDENTITY) < prompt.index(prompts.BENCHMARK_SYSTEM_PROMPT)


def test_load_prefill_messages_validates_schema(tmp_path):
    path = tmp_path / "prefill.json"
    payload = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert run_mvp._load_prefill_messages(path) == payload


def test_load_prefill_messages_rejects_non_array(tmp_path):
    path = tmp_path / "prefill.json"
    path.write_text(json.dumps({"role": "user", "content": "Hi"}), encoding="utf-8")

    with pytest.raises(ValueError):
        run_mvp._load_prefill_messages(path)


def test_parse_reasoning_config_none_disables_provider_reasoning():
    assert run_mvp._parse_reasoning_config("none") == {"effort": "none"}


def test_parse_reasoning_config_preserves_effort():
    assert run_mvp._parse_reasoning_config("high") == {"enabled": True, "effort": "high"}


def test_parse_reasoning_config_accepts_deepseek_max_effort():
    assert run_mvp._parse_reasoning_config("max") == {"enabled": True, "effort": "max"}


def test_parse_reasoning_config_rejects_unknown_effort():
    with pytest.raises(ValueError):
        run_mvp._parse_reasoning_config("maximum")


def test_resolve_extra_body_merges_reasoning_override():
    args = Namespace(
        extra_body_json='{"provider":{"ignore":["deepinfra"]}}',
        reasoning_config={"effort": "none"},
    )

    assert run_mvp._resolve_extra_body(args) == {
        "provider": {"ignore": ["deepinfra"]},
        "reasoning": {"effort": "none"},
        "reasoning_effort": "none",
    }


def test_resolve_extra_body_adds_vllm_reasoning_effort():
    args = Namespace(
        extra_body_json=None,
        reasoning_config={"enabled": True, "effort": "xhigh"},
    )

    assert run_mvp._resolve_extra_body(args) == {
        "reasoning": {"enabled": True, "effort": "xhigh"},
        "reasoning_effort": "xhigh",
    }


def test_api_trace_sink_counts_reasoning_text_without_reasoning_token_details():
    sink = run_mvp.ApiTraceSink(
        path=None,
        prompt_cost_per_million=0.0,
        completion_cost_per_million=0.0,
    )

    sink(
        {
            "reasoning": "private scratchpad",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "completion_tokens_details": None,
            },
        }
    )

    summary = sink.summary()
    assert summary["reasoning_tokens"] == 0
    assert summary["calls_with_reasoning"] == 1
    assert summary["reasoning_chars"] == len("private scratchpad")


def test_render_system_prompt_templates_injects_dsml_tokens_and_tool_schemas():
    rendered = run_mvp._render_system_prompt_templates(
        "token={dsml_token} start={thinking_start_token} end={thinking_end_token}\n{tool_schemas}",
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert "｜DSML｜" in rendered
    assert "<think>" in rendered
    assert "</think>" in rendered
    assert '"name": "read_file"' in rendered


def test_prompt_caching_auto_enables_for_nous_claude():
    args = Namespace(
        prompt_caching=None,
        base_url="https://inference-api.nousresearch.com/v1",
        model="anthropic/claude-opus-4.6",
    )

    assert run_mvp._prompt_caching_enabled(args)


def test_prompt_caching_nous_transform_marks_tool_messages():
    args = Namespace(
        prompt_caching=None,
        base_url="https://inference-api.nousresearch.com/v1",
        model="anthropic/claude-opus-4.6",
        prompt_cache_ttl="5m",
    )
    transform = run_mvp._request_transform(args)
    payload = transform(
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ]
        }
    )

    assert payload["messages"][-1]["cache_control"] == {"type": "ephemeral"}


def test_prompt_caching_openrouter_transform_skips_tool_messages():
    args = Namespace(
        prompt_caching=None,
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-opus-4.6",
        prompt_cache_ttl="5m",
    )
    transform = run_mvp._request_transform(args)
    payload = transform(
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ]
        }
    )

    assert "cache_control" not in payload["messages"][-1]
