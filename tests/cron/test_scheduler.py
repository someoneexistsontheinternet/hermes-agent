"""Tests for cron/scheduler.py."""

import json
import os
import sys
import types

import pytest

import cron.scheduler as scheduler
from model_runtime_config import ModelRuntimeConfig


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    (tmp_path / ".env").write_text("", encoding="utf-8")
    return tmp_path


def _install_fake_agent(monkeypatch, result, captures):
    module = types.ModuleType("run_agent")

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captures.append(kwargs)

        def run_conversation(self, prompt):
            captures[-1]["prompt"] = prompt
            return result

    module.AIAgent = FakeAIAgent
    monkeypatch.setitem(sys.modules, "run_agent", module)


class TestResolveOrigin:
    def test_full_origin(self):
        job = {
            "origin": {
                "platform": "telegram",
                "chat_id": "123456",
                "chat_name": "Test Chat",
            }
        }
        result = scheduler._resolve_origin(job)
        assert result is not None
        assert result["platform"] == "telegram"
        assert result["chat_id"] == "123456"

    def test_no_origin(self):
        assert scheduler._resolve_origin({}) is None
        assert scheduler._resolve_origin({"origin": None}) is None

    def test_missing_platform(self):
        job = {"origin": {"chat_id": "123"}}
        assert scheduler._resolve_origin(job) is None

    def test_missing_chat_id(self):
        job = {"origin": {"platform": "telegram"}}
        assert scheduler._resolve_origin(job) is None

    def test_empty_origin(self):
        job = {"origin": {}}
        assert scheduler._resolve_origin(job) is None


class TestRunJob:
    def test_prefers_openrouter_key_for_openrouter_jobs(self, hermes_home, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
        monkeypatch.setattr(
            scheduler,
            "load_model_runtime_config",
            lambda *args, **kwargs: ModelRuntimeConfig(
                model="anthropic/claude-opus-4.6",
                base_url="https://openrouter.ai/api/v1",
                provider="openrouter",
                extra_body=None,
            ),
        )

        captures = []
        _install_fake_agent(
            monkeypatch,
            {"final_response": "ok", "failed": False, "error": None},
            captures,
        )

        success, output, final_response, error = scheduler.run_job(
            {
                "id": "job-123",
                "name": "Daily Summary",
                "prompt": "Summarize the day.",
                "schedule_display": "0 9 * * *",
            }
        )

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert "## Response" in output
        assert captures[0]["api_key"] == "openrouter-key"
        assert captures[0]["base_url"] == "https://openrouter.ai/api/v1"

    def test_agent_failure_stays_failed(self, hermes_home, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
        monkeypatch.setattr(
            scheduler,
            "load_model_runtime_config",
            lambda *args, **kwargs: ModelRuntimeConfig(
                model="anthropic/claude-opus-4.6",
                base_url="https://openrouter.ai/api/v1",
                provider="openrouter",
                extra_body=None,
            ),
        )

        captures = []
        _install_fake_agent(
            monkeypatch,
            {
                "final_response": None,
                "failed": True,
                "error": "Error code: 401 - Missing Authentication header",
            },
            captures,
        )

        success, output, final_response, error = scheduler.run_job(
            {
                "id": "job-123",
                "name": "Daily Summary",
                "prompt": "Summarize the day.",
                "schedule_display": "0 9 * * *",
            }
        )

        assert success is False
        assert final_response == ""
        assert error == "Error code: 401 - Missing Authentication header"
        assert "# Cron Job: Daily Summary (FAILED)" in output
        assert "Missing Authentication header" in output
        assert "(No response generated)" not in output

    def test_passes_workspace_context_cwd_to_agent(self, hermes_home, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        (hermes_home / "AGENTS.md").write_text("workspace rules\n", encoding="utf-8")

        captures = []
        _install_fake_agent(
            monkeypatch,
            {"final_response": "ok", "failed": False, "error": None},
            captures,
        )

        success, output, final_response, error = scheduler.run_job(
            {
                "id": "job-ctx",
                "name": "Context Job",
                "prompt": "Ping",
                "schedule_display": "every hour",
            }
        )

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert "## Response" in output
        assert captures[0]["context_cwd"] == str(hermes_home)

    def test_normalizes_terminal_env_from_workspace_config(self, hermes_home, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("MESSAGING_CWD", "/workspace/docker_path")
        monkeypatch.setenv("TERMINAL_CWD", "~")
        (hermes_home / "config.yaml").write_text(
            "terminal:\n"
            "  backend: docker\n"
            "  docker_volumes:\n"
            "    - /host/data:/workspace/docker_path\n",
            encoding="utf-8",
        )

        captures = []
        _install_fake_agent(
            monkeypatch,
            {"final_response": "ok", "failed": False, "error": None},
            captures,
        )

        success, _, _, error = scheduler.run_job(
            {
                "id": "job-env",
                "name": "Env Job",
                "prompt": "Ping",
                "schedule_display": "every hour",
            }
        )

        assert success is True
        assert error is None
        assert os.environ["TERMINAL_ENV"] == "docker"
        assert os.environ["TERMINAL_CWD"] == "/workspace/docker_path"
        assert os.environ["TERMINAL_DOCKER_VOLUMES"] == "[\"/host/data:/workspace/docker_path\"]"

    def test_applies_platform_runtime_overrides_from_workspace_config(self, hermes_home, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
        (hermes_home / "prefill.json").write_text(
            json.dumps(
                [
                    {"role": "user", "content": "prefill user"},
                    {"role": "assistant", "content": "prefill assistant"},
                ]
            ),
            encoding="utf-8",
        )
        (hermes_home / "config.yaml").write_text(
            "agent:\n"
            "  reasoning_effort: none\n"
            "  system_prompt: Keep it serious.\n"
            "platform_toolsets:\n"
            "  discord:\n"
            "    - web\n"
            "    - discord_search\n"
            "prefill_messages_file: prefill.json\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            scheduler,
            "load_model_runtime_config",
            lambda *args, **kwargs: ModelRuntimeConfig(
                model="anthropic/claude-opus-4.6",
                base_url="https://openrouter.ai/api/v1",
                provider="openrouter",
                extra_body=None,
            ),
        )

        captures = []
        _install_fake_agent(
            monkeypatch,
            {"final_response": "ok", "failed": False, "error": None},
            captures,
        )

        success, _, final_response, error = scheduler.run_job(
            {
                "id": "job-platform",
                "name": "Discord Job",
                "prompt": "Ping",
                "schedule_display": "every hour",
                "origin": {
                    "platform": "discord",
                    "chat_id": "123",
                    "chat_name": "daily-slop-summary",
                    "chat_type": "channel",
                    "thread_id": "456",
                },
            }
        )

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert captures[0]["enabled_toolsets"] == ["web", "discord_search"]
        assert captures[0]["reasoning_config"] == {"enabled": False}
        assert captures[0]["platform"] == "discord"
        assert captures[0]["ephemeral_system_prompt"] == "Keep it serious."
        assert captures[0]["prefill_messages"] == [
            {"role": "user", "content": "prefill user"},
            {"role": "assistant", "content": "prefill assistant"},
        ]
        assert "HERMES_SESSION_PLATFORM" not in os.environ
        assert "HERMES_SESSION_CHAT_ID" not in os.environ
        assert "HERMES_SESSION_CHAT_TYPE" not in os.environ
        assert "HERMES_SESSION_CHAT_NAME" not in os.environ
        assert "HERMES_SESSION_THREAD_ID" not in os.environ
