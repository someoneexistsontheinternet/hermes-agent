"""Regression tests for Modal worker cleanup and benchmark runner failures."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mini-swe-agent" / "src"))

from tools.environments.modal import _AsyncWorker


def _load_run_mvp_module():
    module_path = PROJECT_ROOT / "environments" / "benchmarks" / "swe_rebench" / "run_mvp.py"
    module_name = "test_run_mvp_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


run_mvp = _load_run_mvp_module()


def test_async_worker_closes_event_loop_on_stop():
    worker = _AsyncWorker()
    worker.start()

    loop = worker._loop
    assert loop is not None
    assert loop.is_running()

    worker.stop()

    assert loop.is_closed()


def test_async_worker_surfaces_startup_errors(monkeypatch):
    def _boom():
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(asyncio, "new_event_loop", _boom)

    worker = _AsyncWorker()
    with pytest.raises(RuntimeError, match="AsyncWorker failed to start") as excinfo:
        worker.start()

    assert isinstance(excinfo.value.__cause__, OSError)


def test_workspace_spend_limit_detection_raises_on_nested_results():
    payload = {
        "exit_code": -1,
        "error": "Failed to execute command: Workspace has exceeded its spend limit",
        "output": "",
    }

    assert run_mvp._contains_workspace_spend_limit(payload)

    with pytest.raises(run_mvp.WorkspaceSpendLimitError, match="Workspace has exceeded its spend limit"):
        run_mvp._raise_if_workspace_spend_limit(payload, context="patch extraction")


class _FakeToolContext:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.commands: list[tuple[str, int, dict]] = []
        self.uploads: list[tuple[str, str]] = []
        self.writes: dict[str, str] = {}
        self.cleaned = False
        self._status_polls = 0

    def terminal(self, command: str, timeout: int = 180, **kwargs):
        self.commands.append((command, timeout, kwargs))
        if "run_score_wrapper.sh" in command:
            return {"exit_code": 0, "session_id": "proc_fake", "output": "Background process started"}
        if "__PENDING__" in command:
            self._status_polls += 1
            if self._status_polls == 1:
                return {"exit_code": 0, "output": "__PENDING__\n"}
            return {"exit_code": 0, "output": "0\n"}
        return {"exit_code": 0, "output": ""}

    def call_tool(self, tool_name: str, arguments: dict):
        assert tool_name == "process"
        assert arguments["session_id"] == "proc_fake"
        return json.dumps({"status": "running", "session_id": "proc_fake"})

    def upload_file(self, local_path: str, remote_path: str):
        self.uploads.append((local_path, remote_path))
        return {"exit_code": 0, "output": ""}

    def write_file(self, path: str, content: str):
        self.writes[path] = content
        return {"error": None}

    def cleanup(self):
        self.cleaned = True


def test_run_background_scorer_polls_until_exit_code_ready(monkeypatch):
    class _PollingContext:
        def __init__(self):
            self.calls = 0
            self.status_calls = 0
            self.commands = []
            self.polls = []

        def terminal(self, command: str, timeout: int = 180, **kwargs):
            self.commands.append((command, timeout, kwargs))
            self.calls += 1
            if self.calls == 1:
                return {"exit_code": 0, "session_id": "proc_test", "output": "Background process started"}
            self.status_calls += 1
            if self.status_calls == 1:
                return {"exit_code": 0, "output": "__PENDING__\n"}
            return {"exit_code": 0, "output": "17\n"}

        def call_tool(self, tool_name: str, arguments: dict):
            self.polls.append((tool_name, arguments))
            return json.dumps({"status": "running", "session_id": arguments["session_id"]})

    monkeypatch.setattr(run_mvp.time, "sleep", lambda *_args, **_kwargs: None)

    ctx = _PollingContext()
    exit_code = run_mvp._run_background_scorer(
        ctx,
        launch_command="launch scorer",
        remote_exit="/tmp/exit.txt",
        timeout_seconds=30,
        poll_interval_seconds=0,
    )

    assert exit_code == 17
    assert ctx.commands[0][0] == "launch scorer"
    assert ctx.commands[0][2]["background"] is True
    assert ctx.polls == [("process", {"action": "poll", "session_id": "proc_test"})]


def test_score_in_fresh_sandbox_uses_background_launch(monkeypatch, tmp_path):
    created_contexts: list[_FakeToolContext] = []

    def _fake_tool_context(task_id: str):
        ctx = _FakeToolContext(task_id)
        created_contexts.append(ctx)
        return ctx

    def _fake_download(ctx, remote_path, local_path, *, chunk_bytes=24000):
        Path(local_path).write_text("fake scorer log\n", encoding="utf-8")
        return {"success": True, "bytes": 16}

    def _fake_score(_row, _parsed, *, exit_code, log_path, log_text=""):
        return {
            "correct": exit_code == 0,
            "exit_code": exit_code,
            "scorable": True,
            "failure_category": None,
            "failure_reason": "",
            "from_fail_to_pass": [],
            "failed_from_pass_to_pass": [],
            "passed_actual": [],
            "error": "",
            "log_path": log_path,
        }

    monkeypatch.setattr(run_mvp, "ToolContext", _fake_tool_context)
    monkeypatch.setattr(run_mvp, "register_task_env_overrides", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mvp, "clear_task_env_overrides", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mvp, "_download_large_file", _fake_download)
    monkeypatch.setattr(run_mvp, "parse_log", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_mvp, "score_parsed_log", _fake_score)
    monkeypatch.setattr(run_mvp.time, "sleep", lambda *_args, **_kwargs: None)

    instance_dir = tmp_path / "instance"
    instance_dir.mkdir()
    (instance_dir / "model_patch.diff").write_text(
        """diff --git a/src/demo.py b/src/demo.py
index 1111111..2222222 100644
--- a/src/demo.py
+++ b/src/demo.py
@@ -1 +1 @@
-old
+new
diff --git a/tests/test_demo.py b/tests/test_demo.py
index 3333333..4444444 100644
--- a/tests/test_demo.py
+++ b/tests/test_demo.py
@@ -1 +1 @@
-assert False
+assert True
""",
        encoding="utf-8",
    )

    row = {
        "instance_id": "demo__case-1",
        "image_name": "demo-image",
        "repo_workdir": "/repo",
        "base_commit": "abc123",
        "test_patch": """diff --git a/tests/test_demo.py b/tests/test_demo.py
index 5555555..6666666 100644
--- a/tests/test_demo.py
+++ b/tests/test_demo.py
@@ -2 +2 @@
-assert False
+assert True
""",
        "install_config": {
            "test_cmd": ["./vendor/bin/phpunit --colors=never"],
            "log_parser": "parse_log_phpunit",
        },
    }

    score = run_mvp._score_in_fresh_sandbox(
        mode="no_web",
        row=row,
        swe_rebench_src=tmp_path,
        instance_dir=instance_dir,
        scorer_timeout=600,
    )

    assert score["exit_code"] == 0
    assert created_contexts
    ctx = created_contexts[0]
    scorer_launches = [
        (command, kwargs)
        for command, _timeout, kwargs in ctx.commands
        if "run_score_wrapper.sh" in command and kwargs.get("background")
    ]
    assert scorer_launches
    assert all("nohup" not in command for command, _kwargs in scorer_launches)
    assert scorer_launches[0][1]["background"] is True
    assert any(path.endswith("run_score_wrapper.sh") for path in ctx.writes)
    assert ctx.uploads[0][0].endswith("model_patch.sanitized.diff")
    sanitized_patch = Path(ctx.uploads[0][0]).read_text(encoding="utf-8")
    assert "diff --git a/src/demo.py b/src/demo.py" in sanitized_patch
    assert "diff --git a/tests/test_demo.py b/tests/test_demo.py" not in sanitized_patch
