"""Tests for the file tools module (schema, handler wiring, error paths).

Tests verify tool schemas, handler dispatch, validation logic, and error
handling without requiring a running terminal environment.
"""

import json
import importlib
from unittest.mock import MagicMock, patch

from tools.file_tools import (
    FILE_TOOLS,
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    PATCH_SCHEMA,
    SEARCH_FILES_SCHEMA,
)


class TestFileToolsList:
    def test_has_expected_entries(self):
        names = {t["name"] for t in FILE_TOOLS}
        assert names == {"read_file", "write_file", "patch", "search_files"}

    def test_each_entry_has_callable_function(self):
        for tool in FILE_TOOLS:
            assert callable(tool["function"]), f"{tool['name']} missing callable"

    def test_schemas_have_required_fields(self):
        """All schemas must have name, description, and parameters with properties."""
        for schema in [READ_FILE_SCHEMA, WRITE_FILE_SCHEMA, PATCH_SCHEMA, SEARCH_FILES_SCHEMA]:
            assert "name" in schema
            assert "description" in schema
            assert "properties" in schema["parameters"]


class TestReadFileHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_returns_file_content(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"content": "line1\nline2", "total_lines": 2}
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool("/tmp/test.txt"))
        assert result["content"] == "line1\nline2"
        assert result["total_lines"] == 2
        mock_ops.read_file.assert_called_once_with("/tmp/test.txt", 1, 500)

    @patch("tools.file_tools._get_file_ops")
    def test_custom_offset_and_limit(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"content": "line10", "total_lines": 50}
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool
        read_file_tool("/tmp/big.txt", offset=10, limit=20)
        mock_ops.read_file.assert_called_once_with("/tmp/big.txt", 10, 20)

    @patch("tools.file_tools._get_file_ops")
    def test_exception_returns_error_json(self, mock_get):
        mock_get.side_effect = RuntimeError("terminal not available")

        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool("/tmp/test.txt"))
        assert "error" in result
        assert "terminal not available" in result["error"]

    @patch("tools.terminal_tool._get_env_config")
    def test_rejects_host_path_on_docker_backend(self, mock_env):
        mock_env.return_value = {"env_type": "docker"}

        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool("/Users/chen/notes/INTC/trades.md"))
        assert "error" in result
        assert "Host path not accessible from docker backend" in result["error"]


class TestWriteFileHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_writes_content(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "path": "/tmp/out.txt", "bytes": 13}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool("/tmp/out.txt", "hello world!\n"))
        assert result["status"] == "ok"
        mock_ops.write_file.assert_called_once_with("/tmp/out.txt", "hello world!\n")

    @patch("tools.file_tools._get_file_ops")
    def test_exception_returns_error_json(self, mock_get):
        mock_get.side_effect = PermissionError("read-only filesystem")

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool("/tmp/out.txt", "data"))
        assert "error" in result
        assert "read-only" in result["error"]

    @patch("tools.terminal_tool._get_env_config")
    def test_rejects_host_path_on_remote_backend(self, mock_env):
        mock_env.return_value = {"env_type": "ssh"}

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool("/Users/chen/notes/out.txt", "data"))
        assert "error" in result
        assert "Host path not accessible from ssh backend" in result["error"]


class TestPatchHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_calls_patch_replace(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "replacements": 1}
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(
            mode="replace", path="/tmp/f.py",
            old_string="foo", new_string="bar"
        ))
        assert result["status"] == "ok"
        mock_ops.patch_replace.assert_called_once_with("/tmp/f.py", "foo", "bar", False)

    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_replace_all_flag(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "replacements": 5}
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        patch_tool(mode="replace", path="/tmp/f.py",
                   old_string="x", new_string="y", replace_all=True)
        mock_ops.patch_replace.assert_called_once_with("/tmp/f.py", "x", "y", True)

    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_missing_path_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="replace", path=None, old_string="a", new_string="b"))
        assert "error" in result

    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_missing_strings_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="replace", path="/tmp/f.py", old_string=None, new_string="b"))
        assert "error" in result

    @patch("tools.file_tools._get_file_ops")
    def test_patch_mode_calls_patch_v4a(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "operations": 1}
        mock_ops.patch_v4a.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="patch", patch="*** Begin Patch\n..."))
        assert result["status"] == "ok"
        mock_ops.patch_v4a.assert_called_once()

    @patch("tools.file_tools._get_file_ops")
    def test_patch_mode_missing_content_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="patch", patch=None))
        assert "error" in result

    @patch("tools.file_tools._get_file_ops")
    def test_unknown_mode_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="invalid_mode"))
        assert "error" in result
        assert "Unknown mode" in result["error"]


class TestSearchHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_search_calls_file_ops(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"matches": ["file1.py:3:match"]}
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        result = json.loads(search_tool(pattern="TODO", target="content", path="."))
        assert "matches" in result
        mock_ops.search.assert_called_once()

    @patch("tools.file_tools._get_file_ops")
    def test_search_passes_all_params(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"matches": []}
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        search_tool(pattern="class", target="files", path="/src",
                    file_glob="*.py", limit=10, offset=5, output_mode="count", context=2)
        mock_ops.search.assert_called_once_with(
            pattern="class", path="/src", target="files", file_glob="*.py",
            limit=10, offset=5, output_mode="count", context=2,
        )

    @patch("tools.file_tools._get_file_ops")
    def test_search_exception_returns_error(self, mock_get):
        mock_get.side_effect = RuntimeError("no terminal")

        from tools.file_tools import search_tool
        result = json.loads(search_tool(pattern="x"))
        assert "error" in result

    @patch("tools.terminal_tool._get_env_config")
    def test_rejects_host_path_search_on_docker_backend(self, mock_env):
        mock_env.return_value = {"env_type": "docker"}

        from tools.file_tools import search_tool
        result = json.loads(search_tool(pattern="*", target="files", path="/Users/chen/notes"))
        assert "error" in result
        assert "Host path not accessible from docker backend" in result["error"]


class TestGetFileOpsEnvironmentCreation:
    def test_get_file_ops_passes_task_id_and_docker_volumes(self, monkeypatch):
        from tools import file_tools
        terminal_tool = importlib.import_module("tools.terminal_tool")

        fake_env = MagicMock(cwd="/workspace/docker_path")
        created = {}

        monkeypatch.setattr(file_tools, "_file_ops_cache", {})
        monkeypatch.setattr(terminal_tool, "_active_environments", {})
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(terminal_tool, "_creation_locks", {})
        monkeypatch.setattr(
            terminal_tool,
            "_get_env_config",
            lambda: {
                "env_type": "docker",
                "docker_image": "test-image",
                "cwd": "/workspace/docker_path",
                "timeout": 180,
                "container_cpu": 2,
                "container_memory": 4096,
                "container_disk": 102400,
                "container_persistent": True,
                "docker_volumes": ["/host/docker_path:/workspace/docker_path"],
            },
        )

        def _fake_create_environment(**kwargs):
            created.update(kwargs)
            return fake_env

        monkeypatch.setattr(terminal_tool, "_create_environment", _fake_create_environment)
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(terminal_tool, "_check_disk_usage_warning", lambda: None)

        file_ops = file_tools._get_file_ops("gw-test-docker")

        assert file_ops.env is fake_env
        assert created["task_id"] == "gw-test-docker"
        assert created["container_config"]["docker_volumes"] == [
            "/host/docker_path:/workspace/docker_path"
        ]

    def test_get_file_ops_passes_ssh_config(self, monkeypatch):
        from tools import file_tools
        terminal_tool = importlib.import_module("tools.terminal_tool")

        fake_env = MagicMock(cwd="~/workspace")
        created = {}

        monkeypatch.setattr(file_tools, "_file_ops_cache", {})
        monkeypatch.setattr(terminal_tool, "_active_environments", {})
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(terminal_tool, "_creation_locks", {})
        monkeypatch.setattr(
            terminal_tool,
            "_get_env_config",
            lambda: {
                "env_type": "ssh",
                "cwd": "~/workspace",
                "timeout": 90,
                "ssh_host": "example.com",
                "ssh_user": "chen",
                "ssh_port": 2222,
                "ssh_key": "~/.ssh/id_ed25519",
            },
        )

        def _fake_create_environment(**kwargs):
            created.update(kwargs)
            return fake_env

        monkeypatch.setattr(terminal_tool, "_create_environment", _fake_create_environment)
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(terminal_tool, "_check_disk_usage_warning", lambda: None)

        file_ops = file_tools._get_file_ops("gw-test-ssh")

        assert file_ops.env is fake_env
        assert created["task_id"] == "gw-test-ssh"
        assert created["ssh_config"] == {
            "host": "example.com",
            "user": "chen",
            "port": 2222,
            "key": "~/.ssh/id_ed25519",
        }
        assert created["container_config"] is None
