from pathlib import Path

from environments.benchmarks.swe_rebench import scoring


def test_sanitize_model_patch_removes_overlapping_test_patch_files():
    model_patch = """diff --git a/src/pkg.py b/src/pkg.py
index 1111111..2222222 100644
--- a/src/pkg.py
+++ b/src/pkg.py
@@ -1 +1 @@
-old
+new
diff --git a/tests/test_pkg.py b/tests/test_pkg.py
index 3333333..4444444 100644
--- a/tests/test_pkg.py
+++ b/tests/test_pkg.py
@@ -1 +1 @@
-assert False
+assert True
"""
    test_patch = """diff --git a/tests/test_pkg.py b/tests/test_pkg.py
index 5555555..6666666 100644
--- a/tests/test_pkg.py
+++ b/tests/test_pkg.py
@@ -2 +2 @@
-old
+new
"""

    sanitized = scoring.sanitize_model_patch(model_patch, test_patch)

    assert sanitized["changed"] is True
    assert sanitized["removed_files"] == ["tests/test_pkg.py"]
    assert sanitized["kept_files"] == ["src/pkg.py"]
    assert "diff --git a/src/pkg.py b/src/pkg.py" in sanitized["patch"]
    assert "diff --git a/tests/test_pkg.py b/tests/test_pkg.py" not in sanitized["patch"]


def test_write_sanitized_model_patch_writes_sidecar_file(tmp_path: Path):
    model_patch_path = tmp_path / "model_patch.diff"
    model_patch_path.write_text(
        """diff --git a/testing/helpers.py b/testing/helpers.py
index 1111111..2222222 100644
--- a/testing/helpers.py
+++ b/testing/helpers.py
@@ -1 +1 @@
-old
+new
""",
        encoding="utf-8",
    )

    result = scoring.write_sanitized_model_patch(
        model_patch_path,
        """diff --git a/testing/helpers.py b/testing/helpers.py
index 3333333..4444444 100644
--- a/testing/helpers.py
+++ b/testing/helpers.py
@@ -2 +2 @@
-old
+new
""",
    )

    sanitized_path = Path(result["sanitized_patch_path"])
    assert sanitized_path.exists()
    assert sanitized_path.read_text(encoding="utf-8") == ""
    assert result["removed_files"] == ["testing/helpers.py"]


def test_classify_execution_issue_detects_missing_pytest_plugin():
    issue = scoring.classify_execution_issue(
        log="ERROR: Missing required plugins: pytest-subtests\n",
        exit_code=4,
    )

    assert issue == {
        "category": "environment_missing_plugin",
        "reason": "pytest-subtests",
        "exit_code": "4",
    }


def test_score_parsed_log_marks_environment_issues_unscorable():
    spec = {
        "instance_id": "demo__case-1",
        "FAIL_TO_PASS": ["tests/test_demo.py::test_it"],
        "PASS_TO_PASS": ["tests/test_demo.py::test_still_ok"],
        "install_config": {
            "log_parser": "parse_log_pytest",
            "test_cmd": ["pytest tests/test_demo.py"],
        },
    }

    score = scoring.score_parsed_log(
        spec,
        {},
        exit_code=4,
        log_path="/tmp/test_log.txt",
        log_text="ERROR: Missing required plugins: pytest-subtests\n",
    )

    assert score["correct"] is False
    assert score["scorable"] is False
    assert score["failure_category"] == "environment_missing_plugin"
    assert score["failure_reason"] == "pytest-subtests"
    assert score["from_fail_to_pass"] == []
    assert score["failed_from_pass_to_pass"] == []
