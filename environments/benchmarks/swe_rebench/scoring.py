"""Scoring helpers for SWE-rebench MVP."""

from __future__ import annotations

import importlib.util
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List


_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]
_DIFF_GIT_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
_EXECUTION_ISSUE_PATTERNS = [
    (
        "environment_missing_plugin",
        re.compile(r"Missing required plugins:\s*(?P<reason>[^\n]+)", re.IGNORECASE),
    ),
    (
        "environment_pytest_config",
        re.compile(r"pytest:\s*error:\s*unrecognized arguments:\s*(?P<reason>[^\n]*--nf[^\n]*)", re.IGNORECASE),
    ),
    (
        "infrastructure_image_build",
        re.compile(r"Image build for .* failed", re.IGNORECASE),
    ),
]


def normalize_test_name(name: str) -> str:
    """Strip timing suffixes/infixes from expected and actual test names."""
    normalized = name
    for pattern in _TIMING_NORMALIZE_RES:
        normalized = pattern.sub("", normalized)
    return normalized.strip()


def patch_files_from_diff(diff_text: str) -> List[str]:
    """Return the file paths touched by a unified git diff."""
    seen = set()
    files: List[str] = []
    for match in _DIFF_GIT_HEADER_RE.finditer(diff_text or ""):
        path = match.group(2)
        if path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def sanitize_model_patch(model_patch_text: str, test_patch_text: str) -> Dict[str, Any]:
    """Remove any file-level model diff chunks that overlap evaluator-owned test_patch files."""
    model_patch_text = model_patch_text or ""
    test_patch_text = test_patch_text or ""
    test_patch_files = set(patch_files_from_diff(test_patch_text))

    if not model_patch_text.strip():
        return {
            "patch": model_patch_text,
            "changed": False,
            "removed_files": [],
            "kept_files": [],
            "test_patch_files": sorted(test_patch_files),
        }

    parts = re.split(r"(?=^diff --git )", model_patch_text, flags=re.MULTILINE)
    preamble = ""
    chunks = parts
    if parts and not parts[0].startswith("diff --git "):
        preamble = parts[0]
        chunks = parts[1:]

    kept_chunks: List[str] = []
    kept_files: List[str] = []
    removed_files: List[str] = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        first_line = chunk.splitlines()[0] if chunk.splitlines() else ""
        match = re.match(r"^diff --git a/(.+?) b/(.+)$", first_line)
        if match is None:
            kept_chunks.append(chunk)
            continue
        path = match.group(2)
        if path in test_patch_files:
            removed_files.append(path)
            continue
        kept_chunks.append(chunk)
        kept_files.append(path)

    sanitized_patch = preamble + "".join(kept_chunks)
    return {
        "patch": sanitized_patch,
        "changed": sanitized_patch != model_patch_text,
        "removed_files": removed_files,
        "kept_files": kept_files,
        "test_patch_files": sorted(test_patch_files),
    }


def write_sanitized_model_patch(
    model_patch_path: Path,
    test_patch_text: str,
    *,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    """Write the scorer-facing patch after dropping model edits to evaluator-owned test files."""
    model_patch_text = model_patch_path.read_text(encoding="utf-8", errors="replace") if model_patch_path.exists() else ""
    result = sanitize_model_patch(model_patch_text, test_patch_text)
    sanitized_path = output_path or model_patch_path.with_name(
        f"{model_patch_path.stem}.sanitized{model_patch_path.suffix}"
    )
    sanitized_path.write_text(result["patch"], encoding="utf-8")
    result["original_patch_path"] = str(model_patch_path)
    result["sanitized_patch_path"] = str(sanitized_path)
    return result


def classify_execution_issue(*, log: str = "", error: str = "", exit_code: int | None = None) -> Dict[str, str] | None:
    """Identify clear non-model environment/infrastructure failures from scorer output."""
    haystack = "\n".join(part for part in (error, log) if part)
    if not haystack:
        return None
    for category, pattern in _EXECUTION_ISSUE_PATTERNS:
        match = pattern.search(haystack)
        if match is None:
            continue
        reason = (match.groupdict().get("reason") or match.group(0)).strip()
        return {
            "category": category,
            "reason": reason,
            "exit_code": "" if exit_code is None else str(exit_code),
        }
    return None


@lru_cache(maxsize=4)
def _load_official_log_parsers(repo_root_str: str):
    repo_root = Path(repo_root_str)
    module_path = repo_root / "lib" / "agent" / "log_parsers.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Official log parsers not found at {module_path}")

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    module_name = "swe_rebench_official_log_parsers"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_parser(repo_root: Path, parser_name: str) -> Callable[[str], Dict[str, str]]:
    """Resolve an official SWE-rebench log parser by name."""
    module = _load_official_log_parsers(str(repo_root))
    parser = getattr(module, "NAME_TO_PARSER", {}).get(parser_name)
    if parser is None:
        parser = getattr(module, parser_name, None)
    if parser is None:
        raise ValueError(f"Unknown log parser: {parser_name}")
    return parser


def parse_log(repo_root: Path, parser_name: str, log: str) -> Dict[str, str]:
    """Parse and normalize a test log."""
    parser = get_parser(repo_root, parser_name)
    parsed = parser(log)
    if not isinstance(parsed, dict):
        raise ValueError(f"Parser {parser_name} returned non-dict result")
    return {normalize_test_name(str(name)): str(status) for name, status in parsed.items()}


def score_parsed_log(
    spec: Dict[str, Any],
    parsed: Dict[str, str],
    *,
    exit_code: int,
    log_path: str,
    log_text: str = "",
) -> Dict[str, Any]:
    """Collapse parsed test results to the binary benchmark label plus details."""
    fail_to_pass_expected = {normalize_test_name(name) for name in spec.get("FAIL_TO_PASS", [])}
    pass_to_pass_expected = {normalize_test_name(name) for name in spec.get("PASS_TO_PASS", [])}

    passed_actual = sorted(name for name, status in parsed.items() if status == "PASSED")
    failed_actual = sorted(name for name, status in parsed.items() if status == "FAILED")
    passed_actual_set = set(passed_actual)
    expected_passed = sorted(fail_to_pass_expected | pass_to_pass_expected)
    from_fail_to_pass = sorted(passed_actual_set.intersection(fail_to_pass_expected))
    failed_from_pass_to_pass = sorted(pass_to_pass_expected.difference(passed_actual_set))
    passed_match = sorted(passed_actual_set) == expected_passed
    issue = classify_execution_issue(log=log_text, exit_code=exit_code)
    scorable = issue is None
    if not scorable:
        from_fail_to_pass = []
        failed_from_pass_to_pass = []
        passed_match = False

    return {
        "instance_id": spec.get("instance_id"),
        "correct": passed_match,
        "passed_match": passed_match,
        "scorable": scorable,
        "failure_category": None if issue is None else issue["category"],
        "failure_reason": "" if issue is None else issue["reason"],
        "exit_code": exit_code,
        "from_fail_to_pass": from_fail_to_pass,
        "failed_from_pass_to_pass": failed_from_pass_to_pass,
        "passed_actual": passed_actual,
        "failed_actual": failed_actual,
        "passed_expected": expected_passed,
        "log_parser": spec.get("install_config", {}).get("log_parser"),
        "test_cmd": list(spec.get("install_config", {}).get("test_cmd") or []),
        "log_path": log_path,
        "error": "",
    }
