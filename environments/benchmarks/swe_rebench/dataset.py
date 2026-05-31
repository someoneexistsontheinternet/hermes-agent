"""Dataset helpers for local SWE-rebench JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def repo_workdir(repo: str) -> str:
    """Derive the canonical repo workdir used by SWE-rebench images."""
    if not repo or "/" not in repo:
        raise ValueError(f"Invalid repo value: {repo!r}")
    return f"/{repo.split('/', 1)[1]}"


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one dataset row to the shapes the runner expects."""
    normalized = dict(row)
    install_config = dict(normalized.get("install_config") or {})

    test_cmd = install_config.get("test_cmd", [])
    if isinstance(test_cmd, str):
        test_cmd = [test_cmd]
    elif not isinstance(test_cmd, list):
        raise ValueError(f"install_config.test_cmd has unsupported type: {type(test_cmd).__name__}")
    install_config["test_cmd"] = [cmd for cmd in test_cmd if isinstance(cmd, str) and cmd.strip()]

    normalized["install_config"] = install_config
    normalized["repo_workdir"] = repo_workdir(str(normalized.get("repo", "")))
    normalized["FAIL_TO_PASS"] = list(normalized.get("FAIL_TO_PASS") or [])
    normalized["PASS_TO_PASS"] = list(normalized.get("PASS_TO_PASS") or [])
    return normalized


def iter_jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            yield normalize_row(payload)


def load_rows(
    path: Path,
    *,
    offset: int = 0,
    max_samples: Optional[int] = None,
    instance_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load and optionally slice/filter rows from a local JSONL dataset."""
    wanted = set(instance_ids or [])
    rows: List[Dict[str, Any]] = []

    for row in iter_jsonl_rows(path):
        if wanted and row.get("instance_id") not in wanted:
            continue
        rows.append(row)

    if offset:
        rows = rows[offset:]
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows
