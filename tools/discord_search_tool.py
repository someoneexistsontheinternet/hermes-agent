#!/usr/bin/env python3
"""Discord archive SQL search tool (SQLite)."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.discord_archive import default_archive_db_path
from tools.registry import registry


def _archive_db_path() -> Path:
    path = (os.getenv("DISCORD_ARCHIVE_DB_PATH", "") or "").strip()
    if path:
        return Path(path).expanduser()
    return default_archive_db_path()


def check_discord_search_requirements() -> bool:
    """Enable only when the archive DB file exists."""
    return _archive_db_path().exists()


_SQL_READ_PREFIXES = ("select", "with")
_SQL_READ_ACTIONS = {
    getattr(sqlite3, "SQLITE_READ", -1),
    getattr(sqlite3, "SQLITE_SELECT", -1),
    getattr(sqlite3, "SQLITE_FUNCTION", -1),
    getattr(sqlite3, "SQLITE_RECURSIVE", -1),
}
_SQL_FORBIDDEN_CLAUSE = re.compile(
    r"\b(insert|update|delete|replace|create|drop|alter|attach|detach|vacuum|reindex|analyze)\b",
    flags=re.IGNORECASE,
)
_ALLOWED_INTERNAL_PRAGMAS = {"data_version"}
_DEFAULT_SQL_LIMIT = 100
_MAX_SQL_ROWS = 500
_TRUNCATED_TEXT_LIMIT = 200
_FULL_TEXT_SQL_ROW_THRESHOLD = 5
_MAX_MESSAGE_IDS = 50


def _normalize_read_sql(raw_sql: str) -> Tuple[Optional[str], Optional[str]]:
    text = (raw_sql or "").strip()
    if not text:
        return None, "SQL cannot be empty."

    if ";" in text[:-1]:
        return None, "Only a single SQL statement is allowed."
    if text.endswith(";"):
        text = text[:-1].rstrip()
    if not text:
        return None, "SQL cannot be empty."

    first_token = text.split(None, 1)[0].lower()
    if first_token not in _SQL_READ_PREFIXES:
        return None, "Only read SQL is allowed (SELECT/WITH)."

    if _SQL_FORBIDDEN_CLAUSE.search(text):
        return None, "Write SQL clauses are not allowed."

    return text, None


def _has_top_level_limit(sql_text: str) -> bool:
    depth = 0
    idx = 0
    n = len(sql_text)

    while idx < n:
        ch = sql_text[idx]
        nxt = sql_text[idx + 1] if idx + 1 < n else ""

        if ch == "-" and nxt == "-":
            idx += 2
            while idx < n and sql_text[idx] not in "\r\n":
                idx += 1
            continue

        if ch == "/" and nxt == "*":
            idx += 2
            while idx + 1 < n and not (sql_text[idx] == "*" and sql_text[idx + 1] == "/"):
                idx += 1
            idx += 2
            continue

        if ch == "'":
            idx += 1
            while idx < n:
                if sql_text[idx] == "'":
                    if idx + 1 < n and sql_text[idx + 1] == "'":
                        idx += 2
                        continue
                    idx += 1
                    break
                idx += 1
            continue

        if ch == '"':
            idx += 1
            while idx < n:
                if sql_text[idx] == '"':
                    if idx + 1 < n and sql_text[idx + 1] == '"':
                        idx += 2
                        continue
                    idx += 1
                    break
                idx += 1
            continue

        if ch == "[":
            idx += 1
            while idx < n and sql_text[idx] != "]":
                idx += 1
            idx += 1
            continue

        if ch == "(":
            depth += 1
            idx += 1
            continue

        if ch == ")":
            depth = max(0, depth - 1)
            idx += 1
            continue

        if depth == 0 and (ch.isalpha() or ch == "_"):
            start = idx
            idx += 1
            while idx < n and (sql_text[idx].isalnum() or sql_text[idx] == "_"):
                idx += 1
            if sql_text[start:idx].casefold() == "limit":
                return True
            continue

        idx += 1

    return False


def _apply_default_limit(sql_text: str) -> str:
    if _has_top_level_limit(sql_text):
        return sql_text
    return f"{sql_text}\nLIMIT {_DEFAULT_SQL_LIMIT}"


def _parse_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = re.split(r"[\n,]", str(value))

    out: List[str] = []
    seen = set()
    for item in raw_items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _truncate_text(value: str, limit: int = _TRUNCATED_TEXT_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _sql_json_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _authorizer_read_only(
    action: int,
    arg1: Optional[str],
    _arg2: Optional[str],
    _db_name: Optional[str],
    _trigger_or_view: Optional[str],
) -> int:
    if action in _SQL_READ_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == getattr(sqlite3, "SQLITE_PRAGMA", -1):
        pragma_name = str(arg1 or "").casefold()
        if pragma_name in _ALLOWED_INTERNAL_PRAGMAS:
            return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _format_row_value(value: Any, *, truncate_text: bool) -> Any:
    value = _sql_json_value(value)
    if truncate_text and isinstance(value, str):
        return _truncate_text(value)
    return value


def _run_read_sql(db_path: Path, sql_text: str, *, truncate_text: bool = True) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_authorizer_read_only)
        cur = conn.execute(sql_text)
        if cur.description is None:
            return {"columns": [], "rows": [], "truncated": False}
        columns = [str(col[0]) for col in cur.description]
        fetched_rows = cur.fetchmany(_MAX_SQL_ROWS + 1)
        rows = fetched_rows[:_MAX_SQL_ROWS]
        truncate_text = truncate_text and len(rows) > _FULL_TEXT_SQL_ROW_THRESHOLD

        row_dicts: List[Dict[str, Any]] = []
        for row in rows:
            row_dicts.append(
                {
                    col: _format_row_value(row[col], truncate_text=truncate_text)
                    for col in columns
                }
            )
        return {
            "columns": columns,
            "rows": row_dicts,
            "truncated": len(fetched_rows) > _MAX_SQL_ROWS,
        }
    finally:
        conn.close()


def _fetch_full_messages(db_path: Path, message_ids: List[str]) -> Dict[str, Any]:
    ordered_ids = message_ids[:_MAX_MESSAGE_IDS]
    if not ordered_ids:
        return {
            "columns": ["message_id", "created_at", "channel_id", "author_name", "content"],
            "rows": [],
        }

    placeholders = ",".join("?" for _ in ordered_ids)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_authorizer_read_only)
        cur = conn.execute(
            f"""
            SELECT message_id, created_at, channel_id, author_name, content
            FROM messages
            WHERE message_id IN ({placeholders})
            """,
            ordered_ids,
        )
        by_id = {str(row["message_id"]): row for row in cur.fetchall()}
        rows: List[Dict[str, Any]] = []
        for message_id in ordered_ids:
            row = by_id.get(message_id)
            if row is None:
                continue
            rows.append(
                {
                    "message_id": _sql_json_value(row["message_id"]),
                    "created_at": _sql_json_value(row["created_at"]),
                    "channel_id": _sql_json_value(row["channel_id"]),
                    "author_name": _sql_json_value(row["author_name"]),
                    "content": _sql_json_value(row["content"]),
                }
            )
        return {
            "columns": ["message_id", "created_at", "channel_id", "author_name", "content"],
            "rows": rows,
        }
    finally:
        conn.close()


def _rows_to_csv(columns: List[str], rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row.get(col, "") for col in columns])
    return buf.getvalue().rstrip("\n")


def discord_search(sql: str = "", message_ids: Optional[List[str]] = None) -> str:
    """Search the Discord archive or fetch full messages by ID."""
    raw_sql = (sql or "").strip()
    normalized_ids = _parse_str_list(message_ids)
    if raw_sql and normalized_ids:
        return json.dumps(
            {
                "success": False,
                "error": "Provide either 'sql' or 'message_ids', not both.",
            },
            ensure_ascii=False,
        )

    if not raw_sql:
        if not normalized_ids:
            return json.dumps(
                {
                    "success": False,
                    "error": "Provide either 'sql' or 'message_ids'.",
                },
                ensure_ascii=False,
            )

    db_path = _archive_db_path()
    if not db_path.exists():
        return json.dumps(
            {
                "success": False,
                "error": f"Discord archive DB not found: {db_path}",
            },
            ensure_ascii=False,
        )

    if normalized_ids:
        try:
            result = _fetch_full_messages(db_path, normalized_ids)
            return _rows_to_csv(columns=result["columns"], rows=result["rows"])
        except Exception as e:
            return json.dumps(
                {"success": False, "error": f"Discord message fetch failed: {e}"},
                ensure_ascii=False,
            )

    sql_text, sql_error = _normalize_read_sql(raw_sql)
    if sql_error:
        return json.dumps({"success": False, "error": sql_error}, ensure_ascii=False)

    sql_text = _apply_default_limit(sql_text or "")
    try:
        sql_result = _run_read_sql(db_path, sql_text, truncate_text=True)
        return _rows_to_csv(columns=sql_result["columns"], rows=sql_result["rows"])
    except Exception as e:
        return json.dumps({"success": False, "error": f"Discord SQL read failed: {e}"}, ensure_ascii=False)


DISCORD_SEARCH_SCHEMA = {
    "name": "discord_search",
    "description": (
        "Query the local Discord archive in two ways. "
        "1) Search/aggregate with a single read-only SQL SELECT/WITH statement via `sql`. "
        "2) Fetch full untruncated messages for specific IDs via `message_ids`. "
        "The tool always returns CSV with headers.\n\n"
        f"When using `sql`, text cells are returned in full when the result has {_FULL_TEXT_SQL_ROW_THRESHOLD} rows or fewer. "
        f"For larger result sets, text cells longer than {_TRUNCATED_TEXT_LIMIT} characters are truncated with `...` "
        "to keep results compact. If you need the full body of specific hits from larger result sets, call the tool again with "
        "`message_ids` using the message IDs from the search results. "
        f"If your SQL has no top-level LIMIT, the tool automatically adds LIMIT {_DEFAULT_SQL_LIMIT}. "
        f"Regardless of the query text, at most {_MAX_SQL_ROWS} rows are returned. "
        f"`message_ids` fetches at most {_MAX_MESSAGE_IDS} messages and preserves the requested order.\n\n"
        "Useful tables:\n"
        "- `messages`: archived Discord messages.\n"
        "- `message_changes`: edit/delete history.\n"
        "- `discord_messages_fts`: FTS5 virtual table for content search; join it to `messages` on `rowid`.\n\n"
        "Search examples:\n"
        "Recent messages in one channel:\n"
        "SELECT message_id, created_at, author_name, content\n"
        "FROM messages\n"
        "WHERE channel_id = '123' AND deleted = 0\n"
        "ORDER BY created_at DESC\n"
        "LIMIT 50\n\n"
        "Recent edits/deletes in one channel:\n"
        "SELECT message_id, change_type, changed_at, author_name, before_content, after_content\n"
        "FROM message_changes\n"
        "WHERE channel_id = '123'\n"
        "ORDER BY changed_at DESC\n"
        "LIMIT 50\n\n"
        "Full-text search with FTS:\n"
        "SELECT m.message_id, m.created_at, m.channel_id, m.author_name,\n"
        "       snippet(discord_messages_fts, 0, '>>>', '<<<', '...', 28) AS snippet\n"
        "FROM discord_messages_fts\n"
        "JOIN messages m ON m.rowid = discord_messages_fts.rowid\n"
        "WHERE discord_messages_fts MATCH 'deployment'\n"
        "  AND m.deleted = 0\n"
        "ORDER BY rank\n"
        "LIMIT 25\n\n"
        "Full-message expansion example:\n"
        "{\n"
        '  "message_ids": ["1234567890", "1234567891"]\n'
        "}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "Single read-only SELECT/WITH query. Use this for search, filtering, aggregation, or FTS. "
                    f"If no top-level LIMIT is present, the tool adds LIMIT {_DEFAULT_SQL_LIMIT}. "
                    f"Text cells are returned in full when the result has {_FULL_TEXT_SQL_ROW_THRESHOLD} rows or fewer; "
                    f"otherwise long text cells are truncated to {_TRUNCATED_TEXT_LIMIT} characters with `...`."
                ),
            },
            "message_ids": {
                "type": ["array", "string"],
                "items": {"type": "string"},
                "description": (
                    "Optional list of message IDs to fetch in full, without truncation. "
                    "Use this after a search when you want the exact full content of specific hits. "
                    f"Maximum {_MAX_MESSAGE_IDS} IDs."
                ),
            },
        },
        "required": [],
    },
}


registry.register(
    name="discord_search",
    toolset="discord_search",
    schema=DISCORD_SEARCH_SCHEMA,
    handler=lambda args, **kw: discord_search(
        sql=args.get("sql", ""),
        message_ids=args.get("message_ids"),
    ),
    check_fn=check_discord_search_requirements,
)
