#!/usr/bin/env python3
"""Discord archive search tool (SQLite FTS5)."""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.discord_archive import DiscordArchiveDB, default_archive_db_path
from tools.registry import registry


def _parse_time_range(value: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse time range string.

    Supported formats:
    - "7d" / "30d" / "24h" / "90m"
    - "YYYY-MM-DD..YYYY-MM-DD"
    - "<unix_ts>..<unix_ts>"
    """
    text = (value or "").strip()
    if not text:
        return None, None

    now = datetime.now()

    def _to_ts(raw: str) -> Optional[float]:
        raw = raw.strip()
        if not raw:
            return None
        if raw.replace(".", "", 1).isdigit():
            return float(raw)
        try:
            return datetime.fromisoformat(raw).timestamp()
        except Exception:
            pass
        try:
            return datetime.strptime(raw, "%Y-%m-%d").timestamp()
        except Exception:
            return None

    if text.endswith("d") and text[:-1].isdigit():
        days = int(text[:-1])
        return (now - timedelta(days=days)).timestamp(), now.timestamp()
    if text.endswith("h") and text[:-1].isdigit():
        hours = int(text[:-1])
        return (now - timedelta(hours=hours)).timestamp(), now.timestamp()
    if text.endswith("m") and text[:-1].isdigit():
        minutes = int(text[:-1])
        return (now - timedelta(minutes=minutes)).timestamp(), now.timestamp()

    if ".." in text:
        left, right = text.split("..", 1)
        return _to_ts(left), _to_ts(right)

    # Unknown format: ignore instead of erroring.
    return None, None


def _fmt_ts(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")
    except Exception:
        return None


def _archive_db_path() -> Path:
    path = (os.getenv("DISCORD_ARCHIVE_DB_PATH", "") or "").strip()
    if path:
        return Path(path).expanduser()
    return default_archive_db_path()


def check_discord_search_requirements() -> bool:
    """Enable only when the archive DB file exists."""
    return _archive_db_path().exists()


_SQL_READ_PREFIXES = ("select", "with", "explain")
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


def _parse_sql_limit(value: Any, default: int = 200) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(1, min(parsed, 1000))


def _parse_bounded_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(min_value, min(parsed, max_value))


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return default


def _parse_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    parts: List[str] = []
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = re.split(r"[\n,]", str(value))
    for item in raw_items:
        text = str(item or "").strip()
        if text:
            parts.append(text)
    return parts


def _dedupe(values: List[str], lower: bool = False) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        key = value.casefold() if lower else value
        if key in seen:
            continue
        seen.add(key)
        out.append(key if lower else value)
    return out


def _normalize_user_filters(
    author_id: Any,
    author_ids: Any,
    author_name: Any,
    author_names: Any,
) -> Tuple[List[str], List[str]]:
    id_list = _parse_str_list(author_ids)
    single_id = str(author_id or "").strip()
    if single_id:
        id_list.append(single_id)

    name_list = _parse_str_list(author_names)
    single_name = str(author_name or "").strip()
    if single_name:
        name_list.append(single_name)

    normalized_names = []
    for name in name_list:
        trimmed = name.strip()
        if trimmed.startswith("@"):
            trimmed = trimmed[1:].strip()
        if trimmed:
            normalized_names.append(trimmed.casefold())

    return _dedupe(id_list, lower=False), _dedupe(normalized_names, lower=True)


def _message_matches_user(msg: Dict[str, Any], author_ids: List[str], author_names_cf: List[str]) -> bool:
    if not author_ids and not author_names_cf:
        return True

    if author_ids:
        current_id = str(msg.get("author_id") or "").strip()
        if current_id and current_id in author_ids:
            return True

    if author_names_cf:
        for key in ("author_name", "author_display"):
            raw = str(msg.get(key) or "").strip()
            if raw and raw.casefold() in author_names_cf:
                return True

    return False


def _parse_optional_ts(value: Any, label: str) -> Tuple[Optional[float], Optional[str]]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    try:
        return float(text), None
    except Exception:
        return None, f"{label} must be a unix timestamp (seconds)."


def _normalize_read_sql(raw_sql: str) -> Tuple[Optional[str], Optional[str]]:
    text = (raw_sql or "").strip()
    if not text:
        return None, "SQL cannot be empty."

    # sqlite3 cursor.execute() accepts only one statement; enforce that explicitly.
    if ";" in text[:-1]:
        return None, "Only a single SQL statement is allowed."
    if text.endswith(";"):
        text = text[:-1].rstrip()
    if not text:
        return None, "SQL cannot be empty."

    first_token = text.split(None, 1)[0].lower()
    if first_token not in _SQL_READ_PREFIXES:
        return None, "Only read SQL is allowed (SELECT/WITH/EXPLAIN)."

    # Block known write primitives even inside WITH statements.
    if _SQL_FORBIDDEN_CLAUSE.search(text):
        return None, "Write SQL clauses are not allowed."

    return text, None


def _sql_json_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _authorizer_read_only(
    action: int,
    _arg1: Optional[str],
    _arg2: Optional[str],
    _db_name: Optional[str],
    _trigger_or_view: Optional[str],
) -> int:
    if action in _SQL_READ_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _run_read_sql(db_path: Path, sql_text: str, sql_limit: int) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_authorizer_read_only)
        cur = conn.execute(sql_text)
        if cur.description is None:
            return {"columns": [], "rows": [], "truncated": False}
        columns = [str(col[0]) for col in cur.description]
        fetched_rows = cur.fetchmany(sql_limit + 1)
        truncated = len(fetched_rows) > sql_limit
        rows = fetched_rows[:sql_limit]

        row_dicts: List[Dict[str, Any]] = []
        for row in rows:
            row_dicts.append({col: _sql_json_value(row[col]) for col in columns})
        return {"columns": columns, "rows": row_dicts, "truncated": truncated}
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {str(k): row[k] for k in row.keys()}


def _compact_message_row(msg: Dict[str, Any], kind: str, hit: int = 1) -> Dict[str, Any]:
    return {
        "kind": kind,
        "hit": int(hit),
        "message_id": msg.get("message_id"),
        "created_at": _fmt_ts(msg.get("created_at")),
        "channel_id": msg.get("channel_id"),
        "author": msg.get("author_name") or msg.get("author_id") or "",
        "content": msg.get("content") or "",
    }


def _rows_to_csv(columns: List[str], rows: List[Dict[str, Any]], include_headers: bool = True) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    if include_headers:
        writer.writerow(columns)
    for row in rows:
        writer.writerow([row.get(col, "") for col in columns])
    return buf.getvalue().rstrip("\n")


def _flatten_fts_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        for msg in row.get("context_before", []) or []:
            flat.append(_compact_message_row(msg, kind="before", hit=idx))
        hit = row.get("hit") or {}
        flat.append(_compact_message_row(hit, kind="hit", hit=idx))
        for msg in row.get("context_after", []) or []:
            flat.append(_compact_message_row(msg, kind="after", hit=idx))
    return flat


def _fetch_anchor_window(
    db_path: Path,
    anchor_message_id: str,
    before: int,
    after: int,
    include_bots: bool,
    author_ids: List[str],
    author_names_cf: List[str],
) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        anchor = conn.execute(
            """
            SELECT *
            FROM messages
            WHERE message_id = ? AND deleted = 0
            LIMIT 1
            """,
            (anchor_message_id,),
        ).fetchone()
        if anchor is None:
            return {"success": False, "error": f"Anchor message not found: {anchor_message_id}"}

        anchor_dict = _row_to_dict(anchor)
        channel_id = str(anchor_dict.get("channel_id") or "")
        created_at = float(anchor_dict.get("created_at") or 0.0)

        bot_clause = "" if include_bots else "AND author_is_bot = 0"

        before_rows = conn.execute(
            f"""
            SELECT *
            FROM messages
            WHERE channel_id = ?
              AND deleted = 0
              {bot_clause}
              AND created_at < ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (channel_id, created_at, int(before)),
        ).fetchall()
        after_rows = conn.execute(
            f"""
            SELECT *
            FROM messages
            WHERE channel_id = ?
              AND deleted = 0
              {bot_clause}
              AND created_at > ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (channel_id, created_at, int(after)),
        ).fetchall()

        before_dicts = [_row_to_dict(r) for r in before_rows]
        before_dicts.reverse()
        after_dicts = [_row_to_dict(r) for r in after_rows]

        flat_rows: List[Dict[str, Any]] = []
        for msg in before_dicts:
            if _message_matches_user(msg, author_ids, author_names_cf):
                flat_rows.append(_compact_message_row(msg, kind="before", hit=1))
        flat_rows.append(_compact_message_row(anchor_dict, kind="anchor", hit=1))
        for msg in after_dicts:
            if _message_matches_user(msg, author_ids, author_names_cf):
                flat_rows.append(_compact_message_row(msg, kind="after", hit=1))

        return {
            "success": True,
            "channel_id": channel_id,
            "anchor_message_id": anchor_message_id,
            "count": len(flat_rows),
            "rows": flat_rows,
        }
    finally:
        conn.close()


def discord_search(
    query: str = "",
    guild_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    author_id: Optional[str] = None,
    author_ids: Optional[List[str]] = None,
    author_name: Optional[str] = None,
    author_names: Optional[List[str]] = None,
    time_range: Optional[str] = None,
    since_ts: Optional[float] = None,
    until_ts: Optional[float] = None,
    k: int = 5,
    around: int = 1,
    anchor_message_id: Optional[str] = None,
    before: int = 20,
    after: int = 20,
    include_bots: bool = False,
    include_headers: bool = True,
    sql: Optional[str] = None,
    sql_limit: int = 200,
) -> str:
    """Search local Discord archive via FTS5 or run read-only SQL."""
    query = (query or "").strip()
    raw_sql = (sql or "").strip()
    anchor_message_id = str(anchor_message_id or "").strip()
    include_headers = _parse_bool(include_headers, default=True)
    include_bots = _parse_bool(include_bots, default=False)
    author_ids, author_names_cf = _normalize_user_filters(
        author_id=author_id,
        author_ids=author_ids,
        author_name=author_name,
        author_names=author_names,
    )

    if not query and not raw_sql and not anchor_message_id:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "Provide one of: 'query' (FTS), 'sql' (raw read SQL), "
                    "or 'anchor_message_id' (+ before/after)."
                ),
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

    if raw_sql:
        sql_text, sql_error = _normalize_read_sql(raw_sql)
        if sql_error:
            return json.dumps({"success": False, "error": sql_error}, ensure_ascii=False)
        sql_limit = _parse_sql_limit(sql_limit, default=200)
        try:
            sql_result = _run_read_sql(db_path, sql_text or "", sql_limit)
            return _rows_to_csv(
                columns=sql_result["columns"],
                rows=sql_result["rows"],
                include_headers=include_headers,
            )
        except Exception as e:
            return json.dumps({"success": False, "error": f"Discord SQL read failed: {e}"}, ensure_ascii=False)

    if anchor_message_id:
        before_n = _parse_bounded_int(before, default=20, min_value=0, max_value=200)
        after_n = _parse_bounded_int(after, default=20, min_value=0, max_value=200)
        window = _fetch_anchor_window(
            db_path=db_path,
            anchor_message_id=anchor_message_id,
            before=before_n,
            after=after_n,
            include_bots=include_bots,
            author_ids=author_ids,
            author_names_cf=author_names_cf,
        )
        if not window.get("success"):
            return json.dumps({"success": False, "error": window.get("error", "Anchor lookup failed.")}, ensure_ascii=False)
        return _rows_to_csv(
            columns=["kind", "hit", "message_id", "created_at", "channel_id", "author", "content"],
            rows=window.get("rows", []),
            include_headers=include_headers,
        )

    k = max(1, min(int(k), 25))
    around = max(0, min(int(around), 8))
    parsed_since_ts, parsed_until_ts = _parse_time_range(time_range)
    if since_ts is not None:
        parsed_since_ts, err = _parse_optional_ts(since_ts, "since_ts")
        if err:
            return json.dumps({"success": False, "error": err}, ensure_ascii=False)
    if until_ts is not None:
        parsed_until_ts, err = _parse_optional_ts(until_ts, "until_ts")
        if err:
            return json.dumps({"success": False, "error": err}, ensure_ascii=False)
    if (
        parsed_since_ts is not None
        and parsed_until_ts is not None
        and parsed_since_ts > parsed_until_ts
    ):
        return json.dumps({"success": False, "error": "since_ts cannot be greater than until_ts."}, ensure_ascii=False)

    db = DiscordArchiveDB(db_path)
    try:
        fetch_limit = k if (not author_ids and not author_names_cf) else min(200, max(k, k * 8))
        raw_rows = db.search_messages(
            query=query,
            guild_id=(guild_id or None),
            channel_id=(channel_id or None),
            since_ts=parsed_since_ts,
            until_ts=parsed_until_ts,
            limit=fetch_limit,
            around=around,
        )
        rows: List[Dict[str, Any]] = []
        for row in raw_rows:
            hit = row.get("hit") or {}
            if not include_bots and bool(hit.get("author_is_bot")):
                continue
            if not _message_matches_user(hit, author_ids, author_names_cf):
                continue
            context_before = row.get("context_before") or []
            context_after = row.get("context_after") or []
            if not include_bots:
                context_before = [m for m in context_before if not bool(m.get("author_is_bot"))]
                context_after = [m for m in context_after if not bool(m.get("author_is_bot"))]
            if author_ids or author_names_cf:
                context_before = [m for m in context_before if _message_matches_user(m, author_ids, author_names_cf)]
                context_after = [m for m in context_after if _message_matches_user(m, author_ids, author_names_cf)]
            rows.append(
                {
                    "hit": hit,
                    "context_before": context_before,
                    "context_after": context_after,
                }
            )
            if len(rows) >= k:
                break

        compact_rows = _flatten_fts_rows(rows)
        return _rows_to_csv(
            columns=["kind", "hit", "message_id", "created_at", "channel_id", "author", "content"],
            rows=compact_rows,
            include_headers=include_headers,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": f"Discord search failed: {e}"}, ensure_ascii=False)
    finally:
        db.close()


DISCORD_SEARCH_SCHEMA = {
    "name": "discord_search",
    "description": (
        "Search local Discord channel history stored in SQLite FTS5, or run raw read-only SQL "
        "against the same archive DB. Use `query` for FTS recall, `sql` for direct inspection, "
        "or `anchor_message_id` + before/after for direct window retrieval. Use this tool to "
        "personalize responses or gather more context about the conversation at hand."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "FTS query string (keywords, phrase, boolean). Optional when `sql` is provided.",
            },
            "sql": {
                "type": "string",
                "description": (
                    "Optional raw SQL (single statement, read-only). Allowed classes: SELECT/WITH/EXPLAIN. "
                    "Write SQL is rejected."
                ),
            },
            "sql_limit": {
                "type": "integer",
                "description": "Max rows returned for raw SQL mode (default 200, max 1000).",
                "default": 200,
            },
            "guild_id": {
                "type": "string",
                "description": "Optional Discord guild ID filter.",
            },
            "channel_id": {
                "type": "string",
                "description": "Optional Discord channel ID filter.",
            },
            "author_id": {
                "type": "string",
                "description": "Optional exact Discord author ID filter.",
            },
            "author_ids": {
                "type": ["array", "string"],
                "items": {"type": "string"},
                "description": "Optional author ID list filter (array or comma-separated string).",
            },
            "author_name": {
                "type": "string",
                "description": "Optional exact author username/display filter (case-insensitive).",
            },
            "author_names": {
                "type": ["array", "string"],
                "items": {"type": "string"},
                "description": "Optional author name list filter (array or comma-separated string).",
            },
            "time_range": {
                "type": "string",
                "description": (
                    "Optional time filter: e.g. '7d', '24h', or '2026-01-01..2026-01-31'."
                ),
            },
            "since_ts": {
                "type": "number",
                "description": "Optional lower bound unix timestamp (seconds). Overrides time_range start.",
            },
            "until_ts": {
                "type": "number",
                "description": "Optional upper bound unix timestamp (seconds). Overrides time_range end.",
            },
            "k": {
                "type": "integer",
                "description": "Number of top hits to return (default 5, max 25).",
                "default": 5,
            },
            "around": {
                "type": "integer",
                "description": "Neighbor messages before/after each hit (default 1, max 8).",
                "default": 1,
            },
            "anchor_message_id": {
                "type": "string",
                "description": "Optional anchor message ID for direct window retrieval.",
            },
            "before": {
                "type": "integer",
                "description": "When anchor_message_id is set: number of messages before anchor (max 200).",
                "default": 20,
            },
            "after": {
                "type": "integer",
                "description": "When anchor_message_id is set: number of messages after anchor (max 200).",
                "default": 20,
            },
            "include_bots": {
                "type": "boolean",
                "description": "Include bot-authored messages (default false).",
                "default": False,
            },
            "include_headers": {
                "type": "boolean",
                "description": "Include CSV header row (default true).",
                "default": True,
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
        query=args.get("query", ""),
        since_ts=args.get("since_ts"),
        until_ts=args.get("until_ts"),
        author_id=args.get("author_id"),
        author_ids=args.get("author_ids"),
        author_name=args.get("author_name"),
        author_names=args.get("author_names"),
        anchor_message_id=args.get("anchor_message_id"),
        before=args.get("before", 20),
        after=args.get("after", 20),
        include_bots=args.get("include_bots", False),
        include_headers=args.get("include_headers", True),
        sql=args.get("sql"),
        sql_limit=args.get("sql_limit", 200),
        guild_id=args.get("guild_id"),
        channel_id=args.get("channel_id"),
        time_range=args.get("time_range"),
        k=args.get("k", 5),
        around=args.get("around", 1),
    ),
    check_fn=check_discord_search_requirements,
)
