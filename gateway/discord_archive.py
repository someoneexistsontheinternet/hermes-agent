"""
Discord archive storage (SQLite + FTS5).

Stores Discord channel messages for local search/retrieval and keeps enough
metadata for channel-scoped filtering.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def default_archive_db_path() -> Path:
    """Default archive DB location under Hermes home."""
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return hermes_home / "discord_data.sqlite"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    guild_id TEXT,
    guild_name TEXT,
    channel_id TEXT NOT NULL,
    channel_name TEXT,
    thread_id TEXT,
    author_id TEXT,
    author_name TEXT,
    author_display TEXT,
    author_is_bot INTEGER NOT NULL DEFAULT 0,
    content TEXT,
    attachments_json TEXT,
    created_at REAL NOT NULL,
    edited_at REAL,
    deleted INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS message_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,
    guild_id TEXT,
    channel_id TEXT NOT NULL,
    author_id TEXT,
    author_name TEXT,
    author_display TEXT,
    author_is_bot INTEGER NOT NULL DEFAULT 0,
    original_created_at REAL,
    changed_at REAL NOT NULL,
    change_type TEXT NOT NULL,
    before_content TEXT,
    after_content TEXT
);

CREATE TABLE IF NOT EXISTS channel_state (
    channel_id TEXT PRIMARY KEY,
    last_message_id TEXT,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS channel_turn_state (
    channel_id TEXT PRIMARY KEY,
    last_context_message_id TEXT,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS channel_backfill_state (
    channel_id TEXT PRIMARY KEY,
    oldest_message_id TEXT,
    oldest_created_at REAL,
    complete INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_channel_created
    ON messages(channel_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_guild_channel_created
    ON messages(guild_id, channel_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_author
    ON messages(author_id);
CREATE INDEX IF NOT EXISTS idx_message_changes_channel_changed
    ON message_changes(channel_id, changed_at);
CREATE INDEX IF NOT EXISTS idx_message_changes_message
    ON message_changes(message_id, changed_at);
CREATE INDEX IF NOT EXISTS idx_channel_backfill_complete
    ON channel_backfill_state(complete, updated_at);
"""


FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS discord_messages_fts USING fts5(
    content,
    content=messages,
    content_rowid=rowid
);

CREATE TRIGGER IF NOT EXISTS discord_messages_fts_insert
AFTER INSERT ON messages BEGIN
    INSERT INTO discord_messages_fts(rowid, content)
    VALUES (new.rowid, COALESCE(new.content, ''));
END;

CREATE TRIGGER IF NOT EXISTS discord_messages_fts_delete
AFTER DELETE ON messages BEGIN
    INSERT INTO discord_messages_fts(discord_messages_fts, rowid, content)
    VALUES ('delete', old.rowid, COALESCE(old.content, ''));
END;

CREATE TRIGGER IF NOT EXISTS discord_messages_fts_update
AFTER UPDATE ON messages BEGIN
    INSERT INTO discord_messages_fts(discord_messages_fts, rowid, content)
    VALUES ('delete', old.rowid, COALESCE(old.content, ''));
    INSERT INTO discord_messages_fts(rowid, content)
    VALUES (new.rowid, COALESCE(new.content, ''));
END;
"""


class DiscordArchiveDB:
    """SQLite-backed Discord archive with FTS5 search."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript(SCHEMA_SQL)
        try:
            cursor.execute("SELECT * FROM discord_messages_fts LIMIT 0")
        except sqlite3.OperationalError:
            cursor.executescript(FTS_SQL)
        self._conn.commit()

    def get_channel_cursor(self, channel_id: str) -> Optional[str]:
        cursor = self._conn.execute(
            "SELECT last_message_id FROM channel_state WHERE channel_id = ?",
            (str(channel_id),),
        )
        row = cursor.fetchone()
        return row["last_message_id"] if row else None

    def get_turn_anchor(self, channel_id: str) -> Optional[str]:
        cursor = self._conn.execute(
            "SELECT last_context_message_id FROM channel_turn_state WHERE channel_id = ?",
            (str(channel_id),),
        )
        row = cursor.fetchone()
        return row["last_context_message_id"] if row else None

    def set_turn_anchor(self, channel_id: str, message_id: Optional[str]) -> None:
        now = time.time()
        channel_id = str(channel_id)
        anchor = str(message_id) if message_id else None

        # Keep anchor monotonic for numeric Discord IDs so concurrent/out-of-order
        # handlers cannot move the channel delta cursor backwards.
        if anchor and anchor.isdigit():
            row = self._conn.execute(
                "SELECT last_context_message_id FROM channel_turn_state WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
            current = str(row["last_context_message_id"]) if row and row["last_context_message_id"] else None
            if current and current.isdigit() and int(current) > int(anchor):
                anchor = current

        self._conn.execute(
            """
            INSERT INTO channel_turn_state(channel_id, last_context_message_id, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                last_context_message_id = excluded.last_context_message_id,
                updated_at = excluded.updated_at
            """,
            (channel_id, anchor, now),
        )
        self._conn.commit()

    def clear_turn_anchor(self, channel_id: str) -> None:
        """Clear the turn anchor for a channel (used by /new and /reset)."""
        self._conn.execute(
            "DELETE FROM channel_turn_state WHERE channel_id = ?",
            (str(channel_id),),
        )
        self._conn.commit()

    def get_oldest_message(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Return the oldest archived message metadata for one channel."""
        row = self._conn.execute(
            """
            SELECT message_id, created_at
            FROM messages
            WHERE channel_id = ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (str(channel_id),),
        ).fetchone()
        if not row:
            return None
        return {
            "message_id": row["message_id"],
            "created_at": row["created_at"],
        }

    def get_backfill_state(self, channel_id: str) -> Dict[str, Any]:
        """Return persisted backward-fill state for one channel."""
        row = self._conn.execute(
            """
            SELECT oldest_message_id, oldest_created_at, complete, updated_at
            FROM channel_backfill_state
            WHERE channel_id = ?
            """,
            (str(channel_id),),
        ).fetchone()
        if not row:
            return {
                "channel_id": str(channel_id),
                "oldest_message_id": None,
                "oldest_created_at": None,
                "complete": False,
                "updated_at": None,
            }
        return {
            "channel_id": str(channel_id),
            "oldest_message_id": row["oldest_message_id"],
            "oldest_created_at": row["oldest_created_at"],
            "complete": bool(row["complete"]),
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _is_older_id(candidate: str, current: str) -> bool:
        """Best-effort Discord snowflake ordering: lower numeric IDs are older."""
        if not candidate:
            return False
        if not current:
            return True
        if candidate.isdigit() and current.isdigit():
            return int(candidate) < int(current)
        return candidate < current

    def upsert_backfill_state(
        self,
        channel_id: str,
        *,
        oldest_message_id: Optional[str] = None,
        oldest_created_at: Optional[float] = None,
        complete: Optional[bool] = None,
    ) -> None:
        """
        Upsert backward-fill cursor for a channel.

        - oldest_message_id is kept monotonic (only moves to older IDs).
        - complete can be toggled explicitly when provided.
        """
        now = time.time()
        channel_id = str(channel_id)
        if not channel_id:
            return

        current = self._conn.execute(
            """
            SELECT oldest_message_id, oldest_created_at, complete
            FROM channel_backfill_state
            WHERE channel_id = ?
            """,
            (channel_id,),
        ).fetchone()

        current_oldest = str(current["oldest_message_id"]) if current and current["oldest_message_id"] else ""
        current_created = (
            float(current["oldest_created_at"])
            if current and current["oldest_created_at"] is not None
            else None
        )
        current_complete = bool(current["complete"]) if current else False

        candidate_oldest = str(oldest_message_id or "").strip()
        next_oldest = current_oldest or None
        next_created = current_created

        if candidate_oldest and self._is_older_id(candidate_oldest, current_oldest):
            next_oldest = candidate_oldest
            next_created = float(oldest_created_at) if oldest_created_at is not None else None
        elif not current_oldest and candidate_oldest:
            next_oldest = candidate_oldest
            next_created = float(oldest_created_at) if oldest_created_at is not None else None

        if complete is None:
            next_complete = current_complete
        else:
            next_complete = bool(complete)

        self._conn.execute(
            """
            INSERT INTO channel_backfill_state(
                channel_id, oldest_message_id, oldest_created_at, complete, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                oldest_message_id = excluded.oldest_message_id,
                oldest_created_at = excluded.oldest_created_at,
                complete = excluded.complete,
                updated_at = excluded.updated_at
            """,
            (
                channel_id,
                next_oldest,
                next_created,
                1 if next_complete else 0,
                now,
            ),
        )
        self._conn.commit()

    def mark_backfill_complete(self, channel_id: str, complete: bool = True) -> None:
        """Mark backward-fill status for one channel."""
        self.upsert_backfill_state(channel_id, complete=complete)

    def _update_channel_cursor(self, channel_id: str, message_id: Optional[str]) -> None:
        if not channel_id:
            return
        now = time.time()
        channel_id = str(channel_id)
        message_id = str(message_id) if message_id is not None else None

        if message_id and message_id.isdigit():
            self._conn.execute(
                """
                INSERT INTO channel_state(channel_id, last_message_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    last_message_id = CASE
                        WHEN channel_state.last_message_id IS NULL THEN excluded.last_message_id
                        WHEN CAST(excluded.last_message_id AS INTEGER) > CAST(channel_state.last_message_id AS INTEGER)
                            THEN excluded.last_message_id
                        ELSE channel_state.last_message_id
                    END,
                    updated_at = excluded.updated_at
                """,
                (channel_id, message_id, now),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO channel_state(channel_id, last_message_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    last_message_id = excluded.last_message_id,
                    updated_at = excluded.updated_at
                """,
                (channel_id, message_id, now),
            )

    def upsert_message(self, msg: Dict[str, Any]) -> None:
        """
        Upsert one message row.

        Expected keys include: message_id, channel_id, content, created_at.
        """
        now = time.time()
        message_id = str(msg.get("message_id", "")).strip()
        if not message_id:
            return

        channel_id = str(msg.get("channel_id", "")).strip()
        if not channel_id:
            return

        created_at = float(msg.get("created_at", now))
        edited_at = msg.get("edited_at")
        edited_at = float(edited_at) if edited_at is not None else None
        deleted = 1 if msg.get("deleted") else 0

        attachments = msg.get("attachments_json")
        if attachments is not None and not isinstance(attachments, str):
            attachments = json.dumps(attachments, ensure_ascii=False)

        self._conn.execute(
            """
            INSERT INTO messages(
                message_id, guild_id, guild_name, channel_id, channel_name, thread_id,
                author_id, author_name, author_display, author_is_bot,
                content, attachments_json, created_at, edited_at, deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                guild_id = excluded.guild_id,
                guild_name = excluded.guild_name,
                channel_id = excluded.channel_id,
                channel_name = excluded.channel_name,
                thread_id = excluded.thread_id,
                author_id = excluded.author_id,
                author_name = excluded.author_name,
                author_display = excluded.author_display,
                author_is_bot = excluded.author_is_bot,
                content = excluded.content,
                attachments_json = excluded.attachments_json,
                edited_at = excluded.edited_at,
                deleted = excluded.deleted
            """,
            (
                message_id,
                msg.get("guild_id"),
                msg.get("guild_name"),
                channel_id,
                msg.get("channel_name"),
                msg.get("thread_id"),
                msg.get("author_id"),
                msg.get("author_name"),
                msg.get("author_display"),
                1 if msg.get("author_is_bot") else 0,
                msg.get("content"),
                attachments,
                created_at,
                edited_at,
                deleted,
            ),
        )
        self._update_channel_cursor(channel_id, message_id)
        self._conn.commit()

    @staticmethod
    def _normalize_content(content: Optional[str]) -> str:
        return " ".join((content or "").split()).strip()

    def record_message_edit(
        self,
        *,
        message_id: str,
        channel_id: str,
        guild_id: Optional[str] = None,
        author_id: Optional[str] = None,
        author_name: Optional[str] = None,
        author_display: Optional[str] = None,
        author_is_bot: bool = False,
        original_created_at: Optional[float] = None,
        changed_at: Optional[float] = None,
        before_content: Optional[str] = None,
        after_content: Optional[str] = None,
    ) -> None:
        """
        Record one edit event as `before -> after`.

        No-op content edits are ignored to avoid noisy context payloads.
        """
        message_id = str(message_id or "").strip()
        channel_id = str(channel_id or "").strip()
        if not message_id or not channel_id:
            return

        before_norm = self._normalize_content(before_content)
        after_norm = self._normalize_content(after_content)
        if before_norm == after_norm:
            return

        changed_ts = float(changed_at) if changed_at is not None else time.time()
        original_ts = float(original_created_at) if original_created_at is not None else None

        self._conn.execute(
            """
            INSERT INTO message_changes(
                message_id, guild_id, channel_id,
                author_id, author_name, author_display, author_is_bot,
                original_created_at, changed_at, change_type,
                before_content, after_content
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'edit', ?, ?)
            """,
            (
                message_id,
                guild_id,
                channel_id,
                author_id,
                author_name,
                author_display,
                1 if author_is_bot else 0,
                original_ts,
                changed_ts,
                before_content,
                after_content,
            ),
        )
        self._conn.commit()

    def mark_deleted(
        self,
        message_id: str,
        channel_id: Optional[str] = None,
        guild_id: Optional[str] = None,
    ) -> None:
        """Mark a message as deleted, inserting a tombstone if needed."""
        now = time.time()
        message_id = str(message_id)
        cursor = self._conn.execute(
            """
            SELECT
                message_id,
                guild_id,
                channel_id,
                author_id,
                author_name,
                author_display,
                author_is_bot,
                content,
                created_at,
                deleted
            FROM messages
            WHERE message_id = ?
            """,
            (message_id,),
        )
        row = cursor.fetchone()
        exists = row is not None

        resolved_channel = str(
            channel_id or (row["channel_id"] if row and row["channel_id"] else "") or "unknown"
        )
        resolved_guild = guild_id or (row["guild_id"] if row else None)

        if exists:
            # Record one delete transition for context rendering.
            if not bool(row["deleted"]):
                self._conn.execute(
                    """
                    INSERT INTO message_changes(
                        message_id, guild_id, channel_id,
                        author_id, author_name, author_display, author_is_bot,
                        original_created_at, changed_at, change_type,
                        before_content, after_content
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'delete', ?, '[Deleted]')
                    """,
                    (
                        message_id,
                        resolved_guild,
                        resolved_channel,
                        row["author_id"],
                        row["author_name"],
                        row["author_display"],
                        int(row["author_is_bot"] or 0),
                        float(row["created_at"]) if row["created_at"] is not None else None,
                        now,
                        row["content"],
                    ),
                )
            self._conn.execute(
                "UPDATE messages SET deleted = 1, edited_at = COALESCE(edited_at, ?) WHERE message_id = ?",
                (now, message_id),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO message_changes(
                    message_id, guild_id, channel_id,
                    original_created_at, changed_at, change_type,
                    before_content, after_content
                )
                VALUES (?, ?, ?, NULL, ?, 'delete', NULL, '[Deleted]')
                """,
                (message_id, resolved_guild, resolved_channel, now),
            )
            self._conn.execute(
                """
                INSERT INTO messages(
                    message_id, guild_id, channel_id, content, created_at, edited_at, deleted
                )
                VALUES (?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(message_id) DO UPDATE SET
                    deleted = 1,
                    edited_at = excluded.edited_at
                """,
                (message_id, resolved_guild, resolved_channel, None, now, now),
            )

        if resolved_channel:
            self._update_channel_cursor(str(resolved_channel), message_id)
        self._conn.commit()

    def _message_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        attachments: List[Dict[str, Any]] = []
        raw_attachments = row["attachments_json"]
        if raw_attachments:
            try:
                if isinstance(raw_attachments, str):
                    decoded = json.loads(raw_attachments)
                else:
                    decoded = raw_attachments
                if isinstance(decoded, list):
                    attachments = [att for att in decoded if isinstance(att, dict)]
            except Exception:
                attachments = []

        return {
            "message_id": row["message_id"],
            "guild_id": row["guild_id"],
            "guild_name": row["guild_name"],
            "channel_id": row["channel_id"],
            "channel_name": row["channel_name"],
            "thread_id": row["thread_id"],
            "author_id": row["author_id"],
            "author_name": row["author_name"],
            "author_display": row["author_display"],
            "author_is_bot": bool(row["author_is_bot"]),
            "content": row["content"] or "",
            "attachments_json": raw_attachments,
            "attachments": attachments,
            "created_at": row["created_at"],
            "edited_at": row["edited_at"],
            "deleted": bool(row["deleted"]),
        }

    def _change_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "message_id": row["message_id"],
            "guild_id": row["guild_id"],
            "channel_id": row["channel_id"],
            "author_id": row["author_id"],
            "author_name": row["author_name"],
            "author_display": row["author_display"],
            "author_is_bot": bool(row["author_is_bot"]),
            "original_created_at": row["original_created_at"],
            "changed_at": row["changed_at"],
            "change_type": row["change_type"],
            "before_content": row["before_content"] or "",
            "after_content": row["after_content"] or "",
        }

    def list_recent_messages(
        self,
        channel_id: str,
        limit: int = 20,
        include_bots: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return recent messages in chronological order."""
        limit = max(1, min(int(limit), 500))
        where = ["channel_id = ?", "deleted = 0"]
        params: List[Any] = [str(channel_id)]
        if not include_bots:
            where.append("author_is_bot = 0")

        sql = f"""
            SELECT * FROM messages
            WHERE {' AND '.join(where)}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        out = [self._message_row_to_dict(r) for r in rows]
        out.reverse()
        return out

    def list_messages_after(
        self,
        channel_id: str,
        after_message_id: str,
        limit: int = 200,
        include_bots: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return messages in a channel after a message ID, oldest first."""
        limit = max(1, min(int(limit), 1000))
        channel_id = str(channel_id)
        after_message_id = str(after_message_id or "").strip()
        if not after_message_id:
            return self.list_recent_messages(
                channel_id=channel_id,
                limit=limit,
                include_bots=include_bots,
            )

        where = ["channel_id = ?", "deleted = 0"]
        params: List[Any] = [channel_id]
        if not include_bots:
            where.append("author_is_bot = 0")

        if after_message_id.isdigit():
            where.append("CAST(message_id AS INTEGER) > CAST(? AS INTEGER)")
            params.append(after_message_id)
        else:
            row = self._conn.execute(
                "SELECT created_at FROM messages WHERE channel_id = ? AND message_id = ?",
                (channel_id, after_message_id),
            ).fetchone()
            if row:
                where.append("created_at > ?")
                params.append(float(row["created_at"]))

        sql = f"""
            SELECT * FROM messages
            WHERE {' AND '.join(where)}
            ORDER BY created_at ASC
            LIMIT ?
        """
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._message_row_to_dict(r) for r in rows]

    def list_changes_since_anchor(
        self,
        channel_id: str,
        anchor_message_id: str,
        limit: int = 200,
        include_bots: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return message edit/delete events that happened after the anchor message time.

        This catches edits/deletes to older messages that would otherwise be invisible
        when filtering strictly by message_id.
        """
        limit = max(1, min(int(limit), 1000))
        channel_id = str(channel_id)
        anchor_message_id = str(anchor_message_id or "").strip()
        if not anchor_message_id:
            return []

        anchor_row = self._conn.execute(
            "SELECT created_at FROM messages WHERE channel_id = ? AND message_id = ?",
            (channel_id, anchor_message_id),
        ).fetchone()
        if not anchor_row:
            return []

        where = ["channel_id = ?", "changed_at > ?"]
        params: List[Any] = [channel_id, float(anchor_row["created_at"])]
        if not include_bots:
            where.append("author_is_bot = 0")

        sql = f"""
            SELECT * FROM message_changes
            WHERE {' AND '.join(where)}
            ORDER BY changed_at ASC
            LIMIT ?
        """
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._change_row_to_dict(r) for r in rows]

    def count_new_non_bot_messages(
        self,
        channel_id: str,
        after_message_id: Optional[str],
    ) -> int:
        """Count non-bot, non-deleted messages after the anchor message ID."""
        after_message_id = str(after_message_id or "").strip()
        if not after_message_id:
            return 0

        channel_id = str(channel_id)
        if after_message_id.isdigit():
            row = self._conn.execute(
                """
                SELECT COUNT(1) AS c
                FROM messages
                WHERE channel_id = ?
                  AND deleted = 0
                  AND author_is_bot = 0
                  AND CAST(message_id AS INTEGER) > CAST(? AS INTEGER)
                """,
                (channel_id, after_message_id),
            ).fetchone()
            return int(row["c"]) if row else 0

        row = self._conn.execute(
            "SELECT created_at FROM messages WHERE channel_id = ? AND message_id = ?",
            (channel_id, after_message_id),
        ).fetchone()
        if not row:
            return 0

        count_row = self._conn.execute(
            """
            SELECT COUNT(1) AS c
            FROM messages
            WHERE channel_id = ?
              AND deleted = 0
              AND author_is_bot = 0
              AND created_at > ?
            """,
            (channel_id, float(row["created_at"])),
        ).fetchone()
        return int(count_row["c"]) if count_row else 0

    def _neighbors(self, channel_id: str, created_at: float, around: int) -> Dict[str, List[Dict[str, Any]]]:
        if around <= 0:
            return {"before": [], "after": []}

        before_cur = self._conn.execute(
            """
            SELECT * FROM messages
            WHERE channel_id = ? AND deleted = 0 AND created_at < ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (channel_id, created_at, around),
        )
        after_cur = self._conn.execute(
            """
            SELECT * FROM messages
            WHERE channel_id = ? AND deleted = 0 AND created_at > ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (channel_id, created_at, around),
        )
        before = [self._message_row_to_dict(r) for r in before_cur.fetchall()]
        before.reverse()
        after = [self._message_row_to_dict(r) for r in after_cur.fetchall()]
        return {"before": before, "after": after}

    def search_messages(
        self,
        query: str,
        guild_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        since_ts: Optional[float] = None,
        until_ts: Optional[float] = None,
        limit: int = 5,
        around: int = 1,
    ) -> List[Dict[str, Any]]:
        """FTS5 search with optional guild/channel/time filters and neighbors."""
        query = (query or "").strip()
        if not query:
            return []

        limit = max(1, min(int(limit), 200))
        around = max(0, min(int(around), 10))

        where = ["discord_messages_fts MATCH ?", "m.deleted = 0"]
        params: List[Any] = [query]

        if guild_id:
            where.append("m.guild_id = ?")
            params.append(str(guild_id))
        if channel_id:
            where.append("m.channel_id = ?")
            params.append(str(channel_id))
        if since_ts is not None:
            where.append("m.created_at >= ?")
            params.append(float(since_ts))
        if until_ts is not None:
            where.append("m.created_at <= ?")
            params.append(float(until_ts))

        params.append(limit)
        sql = f"""
            SELECT
                m.*,
                snippet(discord_messages_fts, 0, '>>>', '<<<', '...', 28) AS snippet
            FROM discord_messages_fts
            JOIN messages m ON m.rowid = discord_messages_fts.rowid
            WHERE {' AND '.join(where)}
            ORDER BY rank
            LIMIT ?
        """
        cur = self._conn.execute(sql, params)
        rows = cur.fetchall()

        results = []
        for row in rows:
            hit = self._message_row_to_dict(row)
            hit["snippet"] = row["snippet"]
            ctx = self._neighbors(hit["channel_id"], float(hit["created_at"]), around)
            results.append(
                {
                    "hit": hit,
                    "context_before": ctx["before"],
                    "context_after": ctx["after"],
                }
            )
        return results
