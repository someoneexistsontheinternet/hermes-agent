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
    reactions_json TEXT,
    reply_to_message_id TEXT,
    reply_to_channel_id TEXT,
    reply_to_guild_id TEXT,
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

CREATE TABLE IF NOT EXISTS reaction_changes (
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
    emoji_key TEXT,
    emoji_name TEXT,
    emoji_display TEXT,
    emoji_id TEXT,
    message_author_id TEXT,
    message_author_name TEXT,
    message_author_display TEXT
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

CREATE TABLE IF NOT EXISTS thread_context_seed (
    thread_id TEXT PRIMARY KEY,
    guild_id TEXT,
    parent_channel_id TEXT,
    anchor_message_id TEXT,
    seed_text TEXT,
    seed_kind TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS thread_scrape_state (
    parent_channel_id TEXT PRIMARY KEY,
    public_before_ts REAL,
    private_before_ts REAL,
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
CREATE INDEX IF NOT EXISTS idx_reaction_changes_channel_changed
    ON reaction_changes(channel_id, changed_at);
CREATE INDEX IF NOT EXISTS idx_reaction_changes_message
    ON reaction_changes(message_id, changed_at);
CREATE INDEX IF NOT EXISTS idx_channel_backfill_complete
    ON channel_backfill_state(complete, updated_at);
CREATE INDEX IF NOT EXISTS idx_thread_seed_parent
    ON thread_context_seed(parent_channel_id, updated_at);
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
        self._ensure_column("messages", "reactions_json", "TEXT")
        self._ensure_column("messages", "reply_to_message_id", "TEXT")
        self._ensure_column("messages", "reply_to_channel_id", "TEXT")
        self._ensure_column("messages", "reply_to_guild_id", "TEXT")
        try:
            cursor.execute("SELECT * FROM discord_messages_fts LIMIT 0")
        except sqlite3.OperationalError:
            cursor.executescript(FTS_SQL)
        self._conn.commit()

    def _ensure_column(self, table_name: str, column_name: str, column_ddl: str) -> None:
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing = {str(row["name"]) for row in rows}
        if column_name in existing:
            return
        self._conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_ddl}"
        )

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

    def get_message(self, channel_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        """Return one archived message row for a channel/message pair."""
        row = self._conn.execute(
            """
            SELECT * FROM messages
            WHERE channel_id = ? AND message_id = ?
            LIMIT 1
            """,
            (str(channel_id), str(message_id)),
        ).fetchone()
        if not row:
            return None
        return self._message_row_to_dict(row)

    def get_reply_context(
        self,
        *,
        message_id: str,
        channel_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return author + preview fields for one reply target message."""
        reply_id = str(message_id or "").strip()
        if not reply_id:
            return None

        row = None
        if channel_id:
            row = self._conn.execute(
                """
                SELECT
                    message_id,
                    author_id,
                    author_name,
                    author_display,
                    author_is_bot,
                    content,
                    deleted
                FROM messages
                WHERE channel_id = ? AND message_id = ?
                LIMIT 1
                """,
                (str(channel_id), reply_id),
            ).fetchone()

        if row is None:
            row = self._conn.execute(
                """
                SELECT
                    message_id,
                    author_id,
                    author_name,
                    author_display,
                    author_is_bot,
                    content,
                    deleted
                FROM messages
                WHERE message_id = ?
                LIMIT 1
                """,
                (reply_id,),
            ).fetchone()

        if not row:
            return None

        author_display = (
            row["author_display"]
            or row["author_name"]
            or row["author_id"]
            or "unknown"
        )
        deleted = bool(row["deleted"])
        return {
            "message_id": str(row["message_id"]),
            "author_id": row["author_id"],
            "author_name": row["author_name"],
            "author_display": author_display,
            "author_is_bot": bool(row["author_is_bot"]),
            "deleted": deleted,
            "preview": self._reply_preview(row["content"], deleted),
        }

    def get_thread_seed(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Return persisted seed context for one thread."""
        row = self._conn.execute(
            """
            SELECT
                thread_id,
                guild_id,
                parent_channel_id,
                anchor_message_id,
                seed_text,
                seed_kind,
                created_at,
                updated_at
            FROM thread_context_seed
            WHERE thread_id = ?
            """,
            (str(thread_id),),
        ).fetchone()
        if not row:
            return None
        return {
            "thread_id": row["thread_id"],
            "guild_id": row["guild_id"],
            "parent_channel_id": row["parent_channel_id"],
            "anchor_message_id": row["anchor_message_id"],
            "seed_text": row["seed_text"] or "",
            "seed_kind": row["seed_kind"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def upsert_thread_seed(
        self,
        *,
        thread_id: str,
        seed_text: str,
        seed_kind: str,
        guild_id: Optional[str] = None,
        parent_channel_id: Optional[str] = None,
        anchor_message_id: Optional[str] = None,
    ) -> None:
        """Persist one thread seed payload."""
        tid = str(thread_id or "").strip()
        if not tid:
            return
        now = time.time()
        existing = self._conn.execute(
            "SELECT created_at FROM thread_context_seed WHERE thread_id = ?",
            (tid,),
        ).fetchone()
        created_at = float(existing["created_at"]) if existing and existing["created_at"] is not None else now
        self._conn.execute(
            """
            INSERT INTO thread_context_seed(
                thread_id, guild_id, parent_channel_id, anchor_message_id,
                seed_text, seed_kind, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                guild_id = excluded.guild_id,
                parent_channel_id = excluded.parent_channel_id,
                anchor_message_id = excluded.anchor_message_id,
                seed_text = excluded.seed_text,
                seed_kind = excluded.seed_kind,
                updated_at = excluded.updated_at
            """,
            (
                tid,
                str(guild_id) if guild_id is not None else None,
                str(parent_channel_id) if parent_channel_id is not None else None,
                str(anchor_message_id) if anchor_message_id is not None else None,
                str(seed_text or ""),
                str(seed_kind or "").strip().lower() or "unanchored",
                created_at,
                now,
            ),
        )
        self._conn.commit()

    def get_thread_scrape_state(self, parent_channel_id: str) -> Dict[str, Any]:
        """Return archived-thread scan cursors for one parent channel."""
        parent_id = str(parent_channel_id or "").strip()
        if not parent_id:
            return {
                "parent_channel_id": "",
                "public_before_ts": None,
                "private_before_ts": None,
                "updated_at": None,
            }
        row = self._conn.execute(
            """
            SELECT parent_channel_id, public_before_ts, private_before_ts, updated_at
            FROM thread_scrape_state
            WHERE parent_channel_id = ?
            """,
            (parent_id,),
        ).fetchone()
        if not row:
            return {
                "parent_channel_id": parent_id,
                "public_before_ts": None,
                "private_before_ts": None,
                "updated_at": None,
            }
        return {
            "parent_channel_id": row["parent_channel_id"],
            "public_before_ts": float(row["public_before_ts"]) if row["public_before_ts"] is not None else None,
            "private_before_ts": float(row["private_before_ts"]) if row["private_before_ts"] is not None else None,
            "updated_at": float(row["updated_at"]) if row["updated_at"] is not None else None,
        }

    def upsert_thread_scrape_state(
        self,
        parent_channel_id: str,
        *,
        public_before_ts: Any = ...,
        private_before_ts: Any = ...,
    ) -> None:
        """
        Upsert archived-thread scan cursors.

        Pass `...` to keep a field unchanged, or `None` to clear/reset a field.
        """
        parent_id = str(parent_channel_id or "").strip()
        if not parent_id:
            return
        current = self.get_thread_scrape_state(parent_id)
        next_public = (
            current.get("public_before_ts")
            if public_before_ts is ...
            else (float(public_before_ts) if public_before_ts is not None else None)
        )
        next_private = (
            current.get("private_before_ts")
            if private_before_ts is ...
            else (float(private_before_ts) if private_before_ts is not None else None)
        )
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO thread_scrape_state(parent_channel_id, public_before_ts, private_before_ts, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(parent_channel_id) DO UPDATE SET
                public_before_ts = excluded.public_before_ts,
                private_before_ts = excluded.private_before_ts,
                updated_at = excluded.updated_at
            """,
            (parent_id, next_public, next_private, now),
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
        reactions = msg.get("reactions_json", msg.get("reactions"))
        if reactions is not None and not isinstance(reactions, str):
            reactions = json.dumps(reactions, ensure_ascii=False)
        reply_to_message_id = str(msg.get("reply_to_message_id") or "").strip() or None
        reply_to_channel_id = str(msg.get("reply_to_channel_id") or "").strip() or None
        reply_to_guild_id = str(msg.get("reply_to_guild_id") or "").strip() or None

        self._conn.execute(
            """
            INSERT INTO messages(
                message_id, guild_id, guild_name, channel_id, channel_name, thread_id,
                author_id, author_name, author_display, author_is_bot,
                content, attachments_json, reactions_json,
                reply_to_message_id, reply_to_channel_id, reply_to_guild_id,
                created_at, edited_at, deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                reactions_json = COALESCE(excluded.reactions_json, messages.reactions_json),
                reply_to_message_id = excluded.reply_to_message_id,
                reply_to_channel_id = excluded.reply_to_channel_id,
                reply_to_guild_id = excluded.reply_to_guild_id,
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
                reactions,
                reply_to_message_id,
                reply_to_channel_id,
                reply_to_guild_id,
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

    @staticmethod
    def _reply_preview(content: Optional[str], deleted: bool) -> str:
        if deleted:
            return "[Deleted]"
        normalized = " ".join((content or "").split()).strip()
        if not normalized:
            return "[non-text message]"
        return normalized[:50]

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

    def record_reaction_change(
        self,
        *,
        message_id: str,
        channel_id: str,
        change_type: str,
        changed_at: Optional[float] = None,
        guild_id: Optional[str] = None,
        author_id: Optional[str] = None,
        author_name: Optional[str] = None,
        author_display: Optional[str] = None,
        author_is_bot: bool = False,
        original_created_at: Optional[float] = None,
        emoji_key: Optional[str] = None,
        emoji_name: Optional[str] = None,
        emoji_display: Optional[str] = None,
        emoji_id: Optional[str] = None,
        message_author_id: Optional[str] = None,
        message_author_name: Optional[str] = None,
        message_author_display: Optional[str] = None,
    ) -> None:
        """Record one reaction add/remove/clear event."""
        message_id = str(message_id or "").strip()
        channel_id = str(channel_id or "").strip()
        change_type = str(change_type or "").strip().lower()
        if not message_id or not channel_id or not change_type:
            return

        changed_ts = float(changed_at) if changed_at is not None else time.time()
        original_ts = float(original_created_at) if original_created_at is not None else None
        self._conn.execute(
            """
            INSERT INTO reaction_changes(
                message_id, guild_id, channel_id,
                author_id, author_name, author_display, author_is_bot,
                original_created_at, changed_at, change_type,
                emoji_key, emoji_name, emoji_display, emoji_id,
                message_author_id, message_author_name, message_author_display
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                change_type,
                emoji_key,
                emoji_name,
                emoji_display,
                emoji_id,
                message_author_id,
                message_author_name,
                message_author_display,
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

        reactions: List[Dict[str, Any]] = []
        raw_reactions = row["reactions_json"] if "reactions_json" in row.keys() else None
        if raw_reactions:
            try:
                if isinstance(raw_reactions, str):
                    decoded = json.loads(raw_reactions)
                else:
                    decoded = raw_reactions
                if isinstance(decoded, list):
                    reactions = [entry for entry in decoded if isinstance(entry, dict)]
            except Exception:
                reactions = []

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
            "reactions_json": raw_reactions,
            "reactions": reactions,
            "reply_to_message_id": row["reply_to_message_id"] if "reply_to_message_id" in row.keys() else None,
            "reply_to_channel_id": row["reply_to_channel_id"] if "reply_to_channel_id" in row.keys() else None,
            "reply_to_guild_id": row["reply_to_guild_id"] if "reply_to_guild_id" in row.keys() else None,
            "created_at": row["created_at"],
            "edited_at": row["edited_at"],
            "deleted": bool(row["deleted"]),
        }

    def enrich_reply_context_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach reply target author/preview fields for context rendering."""
        if not rows:
            return rows

        ordered_ids: List[str] = []
        seen_ids = set()
        for row in rows:
            reply_id = str(row.get("reply_to_message_id") or "").strip()
            if not reply_id or reply_id in seen_ids:
                continue
            seen_ids.add(reply_id)
            ordered_ids.append(reply_id)

        if not ordered_ids:
            return rows 

        placeholders = ",".join("?" for _ in ordered_ids)
        cur = self._conn.execute(
            f"""
            SELECT
                message_id,
                author_id,
                author_name,
                author_display,
                content,
                deleted
            FROM messages
            WHERE message_id IN ({placeholders})
            """,
            ordered_ids,
        )
        parent_rows = {
            str(row["message_id"]): {
                "author_display": row["author_display"],
                "author_name": row["author_name"],
                "author_id": row["author_id"],
                "content": row["content"],
                "deleted": bool(row["deleted"]),
            }
            for row in cur.fetchall()
        }

        enriched_rows: List[Dict[str, Any]] = []
        for row in rows:
            enriched = dict(row)
            reply_id = str(row.get("reply_to_message_id") or "").strip()
            if reply_id:
                parent = parent_rows.get(reply_id)
                if parent:
                    enriched["reply_author_display"] = (
                        parent.get("author_display")
                        or parent.get("author_name")
                        or parent.get("author_id")
                        or "unknown"
                    )
                    enriched["reply_preview"] = self._reply_preview(
                        parent.get("content"),
                        bool(parent.get("deleted")),
                    )
            enriched_rows.append(enriched)
        return enriched_rows

    def _change_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        keys = set(row.keys())
        payload = {
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
        if "emoji_key" in keys:
            payload["emoji_key"] = row["emoji_key"] or ""
            payload["emoji_name"] = row["emoji_name"] or ""
            payload["emoji_display"] = row["emoji_display"] or ""
            payload["emoji_id"] = row["emoji_id"] or ""
        if "message_author_id" in keys:
            payload["message_author_id"] = row["message_author_id"] or ""
            payload["message_author_name"] = row["message_author_name"] or ""
            payload["message_author_display"] = row["message_author_display"] or ""
        return payload

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

    def list_reaction_changes_since_anchor(
        self,
        channel_id: str,
        anchor_message_id: str,
        limit: int = 200,
        include_bots: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return reaction events that happened after the anchor message time.

        This catches reactions on older messages that would otherwise be invisible
        in delta-only context windows.
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
            SELECT
                id,
                message_id,
                guild_id,
                channel_id,
                author_id,
                author_name,
                author_display,
                author_is_bot,
                original_created_at,
                changed_at,
                change_type,
                '' AS before_content,
                '' AS after_content,
                emoji_key,
                emoji_name,
                emoji_display,
                emoji_id,
                message_author_id,
                message_author_name,
                message_author_display
            FROM reaction_changes
            WHERE {' AND '.join(where)}
            ORDER BY changed_at ASC
            LIMIT ?
        """
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._change_row_to_dict(r) for r in rows]

    def list_all_changes_since_anchor(
        self,
        channel_id: str,
        anchor_message_id: str,
        limit: int = 200,
        include_bots: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return message and reaction changes ordered chronologically."""
        limit = max(1, min(int(limit), 1000))
        changes = self.list_changes_since_anchor(
            channel_id=channel_id,
            anchor_message_id=anchor_message_id,
            limit=limit,
            include_bots=include_bots,
        )
        reaction_changes = self.list_reaction_changes_since_anchor(
            channel_id=channel_id,
            anchor_message_id=anchor_message_id,
            limit=limit,
            include_bots=include_bots,
        )
        merged = changes + reaction_changes
        merged.sort(key=lambda row: (float(row.get("changed_at") or 0), int(row.get("id") or 0)))
        return merged[:limit]

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
