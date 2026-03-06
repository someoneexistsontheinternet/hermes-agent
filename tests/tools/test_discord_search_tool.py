"""Tests for tools/discord_search_tool.py."""

import csv
import io
import json
import sqlite3
from pathlib import Path

from tools.discord_search_tool import (
    DISCORD_SEARCH_SCHEMA,
    _apply_default_limit,
    discord_search,
)


def _create_archive_db(db_path: Path, row_count: int) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                author_name TEXT,
                content TEXT,
                created_at REAL NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE VIRTUAL TABLE discord_messages_fts USING fts5(
                content,
                content=messages,
                content_rowid=rowid
            );
            CREATE TRIGGER discord_messages_fts_insert
            AFTER INSERT ON messages BEGIN
                INSERT INTO discord_messages_fts(rowid, content)
                VALUES (new.rowid, COALESCE(new.content, ''));
            END;
            """
        )
        for idx in range(row_count):
            conn.execute(
                """
                INSERT INTO messages(message_id, channel_id, author_name, content, created_at, deleted)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (str(idx), "channel-1", f"user-{idx}", f"message {idx}", float(idx)),
            )
        conn.commit()
    finally:
        conn.close()


class TestDiscordSearchSchema:
    def test_schema_exposes_sql_and_message_ids(self):
        properties = DISCORD_SEARCH_SCHEMA["parameters"]["properties"]
        assert set(properties) == {"sql", "message_ids"}
        assert DISCORD_SEARCH_SCHEMA["parameters"]["required"] == []


class TestDiscordSearchValidation:
    def test_rejects_missing_sql_and_message_ids(self):
        result = json.loads(discord_search())
        assert result["success"] is False
        assert "Provide either 'sql' or 'message_ids'" in result["error"]

    def test_rejects_both_sql_and_message_ids(self):
        result = json.loads(discord_search(sql="SELECT 1", message_ids=["1"]))
        assert result["success"] is False
        assert "not both" in result["error"]

    def test_default_limit_is_added_when_missing(self):
        sql = _apply_default_limit("SELECT message_id FROM messages ORDER BY created_at ASC")
        assert sql.endswith("LIMIT 100")

    def test_existing_limit_is_preserved(self):
        original = "SELECT message_id FROM messages ORDER BY created_at ASC LIMIT 25"
        assert _apply_default_limit(original) == original


class TestDiscordSearchExecution:
    def test_default_limit_caps_results(self, monkeypatch, tmp_path):
        db_path = tmp_path / "discord.sqlite"
        _create_archive_db(db_path, row_count=150)
        monkeypatch.setenv("DISCORD_ARCHIVE_DB_PATH", str(db_path))

        output = discord_search("SELECT message_id FROM messages ORDER BY created_at ASC")
        rows = list(csv.reader(io.StringIO(output)))

        assert rows[0] == ["message_id"]
        assert len(rows) == 101
        assert rows[1] == ["0"]
        assert rows[-1] == ["99"]

    def test_explicit_large_limit_is_hard_capped(self, monkeypatch, tmp_path):
        db_path = tmp_path / "discord.sqlite"
        _create_archive_db(db_path, row_count=600)
        monkeypatch.setenv("DISCORD_ARCHIVE_DB_PATH", str(db_path))

        output = discord_search("SELECT message_id FROM messages ORDER BY created_at ASC LIMIT 600")
        rows = list(csv.reader(io.StringIO(output)))

        assert rows[0] == ["message_id"]
        assert len(rows) == 501
        assert rows[1] == ["0"]
        assert rows[-1] == ["499"]

    def test_rejects_write_sql(self, monkeypatch, tmp_path):
        db_path = tmp_path / "discord.sqlite"
        _create_archive_db(db_path, row_count=1)
        monkeypatch.setenv("DISCORD_ARCHIVE_DB_PATH", str(db_path))

        result = json.loads(discord_search("DELETE FROM messages"))
        assert result["success"] is False
        assert "Only read SQL is allowed" in result["error"]

    def test_search_returns_full_long_text_cells_for_five_or_fewer_rows(self, monkeypatch, tmp_path):
        db_path = tmp_path / "discord.sqlite"
        _create_archive_db(db_path, row_count=1)
        monkeypatch.setenv("DISCORD_ARCHIVE_DB_PATH", str(db_path))

        long_text = "x" * 240
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "UPDATE messages SET content = ? WHERE message_id = '0'",
                (long_text,),
            )
            conn.commit()
        finally:
            conn.close()

        output = discord_search("SELECT message_id, content FROM messages WHERE message_id = '0'")
        rows = list(csv.reader(io.StringIO(output)))

        assert rows[0] == ["message_id", "content"]
        assert rows[1][0] == "0"
        assert rows[1][1] == long_text

    def test_search_truncates_long_text_cells_when_more_than_five_rows(self, monkeypatch, tmp_path):
        db_path = tmp_path / "discord.sqlite"
        _create_archive_db(db_path, row_count=6)
        monkeypatch.setenv("DISCORD_ARCHIVE_DB_PATH", str(db_path))

        long_text = "x" * 240
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "UPDATE messages SET content = ? WHERE message_id = '0'",
                (long_text,),
            )
            conn.commit()
        finally:
            conn.close()

        output = discord_search("SELECT message_id, content FROM messages ORDER BY created_at ASC LIMIT 6")
        rows = list(csv.reader(io.StringIO(output)))

        assert rows[0] == ["message_id", "content"]
        assert rows[1][0] == "0"
        assert rows[1][1].endswith("...")
        assert len(rows[1][1]) == 200

    def test_fetch_by_message_ids_returns_full_content(self, monkeypatch, tmp_path):
        db_path = tmp_path / "discord.sqlite"
        _create_archive_db(db_path, row_count=2)
        monkeypatch.setenv("DISCORD_ARCHIVE_DB_PATH", str(db_path))

        long_text = "y" * 240
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "UPDATE messages SET content = ? WHERE message_id = '1'",
                (long_text,),
            )
            conn.commit()
        finally:
            conn.close()

        output = discord_search(message_ids=["1", "0"])
        rows = list(csv.reader(io.StringIO(output)))

        assert rows[0] == ["message_id", "created_at", "channel_id", "author_name", "content"]
        assert rows[1][0] == "1"
        assert rows[1][4] == long_text
        assert rows[2][0] == "0"

    def test_fts_count_query_is_allowed(self, monkeypatch, tmp_path):
        db_path = tmp_path / "discord.sqlite"
        _create_archive_db(db_path, row_count=3)
        monkeypatch.setenv("DISCORD_ARCHIVE_DB_PATH", str(db_path))

        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "UPDATE messages SET author_name = 'vyetic', content = 'fuck this deploy' WHERE message_id = '1'"
            )
            conn.execute(
                "UPDATE messages SET author_name = 'vyetic', content = 'no match here' WHERE message_id = '2'"
            )
            conn.commit()
            conn.execute("INSERT INTO discord_messages_fts(discord_messages_fts) VALUES('rebuild')")
            conn.commit()
        finally:
            conn.close()

        output = discord_search(
            """SELECT COUNT(*) as fuck_count
FROM discord_messages_fts
JOIN messages m ON m.rowid = discord_messages_fts.rowid
WHERE discord_messages_fts MATCH 'fuck'
  AND m.author_name = 'vyetic'
  AND m.deleted = 0"""
        )
        rows = list(csv.reader(io.StringIO(output)))

        assert rows[0] == ["fuck_count"]
        assert rows[1] == ["1"]
