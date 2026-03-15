"""Tests for gateway/discord_archive.py."""

import sqlite3

from gateway.discord_archive import DiscordArchiveDB
from gateway.platforms.discord import DiscordAdapter


def test_set_turn_anchor_is_monotonic_for_numeric_ids(tmp_path):
    db = DiscordArchiveDB(tmp_path / "discord_data.sqlite")
    try:
        db.set_turn_anchor("ch1", "200")
        db.set_turn_anchor("ch1", "150")
        assert db.get_turn_anchor("ch1") == "200"

        db.set_turn_anchor("ch1", "250")
        assert db.get_turn_anchor("ch1") == "250"
    finally:
        db.close()


def test_record_message_edit_is_queryable_since_anchor(tmp_path):
    db = DiscordArchiveDB(tmp_path / "discord_data.sqlite")
    try:
        db.upsert_message(
            {
                "message_id": "100",
                "channel_id": "ch1",
                "guild_id": "g1",
                "author_id": "u1",
                "author_name": "alice",
                "author_display": "alice",
                "created_at": 1000.0,
                "content": "anchor",
                "deleted": False,
            }
        )
        db.upsert_message(
            {
                "message_id": "101",
                "channel_id": "ch1",
                "guild_id": "g1",
                "author_id": "u1",
                "author_name": "alice",
                "author_display": "alice",
                "created_at": 1010.0,
                "content": "before edit",
                "deleted": False,
            }
        )

        db.record_message_edit(
            message_id="101",
            channel_id="ch1",
            guild_id="g1",
            author_id="u1",
            author_name="alice",
            author_display="alice",
            author_is_bot=False,
            original_created_at=1010.0,
            changed_at=1020.0,
            before_content="before edit",
            after_content="after edit",
        )

        changes = db.list_changes_since_anchor(
            channel_id="ch1",
            anchor_message_id="100",
            limit=10,
            include_bots=False,
        )
        assert len(changes) == 1
        assert changes[0]["change_type"] == "edit"
        assert changes[0]["before_content"] == "before edit"
        assert changes[0]["after_content"] == "after edit"
        assert changes[0]["message_id"] == "101"
    finally:
        db.close()


def test_record_message_edit_ignores_noop_text_changes(tmp_path):
    db = DiscordArchiveDB(tmp_path / "discord_data.sqlite")
    try:
        db.upsert_message(
            {
                "message_id": "100",
                "channel_id": "ch1",
                "created_at": 1000.0,
                "content": "anchor",
                "deleted": False,
            }
        )
        db.record_message_edit(
            message_id="101",
            channel_id="ch1",
            original_created_at=1010.0,
            changed_at=1020.0,
            before_content="same text",
            after_content="same text",
        )
        changes = db.list_changes_since_anchor(
            channel_id="ch1",
            anchor_message_id="100",
            limit=10,
            include_bots=False,
        )
        assert changes == []
    finally:
        db.close()


def test_mark_deleted_records_single_delete_change(tmp_path):
    db = DiscordArchiveDB(tmp_path / "discord_data.sqlite")
    try:
        db.upsert_message(
            {
                "message_id": "100",
                "channel_id": "ch1",
                "guild_id": "g1",
                "author_id": "u1",
                "author_name": "alice",
                "author_display": "alice",
                "created_at": 1000.0,
                "content": "anchor",
                "deleted": False,
            }
        )
        db.upsert_message(
            {
                "message_id": "101",
                "channel_id": "ch1",
                "guild_id": "g1",
                "author_id": "u1",
                "author_name": "alice",
                "author_display": "alice",
                "created_at": 1010.0,
                "content": "to be deleted",
                "deleted": False,
            }
        )

        db.mark_deleted("101", channel_id="ch1", guild_id="g1")
        # Repeat delete events should not duplicate the same transition.
        db.mark_deleted("101", channel_id="ch1", guild_id="g1")

        changes = db.list_changes_since_anchor(
            channel_id="ch1",
            anchor_message_id="100",
            limit=10,
            include_bots=False,
        )
        assert len(changes) == 1
        assert changes[0]["change_type"] == "delete"
        assert changes[0]["before_content"] == "to be deleted"
        assert changes[0]["after_content"] == "[Deleted]"
    finally:
        db.close()


def test_init_schema_adds_reply_columns_to_existing_messages_table(tmp_path):
    db_path = tmp_path / "discord_data.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE messages (
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
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    db = DiscordArchiveDB(db_path)
    try:
        columns = {
            str(row["name"])
            for row in db._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        assert "reply_to_message_id" in columns
        assert "reply_to_channel_id" in columns
        assert "reply_to_guild_id" in columns
    finally:
        db.close()


def test_upsert_message_round_trips_reactions(tmp_path):
    db = DiscordArchiveDB(tmp_path / "discord_data.sqlite")
    try:
        db.upsert_message(
            {
                "message_id": "101",
                "channel_id": "ch1",
                "guild_id": "g1",
                "author_id": "u1",
                "author_name": "alice",
                "author_display": "alice",
                "created_at": 1010.0,
                "content": "hello",
                "reactions_json": [
                    {
                        "emoji_key": "custom:55",
                        "emoji_name": "kek",
                        "emoji_display": "kek",
                        "emoji_id": "55",
                        "count": 2,
                        "reactors": [
                            {"user_id": "u2", "user_name": "bob"},
                            {"user_id": "u3", "user_name": "john"},
                        ],
                        "reactors_complete": True,
                    }
                ],
                "deleted": False,
            }
        )

        row = db.get_message("ch1", "101")
        assert row is not None
        assert row["reactions"][0]["emoji_name"] == "kek"
        assert row["reactions"][0]["reactors"] == [
            {"user_id": "u2", "user_name": "bob"},
            {"user_id": "u3", "user_name": "john"},
        ]
    finally:
        db.close()


def test_upsert_message_round_trips_reply_metadata(tmp_path):
    db = DiscordArchiveDB(tmp_path / "discord_data.sqlite")
    try:
        db.upsert_message(
            {
                "message_id": "101",
                "channel_id": "ch1",
                "guild_id": "g1",
                "author_id": "u1",
                "author_name": "alice",
                "author_display": "alice",
                "created_at": 1010.0,
                "content": "replying now",
                "reply_to_message_id": "100",
                "reply_to_channel_id": "ch0",
                "reply_to_guild_id": "g0",
                "deleted": False,
            }
        )

        row = db.get_message("ch1", "101")
        assert row is not None
        assert row["reply_to_message_id"] == "100"
        assert row["reply_to_channel_id"] == "ch0"
        assert row["reply_to_guild_id"] == "g0"
    finally:
        db.close()


def test_enrich_reply_context_rows_resolves_parent_author_and_preview(tmp_path):
    db = DiscordArchiveDB(tmp_path / "discord_data.sqlite")
    try:
        db.upsert_message(
            {
                "message_id": "100",
                "channel_id": "ch1",
                "guild_id": "g1",
                "author_id": "u1",
                "author_name": "alice",
                "author_display": "alice",
                "created_at": 1000.0,
                "content": "01234567890123456789extra words",
                "deleted": False,
            }
        )
        rows = db.enrich_reply_context_rows(
            [
                {
                    "message_id": "101",
                    "channel_id": "ch1",
                    "author_display": "bob",
                    "content": "child message",
                    "reply_to_message_id": "100",
                }
            ]
        )
        assert rows[0]["reply_author_display"] == "alice"
        assert rows[0]["reply_preview"] == "01234567890123456789"
    finally:
        db.close()


def test_list_all_changes_since_anchor_includes_reaction_changes(tmp_path):
    db = DiscordArchiveDB(tmp_path / "discord_data.sqlite")
    try:
        db.upsert_message(
            {
                "message_id": "100",
                "channel_id": "ch1",
                "guild_id": "g1",
                "author_id": "u1",
                "author_name": "alice",
                "author_display": "alice",
                "created_at": 1000.0,
                "content": "anchor",
                "deleted": False,
            }
        )
        db.upsert_message(
            {
                "message_id": "101",
                "channel_id": "ch1",
                "guild_id": "g1",
                "author_id": "u1",
                "author_name": "alice",
                "author_display": "alice",
                "created_at": 1010.0,
                "content": "hello",
                "deleted": False,
            }
        )

        db.record_message_edit(
            message_id="101",
            channel_id="ch1",
            guild_id="g1",
            author_id="u1",
            author_name="alice",
            author_display="alice",
            author_is_bot=False,
            original_created_at=1010.0,
            changed_at=1020.0,
            before_content="hello",
            after_content="hello there",
        )
        db.record_reaction_change(
            message_id="101",
            channel_id="ch1",
            guild_id="g1",
            author_id="u2",
            author_name="bob",
            author_display="bob",
            author_is_bot=False,
            original_created_at=1010.0,
            changed_at=1030.0,
            change_type="reaction_add",
            emoji_key="custom:55",
            emoji_name="kek",
            emoji_display="kek",
            emoji_id="55",
            message_author_id="u1",
            message_author_name="alice",
            message_author_display="alice",
        )

        changes = db.list_all_changes_since_anchor(
            channel_id="ch1",
            anchor_message_id="100",
            limit=10,
            include_bots=False,
        )
        assert [change["change_type"] for change in changes] == ["edit", "reaction_add"]
        assert changes[1]["author_name"] == "bob"
        assert changes[1]["emoji_name"] == "kek"
    finally:
        db.close()


def test_format_archive_history_line_includes_reactor_usernames():
    hour, line = DiscordAdapter._format_archive_history_line(
        {
            "created_at": 1000.0,
            "author_display": "alice",
            "content": "hello",
            "reactions": [
                {
                    "emoji_name": "kek",
                    "emoji_display": "kek",
                    "count": 2,
                    "reactors": [
                        {"user_id": "u2", "user_name": "bob"},
                        {"user_id": "u3", "user_name": "john"},
                    ],
                    "reactors_complete": True,
                },
                {
                    "emoji_name": "👍",
                    "emoji_display": "👍",
                    "count": 1,
                    "reactors": [
                        {"user_id": "u2", "user_name": "bob"},
                    ],
                    "reactors_complete": True,
                },
            ],
        }
    )
    assert hour
    assert line.endswith("hello [reactions: kek (bob, john), 👍 (bob)]")


def test_format_archive_history_line_caps_reactor_names():
    _, line = DiscordAdapter._format_archive_history_line(
        {
            "created_at": 1000.0,
            "author_display": "alice",
            "content": "hello",
            "reactions": [
                {
                    "emoji_name": "kek",
                    "emoji_display": "kek",
                    "count": 5,
                    "reactors": [
                        {"user_id": "u1", "user_name": "amy"},
                        {"user_id": "u2", "user_name": "bob"},
                        {"user_id": "u3", "user_name": "john"},
                    ],
                    "reactors_complete": False,
                }
            ],
        }
    )
    assert line.endswith("hello [reactions: kek (amy, bob, john, +2)]")


def test_format_archive_history_line_includes_reply_suffix():
    _, line = DiscordAdapter._format_archive_history_line(
        {
            "created_at": 1000.0,
            "author_display": "alice",
            "content": "hello",
            "reply_to_message_id": "99",
            "reply_author_display": "bob",
            "reply_preview": "01234567890123456789",
            "reactions": [],
        }
    )
    assert line.endswith(
        "<alice> (replying <bob>: 01234567890123456789): hello"
    )


def test_format_archive_change_line_for_reaction_add():
    _, line = DiscordAdapter._format_archive_change_line(
        {
            "changed_at": 1030.0,
            "change_type": "reaction_add",
            "author_display": "bob",
            "emoji_display": "kek",
            "message_author_display": "alice",
        }
    )
    assert line.endswith("<bob>: added kek to alice's message")


def test_backfill_state_tracks_oldest_id_monotonically(tmp_path):
    db = DiscordArchiveDB(tmp_path / "discord_data.sqlite")
    try:
        db.upsert_backfill_state(
            "ch1",
            oldest_message_id="200",
            oldest_created_at=2000.0,
            complete=False,
        )
        db.upsert_backfill_state(
            "ch1",
            oldest_message_id="300",
            oldest_created_at=3000.0,
            complete=False,
        )
        state = db.get_backfill_state("ch1")
        assert state["oldest_message_id"] == "200"
        assert state["complete"] is False

        db.upsert_backfill_state(
            "ch1",
            oldest_message_id="150",
            oldest_created_at=1500.0,
            complete=False,
        )
        state = db.get_backfill_state("ch1")
        assert state["oldest_message_id"] == "150"
        assert state["oldest_created_at"] == 1500.0

        db.mark_backfill_complete("ch1", complete=True)
        state = db.get_backfill_state("ch1")
        assert state["complete"] is True
    finally:
        db.close()


def test_get_oldest_message_returns_channel_head(tmp_path):
    db = DiscordArchiveDB(tmp_path / "discord_data.sqlite")
    try:
        db.upsert_message(
            {
                "message_id": "900",
                "channel_id": "ch1",
                "created_at": 1900.0,
                "content": "newer",
                "deleted": False,
            }
        )
        db.upsert_message(
            {
                "message_id": "800",
                "channel_id": "ch1",
                "created_at": 1800.0,
                "content": "older",
                "deleted": False,
            }
        )
        row = db.get_oldest_message("ch1")
        assert row is not None
        assert row["message_id"] == "800"
        assert row["created_at"] == 1800.0
    finally:
        db.close()
