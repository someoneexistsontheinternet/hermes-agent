"""Tests for gateway/discord_archive.py."""

from gateway.discord_archive import DiscordArchiveDB


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
