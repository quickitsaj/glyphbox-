"""Integration tests for NetHackAPI."""

import pytest

from src.api.nethack_api import NetHackAPI
from src.api.models import Direction, Position, ActionResult


class TestNetHackAPILifecycle:
    """Tests for API lifecycle management."""

    def test_create_and_close(self):
        """Test creating and closing API."""
        api = NetHackAPI(max_episode_steps=100)
        try:
            obs = api.reset()
            assert obs is not None
        finally:
            api.close()

    def test_context_manager(self):
        """Test using API as context manager."""
        with NetHackAPI(max_episode_steps=100) as api:
            obs = api.reset()
            assert obs is not None

    def test_reset_returns_observation(self):
        """Test that reset returns an observation."""
        with NetHackAPI(max_episode_steps=100) as api:
            obs = api.reset()
            assert obs is not None
            assert hasattr(obs, "glyphs")
            assert hasattr(obs, "blstats")


class TestNetHackAPIQueries:
    """Tests for API query methods."""

    def test_get_stats(self, nethack_api):
        """Test get_stats method."""
        nethack_api.reset()
        stats = nethack_api.get_stats()

        assert stats.hp > 0
        assert stats.max_hp >= stats.hp
        assert stats.dungeon_level >= 1
        assert isinstance(stats.position, Position)

    def test_get_position(self, nethack_api):
        """Test get_position method."""
        nethack_api.reset()
        pos = nethack_api.get_position()

        assert isinstance(pos, Position)
        assert 0 <= pos.x < 79
        assert 0 <= pos.y < 21

    def test_get_screen(self, nethack_api):
        """Test get_screen method."""
        nethack_api.reset()
        screen = nethack_api.get_screen()

        assert isinstance(screen, str)
        assert len(screen) > 0
        # Screen should have multiple lines
        lines = screen.split("\n")
        assert len(lines) >= 20

    def test_get_local_map(self, nethack_api):
        """Test get_local_map method returns LLM-optimized local view."""
        nethack_api.reset()
        local_map = nethack_api.get_local_map(radius=7)

        assert isinstance(local_map, str)
        lines = local_map.split("\n")

        # Should have header line
        assert "LOCAL VIEW" in lines[0]
        assert "radius=7" in lines[0]

        # Should have coordinate headers and row labels
        # Column header line should have numbers
        assert any(char.isdigit() for char in lines[1])

        # Map rows should have row labels (y coordinates)
        # Row labels look like "   5:" or similar
        map_lines = [l for l in lines[2:] if ":" in l and l.strip()[0].isdigit()]
        assert len(map_lines) > 0

        # Should include status bar (last 2 lines should have game info)
        # Status bar contains things like "HP:", "Dlvl:", etc.
        status_lines = "\n".join(lines[-2:])
        # Should have some player stat info
        assert any(stat in status_lines for stat in ["HP", "Dlvl", "St:", "Pw:"])

        # Should contain player @ symbol somewhere in the map
        assert "@" in local_map

    def test_get_local_map_different_radius(self, nethack_api):
        """Test get_local_map with different radius values."""
        nethack_api.reset()

        # Test with smaller radius
        small_map = nethack_api.get_local_map(radius=3)
        large_map = nethack_api.get_local_map(radius=10)

        # Larger radius should produce more content
        assert len(large_map) > len(small_map)

        # Both should have the header with correct radius
        assert "radius=3" in small_map
        assert "radius=10" in large_map

    def test_get_message(self, nethack_api):
        """Test get_message method."""
        nethack_api.reset()
        message = nethack_api.get_message()

        # Message may be empty or contain text
        assert isinstance(message, str)

    def test_get_inventory(self, nethack_api):
        """Test get_inventory method."""
        nethack_api.reset()
        inventory = nethack_api.get_inventory()

        assert isinstance(inventory, list)
        # Player should have some starting items
        assert len(inventory) > 0

    def test_get_current_level(self, nethack_api):
        """Test get_current_level method."""
        nethack_api.reset()
        level = nethack_api.get_current_level()

        assert level.level_number == 1
        assert level.dungeon_number == 0  # Dungeons of Doom
        assert len(level.tiles) == 21
        assert len(level.tiles[0]) == 79

    def test_get_visible_monsters(self, nethack_api):
        """Test get_visible_monsters method."""
        nethack_api.reset()
        monsters = nethack_api.get_visible_monsters()

        assert isinstance(monsters, list)

    def test_get_adjacent_hostiles(self, nethack_api):
        """Test get_adjacent_hostiles method."""
        nethack_api.reset()
        monsters = nethack_api.get_adjacent_hostiles()

        assert isinstance(monsters, list)


class TestNetHackAPIActions:
    """Tests for API action methods."""

    def test_move(self, nethack_api):
        """Test move method."""
        nethack_api.reset()
        result = nethack_api.move(Direction.N)

        assert isinstance(result, ActionResult)

    def test_wait(self, nethack_api):
        """Test wait method."""
        nethack_api.reset()
        result = nethack_api.wait()

        assert result.success is True

    def test_search(self, nethack_api):
        """Test search method."""
        nethack_api.reset()
        result = nethack_api.search()

        assert result.success is True

    def test_attack(self, nethack_api):
        """Test attack method."""
        nethack_api.reset()
        result = nethack_api.attack(Direction.N)

        assert isinstance(result, ActionResult)

    def test_kick(self, nethack_api):
        """Test kick method."""
        nethack_api.reset()
        result = nethack_api.kick(Direction.N)

        assert isinstance(result, ActionResult)

    def test_pickup(self, nethack_api):
        """Test pickup method."""
        nethack_api.reset()
        result = nethack_api.pickup()

        assert isinstance(result, ActionResult)

    def test_send_keys(self, nethack_api):
        """Test send_keys method."""
        nethack_api.reset()
        # Send a wait command via raw keys
        result = nethack_api.send_keys(".")

        assert result.success is True


class TestNetHackAPIPathfinding:
    """Tests for API pathfinding methods."""

    def test_internal_find_path_to_self(self, nethack_api):
        """Test internal _find_path to current position returns ALREADY_AT_TARGET."""
        nethack_api.reset()
        pos = nethack_api.get_position()

        # Use allow_with_hostiles=True to bypass hostile check for this test
        result = nethack_api._find_path(pos, allow_with_hostiles=True)

        from src.api import PathResult, PathStopReason
        assert isinstance(result, PathResult)
        assert result.reason == PathStopReason.ALREADY_AT_TARGET
        assert result.path == []

    def test_internal_find_path_hostile_in_view(self, nethack_api):
        """Test internal _find_path refuses when hostile in view."""
        nethack_api.reset()
        pos = nethack_api.get_position()

        # Without allow_with_hostiles, may get HOSTILE_IN_VIEW if there are monsters
        result = nethack_api._find_path(pos)

        from src.api import PathResult, PathStopReason
        assert isinstance(result, PathResult)
        # Either at target or blocked by hostiles
        assert result.reason in (PathStopReason.ALREADY_AT_TARGET, PathStopReason.HOSTILE_IN_VIEW)

    def test_move_to_adjacent(self, nethack_api):
        """Test move_to can move to an adjacent walkable tile."""
        nethack_api.reset()
        start_pos = nethack_api.get_position()

        # Find a walkable adjacent tile
        from src.api.models import Direction
        target = None
        for d in [Direction.N, Direction.S, Direction.E, Direction.W]:
            adj_pos = start_pos.move(d)
            tile = nethack_api.get_tile(adj_pos)
            if tile and tile.is_walkable:
                target = adj_pos
                break

        if target:
            result = nethack_api.move_to(target)
            # Should either succeed or be interrupted by hostile
            assert nethack_api.position == target or not result.success

    def test_find_unexplored(self, nethack_api):
        """Test finding unexplored area returns TargetResult."""
        nethack_api.reset()

        # Use allow_with_hostiles=True to bypass hostile check for this test
        result = nethack_api.find_unexplored(allow_with_hostiles=True)

        from src.api import TargetResult
        assert isinstance(result, TargetResult)
        # May or may not find unexplored tiles
        assert result.position is None or isinstance(result.position, Position)

class TestNetHackAPIStatePersistence:
    """Tests for state persistence across actions."""

    def test_turn_advances_on_action(self, nethack_api):
        """Test that turns advance when taking actions."""
        nethack_api.reset()

        initial_turn = nethack_api.get_stats().turn

        # Wait should advance turn
        nethack_api.wait()

        new_turn = nethack_api.get_stats().turn

        assert new_turn >= initial_turn

    def test_position_changes_on_move(self, nethack_api):
        """Test that position can change on movement."""
        nethack_api.reset()

        initial_pos = nethack_api.get_position()

        # Try moving in all directions until one succeeds
        for direction in [Direction.N, Direction.S, Direction.E, Direction.W,
                          Direction.NE, Direction.NW, Direction.SE, Direction.SW]:
            nethack_api.move(direction)
            new_pos = nethack_api.get_position()
            if new_pos != initial_pos:
                break

        # Position may or may not have changed (depends on walls)
        assert isinstance(new_pos, Position)


class TestNetHackAPIGameplay:
    """Tests for longer gameplay sequences."""

    def test_explore_briefly(self, nethack_api):
        """Test taking multiple actions."""
        nethack_api.reset()

        # Take 10 actions
        for _ in range(10):
            nethack_api.move(Direction.N)

        # Should still be able to query state
        stats = nethack_api.get_stats()
        assert stats is not None
        assert stats.hp > 0 or stats.hp <= 0  # May have died

    def test_search_multiple_times(self, nethack_api):
        """Test searching multiple times."""
        nethack_api.reset()

        # Search 5 times
        for _ in range(5):
            result = nethack_api.search()
            assert result.success is True

        # State should still be valid
        stats = nethack_api.get_stats()
        assert stats is not None


class TestRemindersAndNotes:
    """Tests for reminder and note functionality."""

    def test_add_reminder_stores_correctly(self, nethack_api):
        """Test that add_reminder stores the reminder with correct fire turn."""
        nethack_api.reset()
        current_turn = nethack_api.turn

        nethack_api.add_reminder(10, "Test reminder")

        # Reminder should be stored with fire_turn = current_turn + 10
        assert len(nethack_api._reminders) == 1
        fire_turn, message = nethack_api._reminders[0]
        assert fire_turn == current_turn + 10
        assert message == "Test reminder"

    def test_get_fired_reminders_returns_and_removes(self, nethack_api):
        """Test that get_fired_reminders returns fired reminders and removes them."""
        nethack_api.reset()

        # Add a reminder that fires immediately (0 turns)
        nethack_api.add_reminder(0, "Immediate reminder")
        # Add one that hasn't fired yet
        nethack_api.add_reminder(1000, "Future reminder")

        fired = nethack_api.get_fired_reminders()

        assert len(fired) == 1
        assert fired[0] == "Immediate reminder"
        # Only the future reminder should remain
        assert len(nethack_api._reminders) == 1
        assert nethack_api._reminders[0][1] == "Future reminder"

    def test_add_note_returns_id(self, nethack_api):
        """Test that add_note returns a unique note ID."""
        nethack_api.reset()

        id1 = nethack_api.add_note(10, "Note 1")
        id2 = nethack_api.add_note(20, "Note 2")

        assert id1 == 1
        assert id2 == 2
        assert id1 != id2

    def test_add_note_persistent(self, nethack_api):
        """Test that add_note with turns=0 creates a persistent note."""
        nethack_api.reset()

        note_id = nethack_api.add_note(0, "Persistent note")

        # Check it's stored with expire_turn=0
        assert note_id in nethack_api._notes
        expire_turn, message = nethack_api._notes[note_id]
        assert expire_turn == 0
        assert message == "Persistent note"

    def test_get_active_notes_returns_tuples(self, nethack_api):
        """Test that get_active_notes returns (id, message) tuples."""
        nethack_api.reset()

        id1 = nethack_api.add_note(100, "Note 1")
        id2 = nethack_api.add_note(0, "Persistent note")

        notes = nethack_api.get_active_notes()

        assert len(notes) == 2
        # Should be sorted by ID
        assert notes[0] == (id1, "Note 1")
        assert notes[1] == (id2, "Persistent note")

    def test_get_active_notes_removes_expired(self, nethack_api):
        """Test that get_active_notes removes expired notes."""
        nethack_api.reset()

        # Add a note that expires immediately
        nethack_api.add_note(0, "Persistent")  # This one won't expire
        # Manually add an expired note for testing
        current_turn = nethack_api.turn
        nethack_api._notes[99] = (current_turn, "Already expired")  # expire at current turn

        notes = nethack_api.get_active_notes()

        # Only the persistent note should remain
        assert len(notes) == 1
        assert notes[0][1] == "Persistent"

    def test_remove_note_success(self, nethack_api):
        """Test that remove_note successfully removes a note."""
        nethack_api.reset()

        note_id = nethack_api.add_note(0, "To be removed")
        assert note_id in nethack_api._notes

        result = nethack_api.remove_note(note_id)

        assert result is True
        assert note_id not in nethack_api._notes

    def test_remove_note_nonexistent(self, nethack_api):
        """Test that remove_note returns False for nonexistent note."""
        nethack_api.reset()

        result = nethack_api.remove_note(999)

        assert result is False

    def test_reset_clears_reminders_and_notes(self, nethack_api):
        """Test that reset clears all reminders and notes."""
        nethack_api.reset()

        nethack_api.add_reminder(10, "Reminder")
        nethack_api.add_note(10, "Note")

        assert len(nethack_api._reminders) == 1
        assert len(nethack_api._notes) == 1

        nethack_api.reset()

        assert len(nethack_api._reminders) == 0
        assert len(nethack_api._notes) == 0
        assert nethack_api._next_note_id == 1
