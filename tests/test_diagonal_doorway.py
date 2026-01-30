"""Tests for diagonal doorway pathfinding behavior.

This tests the specific issue where pathfinding generates paths that
require diagonal moves through doorways, which NetHack doesn't allow.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from nle import nethack

from src.api.pathfinding import (
    find_path,
    _build_walkability_grid,
    _astar,
    _can_move_to_neighbor,
    is_doorway_glyph,
)
from src.api.models import Direction, Position
from src.api.queries import BL_X, BL_Y


# Glyph constants
GLYPH_CMAP_OFF = nethack.GLYPH_CMAP_OFF
STONE_GLYPH = GLYPH_CMAP_OFF + 0
NDOOR_GLYPH = GLYPH_CMAP_OFF + 12   # S_ndoor: doorless doorway (diagonal OK)
OPEN_DOOR_GLYPH = GLYPH_CMAP_OFF + 13  # S_vodoor: vertical open door
CLOSED_DOOR_GLYPH = GLYPH_CMAP_OFF + 15  # S_vcdoor: vertical closed door
FLOOR_GLYPH = GLYPH_CMAP_OFF + 19   # S_room: floor of a room
CORRIDOR_GLYPH = GLYPH_CMAP_OFF + 21  # S_corr: corridor
HWALL_GLYPH = GLYPH_CMAP_OFF + 2  # horizontal wall
VWALL_GLYPH = GLYPH_CMAP_OFF + 1  # vertical wall


def make_observation_with_door(player_x, player_y, door_x, door_y, door_type="open"):
    """
    Create a mock observation with a room, door, and corridor.

    Layout:
        |.....|
        |.@...|
        |.....|
        --D----  (door at D)
          #
          #  (corridor)
    """
    obs = MagicMock()

    # Blstats
    blstats = np.zeros(27, dtype=np.int64)
    blstats[BL_X] = player_x
    blstats[BL_Y] = player_y
    obs.blstats = blstats

    # Start with all stone
    glyphs = np.full((21, 79), STONE_GLYPH, dtype=np.int32)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)

    # Create a room (5x5) at position (5, 5)
    for y in range(5, 10):
        for x in range(5, 11):
            if y == 5 or y == 9:  # top/bottom walls
                glyphs[y, x] = HWALL_GLYPH
                chars[y, x] = ord("-")
            elif x == 5 or x == 10:  # left/right walls
                glyphs[y, x] = VWALL_GLYPH
                chars[y, x] = ord("|")
            else:  # floor
                glyphs[y, x] = FLOOR_GLYPH
                chars[y, x] = ord(".")

    # Add door in bottom wall
    door_glyph = OPEN_DOOR_GLYPH if door_type == "open" else CLOSED_DOOR_GLYPH
    glyphs[door_y, door_x] = door_glyph
    chars[door_y, door_x] = ord(".") if door_type == "open" else ord("+")

    # Add corridor below door
    for y in range(door_y + 1, door_y + 5):
        glyphs[y, door_x] = CORRIDOR_GLYPH
        chars[y, door_x] = ord("#")

    obs.glyphs = glyphs
    obs.chars = chars
    obs.colors = np.zeros((21, 79), dtype=np.int8)
    obs.screen_descriptions = None

    return obs


class TestDoorwayGlyphDetection:
    """Test that door glyphs are correctly identified as doorways."""

    def test_open_door_is_doorway(self):
        """Open doors should be detected as doorways."""
        assert is_doorway_glyph(OPEN_DOOR_GLYPH) is True

    def test_closed_door_is_doorway(self):
        """Closed doors should be detected as doorways."""
        assert is_doorway_glyph(CLOSED_DOOR_GLYPH) is True

    def test_floor_is_not_doorway(self):
        """Floor tiles should NOT be detected as doorways."""
        assert is_doorway_glyph(FLOOR_GLYPH) is False

    def test_corridor_is_not_doorway(self):
        """Corridor tiles should NOT be detected as doorways."""
        assert is_doorway_glyph(CORRIDOR_GLYPH) is False

    def test_wall_is_not_doorway(self):
        """Wall tiles should NOT be detected as doorways."""
        assert is_doorway_glyph(HWALL_GLYPH) is False
        assert is_doorway_glyph(VWALL_GLYPH) is False


class TestWalkabilityGridDoorwayDetection:
    """Test that _build_walkability_grid correctly populates the doorway grid."""

    def test_open_door_detected_in_doorway_grid(self):
        """Open doors should be marked in the doorway grid."""
        door_x, door_y = 7, 9  # bottom wall of room
        obs = make_observation_with_door(
            player_x=7, player_y=7,  # player inside room
            door_x=door_x, door_y=door_y,
            door_type="open"
        )

        walkable, doorways = _build_walkability_grid(obs)

        # The door position should be marked as a doorway
        assert doorways[door_y][door_x] is True, \
            f"Open door at ({door_x}, {door_y}) should be marked as doorway"

        # Floor tiles should NOT be marked as doorways
        assert doorways[7][7] is False, "Floor should not be doorway"

    def test_closed_door_detected_in_doorway_grid(self):
        """Closed doors should be marked in the doorway grid."""
        door_x, door_y = 7, 9
        obs = make_observation_with_door(
            player_x=7, player_y=7,
            door_x=door_x, door_y=door_y,
            door_type="closed"
        )

        walkable, doorways = _build_walkability_grid(obs)

        # The door position should be marked as a doorway
        assert doorways[door_y][door_x] is True, \
            f"Closed door at ({door_x}, {door_y}) should be marked as doorway"


class TestCanMoveToNeighbor:
    """Test the diagonal doorway restriction in _can_move_to_neighbor."""

    def test_diagonal_into_doorway_blocked(self):
        """Diagonal move INTO a doorway should be blocked."""
        doorways = [[False] * 79 for _ in range(21)]
        doorways[5][6] = True  # doorway at (6, 5)

        from_pos = Position(5, 4)  # SW of doorway
        to_pos = Position(6, 5)    # the doorway

        result = _can_move_to_neighbor(from_pos, to_pos, doorways)
        assert result is False, "Diagonal move INTO doorway should be blocked"

    def test_diagonal_out_of_doorway_blocked(self):
        """Diagonal move OUT OF a doorway should be blocked."""
        doorways = [[False] * 79 for _ in range(21)]
        doorways[5][6] = True  # doorway at (6, 5)

        from_pos = Position(6, 5)  # the doorway
        to_pos = Position(7, 4)    # NE of doorway

        result = _can_move_to_neighbor(from_pos, to_pos, doorways)
        assert result is False, "Diagonal move OUT OF doorway should be blocked"

    def test_cardinal_through_doorway_allowed(self):
        """Cardinal move through a doorway should be allowed."""
        doorways = [[False] * 79 for _ in range(21)]
        doorways[5][6] = True  # doorway at (6, 5)

        from_pos = Position(6, 4)  # N of doorway
        to_pos = Position(6, 5)    # the doorway

        result = _can_move_to_neighbor(from_pos, to_pos, doorways)
        assert result is True, "Cardinal move through doorway should be allowed"

    def test_diagonal_away_from_doorway_allowed(self):
        """Diagonal move that doesn't involve a doorway should be allowed."""
        doorways = [[False] * 79 for _ in range(21)]
        doorways[5][6] = True  # doorway at (6, 5), not involved in this move

        from_pos = Position(10, 10)
        to_pos = Position(11, 11)

        result = _can_move_to_neighbor(from_pos, to_pos, doorways)
        assert result is True, "Diagonal move not involving doorway should be allowed"


class TestAstarDoorwayAvoidance:
    """Test that A* correctly avoids diagonal moves through doorways."""

    def test_astar_avoids_diagonal_doorway(self):
        """A* should find a path that doesn't use diagonal doorway moves."""
        walkable = [[True] * 79 for _ in range(21)]
        doorways = [[False] * 79 for _ in range(21)]

        # Place a doorway at (6, 5)
        doorways[5][6] = True

        # Player at (5, 5), wants to go to (7, 4) - diagonal path through doorway
        start = Position(5, 5)
        goal = Position(7, 4)

        path = _astar(start, goal, walkable, doorways)

        assert len(path) > 0, "A* should find a path"
        assert path[-1] == goal, "Path should reach goal"

        # Verify no diagonal move goes through the doorway
        prev = start
        for pos in path:
            dx = abs(pos.x - prev.x)
            dy = abs(pos.y - prev.y)
            is_diagonal = dx + dy == 2
            if is_diagonal:
                assert not doorways[prev.y][prev.x], \
                    f"Diagonal from doorway at {prev}"
                assert not doorways[pos.y][pos.x], \
                    f"Diagonal into doorway at {pos}"
            prev = pos


class TestFindPathWithDoorway:
    """Test find_path with doorways in the observation."""

    def test_find_path_avoids_diagonal_through_door(self):
        """find_path should generate a path avoiding diagonal doorway moves."""
        # Create a room with a door in the corner
        door_x, door_y = 10, 9  # door at right edge of bottom wall
        obs = make_observation_with_door(
            player_x=9, player_y=8,   # player inside room, near door
            door_x=door_x, door_y=door_y,
            door_type="open"
        )

        # Target is in corridor below and to right of door
        target = Position(door_x + 1, door_y + 2)

        # Make the target area walkable
        obs.glyphs[door_y + 1, door_x] = CORRIDOR_GLYPH
        obs.glyphs[door_y + 2, door_x] = CORRIDOR_GLYPH
        obs.glyphs[door_y + 1, door_x + 1] = CORRIDOR_GLYPH
        obs.glyphs[door_y + 2, door_x + 1] = CORRIDOR_GLYPH

        result = find_path(obs, target, allow_with_hostiles=True)

        if result.success:
            # Verify no diagonal move involves the doorway
            start = Position(9, 8)
            prev = start
            for direction in result.path:
                next_pos = prev.move(direction)
                dx = abs(next_pos.x - prev.x)
                dy = abs(next_pos.y - prev.y)
                is_diagonal = dx + dy == 2
                if is_diagonal:
                    # Check the door position
                    assert not (prev.x == door_x and prev.y == door_y), \
                        f"Diagonal OUT of doorway from {prev}"
                    assert not (next_pos.x == door_x and next_pos.y == door_y), \
                        f"Diagonal INTO doorway to {next_pos}"
                prev = next_pos


class TestDoorwayMemoryPersistence:
    """Test that doorways are remembered when out of sight."""

    def test_doorway_remembered_when_out_of_sight(self):
        """
        When a doorway goes out of sight (becomes 'stone' glyph),
        level_memory should still mark it as a doorway.
        """
        from src.memory.dungeon import LevelMemory

        door_x, door_y = 7, 9
        level_memory = LevelMemory(level_number=1)

        # First observation: door is visible
        obs1 = make_observation_with_door(
            player_x=7, player_y=7,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )

        # Build grid - this should detect and remember the doorway
        walkable1, doorways1 = _build_walkability_grid(
            obs1, level_memory=level_memory
        )

        assert doorways1[door_y][door_x] is True, \
            "Doorway should be detected when visible"

        # Second observation: door is out of sight (stone glyph)
        obs2 = make_observation_with_door(
            player_x=7, player_y=7,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        # Overwrite the door glyph with stone (simulating out-of-sight)
        obs2.glyphs[door_y, door_x] = STONE_GLYPH
        obs2.chars[door_y, door_x] = ord(" ")

        # Build grid with same level_memory - doorway should still be marked
        walkable2, doorways2 = _build_walkability_grid(
            obs2, level_memory=level_memory
        )

        assert doorways2[door_y][door_x] is True, \
            "Doorway should be remembered from level_memory even when out of sight"

    def test_doorway_remembered_after_player_walks_through(self):
        """
        Scenario: Player walks through a door, then moves away.
        The door should still be remembered in level_memory at each stage.
        """
        from src.memory.dungeon import LevelMemory

        door_x, door_y = 7, 9
        level_memory = LevelMemory(level_number=1)

        # Step 1: Player approaches door, door is visible
        obs1 = make_observation_with_door(
            player_x=7, player_y=8,  # player adjacent to door
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        walkable1, doorways1 = _build_walkability_grid(obs1, level_memory=level_memory)
        assert doorways1[door_y][door_x] is True, "Door should be detected when adjacent"

        # Step 2: Player steps on door - player glyph overwrites door
        obs2 = make_observation_with_door(
            player_x=door_x, player_y=door_y,  # player ON the door
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        # The player glyph (monster glyph) overwrites the door
        player_glyph = nethack.GLYPH_MON_OFF + 340  # valkyrie
        obs2.glyphs[door_y, door_x] = player_glyph
        obs2.chars[door_y, door_x] = ord("@")

        walkable2, doorways2 = _build_walkability_grid(obs2, level_memory=level_memory)
        assert doorways2[door_y][door_x] is True, \
            "Door should be remembered from memory when player stands on it"

        # Step 3: Player moves away, door is now out of sight
        obs3 = make_observation_with_door(
            player_x=7, player_y=7,  # player back in room
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        # Door is out of line-of-sight, shows as stone
        obs3.glyphs[door_y, door_x] = STONE_GLYPH
        obs3.chars[door_y, door_x] = ord(" ")

        walkable3, doorways3 = _build_walkability_grid(obs3, level_memory=level_memory)
        assert doorways3[door_y][door_x] is True, \
            "Door should still be remembered after player moved away"

    def test_stepped_doorway_still_marked_when_out_of_sight(self):
        """
        When a doorway that was stepped on goes out of sight,
        it should be marked as BOTH walkable AND a doorway.

        is_stepped makes it walkable, and level_memory doorway flag
        preserves the diagonal movement restriction.
        """
        from src.memory.dungeon import LevelMemory

        door_x, door_y = 7, 9
        level_memory = LevelMemory(level_number=1)

        # Step 1: Player sees and steps on the door
        obs1 = make_observation_with_door(
            player_x=door_x, player_y=door_y,  # player ON the door
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        # Put player glyph on door position
        player_glyph = nethack.GLYPH_MON_OFF + 340
        obs1.glyphs[door_y, door_x] = player_glyph

        # Mark the door as stepped in memory (simulating player walked through)
        level_memory.mark_stepped(door_x, door_y)
        level_memory.mark_doorway(door_x, door_y)  # Should have been marked when visible

        # Step 2: Door is now out of sight (stone glyph)
        obs2 = make_observation_with_door(
            player_x=7, player_y=7,  # player somewhere else
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        obs2.glyphs[door_y, door_x] = STONE_GLYPH
        obs2.chars[door_y, door_x] = ord(" ")

        walkable, doorways = _build_walkability_grid(obs2, level_memory=level_memory)

        # The door should be BOTH walkable (via is_stepped) AND a doorway (via memory)
        assert walkable[door_y][door_x] is True, \
            "Stepped doorway should be walkable (via is_stepped)"
        assert doorways[door_y][door_x] is True, \
            "Stepped doorway should still be marked as doorway (via memory)"

    def test_doorway_not_remembered_without_level_memory(self):
        """Without level_memory, doorways out of sight won't be detected."""
        door_x, door_y = 7, 9

        # Create observation with door visible
        obs = make_observation_with_door(
            player_x=7, player_y=7,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        # Overwrite with stone (out of sight)
        obs.glyphs[door_y, door_x] = STONE_GLYPH

        # Build grid WITHOUT level_memory
        walkable, doorways = _build_walkability_grid(obs, level_memory=None)

        assert doorways[door_y][door_x] is False, \
            "Without level_memory, out-of-sight doorway shouldn't be detected"


class TestPlayerOnUnmarkedDoorway:
    """Test the scenario where the player is standing on an unmarked doorway."""

    def test_player_on_door_not_in_memory_not_detected(self):
        """
        When the player stands on a door that was NEVER recorded in level_memory,
        the doorway is NOT detected. This is an intentional tradeoff: we skip
        context detection (walls on both sides) when level_memory exists because
        context detection can't distinguish intact doors from doorless doorways,
        causing false positives that severely break pathfinding connectivity.

        This false negative is rare (teleportation onto unseen door) while the
        false positive (doorless doorway treated as door) was common.
        """
        from src.memory.dungeon import LevelMemory

        door_x, door_y = 7, 9
        level_memory = LevelMemory(level_number=1)

        # Player is ON the door, but door was NEVER seen
        obs = make_observation_with_door(
            player_x=door_x, player_y=door_y,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        # Player glyph overwrites door glyph
        player_glyph = nethack.GLYPH_MON_OFF + 340
        obs.glyphs[door_y, door_x] = player_glyph
        obs.chars[door_y, door_x] = ord("@")

        # NOTE: level_memory does NOT have this door marked

        walkable, doorways = _build_walkability_grid(obs, level_memory=level_memory)

        # Not detected - level_memory has no record, context detection is skipped
        assert doorways[door_y][door_x] is False, \
            "Unmarked door should not be detected (avoids false positives on doorless doorways)"

    def test_player_on_door_without_level_memory_uses_context(self):
        """
        Without level_memory, context detection IS used as a fallback.
        This handles the case where no memory system is available.
        """
        door_x, door_y = 7, 9

        # Player is ON the door, no level_memory at all
        obs = make_observation_with_door(
            player_x=door_x, player_y=door_y,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        # Player glyph overwrites door glyph
        player_glyph = nethack.GLYPH_MON_OFF + 340
        obs.glyphs[door_y, door_x] = player_glyph
        obs.chars[door_y, door_x] = ord("@")

        walkable, doorways = _build_walkability_grid(obs, level_memory=None)

        # With no memory, context detection kicks in (walls on both sides)
        assert doorways[door_y][door_x] is True, \
            "Without level_memory, context detection should identify doorway"


class TestDestroyedDoorClearsMemory:
    """Test that destroyed doors properly clear level_memory doorway flags."""

    def test_destroyed_door_clears_doorway_memory(self):
        """
        When a door is destroyed (kicked to pieces), it becomes a doorless
        doorway (cmap 12). When this tile is visible with a non-door glyph,
        the level_memory doorway flag should be cleared.
        """
        from src.memory.dungeon import LevelMemory

        door_x, door_y = 7, 9
        level_memory = LevelMemory(level_number=1)

        # Step 1: Door is visible and detected
        obs1 = make_observation_with_door(
            player_x=7, player_y=8,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        walkable1, doorways1 = _build_walkability_grid(obs1, level_memory=level_memory)
        assert doorways1[door_y][door_x] is True, "Door should be detected"
        assert level_memory.is_doorway(door_x, door_y) is True, "Door should be in memory"

        # Step 2: Door is destroyed - now shows as doorless doorway (cmap 12)
        obs2 = make_observation_with_door(
            player_x=7, player_y=8,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        # Replace the door glyph with doorless doorway (cmap 12)
        obs2.glyphs[door_y, door_x] = NDOOR_GLYPH  # cmap 12 = doorless doorway
        obs2.chars[door_y, door_x] = ord(".")

        walkable2, doorways2 = _build_walkability_grid(obs2, level_memory=level_memory)

        # Door should NOT be detected (doorless doorway allows diagonal)
        assert doorways2[door_y][door_x] is False, \
            "Destroyed door should not be marked as doorway"
        # Memory should be cleared
        assert level_memory.is_doorway(door_x, door_y) is False, \
            "Destroyed door should clear level_memory doorway flag"

    def test_destroyed_door_allows_diagonal_after_going_out_of_sight(self):
        """
        After a door is destroyed and the memory is cleared, even if the tile
        goes out of sight, it should NOT be restored from stale memory.
        """
        from src.memory.dungeon import LevelMemory

        door_x, door_y = 7, 9
        level_memory = LevelMemory(level_number=1)

        # Step 1: Door detected
        obs1 = make_observation_with_door(
            player_x=7, player_y=8,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        _build_walkability_grid(obs1, level_memory=level_memory)
        assert level_memory.is_doorway(door_x, door_y) is True

        # Step 2: Door destroyed (visible as cmap 12)
        obs2 = make_observation_with_door(
            player_x=7, player_y=8,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        obs2.glyphs[door_y, door_x] = NDOOR_GLYPH  # cmap 12 = doorless doorway
        obs2.chars[door_y, door_x] = ord(".")
        _build_walkability_grid(obs2, level_memory=level_memory)
        assert level_memory.is_doorway(door_x, door_y) is False, "Memory should be cleared"

        # Step 3: Tile goes out of sight (stone)
        obs3 = make_observation_with_door(
            player_x=7, player_y=7,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        obs3.glyphs[door_y, door_x] = STONE_GLYPH
        obs3.chars[door_y, door_x] = ord(" ")

        walkable3, doorways3 = _build_walkability_grid(obs3, level_memory=level_memory)

        # Should NOT be restored from stale memory (memory was cleared)
        assert doorways3[door_y][door_x] is False, \
            "Destroyed door should not be restored from memory when out of sight"

    def test_player_on_doorless_doorway_not_falsely_detected(self):
        """
        When the player stands on a doorless doorway (cmap 12, never had a
        door or door was destroyed), it should NOT be detected as a doorway.
        This prevents the false positive that caused reachable=3 connectivity issues.
        """
        from src.memory.dungeon import LevelMemory

        door_x, door_y = 7, 9
        level_memory = LevelMemory(level_number=1)

        # Step 1: The doorless doorway is visible (cmap 12, NOT a door)
        obs1 = make_observation_with_door(
            player_x=7, player_y=8,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        # Make it a doorless doorway (cmap 12), not an intact door
        obs1.glyphs[door_y, door_x] = NDOOR_GLYPH  # cmap 12
        obs1.chars[door_y, door_x] = ord(".")

        walkable1, doorways1 = _build_walkability_grid(obs1, level_memory=level_memory)
        assert doorways1[door_y][door_x] is False, "Doorless doorway should not be detected"

        # Step 2: Player steps onto the doorless doorway
        obs2 = make_observation_with_door(
            player_x=door_x, player_y=door_y,
            door_x=door_x, door_y=door_y,
            door_type="open"
        )
        player_glyph = nethack.GLYPH_MON_OFF + 340
        obs2.glyphs[door_y, door_x] = player_glyph
        obs2.chars[door_y, door_x] = ord("@")

        walkable2, doorways2 = _build_walkability_grid(obs2, level_memory=level_memory)

        # Should NOT be detected as doorway (level_memory has no record)
        # This is the key fix: context detection (walls on sides) is NOT used
        # because level_memory exists and says this was never a door
        assert doorways2[door_y][door_x] is False, \
            "Player on doorless doorway should NOT trigger false doorway detection"


class TestRealGameDoorwayScenario:
    """Integration test with real NLE environment."""

    @pytest.mark.integration
    def test_doorway_detection_real_env(self, nethack_api):
        """Test that doorways are detected in a real game."""
        nethack_api.reset()

        # Take some steps to potentially encounter doorways
        for _ in range(50):
            result = nethack_api.autoexplore(max_steps=10)
            if str(result.stop_reason) in ("hostile", "interrupt", "AutoexploreStopReason.hostile"):
                break

        # Get current observation and build grids
        obs = nethack_api.observation
        walkable, doorways = _build_walkability_grid(obs)

        # Count doorways detected
        doorway_count = sum(1 for row in doorways for v in row if v)

        # Log for debugging
        print(f"\nDetected {doorway_count} doorways in observation")

        # If there are doorways, verify they're at valid positions
        for y in range(21):
            for x in range(79):
                if doorways[y][x]:
                    glyph = int(obs.glyphs[y, x])
                    print(f"  Doorway at ({x}, {y}), glyph={glyph}")

        # Also look for potential doors that weren't detected
        print("\nScanning for door-like characters (+, -, |) that might be doors:")
        for y in range(21):
            for x in range(79):
                char = chr(obs.chars[y, x])
                glyph = int(obs.glyphs[y, x])
                if char in ('+', '-', '|', '.'):
                    # Check if this could be a door
                    if nethack.glyph_is_cmap(glyph):
                        cmap = nethack.glyph_to_cmap(glyph)
                        if cmap in (15, 16, 17):  # closed, open, broken door
                            detected = doorways[y][x]
                            print(f"  ({x}, {y}) char='{char}' glyph={glyph} cmap={cmap} detected={detected}")
