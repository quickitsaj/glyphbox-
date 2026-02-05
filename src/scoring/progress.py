"""
BALROG progression scoring for NetHack.

Vendored from https://github.com/balrog-ai/BALROG
Original file: balrog/environments/nle/progress.py

The progression metric represents win probability - the likelihood that a human
player who reached the same dungeon level or experience level would go on to
ascend, based on historical human gameplay data.

License: MIT (https://github.com/balrog-ai/BALROG/blob/main/LICENSE)
"""

import json
import os
from dataclasses import dataclass, field

# Load achievements data (win probabilities by dungeon/xp level)
with open(os.path.join(os.path.dirname(__file__), "achievements.json")) as f:
    ACHIEVEMENTS = json.load(f)


@dataclass
class Progress:
    """
    Track BALROG progression for a NetHack game.

    Progression is measured as win probability - the likelihood that a human
    player who reached the same state would go on to ascend.
    """

    score: int = 0
    depth: int = 1
    gold: int = 0
    experience_level: int = 1
    time: int = 0
    dlvl_list: list = field(default_factory=list)
    xplvl_list: list = field(default_factory=list)
    highest_achievement: str | None = None
    progression: float = 0.0

    def update(self, depth: int, experience_level: int) -> float:
        """
        Update progression based on current dungeon depth and XP level.

        Args:
            depth: Current dungeon level (1-50+)
            experience_level: Current experience level (1-30)

        Returns:
            Current progression (0.0 to 1.0)
        """
        self.depth = depth
        self.experience_level = experience_level

        # Check XP level achievement
        xp_key = f"Xp:{experience_level}"
        if xp_key not in self.xplvl_list and xp_key in ACHIEVEMENTS:
            self.xplvl_list.append(xp_key)
            if ACHIEVEMENTS[xp_key] > self.progression:
                self.progression = ACHIEVEMENTS[xp_key]
                self.highest_achievement = xp_key

        # Check dungeon level achievement
        dlvl_key = f"Dlvl:{depth}"
        if dlvl_key not in self.dlvl_list and dlvl_key in ACHIEVEMENTS:
            self.dlvl_list.append(dlvl_key)
            if ACHIEVEMENTS[dlvl_key] > self.progression:
                self.progression = ACHIEVEMENTS[dlvl_key]
                self.highest_achievement = dlvl_key

        return self.progression

    @property
    def progression_percent(self) -> float:
        """Get progression as a percentage (0-100)."""
        return self.progression * 100.0


def calculate_progress(depth: int, experience_level: int) -> float:
    """
    Calculate BALROG progression for given dungeon depth and XP level.

    This is a stateless helper for one-off calculations.

    Args:
        depth: Current dungeon level (1-50+)
        experience_level: Current experience level (1-30)

    Returns:
        Progression as percentage (0.0 to 100.0)
    """
    dlvl_key = f"Dlvl:{depth}"
    xp_key = f"Xp:{experience_level}"

    dlvl_prog = ACHIEVEMENTS.get(dlvl_key, 0.0)
    xp_prog = ACHIEVEMENTS.get(xp_key, 0.0)

    return max(dlvl_prog, xp_prog) * 100.0
