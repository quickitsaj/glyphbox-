"""Stats bar widget showing HP, Turn, Level, Score, Hunger, BALROG Progress."""

from rich.text import Text
from textual.widgets import Static

from src.scoring import Progress

from ..events import GameStateUpdated


class StatsBar(Static):
    """
    Horizontal status bar showing key stats.

    Format: HP: 15/15 | Turn: 1234 | DL:1 | XL:1 | BALROG: 0.00% | Hungry
    """

    DEFAULT_CSS = """
    StatsBar {
        background: $surface;
        padding: 0 1;
        height: 3;
        border: solid $primary;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hp = 0
        self._max_hp = 0
        self._turn = 0
        self._level = 1
        self._xp_level = 1
        self._score = 0
        self._hunger = "Not Hungry"
        self._message = ""
        self._progress = Progress()  # BALROG progress tracker

    def on_mount(self) -> None:
        """Initialize display when mounted."""
        self._refresh_display()

    def on_game_state_updated(self, event: GameStateUpdated) -> None:
        """Update stats display."""
        self._hp = event.hp
        self._max_hp = event.max_hp
        self._turn = event.turn
        self._level = event.dungeon_level
        self._xp_level = event.xp_level
        self._score = event.score
        self._hunger = event.hunger
        self._message = event.message
        # Update BALROG progress using absolute depth (not branch-relative level)
        self._progress.update(event.depth, self._xp_level)
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Rebuild the stats display."""
        text = Text()

        # HP with color coding
        hp_ratio = self._hp / self._max_hp if self._max_hp > 0 else 0
        if hp_ratio > 0.5:
            hp_color = "green"
        elif hp_ratio > 0.25:
            hp_color = "yellow"
        else:
            hp_color = "red bold"

        text.append("HP: ", style="bold")
        text.append(f"{self._hp}/{self._max_hp}", style=hp_color)

        text.append(" | Turn: ", style="dim")
        text.append(f"{self._turn}")

        text.append(" | DL:", style="dim")
        text.append(f"{self._level}")

        text.append(" | XL:", style="dim")
        text.append(f"{self._xp_level}")

        # BALROG progress (win probability)
        balrog_pct = self._progress.progression_percent
        text.append(" | BALROG: ", style="dim")
        if balrog_pct >= 50:
            balrog_style = "green bold"
        elif balrog_pct >= 10:
            balrog_style = "cyan"
        elif balrog_pct > 0:
            balrog_style = "yellow"
        else:
            balrog_style = "white"
        text.append(f"{balrog_pct:.2f}%", style=balrog_style)

        # Hunger status with color
        hunger_colors = {
            "satiated": "cyan",
            "not_hungry": "green",
            "hungry": "yellow",
            "weak": "red",
            "fainting": "red bold",
            "fainted": "red bold",
        }
        text.append(" | ", style="dim")
        hunger_style = hunger_colors.get(self._hunger.lower().replace(" ", "_"), "white")
        text.append(self._hunger, style=hunger_style)

        # Current message on second line
        if self._message:
            msg_display = self._message[:76] if len(self._message) > 76 else self._message
            text.append(f"\n{msg_display}", style="italic")

        self.update(text)
