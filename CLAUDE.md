# CLAUDE.md

## Project Overview

Self-programming NetHack agent that uses an LLM to write and execute Python code to play NetHack. The LLM uses the `execute_code` tool call to interact with the game through a high-level Python API. The full game screen is automatically provided before each turn.

## Commands

```bash
# Install dependencies
uv sync

# Run agent in TUI watch mode
uv run python -m src.cli watch

# Record TUI session with asciinema
uv run python -m src.cli watch --record

# Use a different model (cheaper for testing)
uv run python -m src.cli watch --model anthropic/claude-3-haiku-20240307

# Verify setup
uv run python -m src.cli verify

# Run tests (skips integration tests by default)
uv run pytest

# Run specific test file
uv run pytest tests/test_agent_agent.py -v

# Run integration tests (requires API key)
uv run pytest -m integration

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Architecture

```
LLM (strategic layer)
    │ receives: full game screen each turn
    │ tool calls: execute_code
    ▼
SkillSandbox (src/sandbox/manager.py)
    │ validates & executes Python code
    ▼
NetHackAPI (src/api/nethack_api.py)
    │ high-level game interface
    ▼
NLE (NetHack Learning Environment)
```

### Core Components

- **`src/cli.py`**: Entry point. Commands: `watch` (TUI mode), `verify`
- **`src/agent/agent.py`**: `NetHackAgent` class - main orchestration loop. Calls `step()` repeatedly which gets LLM decision and executes it
- **`src/agent/llm_client.py`**: `LLMClient` using OpenAI-compatible API (OpenRouter). Defines tools in `CORE_TOOLS` and `SKILL_TOOLS`
- **`src/sandbox/manager.py`**: `SkillSandbox.execute_code()` runs agent-generated Python in restricted namespace with `nh` (NetHackAPI) available
- **`src/api/nethack_api.py`**: High-level NetHack interface wrapping NLE observations/actions
- **`src/tui/app.py`**: Textual TUI for watching agent play. Keybindings: S=start, Space=pause, Q=quit

### Data Flow

1. `NetHackAgent.step()` builds game state context from `EpisodeMemory`
2. LLM receives context and returns tool call (`execute_code` with Python code)
3. `SkillSandbox.execute_code()` validates code (AST checks, forbidden imports) and runs it
4. Code has access to `nh` (NetHackAPI) and `Direction` enum - all calls are synchronous
5. Results (including game messages, failed API calls) returned to agent for next LLM context

### LLM Context Structure

The agent receives context in this structure:

**System Prompt** (once per session):
- Full NetHackAPI documentation (~4KB) in `SYSTEM_PROMPT_NO_SKILLS`

**Message History** (per turn, managed by `_build_messages_with_compression`):
- Old user messages → Compressed to just `Last Result:` section (game screen stripped)
- Old assistant messages → Kept in full (contain code + reasoning)
- Controlled by `agent.max_history_turns`: 0 = unlimited (compress old), N = sliding window

**Current User Message** (full content each turn):
```
=== CURRENT GAME VIEW ===
Your position: (x, y)
{24x80 ASCII game screen with status bar}

Last Result:
success: {bool}
game_messages:
  - You hit the goblin!
  - The goblin hits! (x3)
failed_api_calls:
  - move() FAILED: blocked by boulder (x5)

What action do you want to take?
```

The game screen comes FIRST so the agent sees spatial context before text feedback. Position is explicitly stated to anchor the agent. Repeated messages are deduplicated with counts (e.g., `(x3)`).

### Sandbox Security

Code validation in `src/sandbox/validation.py`:
- Forbidden imports: `os`, `subprocess`, `socket`, `sys`, etc.
- Forbidden calls: `exec`, `eval`, `compile`, `__import__`, `open`
- Limited builtins whitelist
- Signal-based timeout for infinite loops

## Configuration

`config/default.yaml` - key settings:
- `agent.provider`: "openrouter" or "anthropic"
- `agent.model`: model identifier
- `agent.skills_enabled`: false (core tools only) or true (adds write_skill/invoke_skill)
- `agent.max_history_turns`: 0 = unlimited history (compress old), N = sliding window of N turns

Environment variables:
- `OPENROUTER_KEY` or `OPENROUTER_API_KEY`: Required for OpenRouter
- `NETHACK_AGENT_MODEL`: Override model
- `NETHACK_AGENT_LOG_LEVEL`: Override log level

## Logs & Recordings

**Logs** are stored in `data/logs/` with filenames like `run_2026-01-19_19-34-49.log`.

Logs include:
- LLM requests/responses (prompts sent, tool calls received)
- Game state each turn (HP, position, dungeon level)
- Agent decisions and reasoning
- API call results (successes and failures)
- Pathfinding debug info

**Recordings** (when using `--record`) are stored in `data/recordings/` as `.cast` files.

```bash
# Playback a recording
asciinema play data/recordings/run_2026-01-19_19-34-49.cast

# Upload to asciinema.org for sharing
asciinema upload data/recordings/run_2026-01-19_19-34-49.cast
```

## Key Patterns

### Agent Code Execution

Code runs in sandbox with `nh` available. All API calls are synchronous (no await):
```python
# Example execute_code content
stats = nh.get_stats()
if stats.hp < stats.max_hp * 0.3:
    nh.pray()
else:
    nh.move(Direction.E)
```

### NetHackAPI Methods

State queries: `get_stats()`, `get_position()`, `get_visible_monsters()`, `get_inventory()`, `get_items_here()`, `get_message()`

Actions: `move(direction)`, `attack(direction)`, `pickup()`, `eat()`, `quaff(slot)`, `wear(slot)`, `wield(slot)`

Pathfinding: `find_path(target)`, `find_unexplored()`, `find_stairs_down()`, `autoexplore()`

### Test Markers

- Default: unit tests only (skips `integration` marker)
- `@pytest.mark.integration`: requires API key, tests real game scenarios
