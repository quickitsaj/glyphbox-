"""
Test what feedback appears in Last Result for various actions.

This helps understand why agents might repeat themselves -
if Last Result shows "None" for successful actions, the agent
has no feedback that its action worked.
"""

import pytest
import asyncio

from src.sandbox.manager import SkillSandbox, SandboxConfig


class TestLastResultFeedback:
    """Test what feedback different actions produce."""

    @pytest.fixture
    def sandbox(self):
        return SkillSandbox(SandboxConfig(timeout_seconds=10.0))

    @pytest.fixture
    def api(self, nethack_api):
        nethack_api.reset()
        return nethack_api

    @pytest.mark.asyncio
    async def test_successful_move_feedback(self, sandbox, api):
        """
        What feedback does a successful move produce?

        If this returns empty game_messages, the agent would see
        "Last Result: None" and not know the move succeeded.
        """
        api._message_history = []

        # Try to move in each direction until one succeeds
        code = """
for d in [Direction.N, Direction.S, Direction.E, Direction.W]:
    result = nh.move(d)
    if result.success:
        print(f"Moved {d.name} successfully")
        break
"""
        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== SUCCESSFUL MOVE FEEDBACK ===")
        print(f"result.success: {result.success}")
        print(f"result.result: {result.result}")

        game_messages = result.result.get('game_messages', []) if result.result else []
        failed_calls = result.result.get('failed_api_calls', []) if result.result else []
        stdout = result.result.get('stdout', '') if result.result else ''

        print(f"game_messages: {game_messages}")
        print(f"failed_api_calls: {failed_calls}")
        print(f"stdout: {stdout}")

        # The question: does a successful move produce any messages?
        if not game_messages and not stdout:
            print("\n*** PROBLEM: Successful move produces NO feedback! ***")
            print("Agent would see 'Last Result: None' and not know it moved.")

        assert result.success

    @pytest.mark.asyncio
    async def test_successful_move_no_print(self, sandbox, api):
        """
        What does the agent see when a move succeeds but nothing is printed?

        This simulates realistic agent code that just moves without printing.
        """
        api._message_history = []

        # Try to get a successful move - try all directions
        code = """
# Record starting position
start = nh.get_position()

# Try all directions until we successfully move
for d in [Direction.N, Direction.S, Direction.E, Direction.W, Direction.NE, Direction.NW, Direction.SE, Direction.SW]:
    result = nh.move(d)
    new_pos = nh.get_position()
    if new_pos != start:
        # We successfully moved! Now the question is: what feedback do we get?
        break
"""
        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== SUCCESSFUL MOVE (NO PRINT) ===")
        print(f"result.success: {result.success}")
        print(f"result.result: {result.result}")

        game_messages = result.result.get('game_messages', []) if result.result else []
        failed_calls = result.result.get('failed_api_calls', []) if result.result else []
        stdout = result.result.get('stdout', '') if result.result else ''

        print(f"game_messages: {game_messages}")
        print(f"failed_api_calls: {failed_calls}")
        print(f"stdout: {stdout!r}")

        # Now simulate what agent sees
        from src.agent.prompts import PromptManager
        pm = PromptManager()

        last_result = {
            "tool": "execute_code",
            "success": result.success,
            "error": result.error,
            "messages": game_messages,
            "failed_api_calls": failed_calls,
        }
        if result.stdout:
            last_result["output"] = result.stdout

        result_text = pm.format_last_result(last_result)
        prompt = pm.format_decision_prompt(saved_skills=[], last_result_text=result_text)
        idx = prompt.find("Last Result:")
        last_result_section = prompt[idx:prompt.find("\n\nStudy")] if idx >= 0 else "NOT FOUND"

        print(f"\n=== WHAT AGENT SEES ===")
        print(last_result_section)

        if last_result_section.strip() == "Last Result:\nNone":
            print("\n*** CONFIRMED: Agent sees 'Last Result: None' for successful move! ***")
            print("This is why the agent repeats itself - no feedback that action worked!")

        assert result.success

    @pytest.mark.asyncio
    async def test_wait_now_produces_feedback(self, sandbox, api):
        """
        wait() always succeeds but produces no game message.

        With the new APICallTracker, wait() should now show feedback!
        """
        api._message_history = []

        code = """
nh.wait()
"""
        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== WAIT() FEEDBACK ===")
        print(f"result.success: {result.success}")
        print(f"result.result: {result.result}")

        game_messages = result.result.get('game_messages', []) if result.result else []
        api_calls = result.result.get('api_calls', []) if result.result else []

        print(f"game_messages: {game_messages}")
        print(f"api_calls: {api_calls}")

        # Now simulate what agent sees - using the NEW api_calls field
        from src.agent.prompts import PromptManager
        pm = PromptManager()

        last_result = {
            "tool": "execute_code",
            "success": result.success,
            "error": result.error,
            "messages": game_messages,
            "api_calls": api_calls,  # Use new field
        }

        result_text = pm.format_last_result(last_result)
        prompt = pm.format_decision_prompt(saved_skills=[], last_result_text=result_text)
        idx = prompt.find("Last Result:")
        end_idx = prompt.find("\n\nStudy")
        last_result_section = prompt[idx:end_idx] if idx >= 0 and end_idx >= 0 else prompt[idx:]

        print(f"\n=== WHAT AGENT SEES FOR wait() ===")
        print(last_result_section)

        # Now wait() should produce feedback!
        if "wait()" in last_result_section and "ok" in last_result_section:
            print("\n*** SUCCESS: wait() now produces feedback! ***")
        else:
            print("\n*** PROBLEM: wait() still not producing feedback ***")

        assert result.success
        assert api_calls, "api_calls should contain the wait() call"
        assert api_calls[0]['method'] == 'wait'
        assert api_calls[0]['success'] == True

    @pytest.mark.asyncio
    async def test_failed_move_feedback(self, sandbox, api):
        """What feedback does a failed move (into wall) produce?"""
        api._message_history = []

        # Move north many times until we hit a wall
        code = """
for _ in range(20):
    nh.move(Direction.N)
"""
        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== FAILED MOVE FEEDBACK ===")
        print(f"result.result: {result.result}")

        game_messages = result.result.get('game_messages', []) if result.result else []
        failed_calls = result.result.get('failed_api_calls', []) if result.result else []

        print(f"game_messages: {game_messages}")
        print(f"failed_api_calls: {failed_calls}")

        # Failed moves should produce some feedback
        if not game_messages and not failed_calls:
            print("\n*** PROBLEM: Failed move produces NO feedback! ***")

        assert result.success

    @pytest.mark.asyncio
    async def test_autoexplore_feedback(self, sandbox, api):
        """What feedback does autoexplore produce?"""
        api._message_history = []

        code = """
result = nh.autoexplore(max_steps=10)
print(f"Autoexplore: {result.stop_reason}, {result.steps_taken} steps")
"""
        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== AUTOEXPLORE FEEDBACK ===")
        print(f"result.result: {result.result}")

        autoexplore = result.result.get('autoexplore_result', {}) if result.result else {}
        game_messages = result.result.get('game_messages', []) if result.result else []
        stdout = result.result.get('stdout', '') if result.result else ''

        print(f"autoexplore_result: {autoexplore}")
        print(f"game_messages: {game_messages}")
        print(f"stdout: {stdout}")

        assert result.success

    @pytest.mark.asyncio
    async def test_get_stats_feedback(self, sandbox, api):
        """What feedback does querying stats produce?"""
        api._message_history = []

        code = """
stats = nh.get_stats()
print(f"HP: {stats.hp}/{stats.max_hp}")
"""
        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== GET_STATS FEEDBACK ===")
        print(f"result.result: {result.result}")

        # Queries don't produce game messages, only stdout from print
        game_messages = result.result.get('game_messages', []) if result.result else []
        stdout = result.result.get('stdout', '') if result.result else ''

        print(f"game_messages: {game_messages}")
        print(f"stdout: {stdout}")

        if not game_messages and not stdout:
            print("\n*** Query-only code produces NO feedback! ***")

        assert result.success

    @pytest.mark.asyncio
    async def test_what_agent_sees_as_last_result(self, sandbox, api):
        """
        Simulate what the agent sees as "Last Result:" after various actions.

        This is the critical test - does the agent get useful feedback?
        """
        from src.agent.prompts import PromptManager

        pm = PromptManager()
        api._message_history = []

        # Execute a simple move
        code = """
nh.move(Direction.N)
"""
        result = await sandbox.execute_code(code=code, api=api)

        # Build last_result dict like agent.py does (using new api_calls field)
        game_messages = []
        api_calls = []
        autoexplore_result = None
        if result.result:
            game_messages = result.result.get("game_messages", [])
            api_calls = result.result.get("api_calls", [])
            autoexplore_result = result.result.get("autoexplore_result")

        last_result = {
            "tool": "execute_code",
            "success": result.success,
            "error": result.error,
            "messages": game_messages,
            "api_calls": api_calls,  # Use new field
        }
        if autoexplore_result:
            last_result["autoexplore_result"] = autoexplore_result
        if result.stdout:
            last_result["output"] = result.stdout

        # Format the prompt like agent does
        result_text = pm.format_last_result(last_result)
        prompt = pm.format_decision_prompt(
            saved_skills=[],
            last_result_text=result_text,
        )

        # Extract just the Last Result section
        idx = prompt.find("Last Result:")
        end_idx = prompt.find("\n\nStudy")
        last_result_section = prompt[idx:end_idx] if idx >= 0 and end_idx >= 0 else prompt[idx:]

        print(f"\n=== WHAT AGENT SEES AS LAST RESULT ===")
        print(last_result_section)

        # Now the agent should see the move() action with its result
        assert "move(N)" in last_result_section, "Agent should see the move action"
        print("\n*** SUCCESS: Agent now sees feedback for move()! ***")
