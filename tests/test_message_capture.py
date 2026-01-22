"""
Integration test to investigate message capture timing issues.

This test verifies whether game messages are properly captured during
code execution via the sandbox, and if there's a timing issue where
messages only become available after execution completes.
"""

import pytest
import asyncio

from src.sandbox.manager import SkillSandbox, SandboxConfig


class TestMessageCapture:
    """Tests for message capture during code execution."""

    @pytest.fixture
    def sandbox(self):
        """Create a sandbox instance."""
        return SkillSandbox(SandboxConfig(timeout_seconds=10.0))

    @pytest.fixture
    def api(self, nethack_api):
        """Get a real NetHackAPI instance."""
        nethack_api.reset()
        return nethack_api

    @pytest.mark.asyncio
    async def test_kick_wall_message_capture(self, sandbox, api):
        """
        Test that kicking a wall produces a message that gets captured.

        Kicking a wall should produce "WHAMMM!" or similar message.
        This tests the basic message capture path.
        """
        # Clear message history
        api._message_history = []

        # Code that kicks in a direction (will likely hit a wall)
        code = """
# Try kicking in all directions to find a wall
for d in [Direction.N, Direction.S, Direction.E, Direction.W]:
    result = nh.kick(d)
    # If we get a "Ouch!" or "WHAMMM" we hit something solid
"""

        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== KICK TEST RESULTS ===")
        print(f"Execution success: {result.success}")
        print(f"Result dict: {result.result}")
        print(f"Message history length: {len(api._message_history)}")
        print(f"Message history: {api._message_history}")
        print(f"Current get_message(): {api.get_message()!r}")

        # Check if any messages were captured
        game_messages = result.result.get('game_messages', [])
        print(f"game_messages in result: {game_messages}")

        # The test passes if we got some feedback
        # We want to verify the behavior - success or failure tells us about the issue
        assert result.success, f"Code execution failed: {result.error}"

    @pytest.mark.asyncio
    async def test_move_into_wall_message_capture(self, sandbox, api):
        """
        Test that moving into a wall is detected.

        Moving into a wall should not produce a message but should fail.
        This is a baseline to compare with kick.
        """
        api._message_history = []

        # Try to move north 10 times - will eventually hit a wall
        code = """
for _ in range(10):
    nh.move(Direction.N)
"""

        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== MOVE TEST RESULTS ===")
        print(f"Execution success: {result.success}")
        print(f"Result dict: {result.result}")
        print(f"Message history length: {len(api._message_history)}")
        print(f"Message history: {api._message_history}")
        print(f"Current get_message(): {api.get_message()!r}")

        game_messages = result.result.get('game_messages', [])
        print(f"game_messages in result: {game_messages}")

        assert result.success, f"Code execution failed: {result.error}"

    @pytest.mark.asyncio
    async def test_eat_message_capture(self, sandbox, api):
        """
        Test that eating produces appropriate messages.

        Eating should produce messages like "This tastes like..."
        """
        api._message_history = []

        # Try to eat - will likely say "You don't have anything to eat" or similar
        code = """
nh.eat()
"""

        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== EAT TEST RESULTS ===")
        print(f"Execution success: {result.success}")
        print(f"Result dict: {result.result}")
        print(f"Message history length: {len(api._message_history)}")
        print(f"Message history: {api._message_history}")
        print(f"Current get_message(): {api.get_message()!r}")

        game_messages = result.result.get('game_messages', [])
        print(f"game_messages in result: {game_messages}")

        assert result.success, f"Code execution failed: {result.error}"

    @pytest.mark.asyncio
    async def test_message_after_vs_during_execution(self, sandbox, api):
        """
        CRITICAL TEST: Compare message availability during vs after execution.

        This test specifically checks if messages captured DURING action execution
        (via ActionResult.messages) differ from messages available AFTER execution
        (via api.get_message()).

        If there's a timing issue, get_message() after execution will have content
        that wasn't captured during execution.
        """
        api._message_history = []
        messages_before_len = len(api._message_history)

        # Execute some actions
        code = """
# Do multiple actions that might produce messages
nh.search()  # "You find nothing"
nh.look()    # Description of current position
nh.wait()    # May produce messages if something happens
"""

        result = await sandbox.execute_code(code=code, api=api)

        # Get message AFTER execution
        post_execution_message = api.get_message()

        # Get what was captured during execution
        captured_messages = result.result.get('game_messages', [])

        print(f"\n=== TIMING COMPARISON ===")
        print(f"Messages captured DURING execution: {captured_messages}")
        print(f"Message available AFTER execution (get_message()): {post_execution_message!r}")
        print(f"_message_history: {api._message_history}")

        # Check if there's a discrepancy
        if post_execution_message and post_execution_message not in captured_messages:
            print(f"\n*** TIMING ISSUE DETECTED ***")
            print(f"Message '{post_execution_message}' was available after execution")
            print(f"but was NOT in captured messages: {captured_messages}")

        assert result.success, f"Code execution failed: {result.error}"

    @pytest.mark.asyncio
    async def test_search_message_capture(self, sandbox, api):
        """
        Test that search action messages are captured.

        Search often produces "You find nothing" which should be captured.
        """
        api._message_history = []

        code = """
nh.search()
"""

        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== SEARCH TEST RESULTS ===")
        print(f"Execution success: {result.success}")
        print(f"Result dict: {result.result}")
        print(f"Message history: {api._message_history}")
        print(f"Current get_message(): {api.get_message()!r}")

        game_messages = result.result.get('game_messages', [])
        print(f"game_messages in result: {game_messages}")

        # Search should produce some message
        assert result.success, f"Code execution failed: {result.error}"

    @pytest.mark.asyncio
    async def test_travel_to_message_capture(self, sandbox, api):
        """
        Test that travel_to captures all messages during the journey.

        travel_to may encounter many things during movement - we should
        capture all those messages.
        """
        api._message_history = []

        # Get current position and try to travel somewhere
        code = """
pos = nh.get_position()
# Try to travel 5 tiles east (will likely hit something)
target = (pos.x + 5, pos.y)
result = nh.travel_to(target)
print(f"Travel result: {result}")
"""

        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== TRAVEL_TO TEST RESULTS ===")
        print(f"Execution success: {result.success}")
        print(f"Result dict: {result.result}")
        print(f"Message history length: {len(api._message_history)}")
        print(f"Message history (first 10): {api._message_history[:10]}")
        print(f"Current get_message(): {api.get_message()!r}")

        game_messages = result.result.get('game_messages', [])
        print(f"game_messages count: {len(game_messages)}")
        print(f"game_messages (first 10): {game_messages[:10]}")

        assert result.success, f"Code execution failed: {result.error}"

    @pytest.mark.asyncio
    async def test_autoexplore_message_capture(self, sandbox, api):
        """
        Test that autoexplore captures messages during exploration.

        autoexplore does many moves and may encounter various things.
        """
        api._message_history = []

        code = """
result = nh.autoexplore(max_steps=20)
print(f"Autoexplore result: {result}")
"""

        result = await sandbox.execute_code(code=code, api=api)

        print(f"\n=== AUTOEXPLORE TEST RESULTS ===")
        print(f"Execution success: {result.success}")
        print(f"Result dict keys: {result.result.keys() if result.result else 'None'}")
        print(f"Message history length: {len(api._message_history)}")
        print(f"Message history (first 10): {api._message_history[:10]}")
        print(f"Current get_message(): {api.get_message()!r}")

        game_messages = result.result.get('game_messages', [])
        autoexplore_result = result.result.get('autoexplore_result', {})
        print(f"game_messages count: {len(game_messages)}")
        print(f"game_messages (first 10): {game_messages[:10]}")
        print(f"autoexplore_result: {autoexplore_result}")

        assert result.success, f"Code execution failed: {result.error}"

    @pytest.mark.asyncio
    async def test_message_timing_issue_demonstration(self, sandbox, api):
        """
        CRITICAL: Demonstrates the timing issue where messages appear
        in get_message() after execution but weren't captured during.

        This is the bug we need to fix.
        """
        api._message_history = []

        # Simple code that just does a few basic actions
        code = """
# Just get some state - minimal actions
stats = nh.get_stats()
pos = nh.get_position()
"""

        result = await sandbox.execute_code(code=code, api=api)

        game_messages = result.result.get('game_messages', []) if result.result else []
        post_message = api.get_message()

        print(f"\n=== MESSAGE TIMING ISSUE DEMONSTRATION ===")
        print(f"Messages captured DURING execution: {game_messages}")
        print(f"Message available AFTER execution: {post_message!r}")
        print(f"_message_history: {api._message_history}")

        # This is the bug: if post_message exists but wasn't captured
        if post_message and post_message not in game_messages:
            print(f"\n*** BUG CONFIRMED ***")
            print(f"Message '{post_message}' exists after execution")
            print(f"but was NOT captured in game_messages!")
            print(f"This message would be LOST in Last Result!")

        assert result.success, f"Code execution failed: {result.error}"

    @pytest.mark.asyncio
    async def test_fresh_game_initial_message(self, sandbox, api):
        """
        Test that the initial game message (like 'Be careful! New moon tonight.')
        is properly captured on the first action after reset.

        Many games start with messages that may not be captured.
        """
        # Fresh reset to get initial message
        api.reset()
        api._message_history = []

        # Check what message exists BEFORE any code execution
        initial_message = api.get_message()
        print(f"\n=== INITIAL MESSAGE TEST ===")
        print(f"Message immediately after reset: {initial_message!r}")

        # Now execute minimal code
        code = """
stats = nh.get_stats()
"""
        result = await sandbox.execute_code(code=code, api=api)

        game_messages = result.result.get('game_messages', []) if result.result else []
        post_message = api.get_message()

        print(f"Messages captured during execution: {game_messages}")
        print(f"Message after execution: {post_message!r}")
        print(f"_message_history: {api._message_history}")

        # If there was an initial message, was it captured?
        if initial_message:
            if initial_message in game_messages or initial_message in api._message_history:
                print(f"Initial message WAS captured - good!")
            else:
                print(f"\n*** INITIAL MESSAGE NOT CAPTURED ***")
                print(f"Initial message '{initial_message}' was lost!")

        assert result.success, f"Code execution failed: {result.error}"
