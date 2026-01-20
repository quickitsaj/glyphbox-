"""Tests for the prompt manager."""

import pytest
from src.agent.prompts import PromptManager


class TestPromptManager:
    """Tests for PromptManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PromptManager()  # Default: skills_enabled=False
        self.manager_with_skills = PromptManager(skills_enabled=True)

    def test_get_system_prompt(self):
        """Test getting system prompt."""
        prompt = self.manager.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        # Should contain key concepts
        assert "NetHack" in prompt
        assert "execute_code" in prompt.lower() or "tool" in prompt.lower()

    def test_get_system_prompt_no_skills(self):
        """Test that no-skills prompt excludes skill tools."""
        prompt = self.manager.get_system_prompt()
        assert "1 tool" in prompt
        assert "write_skill" not in prompt
        assert "invoke_skill" not in prompt

    def test_get_system_prompt_with_skills(self):
        """Test that skills-enabled prompt includes skill tools."""
        prompt = self.manager_with_skills.get_system_prompt()
        assert "3 tools" in prompt
        assert "write_skill" in prompt
        assert "invoke_skill" in prompt

    def test_format_decision_prompt_minimal(self):
        """Test formatting decision prompt with minimal data (no skills)."""
        prompt = self.manager.format_decision_prompt(
            saved_skills=[],
            last_result=None,
        )
        assert isinstance(prompt, str)
        # When skills disabled, no "Saved Skills" section
        assert "Saved Skills" not in prompt

    def test_format_decision_prompt_minimal_with_skills(self):
        """Test formatting decision prompt with skills enabled."""
        prompt = self.manager_with_skills.format_decision_prompt(
            saved_skills=[],
            last_result=None,
        )
        assert isinstance(prompt, str)
        assert "Saved Skills" in prompt

    def test_format_decision_prompt_with_saved_skills(self):
        """Test formatting decision prompt with saved skills (skills enabled)."""
        saved_skills = ["explore_corridor", "fight_adjacent"]
        prompt = self.manager_with_skills.format_decision_prompt(
            saved_skills=saved_skills,
            last_result=None,
        )
        assert "explore_corridor" in prompt
        assert "fight_adjacent" in prompt

    def test_format_decision_prompt_with_last_result(self):
        """Test formatting decision prompt with last result."""
        last_result = {
            "tool": "execute_code",
            "success": True,
            "hint": "Moved east successfully",
        }
        prompt = self.manager.format_decision_prompt(
            saved_skills=[],
            last_result=last_result,
        )
        assert "Moved east successfully" in prompt or "success" in prompt.lower()

    def test_format_decision_prompt_with_game_screen(self):
        """Test formatting decision prompt with game screen."""
        game_screen = """
 ------
 |....|
 |..@.|
 ------
"""
        prompt = self.manager.format_decision_prompt(
            saved_skills=[],
            last_result=None,
            game_screen=game_screen,
        )
        assert "@" in prompt
        assert "CURRENT GAME VIEW" in prompt

    def test_format_skill_creation_prompt(self):
        """Test formatting skill creation prompt."""
        prompt = self.manager.format_skill_creation_prompt(
            situation="A floating eye is blocking my path",
            game_state={"hp": 20, "max_hp": 20},
            existing_skills=["explore", "fight"],
        )
        assert isinstance(prompt, str)
        assert "floating eye" in prompt
        assert "explore" in prompt or "fight" in prompt

    def test_format_skill_creation_prompt_with_context(self):
        """Test skill creation with game context."""
        game_state = {
            "hp": 20,
            "max_hp": 20,
        }
        prompt = self.manager.format_skill_creation_prompt(
            situation="Need to eat food",
            game_state=game_state,
            existing_skills=["eat_when_hungry"],
        )
        assert "eat food" in prompt.lower() or "need to eat" in prompt.lower()

    def test_format_skill_creation_prompt_with_failed_attempts(self):
        """Test skill creation with previous failed attempts."""
        failed_attempts = [
            "async def bad_skill(nh): pass  # Didn't check inventory",
        ]
        prompt = self.manager.format_skill_creation_prompt(
            situation="Need new approach",
            game_state={"hp": 15, "max_hp": 20},
            existing_skills=["explore"],
            failed_attempts=failed_attempts,
        )
        assert "Didn't check inventory" in prompt or "previous" in prompt.lower() or "Failed" in prompt

    def test_format_analysis_prompt(self):
        """Test formatting analysis prompt."""
        prompt = self.manager.format_analysis_prompt(
            game_state={
                "hp": 5,
                "max_hp": 20,
            },
            question="Should I fight or flee?",
        )
        assert isinstance(prompt, str)
        assert "fight or flee" in prompt

    def test_prompt_manager_templates_loaded(self):
        """Test that internal templates are loaded."""
        # The manager should have template strings in _templates dict
        assert hasattr(self.manager, '_templates')
        assert "system" in self.manager._templates
        assert "decision" in self.manager._templates
        assert len(self.manager._templates["system"]) > 0
        assert len(self.manager._templates["decision"]) > 0


class TestPromptFormatting:
    """Tests for specific prompt formatting details."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PromptManager()
        self.manager_with_skills = PromptManager(skills_enabled=True)

    def test_decision_prompt_contains_game_view(self):
        """Test that decision prompt contains game view section."""
        prompt = self.manager.format_decision_prompt(
            saved_skills=[],
            last_result=None,
        )
        assert "CURRENT GAME VIEW" in prompt

    def test_skill_creation_prompt_contains_api_info(self):
        """Test that skill creation prompt mentions API."""
        prompt = self.manager.format_skill_creation_prompt(
            situation="Test",
            game_state={"hp": 20, "max_hp": 20},
            existing_skills=["explore"],
        )
        # Should mention the nh API or how to write skills
        assert "nh" in prompt.lower() or "api" in prompt.lower() or "async" in prompt

    def test_prompts_are_reasonable_length(self):
        """Test that prompts aren't excessively long."""
        decision_prompt = self.manager_with_skills.format_decision_prompt(
            saved_skills=[f"skill_{i}" for i in range(10)],
            last_result={"success": True, "hint": "All good"},
        )
        # Should be under 10k characters for reasonable token usage
        assert len(decision_prompt) < 10000


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PromptManager()
        self.manager_with_skills = PromptManager(skills_enabled=True)

    def test_empty_everything(self):
        """Test with all empty inputs."""
        prompt = self.manager.format_decision_prompt(
            saved_skills=[],
            last_result=None,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_special_characters_in_skills(self):
        """Test skills with special characters (skills enabled)."""
        saved_skills = ["skill_with_underscore", "skill-with-dash"]
        prompt = self.manager_with_skills.format_decision_prompt(
            saved_skills=saved_skills,
            last_result=None,
        )
        # Should not crash and should contain the content
        assert "skill_with_underscore" in prompt
