"""
Contract Test: ma.prompt.load()
MAHSM v0.1.0

These tests validate the contract requirements for prompt load operations.
Tests MUST FAIL before implementation (TDD).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock


class TestPromptLoadContract:
    """Contract tests for ma.prompt.load() function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test artifact
        self.mahsm_home = Path.home() / ".mahsm" / "prompts"
        self.mahsm_home.mkdir(parents=True, exist_ok=True)

        self.test_artifact = {
            "prompt": "You are an expert assistant.",
            "tools": [
                {
                    "name": "calculate_stats",
                    "description": "Calculate statistics",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"type": "array"}, "measures": {"type": "array"}},
                        "required": ["data", "measures"],
                    },
                }
            ],
            "metadata": {"created_at": "2025-10-07T10:00:00Z", "optimizer": "GEPA", "version": "1.0.0"},
            "task_name": "test_task",
            "task_version": "v1",
        }

        self.artifact_path = self.mahsm_home / "test_task_v1.json"
        with open(self.artifact_path, "w") as f:
            json.dump(self.test_artifact, f)

        # Define matching tool
        def calculate_stats(data: list, measures: list) -> dict:
            """Calculate statistics."""
            return {}

        self.matching_tool = calculate_stats

    def test_load_returns_prompt_string(self):
        """Test that load() returns only the prompt string, not the full artifact."""
        import mahsm as ma

        prompt = ma.prompt.load("test_task_v1")

        assert isinstance(prompt, str)
        assert prompt == "You are an expert assistant."

    def test_load_from_correct_location(self):
        """Test that load() loads from ~/.mahsm/prompts/{name_version}.json."""
        import mahsm as ma

        # This should succeed because we created the file
        prompt = ma.prompt.load("test_task_v1")
        assert prompt is not None

    def test_load_validates_tools_when_provided(self):
        """Test that load() validates tools against saved schemas when validate_tools provided."""
        import mahsm as ma

        # Should succeed with matching tool
        prompt = ma.prompt.load("test_task_v1", validate_tools=[self.matching_tool])
        assert prompt == "You are an expert assistant."

    def test_load_raises_validation_error_for_mismatched_tools(self):
        """Test that load() raises ValidationError when tools don't match schemas."""
        import mahsm as ma
        from mahsm.exceptions import ValidationError

        # Define non-matching tool
        def wrong_tool(different_param: str) -> str:
            """Wrong signature."""
            return ""

        with pytest.raises(ValidationError):
            ma.prompt.load("test_task_v1", validate_tools=[wrong_tool])

    def test_load_raises_file_not_found_for_missing_artifact(self):
        """Test that load() raises FileNotFoundError if artifact doesn't exist."""
        import mahsm as ma

        with pytest.raises(FileNotFoundError):
            ma.prompt.load("nonexistent_task_v1")

    def test_load_handles_corrupted_json(self):
        """Test that load() raises JSONDecodeError for corrupted artifacts."""
        import mahsm as ma
        import json

        # Create corrupted artifact
        corrupted_path = self.mahsm_home / "corrupted_v1.json"
        with open(corrupted_path, "w") as f:
            f.write("{ invalid json content")

        with pytest.raises(json.JSONDecodeError):
            ma.prompt.load("corrupted_v1")

    def test_load_compares_tool_names_exactly(self):
        """Test that load() compares tool names exactly during validation."""
        import mahsm as ma
        from mahsm.exceptions import ValidationError

        # Tool with different name
        def different_name(data: list, measures: list) -> dict:
            """Different name."""
            return {}

        different_name.__name__ = "wrong_name"

        with pytest.raises(ValidationError):
            ma.prompt.load("test_task_v1", validate_tools=[different_name])

    def test_load_compares_parameter_structures(self):
        """Test that load() compares parameter structures during validation."""
        import mahsm as ma
        from mahsm.exceptions import ValidationError

        # Tool with different parameters
        def calculate_stats(wrong_params: str) -> dict:
            """Wrong parameters."""
            return {}

        with pytest.raises(ValidationError):
            ma.prompt.load("test_task_v1", validate_tools=[calculate_stats])

    def teardown_method(self):
        """Clean up test artifacts."""
        if self.artifact_path.exists():
            self.artifact_path.unlink()

        corrupted_path = self.mahsm_home / "corrupted_v1.json"
        if corrupted_path.exists():
            corrupted_path.unlink()


