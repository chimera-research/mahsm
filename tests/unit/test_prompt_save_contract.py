"""
Contract Test: ma.prompt.save()
MAHSM v0.1.0

These tests validate the contract requirements for prompt save operations.
Tests MUST FAIL before implementation (TDD).
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime


class TestPromptSaveContract:
    """Contract tests for ma.prompt.save() function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock DSPy compiled program
        self.mock_compiled_program = Mock()
        self.mock_compiled_program.predict = Mock()
        self.mock_compiled_program.predict.signature = Mock()
        self.mock_compiled_program.predict.signature.instructions = (
            "You are an expert data analyst. Analyze data and provide insights."
        )

        # Mock tools
        def calculate_stats(data: list, measures: list) -> dict:
            """Calculate statistical measures."""
            return {"mean": sum(data) / len(data)}

        def generate_chart(data: list, chart_type: str) -> str:
            """Generate a chart."""
            return f"Generated {chart_type} chart"

        self.tools = [calculate_stats, generate_chart]

    def test_save_extracts_prompt_from_dspy_signature(self):
        """Test that save() extracts prompt from compiled_agent.predict.signature.instructions."""
        # This test will fail until ma.prompt.save() is implemented
        import mahsm as ma

        result_path = ma.prompt.save(
            self.mock_compiled_program, name="test_task", version="v1", tools=self.tools
        )

        # Verify artifact was created and contains the prompt
        assert result_path.exists()
        with open(result_path, "r") as f:
            artifact = json.load(f)

        assert artifact["prompt"] == self.mock_compiled_program.predict.signature.instructions

    def test_save_generates_openai_function_schemas(self):
        """Test that save() generates OpenAI function calling format schemas for tools."""
        import mahsm as ma

        result_path = ma.prompt.save(
            self.mock_compiled_program, name="test_task", version="v1", tools=self.tools
        )

        with open(result_path, "r") as f:
            artifact = json.load(f)

        # Verify tool schemas are in OpenAI format
        assert "tools" in artifact
        assert len(artifact["tools"]) == 2

        for tool_schema in artifact["tools"]:
            assert "name" in tool_schema
            assert "description" in tool_schema
            assert "parameters" in tool_schema
            assert tool_schema["parameters"]["type"] == "object"
            assert "properties" in tool_schema["parameters"]

    def test_save_creates_artifact_with_correct_naming(self):
        """Test that save() creates file with naming convention task_name_version.json."""
        import mahsm as ma

        result_path = ma.prompt.save(
            self.mock_compiled_program, name="data_analysis", version="v2", tools=self.tools
        )

        # Verify file naming
        assert result_path.name == "data_analysis_v2.json"
        assert result_path.parent.name == "prompts"

    def test_save_includes_metadata(self):
        """Test that save() includes creation timestamp and optimizer metadata."""
        import mahsm as ma

        result_path = ma.prompt.save(
            self.mock_compiled_program,
            name="test_task",
            version="v1",
            tools=self.tools,
            metadata={"optimizer": "GEPA"},
        )

        with open(result_path, "r") as f:
            artifact = json.load(f)

        # Verify metadata
        assert "metadata" in artifact
        assert "created_at" in artifact["metadata"]
        assert "optimizer" in artifact["metadata"]
        assert "version" in artifact["metadata"]

        # Verify timestamp is valid ISO format
        datetime.fromisoformat(artifact["metadata"]["created_at"].replace("Z", "+00:00"))

    def test_save_validates_tool_schemas(self):
        """Test that save() validates tool schemas before saving."""
        import mahsm as ma
        from mahsm.exceptions import ValidationError

        # Create invalid tool (no callable)
        invalid_tools = ["not_a_function"]

        with pytest.raises((ValidationError, TypeError, AttributeError)):
            ma.prompt.save(
                self.mock_compiled_program, name="test_task", version="v1", tools=invalid_tools
            )

    def test_save_raises_error_for_missing_prompt(self):
        """Test that save() raises ValueError if compiled_agent has no optimized prompt."""
        import mahsm as ma

        # Create mock without signature.instructions
        bad_program = Mock()
        bad_program.predict = Mock()
        bad_program.predict.signature = None

        with pytest.raises((ValueError, AttributeError)):
            ma.prompt.save(bad_program, name="test_task", version="v1", tools=self.tools)

    def test_save_handles_io_errors(self):
        """Test that save() handles file system errors appropriately."""
        import mahsm as ma

        # Try to save to invalid location
        with pytest.raises(IOError):
            result_path = ma.prompt.save(
                self.mock_compiled_program,
                name="/invalid/path/task",
                version="v1",
                tools=self.tools,
            )

    def test_save_artifact_matches_schema(self):
        """Test that saved artifact matches the expected JSON schema."""
        import mahsm as ma

        result_path = ma.prompt.save(
            self.mock_compiled_program, name="test_task", version="v1", tools=self.tools
        )

        with open(result_path, "r") as f:
            artifact = json.load(f)

        # Verify all required fields exist
        required_fields = ["prompt", "tools", "metadata", "task_name", "task_version"]
        for field in required_fields:
            assert field in artifact, f"Missing required field: {field}"

        # Verify types
        assert isinstance(artifact["prompt"], str)
        assert isinstance(artifact["tools"], list)
        assert isinstance(artifact["metadata"], dict)
        assert isinstance(artifact["task_name"], str)
        assert isinstance(artifact["task_version"], str)

    def teardown_method(self):
        """Clean up test artifacts."""
        # Remove test artifacts if they exist
        import shutil
        from pathlib import Path

        mahsm_home = Path.home() / ".mahsm"
        if mahsm_home.exists():
            test_files = list(mahsm_home.glob("**/test_*.json"))
            for test_file in test_files:
                test_file.unlink()


