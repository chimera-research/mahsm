"""
Integration Test: DSPy Workflow
MAHSM v0.1.0

These tests validate the complete DSPy optimization → save → load workflow.
Tests MUST FAIL before implementation (TDD).
"""

import pytest
from pathlib import Path


@pytest.mark.integration
class TestDSpyIntegration:
    """Integration tests for DSPy workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        # Tool functions for testing
        def calculate_statistics(data: list, measures: list) -> dict:
            """Calculate statistical measures for data."""
            import statistics

            results = {}
            if "mean" in measures:
                results["mean"] = statistics.mean(data)
            if "median" in measures:
                results["median"] = statistics.median(data)
            if "stdev" in measures and len(data) > 1:
                results["stdev"] = statistics.stdev(data)
            return results

        def generate_chart(data: list, chart_type: str) -> str:
            """Generate a chart description."""
            return f"Generated {chart_type} chart for {len(data)} data points"

        self.tools = [calculate_statistics, generate_chart]

    @pytest.mark.slow
    def test_complete_dspy_save_load_workflow(self):
        """Test complete DSPy optimization → save → load workflow."""
        import dspy
        import mahsm as ma

        # Step 1: Configure DSPy
        lm = dspy.OpenAI(model="gpt-4o-mini", max_tokens=250)
        dspy.settings.configure(lm=lm)

        # Step 2: Define signature
        class DataAnalysis(dspy.Signature):
            """Analyze data and provide statistical insights."""

            data = dspy.InputField(desc="Raw data to analyze")
            insights = dspy.OutputField(desc="Statistical insights")

        # Step 3: Create simple program (skip optimization for fast test)
        program = dspy.Predict(DataAnalysis)

        # Step 4: Save with MAHSM
        artifact_path = ma.prompt.save(program, name="dspy_test", version="v1", tools=self.tools)

        # Step 5: Load back and validate
        loaded_prompt = ma.prompt.load("dspy_test_v1", validate_tools=self.tools)

        # Verify
        assert artifact_path.exists()
        assert isinstance(loaded_prompt, str)
        assert len(loaded_prompt) > 0

    def test_prompt_extraction_from_compiled_program(self):
        """Test that prompts are correctly extracted from DSPy compiled programs."""
        import dspy
        import mahsm as ma

        # Create mock compiled program with instructions
        class MockSignature:
            instructions = "You are an expert data analyst. Provide detailed analysis."

        class MockPredict:
            signature = MockSignature()

        class MockProgram:
            predict = MockPredict()

        # Save the mock program
        artifact_path = ma.prompt.save(MockProgram(), name="mock_test", version="v1", tools=self.tools)

        # Load and verify
        loaded_prompt = ma.prompt.load("mock_test_v1")
        assert loaded_prompt == MockSignature.instructions

    def test_tool_schema_compatibility_across_save_load(self):
        """Test that tool schemas remain compatible across save/load cycle."""
        import dspy
        import mahsm as ma

        # Simple DSPy program
        lm = dspy.OpenAI(model="gpt-4o-mini")
        dspy.settings.configure(lm=lm)

        class SimpleTask(dspy.Signature):
            """Simple task."""

            input = dspy.InputField()
            output = dspy.OutputField()

        program = dspy.Predict(SimpleTask)

        # Save with tools
        ma.prompt.save(program, name="tool_test", version="v1", tools=self.tools)

        # Load with same tools - should not raise ValidationError
        prompt = ma.prompt.load("tool_test_v1", validate_tools=self.tools)
        assert prompt is not None

    def test_multiple_tool_functions_handling(self):
        """Test handling of multiple tool functions with DSPy."""
        import dspy
        import mahsm as ma

        lm = dspy.OpenAI(model="gpt-4o-mini")
        dspy.settings.configure(lm=lm)

        class MultiToolTask(dspy.Signature):
            """Task with multiple tools."""

            query = dspy.InputField()
            result = dspy.OutputField()

        program = dspy.Predict(MultiToolTask)

        # Save with multiple tools
        artifact_path = ma.prompt.save(program, name="multi_tool", version="v1", tools=self.tools)

        # Verify both tools are saved
        import json

        with open(artifact_path) as f:
            artifact = json.load(f)

        assert len(artifact["tools"]) == 2
        assert artifact["tools"][0]["name"] == "calculate_statistics"
        assert artifact["tools"][1]["name"] == "generate_chart"

    def teardown_method(self):
        """Clean up test artifacts."""
        mahsm_home = Path.home() / ".mahsm" / "prompts"
        if mahsm_home.exists():
            for test_file in mahsm_home.glob("*_test_*.json"):
                test_file.unlink()


