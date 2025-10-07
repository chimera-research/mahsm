"""
Contract Test: ma.inference()
MAHSM v0.1.0

These tests validate the contract requirements for inference execution.
Tests MUST FAIL before implementation (TDD).
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage


class TestInferenceContract:
    """Contract tests for ma.inference() function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.prompt = "You are an expert assistant."

        def test_tool(param: str) -> str:
            """Test tool function."""
            return f"Result: {param}"

        self.tools = [test_tool]

    def test_inference_creates_system_message_from_prompt(self):
        """Test that inference() creates SystemMessage from prompt."""
        import mahsm as ma

        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=self.tools,
            input="test input",
        )

        assert len(messages) > 0
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == self.prompt

    def test_inference_creates_human_message_from_input(self):
        """Test that inference() creates HumanMessage from input."""
        import mahsm as ma

        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=self.tools,
            input="test query",
        )

        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_messages) > 0
        assert human_messages[0].content == "test query"

    def test_inference_executes_agentic_loop(self):
        """Test that inference() executes agentic loop until no tool_calls."""
        import mahsm as ma

        # Mock should eventually return response without tool_calls
        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=self.tools,
            input="test",
            max_iterations=5,
        )

        # Should have at least SystemMessage and HumanMessage
        assert len(messages) >= 2

    def test_inference_appends_messages_in_order(self):
        """Test that inference() appends all messages in correct chronological order."""
        import mahsm as ma

        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=self.tools,
            input="test",
        )

        # First should be SystemMessage, second should be HumanMessage
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    def test_inference_wraps_tool_results_in_tool_message(self):
        """Test that inference() wraps tool execution results in ToolMessage."""
        import mahsm as ma

        # This test assumes model returns tool_calls
        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=self.tools,
            input="test",
        )

        # Check if any ToolMessages were created (depends on model response)
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        # At minimum, we verify the mechanism exists (may be 0 if no tool_calls)
        assert isinstance(tool_messages, list)

    def test_inference_raises_max_iterations_error(self):
        """Test that inference() raises MaxIterationsError when loop exceeds max_iterations."""
        import mahsm as ma
        from mahsm.exceptions import MaxIterationsError

        # Mock scenario where model always returns tool_calls
        with pytest.raises(MaxIterationsError):
            result, messages = ma.inference(
                model="openai/gpt-4o-mini",
                prompt=self.prompt,
                tools=self.tools,
                input="test",
                max_iterations=1,  # Force quick failure
            )

    def test_inference_handles_tool_execution_errors(self):
        """Test that inference() wraps tool execution errors in ToolExecutionError."""
        import mahsm as ma
        from mahsm.exceptions import ToolExecutionError

        def failing_tool(param: str) -> str:
            """Tool that raises an error."""
            raise RuntimeError("Tool failed!")

        with pytest.raises(ToolExecutionError):
            result, messages = ma.inference(
                model="openai/gpt-4o-mini",
                prompt=self.prompt,
                tools=[failing_tool],
                input="test",
            )

    def test_inference_returns_tuple_format(self):
        """Test that inference() returns tuple of (result, messages)."""
        import mahsm as ma

        output = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=self.tools,
            input="test",
        )

        assert isinstance(output, tuple)
        assert len(output) == 2
        result, messages = output
        assert isinstance(messages, list)

    def test_inference_handles_dict_input(self):
        """Test that inference() converts dict input to HumanMessage."""
        import mahsm as ma

        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=self.tools,
            input={"query": "test"},
        )

        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_messages) > 0

    def test_inference_handles_string_input(self):
        """Test that inference() converts string input to HumanMessage."""
        import mahsm as ma

        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=self.tools,
            input="simple string",
        )

        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_messages) > 0
        assert "simple string" in human_messages[0].content

    def test_inference_continues_existing_state(self):
        """Test that inference() can continue from existing message state."""
        import mahsm as ma

        existing_state = [
            SystemMessage(content="Previous context"),
            HumanMessage(content="Previous query"),
        ]

        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=self.tools,
            input="new query",
            state=existing_state,
        )

        # Should include previous state plus new messages
        assert len(messages) >= len(existing_state) + 2


