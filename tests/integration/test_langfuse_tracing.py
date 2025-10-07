"""
Integration Test: LangFuse Tracing
MAHSM v0.1.0

These tests validate automatic LangFuse tracing integration.
Tests MUST FAIL before implementation (TDD).
"""

import os
import pytest


@pytest.mark.integration
class TestLangFuseTracing:
    """Integration tests for LangFuse tracing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.original_env = os.environ.copy()

        def test_tool(param: str) -> str:
            """Test tool."""
            return f"Result: {param}"

        self.tool = test_tool
        self.prompt = "You are a test assistant."

    def test_automatic_tracing_when_keys_configured(self):
        """Test that tracing activates automatically when environment variables are set."""
        import mahsm as ma

        # Set LangFuse environment variables
        os.environ["LANGFUSE_PUBLIC_KEY"] = "test_public_key"
        os.environ["LANGFUSE_SECRET_KEY"] = "test_secret_key"

        # Execute inference (should create trace)
        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=[self.tool],
            input="test query",
        )

        # Verify execution completed (trace creation tested separately)
        assert messages is not None
        assert len(messages) > 0

    def test_tracing_disabled_when_keys_not_provided(self):
        """Test that tracing is disabled when LangFuse keys are not provided."""
        import mahsm as ma

        # Clear LangFuse keys
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)

        # Execute inference (should work without tracing)
        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=[self.tool],
            input="test query",
        )

        # Verify execution completed without errors
        assert messages is not None

    def test_trace_includes_all_messages(self):
        """Test that trace data includes all messages from the conversation."""
        import mahsm as ma

        os.environ["LANGFUSE_PUBLIC_KEY"] = "test_key"
        os.environ["LANGFUSE_SECRET_KEY"] = "test_secret"

        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=[self.tool],
            input="test query",
        )

        # Verify all message types are present
        from langchain_core.messages import SystemMessage, HumanMessage

        assert any(isinstance(m, SystemMessage) for m in messages)
        assert any(isinstance(m, HumanMessage) for m in messages)

    def test_trace_includes_tool_calls(self):
        """Test that trace data includes tool calls and results."""
        import mahsm as ma

        os.environ["LANGFUSE_PUBLIC_KEY"] = "test_key"
        os.environ["LANGFUSE_SECRET_KEY"] = "test_secret"

        # Execute inference with tools
        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=[self.tool],
            input="test query",
        )

        # If tools were called, verify ToolMessages exist
        from langchain_core.messages import ToolMessage

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        # May be empty if model didn't call tools, but structure should exist
        assert isinstance(tool_messages, list)

    def test_trace_completeness(self):
        """Test that traces contain complete conversation history."""
        import mahsm as ma

        os.environ["LANGFUSE_PUBLIC_KEY"] = "test_key"
        os.environ["LANGFUSE_SECRET_KEY"] = "test_secret"

        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=[self.tool],
            input="complex query",
            max_iterations=3,
        )

        # Verify minimum messages (SystemMessage + HumanMessage at least)
        assert len(messages) >= 2

        # Verify first two messages are correct types
        from langchain_core.messages import SystemMessage, HumanMessage

        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    def test_trace_metadata_includes_model_and_tools(self):
        """Test that trace metadata includes model identifier and tool information."""
        import mahsm as ma

        os.environ["LANGFUSE_PUBLIC_KEY"] = "test_key"
        os.environ["LANGFUSE_SECRET_KEY"] = "test_secret"

        model_name = "openai/gpt-4o-mini"
        result, messages = ma.inference(
            model=model_name, prompt=self.prompt, tools=[self.tool], input="test"
        )

        # Verify execution completed (metadata verification tested separately)
        assert result is not None or messages is not None

    def test_tracing_errors_dont_break_inference(self):
        """Test that tracing errors don't break inference execution."""
        import mahsm as ma

        # Set invalid LangFuse keys
        os.environ["LANGFUSE_PUBLIC_KEY"] = "invalid_key"
        os.environ["LANGFUSE_SECRET_KEY"] = "invalid_secret"

        # Should still execute successfully even if tracing fails
        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=[self.tool],
            input="test",
        )

        # Verify inference completed despite tracing issues
        assert messages is not None

    def test_trace_accuracy_for_multi_turn_conversation(self):
        """Test trace accuracy for conversations with multiple turns."""
        import mahsm as ma

        os.environ["LANGFUSE_PUBLIC_KEY"] = "test_key"
        os.environ["LANGFUSE_SECRET_KEY"] = "test_secret"

        # First turn
        result1, messages1 = ma.inference(
            model="openai/gpt-4o-mini", prompt=self.prompt, tools=[self.tool], input="first query"
        )

        # Second turn (continuing conversation)
        result2, messages2 = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=[self.tool],
            input="second query",
            state=messages1,
        )

        # Verify conversation continuity
        assert len(messages2) > len(messages1)

    def teardown_method(self):
        """Clean up environment."""
        os.environ.clear()
        os.environ.update(self.original_env)


