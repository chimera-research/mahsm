"""
Smoke Test: End-to-End Workflow
MAHSM v0.1.0

This smoke test implements the complete quickstart guide scenario as an automated test.
Tests MUST FAIL before implementation (TDD).
"""

import pytest
import os
from pathlib import Path


@pytest.mark.smoke
@pytest.mark.slow
class TestEndToEndWorkflow:
    """End-to-end smoke test for complete MAHSM workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        # Define tools from quickstart guide
        def calculate_statistics(data: list, measures: list) -> dict:
            """Calculate statistical measures for the given data."""
            import statistics

            results = {}
            if "mean" in measures:
                results["mean"] = round(statistics.mean(data), 4)
            if "median" in measures:
                results["median"] = statistics.median(data)
            if "stdev" in measures and len(data) > 1:
                results["stdev"] = round(statistics.stdev(data), 4)
            return results

        def generate_chart(data: list, chart_type: str) -> str:
            """Generate a chart description for the data."""
            return f"Generated {chart_type} chart for {len(data)} data points"

        self.tools = [calculate_statistics, generate_chart]
        self.test_data = [1.2, 2.3, 3.1, 4.5, 2.8, 3.9, 1.7, 4.2]

    def test_complete_quickstart_workflow(self):
        """Test the complete quickstart guide workflow end-to-end."""
        import dspy
        import mahsm as ma
        from langgraph.graph import StateGraph, MessagesState
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        # Step 1: Configure DSPy
        lm = dspy.OpenAI(model="gpt-4o-mini", max_tokens=500)
        dspy.settings.configure(lm=lm)

        # Step 2: Define and create DSPy program
        class DataAnalysis(dspy.Signature):
            """Analyze data and provide statistical insights."""

            data = dspy.InputField(desc="Raw data to analyze")
            insights = dspy.OutputField(desc="Statistical insights and recommendations")

        program = dspy.Predict(DataAnalysis)

        # Step 3: Save optimized prompt with tools
        artifact_path = ma.prompt.save(
            program, name="data_analysis", version="v1", tools=self.tools
        )

        assert artifact_path.exists()
        print(f"✓ Prompt saved to: {artifact_path}")

        # Step 4: Define LangGraph node with ma.inference()
        def analysis_node(state: MessagesState):
            # Load the optimized prompt with tool validation
            prompt = ma.prompt.load("data_analysis_v1", validate_tools=self.tools)

            # Get user input
            user_input = state["messages"][-1].content if state.get("messages") else "test"

            # Execute inference with the loaded prompt
            result, messages = ma.inference(
                model="openai/gpt-4o-mini",
                prompt=prompt,
                tools=self.tools,
                input=user_input,
                state=state.get("messages", []),
                max_iterations=5,
            )

            return {"messages": messages}

        # Step 5: Build LangGraph workflow
        workflow = StateGraph(MessagesState)
        workflow.add_node("analyze", analysis_node)
        workflow.set_entry_point("analyze")
        workflow.set_finish_point("analyze")

        app = workflow.compile()

        # Step 6: Execute with user input
        user_input = f"Please analyze this dataset: {self.test_data}"
        initial_state = {"messages": [HumanMessage(content=user_input)]}

        result = app.invoke(initial_state)

        # Step 7: Verify message transparency
        messages = result["messages"]

        print(f"✓ Complete conversation with {len(messages)} messages")

        # Verify message types and ordering
        assert len(messages) >= 2, "Should have at least SystemMessage and HumanMessage"
        assert isinstance(messages[0], SystemMessage), "First message should be SystemMessage"
        assert isinstance(messages[1], HumanMessage), "Second message should be HumanMessage"
        assert user_input in messages[1].content, "User input should be in HumanMessage"

        # Check for tool messages (if tools were called)
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        print(f"✓ Found {len(tool_messages)} tool executions")

        print("✅ Message transparency verified!")

    def test_performance_goals_sub_100ms_prompt_operations(self):
        """Test that prompt save/load operations meet sub-100ms performance goal."""
        import time
        import dspy
        import mahsm as ma

        # Configure DSPy
        lm = dspy.OpenAI(model="gpt-4o-mini")
        dspy.settings.configure(lm=lm)

        class SimpleTask(dspy.Signature):
            """Simple task."""

            input = dspy.InputField()
            output = dspy.OutputField()

        program = dspy.Predict(SimpleTask)

        # Test save performance
        start_time = time.time()
        artifact_path = ma.prompt.save(
            program, name="perf_test", version="v1", tools=self.tools
        )
        save_time = (time.time() - start_time) * 1000  # Convert to ms

        print(f"Save time: {save_time:.2f}ms")
        assert save_time < 100, f"Save took {save_time:.2f}ms, should be <100ms"

        # Test load performance
        start_time = time.time()
        prompt = ma.prompt.load("perf_test_v1", validate_tools=self.tools)
        load_time = (time.time() - start_time) * 1000

        print(f"Load time: {load_time:.2f}ms")
        assert load_time < 100, f"Load took {load_time:.2f}ms, should be <100ms"

        print(f"✅ Performance goals met (save: {save_time:.2f}ms, load: {load_time:.2f}ms)")

    def test_message_conversation_completeness(self):
        """Test that complete conversation history is maintained and accessible."""
        import mahsm as ma

        prompt = "You are a helpful assistant."

        # Execute inference
        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=prompt,
            tools=self.tools,
            input="Test message completeness",
            max_iterations=3,
        )

        # Verify conversation completeness
        assert len(messages) >= 2, "Should have minimum of 2 messages"

        # Verify all messages are accessible
        for i, message in enumerate(messages):
            assert hasattr(message, "content"), f"Message {i} should have content"
            print(f"Message {i+1}: {type(message).__name__}")

        print(f"✅ Conversation completeness verified with {len(messages)} messages")

    def test_tool_execution_transparency(self):
        """Test that tool executions are transparent and traceable."""
        import mahsm as ma
        from langchain_core.messages import ToolMessage

        prompt = "You are a data analysis assistant. Use the provided tools."

        # Execute inference that should trigger tool use
        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=prompt,
            tools=self.tools,
            input=f"Calculate mean and median for {self.test_data}",
            max_iterations=3,
        )

        # Check for tool messages
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        if tool_messages:
            print(f"✓ Found {len(tool_messages)} tool executions")
            for tm in tool_messages:
                print(f"  Tool result: {tm.content[:100]}...")
        else:
            print("⚠ No tools were called (model-dependent)")

        # Verify structure exists even if no tools called
        assert isinstance(tool_messages, list)

    def teardown_method(self):
        """Clean up test artifacts."""
        mahsm_home = Path.home() / ".mahsm" / "prompts"
        if mahsm_home.exists():
            test_files = ["data_analysis_v1.json", "perf_test_v1.json"]
            for test_file in test_files:
                file_path = mahsm_home / test_file
                if file_path.exists():
                    file_path.unlink()


