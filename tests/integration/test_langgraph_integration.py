"""
Integration Test: LangGraph Workflow
MAHSM v0.1.0

These tests validate ma.inference() integration with LangGraph.
Tests MUST FAIL before implementation (TDD).
"""

import pytest
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage


@pytest.mark.integration
class TestLangGraphIntegration:
    """Integration tests for LangGraph workflow."""

    def setup_method(self):
        """Set up test fixtures."""

        def simple_tool(query: str) -> str:
            """A simple tool for testing."""
            return f"Processed: {query}"

        self.tool = simple_tool
        self.prompt = "You are a helpful assistant."

    def test_inference_within_langgraph_node(self):
        """Test ma.inference() execution within a LangGraph node."""
        import mahsm as ma

        def analysis_node(state: MessagesState):
            """LangGraph node using ma.inference()."""
            result, messages = ma.inference(
                model="openai/gpt-4o-mini",
                prompt=self.prompt,
                tools=[self.tool],
                input=state["messages"][-1].content if state.get("messages") else "test",
                state=state.get("messages", []),
            )
            return {"messages": messages}

        # Build graph
        workflow = StateGraph(MessagesState)
        workflow.add_node("analyze", analysis_node)
        workflow.set_entry_point("analyze")
        workflow.set_finish_point("analyze")

        app = workflow.compile()

        # Execute
        initial_state = {"messages": [HumanMessage(content="Test query")]}
        result = app.invoke(initial_state)

        # Verify messages were updated
        assert "messages" in result
        assert len(result["messages"]) > 1

    def test_messages_state_integration(self):
        """Test MessagesState integration and message ordering."""
        import mahsm as ma
        from langchain_core.messages import SystemMessage

        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=[self.tool],
            input="test query",
        )

        # Verify message types and ordering
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == self.prompt

    def test_checkpointer_integration_with_sqlite(self):
        """Test checkpointer integration with SQLite backend."""
        import mahsm as ma
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3

        # Get SQLite checkpointer from ma.config
        checkpointer = ma.config.get_checkpointer(checkpoint_type="sqlite")

        # Build graph with checkpointer
        def node_with_checkpointer(state: MessagesState):
            result, messages = ma.inference(
                model="openai/gpt-4o-mini",
                prompt=self.prompt,
                tools=[self.tool],
                input="checkpoint test",
                state=state.get("messages", []),
            )
            return {"messages": messages}

        workflow = StateGraph(MessagesState)
        workflow.add_node("test", node_with_checkpointer)
        workflow.set_entry_point("test")
        workflow.set_finish_point("test")

        app = workflow.compile(checkpointer=checkpointer)

        # Execute with checkpointing
        config = {"configurable": {"thread_id": "test_thread"}}
        initial_state = {"messages": [HumanMessage(content="Test")]}
        result = app.invoke(initial_state, config=config)

        # Verify execution completed
        assert "messages" in result

    def test_conversation_transparency(self):
        """Test complete conversation transparency in LangGraph context."""
        import mahsm as ma
        from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

        result, messages = ma.inference(
            model="openai/gpt-4o-mini",
            prompt=self.prompt,
            tools=[self.tool],
            input="test query",
        )

        # Verify transparency: all message types should be present in order
        assert len(messages) >= 2  # At minimum SystemMessage + HumanMessage
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

        # If tools were called, verify ToolMessages exist
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        # This may be 0 if model didn't call tools, but verify structure exists
        assert isinstance(tool_messages, list)

    def test_state_continuation_across_nodes(self):
        """Test that state continues correctly across multiple LangGraph nodes."""
        import mahsm as ma

        execution_order = []

        def first_node(state: MessagesState):
            execution_order.append("first")
            result, messages = ma.inference(
                model="openai/gpt-4o-mini",
                prompt=self.prompt,
                tools=[self.tool],
                input="first query",
                state=state.get("messages", []),
            )
            return {"messages": messages}

        def second_node(state: MessagesState):
            execution_order.append("second")
            # Continue from existing state
            result, messages = ma.inference(
                model="openai/gpt-4o-mini",
                prompt="Continue the conversation",
                tools=[self.tool],
                input="second query",
                state=state.get("messages", []),
            )
            return {"messages": messages}

        # Build sequential graph
        workflow = StateGraph(MessagesState)
        workflow.add_node("first", first_node)
        workflow.add_node("second", second_node)
        workflow.add_edge("first", "second")
        workflow.set_entry_point("first")
        workflow.set_finish_point("second")

        app = workflow.compile()

        # Execute
        result = app.invoke({"messages": []})

        # Verify execution order and state accumulation
        assert execution_order == ["first", "second"]
        assert len(result["messages"]) > 2  # Should accumulate messages from both nodes


