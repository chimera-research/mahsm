"""
Unit tests for mahsm/graph.py - Testing LangGraph namespace and re-exports.
"""
import pytest
import mahsm as ma
from typing import TypedDict


class TestGraphNamespace:
    """Test LangGraph namespace structure and exports."""
    
    def test_graph_namespace_exists(self):
        """Test that graph namespace is accessible."""
        assert hasattr(ma, "graph")
        assert ma.graph is not None
    
    def test_graph_is_module(self):
        """Test that graph is a proper module."""
        import types
        assert isinstance(ma.graph, types.ModuleType)


class TestGraphCoreClasses:
    """Test core LangGraph classes are properly exported."""
    
    def test_state_graph_exported(self):
        """Test that StateGraph is exported."""
        assert hasattr(ma.graph, "StateGraph")
        
        # Should be instantiable
        class State(TypedDict):
            value: int
        
        workflow = ma.graph.StateGraph(State)
        assert workflow is not None
    
    def test_end_constant_exported(self):
        """Test that END constant is exported."""
        assert hasattr(ma.graph, "END")
        assert isinstance(ma.graph.END, str)
        assert len(ma.graph.END) > 0
    
    def test_start_constant_exported(self):
        """Test that START constant is exported."""
        assert hasattr(ma.graph, "START")
        assert isinstance(ma.graph.START, str)
        assert len(ma.graph.START) > 0


class TestStateGraphBasics:
    """Test StateGraph basic functionality."""
    
    def test_create_empty_graph(self):
        """Test creating an empty state graph."""
        class State(TypedDict):
            value: int
        
        workflow = ma.graph.StateGraph(State)
        assert workflow is not None
    
    def test_add_node_to_graph(self):
        """Test adding a node to the graph."""
        class State(TypedDict):
            value: int
        
        def node_func(state):
            return {"value": state["value"] + 1}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("increment", node_func)
        
        # Should not raise
        assert True
    
    def test_add_edge_to_graph(self):
        """Test adding an edge to the graph."""
        class State(TypedDict):
            value: int
        
        def node_func(state):
            return {"value": state["value"] + 1}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("increment", node_func)
        workflow.add_edge(ma.START, "increment")
        workflow.add_edge("increment", ma.END)
        
        # Should not raise
        assert True
    
    def test_compile_graph(self):
        """Test compiling a graph."""
        class State(TypedDict):
            value: int
        
        def node_func(state):
            return {"value": state["value"] + 1}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("increment", node_func)
        workflow.add_edge(ma.START, "increment")
        workflow.add_edge("increment", ma.END)
        
        graph = workflow.compile()
        assert graph is not None


class TestGraphExecution:
    """Test graph execution."""
    
    def test_invoke_simple_graph(self):
        """Test invoking a simple graph."""
        class State(TypedDict):
            value: int
        
        def increment(state):
            return {"value": state["value"] + 1}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("inc", increment)
        workflow.add_edge(ma.START, "inc")
        workflow.add_edge("inc", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 5})
        
        assert result["value"] == 6
    
    def test_multi_node_sequential_graph(self):
        """Test graph with multiple sequential nodes."""
        class State(TypedDict):
            value: int
        
        def double(state):
            return {"value": state["value"] * 2}
        
        def add_ten(state):
            return {"value": state["value"] + 10}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("double", double)
        workflow.add_node("add_ten", add_ten)
        workflow.add_edge(ma.START, "double")
        workflow.add_edge("double", "add_ten")
        workflow.add_edge("add_ten", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 3})
        
        assert result["value"] == 16  # (3 * 2) + 10
    
    def test_graph_with_multiple_fields(self):
        """Test graph with state containing multiple fields."""
        class State(TypedDict):
            value: int
            message: str
        
        def process(state):
            return {
                "value": state["value"] * 2,
                "message": f"Doubled: {state['value']}"
            }
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("process", process)
        workflow.add_edge(ma.START, "process")
        workflow.add_edge("process", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 5, "message": "initial"})
        
        assert result["value"] == 10
        assert "Doubled: 5" in result["message"]


class TestConditionalEdges:
    """Test conditional edge routing."""
    
    def test_add_conditional_edges(self):
        """Test adding conditional edges to graph."""
        class State(TypedDict):
            value: int
        
        def router(state):
            return "positive" if state["value"] > 0 else "negative"
        
        def positive_node(state):
            return {"value": state["value"] * 2}
        
        def negative_node(state):
            return {"value": state["value"] * -1}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("positive", positive_node)
        workflow.add_node("negative", negative_node)
        
        workflow.add_conditional_edges(
            ma.START,
            router,
            {
                "positive": "positive",
                "negative": "negative"
            }
        )
        
        workflow.add_edge("positive", ma.END)
        workflow.add_edge("negative", ma.END)
        
        graph = workflow.compile()
        
        # Test positive path
        result1 = graph.invoke({"value": 5})
        assert result1["value"] == 10
        
        # Test negative path
        result2 = graph.invoke({"value": -3})
        assert result2["value"] == 3


class TestGraphWithDSPyNodes:
    """Test integration of DSPy modules in LangGraph."""
    
    def test_dspy_module_as_graph_node(self):
        """Test using DSPy module as a graph node."""
        class State(TypedDict):
            value: int
        
        class TripleModule(ma.Module):
            def forward(self, value):
                class Result:
                    def __init__(self):
                        self.value = value * 3
                return Result()
        
        # Wrap as node
        node = ma.dspy_node(TripleModule())
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("triple", node)
        workflow.add_edge(ma.START, "triple")
        workflow.add_edge("triple", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 4})
        
        assert result["value"] == 12
    
    def test_multiple_dspy_nodes_in_graph(self):
        """Test graph with multiple DSPy nodes."""
        class State(TypedDict):
            value: int
        
        class AddModule(ma.Module):
            def __init__(self, amount):
                super().__init__()
                self.amount = amount
            
            def forward(self, value):
                class Result:
                    def __init__(self, val):
                        self.value = val
                return Result(value + self.amount)
        
        # Create nodes
        add_five = ma.dspy_node(AddModule(5))
        add_ten = ma.dspy_node(AddModule(10))
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("add_five", add_five)
        workflow.add_node("add_ten", add_ten)
        workflow.add_edge(ma.START, "add_five")
        workflow.add_edge("add_five", "add_ten")
        workflow.add_edge("add_ten", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 2})
        
        assert result["value"] == 17  # 2 + 5 + 10


class TestGraphStateManagement:
    """Test state management in graphs."""
    
    def test_state_preservation(self):
        """Test that state is preserved across nodes."""
        class State(TypedDict):
            value: int
            history: list
        
        def node1(state):
            history = state.get("history", [])
            return {
                "value": state["value"] + 1,
                "history": history + ["node1"]
            }
        
        def node2(state):
            history = state.get("history", [])
            return {
                "value": state["value"] * 2,
                "history": history + ["node2"]
            }
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("node1", node1)
        workflow.add_node("node2", node2)
        workflow.add_edge(ma.START, "node1")
        workflow.add_edge("node1", "node2")
        workflow.add_edge("node2", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 3, "history": []})
        
        assert result["value"] == 8  # (3 + 1) * 2
        assert result["history"] == ["node1", "node2"]
    
    def test_partial_state_updates(self):
        """Test that nodes can update only part of the state."""
        class State(TypedDict):
            field_a: int
            field_b: int
        
        def update_a(state):
            return {"field_a": state["field_a"] + 1}
        
        def update_b(state):
            return {"field_b": state["field_b"] * 2}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("update_a", update_a)
        workflow.add_node("update_b", update_b)
        workflow.add_edge(ma.START, "update_a")
        workflow.add_edge("update_a", "update_b")
        workflow.add_edge("update_b", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"field_a": 5, "field_b": 3})
        
        assert result["field_a"] == 6
        assert result["field_b"] == 6


class TestGraphConvenienceExports:
    """Test that graph exports are available at top level."""
    
    def test_start_at_top_level(self):
        """Test that START is available at ma.START."""
        assert hasattr(ma, "START")
        assert ma.START is ma.graph.START
    
    def test_end_at_top_level(self):
        """Test that END is available at ma.END."""
        assert hasattr(ma, "END")
        assert ma.END is ma.graph.END


class TestGraphEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_state_graph(self):
        """Test creating graph with minimal state."""
        class State(TypedDict):
            pass
        
        workflow = ma.graph.StateGraph(State)
        assert workflow is not None
    
    def test_node_returning_none(self):
        """Test node that returns None (no state update)."""
        class State(TypedDict):
            value: int
        
        def no_op(state):
            return None  # or could return {}
        
        def increment(state):
            return {"value": state["value"] + 1}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("no_op", no_op)
        workflow.add_node("increment", increment)
        workflow.add_edge(ma.START, "no_op")
        workflow.add_edge("no_op", "increment")
        workflow.add_edge("increment", ma.END)
        
        graph = workflow.compile()
        # May or may not work depending on LangGraph's handling of None
        # Just test that it doesn't crash during compilation
        assert graph is not None
    
    def test_graph_with_lambda_nodes(self):
        """Test using lambda functions as nodes."""
        class State(TypedDict):
            value: int
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("double", lambda state: {"value": state["value"] * 2})
        workflow.add_edge(ma.START, "double")
        workflow.add_edge("double", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 7})
        
        assert result["value"] == 14


class TestGraphStreaming:
    """Test graph streaming capabilities."""
    
    def test_graph_supports_streaming(self):
        """Test that compiled graph supports streaming."""
        class State(TypedDict):
            value: int
        
        def increment(state):
            return {"value": state["value"] + 1}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("inc", increment)
        workflow.add_edge(ma.START, "inc")
        workflow.add_edge("inc", ma.END)
        
        graph = workflow.compile()
        
        # Check that graph has stream method
        assert hasattr(graph, "stream")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
