"""
Integration tests for mahsm - Testing LangGraph integration with dspy_node.
"""
import pytest
import mahsm as ma
from typing import TypedDict


# Define test state types
class SimpleState(TypedDict):
    input: str
    output: str


class MultiNodeState(TypedDict):
    query: str
    processed: str
    result: str


class ConditionalState(TypedDict):
    counter: int
    status: str


def test_single_node_graph():
    """Test a simple graph with a single dspy_node."""
    # Define a simple module
    class Echo(ma.Module):
        def forward(self, input):
            class Result:
                def __init__(self):
                    self.output = f"Echo: {input}"
            return Result()
    
    # Create graph
    workflow = ma.graph.StateGraph(SimpleState)
    workflow.add_node("echo", ma.dspy_node(Echo)())
    workflow.add_edge(ma.START, "echo")
    workflow.add_edge("echo", ma.END)
    
    # Compile and test
    graph = workflow.compile()
    result = graph.invoke({"input": "test"})
    
    assert "output" in result
    assert result["output"] == "Echo: test"


def test_multi_node_sequential_graph():
    """Test a graph with multiple sequential dspy_nodes."""
    # Define modules
    class Processor(ma.Module):
        def forward(self, query):
            class Result:
                def __init__(self):
                    self.processed = f"Processed: {query}"
            return Result()
    
    class Finalizer(ma.Module):
        def forward(self, processed):
            class Result:
                def __init__(self):
                    self.result = f"Final: {processed}"
            return Result()
    
    # Create graph
    workflow = ma.graph.StateGraph(MultiNodeState)
    workflow.add_node("processor", ma.dspy_node(Processor)())
    workflow.add_node("finalizer", ma.dspy_node(Finalizer)())
    workflow.add_edge(ma.START, "processor")
    workflow.add_edge("processor", "finalizer")
    workflow.add_edge("finalizer", ma.END)
    
    # Compile and test
    graph = workflow.compile()
    result = graph.invoke({"query": "hello"})
    
    assert "result" in result
    assert "Processed: hello" in result["result"]


def test_graph_with_conditional_edges():
    """Test a graph with conditional routing."""
    # Define module
    class Counter(ma.Module):
        def forward(self, counter):
            class Result:
                def __init__(self):
                    self.counter = counter + 1
                    self.status = "done" if counter >= 2 else "continue"
            return Result()
    
    # Define routing function
    def should_continue(state: ConditionalState):
        if state.get("status") == "done":
            return ma.END
        return "counter"
    
    # Create graph
    workflow = ma.graph.StateGraph(ConditionalState)
    workflow.add_node("counter", ma.dspy_node(Counter)())
    workflow.add_edge(ma.START, "counter")
    workflow.add_conditional_edges("counter", should_continue)
    
    # Compile and test
    graph = workflow.compile()
    result = graph.invoke({"counter": 0})
    
    assert result["counter"] == 3  # 0 -> 1 -> 2 -> 3
    assert result["status"] == "done"


def test_graph_with_instance_wrapped_node():
    """Test using functional wrapper pattern on an instance."""
    class Doubler(ma.Module):
        def forward(self, value):
            class Result:
                def __init__(self):
                    self.value = value * 2
            return Result()
    
    # Create instance and wrap it
    doubler_instance = Doubler()
    doubler_node = ma.dspy_node(doubler_instance)
    
    # Create graph
    class ValueState(TypedDict):
        value: int
    
    workflow = ma.graph.StateGraph(ValueState)
    workflow.add_node("doubler", doubler_node)
    workflow.add_edge(ma.START, "doubler")
    workflow.add_edge("doubler", ma.END)
    
    # Compile and test
    graph = workflow.compile()
    result = graph.invoke({"value": 5})
    
    assert result["value"] == 10


def test_graph_preserves_existing_state():
    """Test that nodes don't overwrite unrelated state fields."""
    class Adder(ma.Module):
        def forward(self, x):
            class Result:
                def __init__(self):
                    self.y = x + 10
            return Result()
    
    class AdderState(TypedDict):
        x: int
        y: int
        z: int
    
    workflow = ma.graph.StateGraph(AdderState)
    workflow.add_node("adder", ma.dspy_node(Adder)())
    workflow.add_edge(ma.START, "adder")
    workflow.add_edge("adder", ma.END)
    
    # Compile and test
    graph = workflow.compile()
    result = graph.invoke({"x": 5, "z": 100})
    
    assert result["x"] == 5   # Original preserved
    assert result["y"] == 15  # New value added
    assert result["z"] == 100 # Unrelated field preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
