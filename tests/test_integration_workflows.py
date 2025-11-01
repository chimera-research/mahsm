"""
Integration tests for mahsm - End-to-end workflow testing.

Tests the complete integration of DSPy + LangGraph + Tracing working together.
"""
import pytest
import mahsm as ma
from typing import TypedDict
from unittest.mock import patch


class TestBasicWorkflows:
    """Test basic end-to-end workflows."""
    
    def test_simple_dspy_langgraph_pipeline(self):
        """Test simple pipeline: DSPy module -> LangGraph workflow."""
        class State(TypedDict):
            input: int
            output: int
        
        # Create DSPy module
        class DoubleModule(ma.Module):
            def forward(self, input):
                class Result:
                    def __init__(self):
                        self.output = input * 2
                return Result()
        
        # Wrap as node
        double_node = ma.dspy_node(DoubleModule())
        
        # Build workflow
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("double", double_node)
        workflow.add_edge(ma.START, "double")
        workflow.add_edge("double", ma.END)
        
        # Execute
        graph = workflow.compile()
        result = graph.invoke({"input": 5, "output": 0})
        
        assert result["output"] == 10
    
    def test_multi_step_dspy_pipeline(self):
        """Test multi-step pipeline with multiple DSPy modules."""
        class State(TypedDict):
            value: int
        
        # Create multiple modules
        class AddModule(ma.Module):
            def __init__(self, amount):
                super().__init__()
                self.amount = amount
            
            def forward(self, value):
                class Result:
                    def __init__(self, v):
                        self.value = v
                return Result(value + self.amount)
        
        class MultiplyModule(ma.Module):
            def __init__(self, factor):
                super().__init__()
                self.factor = factor
            
            def forward(self, value):
                class Result:
                    def __init__(self, v):
                        self.value = v
                return Result(value * self.factor)
        
        # Create pipeline
        add_five = ma.dspy_node(AddModule(5))
        multiply_three = ma.dspy_node(MultiplyModule(3))
        add_ten = ma.dspy_node(AddModule(10))
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("add_five", add_five)
        workflow.add_node("multiply_three", multiply_three)
        workflow.add_node("add_ten", add_ten)
        
        workflow.add_edge(ma.START, "add_five")
        workflow.add_edge("add_five", "multiply_three")
        workflow.add_edge("multiply_three", "add_ten")
        workflow.add_edge("add_ten", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 2})
        
        # (2 + 5) * 3 + 10 = 31
        assert result["value"] == 31
    
    def test_mixed_node_types_workflow(self):
        """Test workflow mixing DSPy nodes and regular functions."""
        class State(TypedDict):
            value: int
            message: str
        
        # DSPy module
        class ProcessModule(ma.Module):
            def forward(self, value):
                class Result:
                    def __init__(self):
                        self.value = value * 2
                return Result()
        
        # Regular function node
        def add_message(state):
            return {
                "value": state["value"],
                "message": f"Processed: {state['value']}"
            }
        
        # Build workflow
        process_node = ma.dspy_node(ProcessModule())
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("process", process_node)
        workflow.add_node("message", add_message)
        workflow.add_edge(ma.START, "process")
        workflow.add_edge("process", "message")
        workflow.add_edge("message", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 7, "message": ""})
        
        assert result["value"] == 14
        assert "Processed: 14" in result["message"]


class TestTracedWorkflows:
    """Test workflows with tracing integration."""
    
    def test_traced_dspy_module_in_workflow(self):
        """Test DSPy module with @observe in workflow."""
        class State(TypedDict):
            value: int
        
        class TracedModule(ma.Module):
            @ma.tracing.observe()
            def helper(self, x):
                return x * 3
            
            def forward(self, value):
                result = self.helper(value)
                class Result:
                    def __init__(self):
                        self.value = result
                return Result()
        
        node = ma.dspy_node(TracedModule())
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("traced", node)
        workflow.add_edge(ma.START, "traced")
        workflow.add_edge("traced", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 4})
        
        assert result["value"] == 12
    
    def test_traced_function_nodes_in_workflow(self):
        """Test regular functions with @observe in workflow."""
        class State(TypedDict):
            value: int
        
        @ma.tracing.observe()
        def step1(state):
            return {"value": state["value"] + 5}
        
        @ma.tracing.observe()
        def step2(state):
            return {"value": state["value"] * 2}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("step1", step1)
        workflow.add_node("step2", step2)
        workflow.add_edge(ma.START, "step1")
        workflow.add_edge("step1", "step2")
        workflow.add_edge("step2", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 3})
        
        # (3 + 5) * 2 = 16
        assert result["value"] == 16
    
    def test_nested_traced_operations(self):
        """Test nested traced operations in workflow."""
        class State(TypedDict):
            value: int
        
        @ma.tracing.observe(name="outer_helper")
        def outer_helper(x):
            @ma.tracing.observe(name="inner_helper")
            def inner_helper(y):
                return y + 1
            return inner_helper(x * 2)
        
        def node_func(state):
            result = outer_helper(state["value"])
            return {"value": result}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("process", node_func)
        workflow.add_edge(ma.START, "process")
        workflow.add_edge("process", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 5})
        
        # (5 * 2) + 1 = 11
        assert result["value"] == 11


class TestConditionalWorkflows:
    """Test conditional routing workflows."""
    
    def test_conditional_routing_with_dspy_nodes(self):
        """Test conditional routing using DSPy modules."""
        class State(TypedDict):
            value: int
            path: str
        
        # Router function
        def router(state):
            return "positive" if state["value"] > 0 else "negative"
        
        # DSPy modules for each path
        class PositiveModule(ma.Module):
            def forward(self, value):
                class Result:
                    def __init__(self):
                        self.value = value * 2
                        self.path = "took_positive"
                return Result()
        
        class NegativeModule(ma.Module):
            def forward(self, value):
                class Result:
                    def __init__(self):
                        self.value = abs(value)
                        self.path = "took_negative"
                return Result()
        
        # Create nodes
        positive_node = ma.dspy_node(PositiveModule())
        negative_node = ma.dspy_node(NegativeModule())
        
        # Build workflow
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
        result1 = graph.invoke({"value": 5, "path": ""})
        assert result1["value"] == 10
        assert result1["path"] == "took_positive"
        
        # Test negative path
        result2 = graph.invoke({"value": -3, "path": ""})
        assert result2["value"] == 3
        assert result2["path"] == "took_negative"
    
    def test_multi_way_conditional_routing(self):
        """Test multi-way conditional routing."""
        class State(TypedDict):
            value: int
            category: str
        
        def categorize(state):
            val = state["value"]
            if val < 0:
                return "negative"
            elif val == 0:
                return "zero"
            elif val < 10:
                return "small"
            else:
                return "large"
        
        def negative_handler(state):
            return {"value": 0, "category": "negative"}
        
        def zero_handler(state):
            return {"value": 1, "category": "zero"}
        
        def small_handler(state):
            return {"value": state["value"] * 2, "category": "small"}
        
        def large_handler(state):
            return {"value": state["value"] // 2, "category": "large"}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("negative", negative_handler)
        workflow.add_node("zero", zero_handler)
        workflow.add_node("small", small_handler)
        workflow.add_node("large", large_handler)
        
        workflow.add_conditional_edges(
            ma.START,
            categorize,
            {
                "negative": "negative",
                "zero": "zero",
                "small": "small",
                "large": "large"
            }
        )
        
        for node in ["negative", "zero", "small", "large"]:
            workflow.add_edge(node, ma.END)
        
        graph = workflow.compile()
        
        # Test all paths
        assert graph.invoke({"value": -5, "category": ""})["category"] == "negative"
        assert graph.invoke({"value": 0, "category": ""})["category"] == "zero"
        assert graph.invoke({"value": 5, "category": ""})["category"] == "small"
        assert graph.invoke({"value": 20, "category": ""})["category"] == "large"


class TestStatefulWorkflows:
    """Test workflows with complex state management."""
    
    def test_accumulator_workflow(self):
        """Test workflow that accumulates values."""
        class State(TypedDict):
            values: list
            sum: int
        
        def add_value(state):
            values = state.get("values", [])
            return {
                "values": values + [len(values) + 1],
                "sum": state.get("sum", 0) + (len(values) + 1)
            }
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("add1", add_value)
        workflow.add_node("add2", add_value)
        workflow.add_node("add3", add_value)
        
        workflow.add_edge(ma.START, "add1")
        workflow.add_edge("add1", "add2")
        workflow.add_edge("add2", "add3")
        workflow.add_edge("add3", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"values": [], "sum": 0})
        
        assert result["values"] == [1, 2, 3]
        assert result["sum"] == 6
    
    def test_history_tracking_workflow(self):
        """Test workflow that tracks execution history."""
        class State(TypedDict):
            value: int
            history: list
        
        class HistoryModule(ma.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name
            
            def forward(self, value, history):
                class Result:
                    def __init__(self):
                        self.value = value + 1
                        self.history = history + [self.name]
                return Result()
        
        # Create tracked nodes
        node1 = ma.dspy_node(HistoryModule("node1"))
        node2 = ma.dspy_node(HistoryModule("node2"))
        node3 = ma.dspy_node(HistoryModule("node3"))
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("n1", node1)
        workflow.add_node("n2", node2)
        workflow.add_node("n3", node3)
        
        workflow.add_edge(ma.START, "n1")
        workflow.add_edge("n1", "n2")
        workflow.add_edge("n2", "n3")
        workflow.add_edge("n3", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 0, "history": []})
        
        assert result["value"] == 3
        assert result["history"] == ["node1", "node2", "node3"]


class TestComplexIntegrations:
    """Test complex real-world integration scenarios."""
    
    def test_pipeline_with_validation(self):
        """Test pipeline with validation steps."""
        class State(TypedDict):
            data: int
            valid: bool
            result: int
        
        def validate(state):
            return {"valid": state["data"] > 0}
        
        def router(state):
            return "process" if state["valid"] else "error"
        
        class ProcessModule(ma.Module):
            def forward(self, data):
                class Result:
                    def __init__(self):
                        self.result = data * 2
                return Result()
        
        def error_handler(state):
            return {"result": -1}
        
        process_node = ma.dspy_node(ProcessModule())
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("validate", validate)
        workflow.add_node("process", process_node)
        workflow.add_node("error", error_handler)
        
        workflow.add_edge(ma.START, "validate")
        workflow.add_conditional_edges(
            "validate",
            router,
            {"process": "process", "error": "error"}
        )
        workflow.add_edge("process", ma.END)
        workflow.add_edge("error", ma.END)
        
        graph = workflow.compile()
        
        # Valid input
        result1 = graph.invoke({"data": 5, "valid": False, "result": 0})
        assert result1["result"] == 10
        
        # Invalid input
        result2 = graph.invoke({"data": -3, "valid": False, "result": 0})
        assert result2["result"] == -1
    
    def test_parallel_processing_simulation(self):
        """Test simulated parallel processing."""
        class State(TypedDict):
            input: int
            branch_a: int
            branch_b: int
            combined: int
        
        class BranchAModule(ma.Module):
            def forward(self, input):
                class Result:
                    def __init__(self):
                        self.branch_a = input * 2
                return Result()
        
        class BranchBModule(ma.Module):
            def forward(self, input):
                class Result:
                    def __init__(self):
                        self.branch_b = input + 10
                return Result()
        
        def combine(state):
            return {"combined": state["branch_a"] + state["branch_b"]}
        
        # Note: LangGraph executes sequentially, but this simulates parallel pattern
        branch_a_node = ma.dspy_node(BranchAModule())
        branch_b_node = ma.dspy_node(BranchBModule())
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("branch_a", branch_a_node)
        workflow.add_node("branch_b", branch_b_node)
        workflow.add_node("combine", combine)
        
        workflow.add_edge(ma.START, "branch_a")
        workflow.add_edge("branch_a", "branch_b")
        workflow.add_edge("branch_b", "combine")
        workflow.add_edge("combine", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"input": 5, "branch_a": 0, "branch_b": 0, "combined": 0})
        
        # branch_a: 5 * 2 = 10
        # branch_b: 5 + 10 = 15
        # combined: 10 + 15 = 25
        assert result["combined"] == 25


class TestStreamingWorkflows:
    """Test streaming capabilities."""
    
    def test_workflow_supports_streaming(self):
        """Test that workflows support streaming."""
        class State(TypedDict):
            value: int
        
        def increment(state):
            return {"value": state["value"] + 1}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("inc1", increment)
        workflow.add_node("inc2", increment)
        workflow.add_node("inc3", increment)
        
        workflow.add_edge(ma.START, "inc1")
        workflow.add_edge("inc1", "inc2")
        workflow.add_edge("inc2", "inc3")
        workflow.add_edge("inc3", ma.END)
        
        graph = workflow.compile()
        
        # Verify stream method exists
        assert hasattr(graph, "stream")
        
        # Test streaming execution
        events = list(graph.stream({"value": 0}))
        
        # Should have events for each node
        assert len(events) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
