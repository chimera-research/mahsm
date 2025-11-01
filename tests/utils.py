"""
Utility functions and helpers for mahsm tests.
"""
from typing import Any, Dict
import mahsm as ma


# ============================================================================
# Mock Result Generators
# ============================================================================

def create_mock_result(**fields) -> object:
    """
    Create a mock result object with specified fields.
    
    Args:
        **fields: Field names and values for the result object
    
    Returns:
        Object with the specified attributes
    
    Example:
        result = create_mock_result(answer="42", confidence=0.95)
        assert result.answer == "42"
        assert result.confidence == 0.95
    """
    class MockResult:
        def __init__(self):
            for key, value in fields.items():
                setattr(self, key, value)
    
    return MockResult()


def create_mock_dspy_module(forward_return_value=None):
    """
    Create a mock DSPy module that returns a specific value.
    
    Args:
        forward_return_value: Value to return from forward() method
    
    Returns:
        Mock DSPy module instance
    
    Example:
        module = create_mock_dspy_module(
            forward_return_value=create_mock_result(output="test")
        )
    """
    class MockModule(ma.Module):
        def forward(self, *args, **kwargs):
            return forward_return_value or create_mock_result(output="mocked")
    
    return MockModule()


# ============================================================================
# State Builders
# ============================================================================

def build_state(**fields) -> Dict[str, Any]:
    """
    Build a state dictionary with specified fields.
    
    Args:
        **fields: State field names and values
    
    Returns:
        State dictionary
    
    Example:
        state = build_state(query="test", value=42)
        assert state["query"] == "test"
        assert state["value"] == 42
    """
    return dict(fields)


def merge_states(*states: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple state dictionaries.
    
    Args:
        *states: State dictionaries to merge
    
    Returns:
        Merged state dictionary
    
    Example:
        state1 = {"a": 1, "b": 2}
        state2 = {"b": 3, "c": 4}
        merged = merge_states(state1, state2)
        assert merged == {"a": 1, "b": 3, "c": 4}
    """
    result = {}
    for state in states:
        result.update(state)
    return result


# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_has_fields(obj: Any, *field_names: str):
    """
    Assert that an object has all specified fields.
    
    Args:
        obj: Object to check
        *field_names: Names of fields that must exist
    
    Raises:
        AssertionError: If any field is missing
    
    Example:
        result = create_mock_result(answer="42", confidence=0.95)
        assert_has_fields(result, "answer", "confidence")
    """
    missing = [name for name in field_names if not hasattr(obj, name)]
    if missing:
        raise AssertionError(
            f"Object missing fields: {missing}. "
            f"Available fields: {list(obj.__dict__.keys())}"
        )


def assert_no_private_fields(obj: Any):
    """
    Assert that an object has no private fields (starting with '_').
    
    Args:
        obj: Object to check
    
    Raises:
        AssertionError: If any private fields are found
    
    Example:
        result = create_mock_result(answer="42", _internal="hidden")
        assert_no_private_fields(result)  # Would fail
    """
    private_fields = [
        name for name in dir(obj)
        if name.startswith('_') and not name.startswith('__')
    ]
    if private_fields:
        raise AssertionError(
            f"Object has private fields: {private_fields}"
        )


def assert_state_contains(state: Dict[str, Any], **expected):
    """
    Assert that state contains expected key-value pairs.
    
    Args:
        state: State dictionary to check
        **expected: Expected key-value pairs
    
    Raises:
        AssertionError: If any expected pair is missing or incorrect
    
    Example:
        state = {"query": "test", "result": "success"}
        assert_state_contains(state, query="test", result="success")
    """
    for key, expected_value in expected.items():
        if key not in state:
            raise AssertionError(f"State missing key: {key}")
        
        actual_value = state[key]
        if actual_value != expected_value:
            raise AssertionError(
                f"State[{key}] mismatch: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )


def assert_state_preserves(
    original: Dict[str, Any],
    updated: Dict[str, Any],
    *preserved_keys: str
):
    """
    Assert that specific keys are preserved (unchanged) in updated state.
    
    Args:
        original: Original state
        updated: Updated state
        *preserved_keys: Keys that should remain unchanged
    
    Raises:
        AssertionError: If any key was modified
    
    Example:
        original = {"a": 1, "b": 2, "c": 3}
        updated = {"a": 1, "b": 99, "c": 3}
        assert_state_preserves(original, updated, "a", "c")  # Passes
        assert_state_preserves(original, updated, "b")       # Fails
    """
    for key in preserved_keys:
        if key not in original:
            raise AssertionError(f"Key {key} not in original state")
        if key not in updated:
            raise AssertionError(f"Key {key} missing from updated state")
        
        original_value = original[key]
        updated_value = updated[key]
        
        if original_value != updated_value:
            raise AssertionError(
                f"State[{key}] was modified: "
                f"{original_value!r} -> {updated_value!r}"
            )


# ============================================================================
# Graph Testing Helpers
# ============================================================================

def create_simple_graph(node_func, state_type):
    """
    Create a simple single-node graph for testing.
    
    Args:
        node_func: Node function to add to graph
        state_type: TypedDict class defining state structure
    
    Returns:
        Compiled LangGraph
    
    Example:
        graph = create_simple_graph(my_node_func, SimpleState)
        result = graph.invoke({"input": "test"})
    """
    workflow = ma.graph.StateGraph(state_type)
    workflow.add_node("node", node_func)
    workflow.add_edge(ma.START, "node")
    workflow.add_edge("node", ma.END)
    return workflow.compile()


def create_sequential_graph(node_funcs, node_names, state_type):
    """
    Create a sequential multi-node graph for testing.
    
    Args:
        node_funcs: List of node functions
        node_names: List of node names (must match length of node_funcs)
        state_type: TypedDict class defining state structure
    
    Returns:
        Compiled LangGraph
    
    Example:
        graph = create_sequential_graph(
            [node1, node2, node3],
            ["process", "transform", "finalize"],
            MyState
        )
    """
    if len(node_funcs) != len(node_names):
        raise ValueError("node_funcs and node_names must have same length")
    
    workflow = ma.graph.StateGraph(state_type)
    
    # Add nodes
    for name, func in zip(node_names, node_funcs):
        workflow.add_node(name, func)
    
    # Add edges
    workflow.add_edge(ma.START, node_names[0])
    for i in range(len(node_names) - 1):
        workflow.add_edge(node_names[i], node_names[i + 1])
    workflow.add_edge(node_names[-1], ma.END)
    
    return workflow.compile()
