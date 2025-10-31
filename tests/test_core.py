"""
Unit tests for mahsm/core.py - Testing dspy_node decorator/wrapper functionality.
"""
import pytest
import mahsm as ma
import dspy
from typing import TypedDict


# Define a simple state type for testing
class TestState(TypedDict):
    input_text: str
    output_text: str


class SimpleModule(ma.Module):
    """Test module for decorator pattern."""
    def __init__(self):
        super().__init__()
        self.predictor = ma.dspy.Predict("input_text -> output_text")
    
    def forward(self, input_text):
        return self.predictor(input_text=input_text)


def test_dspy_node_decorator_on_class():
    """Test that @ma.dspy_node works as a class decorator."""
    # Decorate the class
    DecoratedModule = ma.dspy_node(SimpleModule)
    
    # Instantiate it
    node_func = DecoratedModule()
    
    # Verify it's callable
    assert callable(node_func)
    
    # Test execution with mock state (without actual LLM call)
    # We'll just verify the structure works
    assert hasattr(node_func, '__call__')


def test_dspy_node_wrapper_on_instance():
    """Test that ma.dspy_node() works as a functional wrapper on instances."""
    # Create a DSPy module instance
    cot = ma.dspy.ChainOfThought("query -> answer")
    
    # Wrap it
    node_func = ma.dspy_node(cot)
    
    # Verify it's callable
    assert callable(node_func)
    
    # Verify it's a function (not a class) - check it's an actual function
    import types
    assert isinstance(node_func, types.FunctionType)


def test_dspy_node_wrapper_on_builtin_modules():
    """Test that ma.dspy_node() works with built-in DSPy modules."""
    # Test with Predict
    predict = ma.dspy.Predict("input -> output")
    predict_node = ma.dspy_node(predict)
    assert callable(predict_node)
    
    # Test with ChainOfThought
    cot = ma.dspy.ChainOfThought("question -> answer")
    cot_node = ma.dspy_node(cot)
    assert callable(cot_node)


def test_dspy_node_invalid_input():
    """Test that dspy_node raises TypeError for invalid inputs."""
    # Test with a non-DSPy class
    class NotAModule:
        pass
    
    with pytest.raises(TypeError, match="dspy.Module"):
        ma.dspy_node(NotAModule)
    
    # Test with a random object
    with pytest.raises(TypeError, match="dspy.Module class or instance"):
        ma.dspy_node("not a module")


def test_dspy_node_state_mapping():
    """Test that dspy_node correctly maps state fields to forward() parameters."""
    # Create a custom module with specific parameters
    class MultiParamModule(ma.Module):
        def forward(self, param1, param2, param3=None):
            # Create a simple result object
            class Result:
                def __init__(self):
                    self.output1 = f"Result from {param1} and {param2}"
                    self.output2 = f"Optional: {param3}"
            return Result()
    
    # Wrap it
    DecoratedModule = ma.dspy_node(MultiParamModule)
    node_func = DecoratedModule()
    
    # Create test state
    state = {
        "param1": "value1",
        "param2": "value2",
        "param3": "value3",
        "extra_field": "ignored"
    }
    
    # Execute
    result = node_func(state)
    
    # Verify output structure
    assert "output1" in result
    assert "output2" in result
    assert "extra_field" not in result  # Should not pass through
    assert "_" not in "".join(result.keys())  # No private fields


def test_dspy_node_excludes_self_parameter():
    """Test that dspy_node correctly excludes 'self' from input fields."""
    class ModuleWithSelf(ma.Module):
        def forward(self, input1, input2):
            class Result:
                def __init__(self):
                    self.combined = f"{input1}+{input2}"
            return Result()
    
    DecoratedModule = ma.dspy_node(ModuleWithSelf)
    node_func = DecoratedModule()
    
    # Should work without 'self' in state
    state = {"input1": "A", "input2": "B"}
    result = node_func(state)
    
    assert "combined" in result
    assert result["combined"] == "A+B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
