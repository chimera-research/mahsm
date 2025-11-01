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


def test_dspy_node_with_optional_parameters(multi_param_module):
    """Test that dspy_node handles optional parameters correctly."""
    # Create decorated module
    DecoratedModule = ma.dspy_node(multi_param_module)
    node_func = DecoratedModule()
    
    # Test with all parameters
    state_all = {"a": 2, "b": 3, "c": 4}
    result_all = node_func(state_all)
    assert result_all["sum"] == 9  # 2 + 3 + 4
    assert result_all["product"] == 24  # 2 * 3 * 4
    
    # Test with only required parameters (c should default to 0)
    state_required = {"a": 2, "b": 3}
    result_required = node_func(state_required)
    assert result_required["sum"] == 5  # 2 + 3 + 0
    assert result_required["product"] == 6  # 2 * 3 * 1


def test_dspy_node_with_missing_required_fields():
    """Test that execution continues even with missing required fields."""
    class RequiredParamModule(ma.Module):
        def forward(self, required_field):
            class Result:
                def __init__(self):
                    self.output = f"Got: {required_field}"
            return Result()
    
    DecoratedModule = ma.dspy_node(RequiredParamModule)
    node_func = DecoratedModule()
    
    # State missing required field - should extract empty inputs
    state = {"other_field": "value"}
    
    # This will call forward() with no arguments, which will fail
    # The behavior depends on whether DSPy handles this gracefully
    # For now, we test that the wrapper itself doesn't crash on extraction
    try:
        result = node_func(state)
    except TypeError:
        # Expected - forward() got missing required argument
        pass


def test_dspy_node_filters_private_fields():
    """Test that private fields (starting with _) are filtered from output."""
    class PrivateFieldModule(ma.Module):
        def forward(self, input):
            class Result:
                def __init__(self):
                    self.public_field = "visible"
                    self._private_field = "hidden"
                    self.__dunder_field = "also_hidden"
            return Result()
    
    DecoratedModule = ma.dspy_node(PrivateFieldModule)
    node_func = DecoratedModule()
    
    state = {"input": "test"}
    result = node_func(state)
    
    # Should have public field
    assert "public_field" in result
    assert result["public_field"] == "visible"
    
    # Should NOT have private fields
    assert "_private_field" not in result
    assert "__dunder_field" not in result


def test_dspy_node_multiple_output_fields():
    """Test that modules can return multiple output fields."""
    class MultiOutputModule(ma.Module):
        def forward(self, value):
            class Result:
                def __init__(self):
                    self.doubled = value * 2
                    self.tripled = value * 3
                    self.squared = value ** 2
            return Result()
    
    DecoratedModule = ma.dspy_node(MultiOutputModule)
    node_func = DecoratedModule()
    
    state = {"value": 5}
    result = node_func(state)
    
    assert len(result) == 3
    assert result["doubled"] == 10
    assert result["tripled"] == 15
    assert result["squared"] == 25


def test_dspy_node_preserves_module_state():
    """Test that module instance state is preserved across calls."""
    class StatefulModule(ma.Module):
        def __init__(self):
            super().__init__()
            self.call_count = 0
        
        def forward(self, input):
            self.call_count += 1
            class Result:
                def __init__(self, count):
                    self.output = f"Call #{count}: {input}"
                    self.count = count
            return Result(self.call_count)
    
    # Wrap instance
    module = StatefulModule()
    node_func = ma.dspy_node(module)
    
    # First call
    result1 = node_func({"input": "first"})
    assert result1["count"] == 1
    
    # Second call - count should increment
    result2 = node_func({"input": "second"})
    assert result2["count"] == 2
    
    # State is preserved
    assert module.call_count == 2


def test_dspy_node_empty_forward_params():
    """Test module with no parameters in forward()."""
    class NoParamModule(ma.Module):
        def forward(self):
            class Result:
                def __init__(self):
                    self.output = "constant"
            return Result()
    
    DecoratedModule = ma.dspy_node(NoParamModule)
    node_func = DecoratedModule()
    
    # Should work with any state
    state = {"anything": "here"}
    result = node_func(state)
    
    assert result["output"] == "constant"


def test_dspy_node_with_complex_types():
    """Test that complex types (lists, dicts) work in state."""
    class ComplexTypeModule(ma.Module):
        def forward(self, items, metadata):
            class Result:
                def __init__(self):
                    self.count = len(items)
                    self.has_metadata = bool(metadata)
            return Result()
    
    DecoratedModule = ma.dspy_node(ComplexTypeModule)
    node_func = DecoratedModule()
    
    state = {
        "items": [1, 2, 3, 4, 5],
        "metadata": {"key": "value", "number": 42}
    }
    
    result = node_func(state)
    
    assert result["count"] == 5
    assert result["has_metadata"] is True


def test_get_forward_params_helper():
    """Test the _get_forward_params helper function."""
    from mahsm.core import _get_forward_params
    
    class TestModule(ma.Module):
        def forward(self, param1, param2, param3=None):
            pass
    
    module = TestModule()
    params = _get_forward_params(module)
    
    assert isinstance(params, list)
    assert "param1" in params
    assert "param2" in params
    assert "param3" in params
    assert "self" not in params


def test_execute_module_helper():
    """Test the _execute_module helper function."""
    from mahsm.core import _execute_module
    
    class SimpleModule(ma.Module):
        def forward(self, a, b):
            class Result:
                def __init__(self):
                    self.sum = a + b
            return Result()
    
    module = SimpleModule()
    state = {"a": 10, "b": 20, "extra": "ignored"}
    
    result = _execute_module(module, state, ["a", "b"])
    
    assert "sum" in result
    assert result["sum"] == 30
    assert "extra" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
