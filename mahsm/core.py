"""
mahsm.core - Core DSPy-LangGraph integration

This module provides the fundamental integration between DSPy modules
and LangGraph state graphs via the @dspy_node decorator.
"""

import dspy
import inspect
from typing import Any, Callable


def dspy_node(module_or_class):
    """
    Decorator/wrapper that transforms a DSPy Module into a LangGraph-compatible node.
    
    The decorator introspects the DSPy module's forward() method signature and
    automatically maps LangGraph state fields to module inputs and outputs.
    
    Usage Patterns:
    
        1. Class Decorator:
            ```python
            @ma.dspy_node
            class Researcher(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.cot = dspy.ChainOfThought("query -> answer")
                
                def forward(self, query):
                    return self.cot(query=query)
            
            # Use in graph
            workflow.add_node("researcher", Researcher())
            ```
        
        2. Instance Wrapper:
            ```python
            # Wrap a built-in DSPy module
            cot = dspy.ChainOfThought("query -> answer")
            node = ma.dspy_node(cot)
            
            # Use in graph
            workflow.add_node("cot", node)
            ```
    
    How it works:
        - Introspects forward() parameters (excluding 'self')
        - Extracts matching fields from LangGraph state
        - Calls module(**inputs) which internally calls forward()
        - Returns module output fields as state updates
    
    Args:
        module_or_class: Either a DSPy Module class or instance
    
    Returns:
        LangGraph-compatible node function or wrapper class
    
    Raises:
        TypeError: If input is not a DSPy Module class or instance
    """
    # Case 1: Decorating a class
    if inspect.isclass(module_or_class):
        if not issubclass(module_or_class, dspy.Module):
            raise TypeError(
                f"@dspy_node can only decorate dspy.Module subclasses. "
                f"Got: {module_or_class.__name__}"
            )
        
        # Return a wrapper that instantiates the module and converts to node
        class WrappedModule:
            def __init__(self, *args, **kwargs):
                self._module = module_or_class(*args, **kwargs)
                self._input_fields = _get_forward_params(self._module)
            
            def __call__(self, state: dict) -> dict:
                return _execute_module(self._module, state, self._input_fields)
        
        return WrappedModule
    
    # Case 2: Wrapping an instance
    elif isinstance(module_or_class, dspy.Module):
        module = module_or_class
        input_fields = _get_forward_params(module)
        
        def node_function(state: dict) -> dict:
            return _execute_module(module, state, input_fields)
        
        return node_function
    
    else:
        raise TypeError(
            f"@dspy_node requires a dspy.Module class or instance. "
            f"Got: {type(module_or_class).__name__}"
        )


def _get_forward_params(module: dspy.Module) -> list[str]:
    """Extract parameter names from a module's forward() method."""
    sig = inspect.signature(module.forward)
    return [name for name in sig.parameters.keys() if name != 'self']


def _execute_module(
    module: dspy.Module,
    state: dict,
    input_fields: list[str]
) -> dict:
    """
    Execute a DSPy module with state inputs and return state updates.
    
    This function:
    1. Extracts required inputs from state
    2. Calls module(**inputs) - the recommended DSPy pattern
    3. Extracts output fields from the result
    4. Returns them as state updates
    """
    # Extract inputs from state
    inputs = {
        field: state[field]
        for field in input_fields
        if field in state
    }
    
    # Execute module using __call__ (which internally calls forward)
    # This is the recommended DSPy pattern vs calling forward() directly
    result = module(**inputs)
    
    # Extract output fields (non-private attributes)
    outputs = {
        key: value
        for key, value in result.__dict__.items()
        if not key.startswith('_')
    }
    
    return outputs


__all__ = ["dspy_node"]