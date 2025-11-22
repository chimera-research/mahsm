"""
mahsm.core - Core DSPy-LangGraph integration

This module provides the fundamental integration between DSPy modules
and LangGraph state graphs via the @dspy_node decorator.
"""

import dspy
import inspect
from typing import Any, Callable, Optional, Union, List, Dict


def dspy_node(
    module_or_class=None,
    *,
    input_fields: Optional[Union[List[str], Dict[str, str]]] = None,
    output_fields: Optional[Union[List[str], Dict[str, str]]] = None,
    input_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    output_transform: Optional[Callable[[Any], Dict[str, Any]]] = None,
    state_key_prefix: str = "",
    pass_full_state: bool = False
):
    """
    Decorator/wrapper that transforms a DSPy Module into a LangGraph-compatible node.
    
    The decorator supports two modes:
    1. Simple auto-introspection (default): Automatically maps state fields to module inputs
    2. Advanced mapping: Custom field mapping and transforms for complex scenarios
    
    Usage Patterns:
    
        1. Simple Class Decorator (auto-introspection):
            ```python
            @ma.dspy_node
            class Researcher(dspy.Module):
                def forward(self, query):
                    return self.cot(query=query)
            
            workflow.add_node("researcher", Researcher())
            ```
        
        2. Instance Wrapper (auto-introspection):
            ```python
            cot = dspy.ChainOfThought("query -> answer")
            node = ma.dspy_node(cot)
            workflow.add_node("cot", node)
            ```
        
        3. Advanced Field Mapping:
            ```python
            # Map state keys to different DSPy parameter names
            node = ma.dspy_node(
                program,
                input_fields={"question": "user_query"},  # DSPy param <- state key
                output_fields={"answer": "bot_response"}   # state key <- prediction attr
            )
            ```
        
        4. Custom Transforms:
            ```python
            node = ma.dspy_node(
                program,
                input_transform=lambda state: {"query": state["messages"][-1][1]},
                output_transform=lambda pred: {"response": pred.answer, "meta": pred.reasoning}
            )
            ```
    
    Args:
        module_or_class: DSPy Module class or instance (required)
        input_fields: Field mapping for inputs. Can be:
            - None (default): Auto-introspect from forward() signature
            - List of field names: ["question"] maps state["question"] -> program(question=...)
            - Dict mapping: {"question": "user_query"} maps state["user_query"] -> program(question=...)
        output_fields: Field mapping for outputs. Can be:
            - None (default): Auto-extract all non-private attributes
            - List of field names: ["answer"] maps prediction.answer -> state["answer"]
            - Dict mapping: {"answer": "result"} maps prediction.answer -> state["result"]
        input_transform: Custom function to extract inputs from state. If provided, input_fields is ignored.
            Signature: func(state: Dict) -> Dict[str, Any] (kwargs for DSPy program)
        output_transform: Custom function to transform DSPy output. If provided, output_fields is ignored.
            Signature: func(prediction: Any) -> Dict[str, Any] (state updates)
        state_key_prefix: Prefix to add to all output keys (e.g., "dspy_" -> "dspy_answer")
        pass_full_state: If True, pass entire state as 'state' kwarg to program
    
    Returns:
        LangGraph-compatible node function or wrapper class
    
    Raises:
        TypeError: If input is not a DSPy Module class or instance
    """
    
    # Handle decorator called without arguments: @ma.dspy_node
    if module_or_class is None:
        # Called with arguments: @ma.dspy_node(input_fields=...)
        def decorator(module_or_class_inner):
            return _create_node_wrapper(
                module_or_class_inner,
                input_fields=input_fields,
                output_fields=output_fields,
                input_transform=input_transform,
                output_transform=output_transform,
                state_key_prefix=state_key_prefix,
                pass_full_state=pass_full_state
            )
        return decorator
    else:
        # Called without arguments: @ma.dspy_node or ma.dspy_node(module)
        return _create_node_wrapper(
            module_or_class,
            input_fields=input_fields,
            output_fields=output_fields,
            input_transform=input_transform,
            output_transform=output_transform,
            state_key_prefix=state_key_prefix,
            pass_full_state=pass_full_state
        )


def _create_node_wrapper(
    module_or_class,
    input_fields,
    output_fields,
    input_transform,
    output_transform,
    state_key_prefix,
    pass_full_state
):
    """Internal function to create the actual wrapper."""
    
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
                self._config = {
                    'input_fields': input_fields,
                    'output_fields': output_fields,
                    'input_transform': input_transform,
                    'output_transform': output_transform,
                    'state_key_prefix': state_key_prefix,
                    'pass_full_state': pass_full_state,
                }
            
            def __call__(self, state: dict) -> dict:
                return _execute_module_enhanced(self._module, state, self._config)
        
        return WrappedModule
    
    # Case 2: Wrapping an instance
    elif isinstance(module_or_class, dspy.Module):
        module = module_or_class
        config = {
            'input_fields': input_fields,
            'output_fields': output_fields,
            'input_transform': input_transform,
            'output_transform': output_transform,
            'state_key_prefix': state_key_prefix,
            'pass_full_state': pass_full_state,
        }
        
        def node_function(state: dict) -> dict:
            return _execute_module_enhanced(module, state, config)
        
        return node_function
    
    else:
        raise TypeError(
            f"@dspy_node requires a dspy.Module class or instance. "
            f"Got: {type(module_or_class).__name__}"
        )


def _execute_module_enhanced(
    module: dspy.Module,
    state: dict,
    config: dict
) -> dict:
    """
    Execute a DSPy module with enhanced field mapping and transforms.
    
    This function:
    1. Extracts inputs from state (via transform, field mapping, or auto-introspection)
    2. Calls module(**inputs)
    3. Transforms outputs (via transform, field mapping, or auto-extraction)
    4. Returns state updates
    """
    input_transform = config.get('input_transform')
    output_transform = config.get('output_transform')
    input_fields = config.get('input_fields')
    output_fields = config.get('output_fields')
    state_key_prefix = config.get('state_key_prefix', '')
    pass_full_state = config.get('pass_full_state', False)
    
    # Step 1: Extract inputs from state
    if input_transform:
        # Custom transformation
        program_inputs = input_transform(state)
    elif input_fields:
        program_inputs = {}
        if isinstance(input_fields, dict):
            # Dict mapping: {"dspy_param": "state_key"}
            for dspy_key, state_key in input_fields.items():
                if state_key in state:
                    program_inputs[dspy_key] = state[state_key]
        else:
            # List: ["param1", "param2"] - direct mapping
            for field in input_fields:
                if field in state:
                    program_inputs[field] = state[field]
    else:
        # Auto-introspection from forward() signature
        sig = inspect.signature(module.forward)
        forward_params = [name for name in sig.parameters.keys() if name != 'self']
        program_inputs = {
            field: state[field]
            for field in forward_params
            if field in state
        }
    
    # Add full state if requested
    if pass_full_state:
        program_inputs['state'] = state
    
    # Step 2: Call the DSPy program
    try:
        prediction = module(**program_inputs)
    except Exception as e:
        raise RuntimeError(
            f"Error calling DSPy program: {e}\n"
            f"Program inputs: {program_inputs}\n"
            f"Available state keys: {list(state.keys())}"
        )
    
    # Step 3: Transform outputs back to state updates
    if output_transform:
        # Custom transformation
        state_updates = output_transform(prediction)
    elif output_fields:
        state_updates = {}
        if isinstance(output_fields, dict):
            # Dict mapping: {"prediction_attr": "state_key"}
            for pred_key, state_key in output_fields.items():
                if hasattr(prediction, pred_key):
                    full_key = f"{state_key_prefix}{state_key}"
                    state_updates[full_key] = getattr(prediction, pred_key)
        else:
            # List: ["answer", "reasoning"] - direct mapping
            for field in output_fields:
                if hasattr(prediction, field):
                    full_key = f"{state_key_prefix}{field}"
                    state_updates[full_key] = getattr(prediction, field)
    else:
        # Auto-extract all non-private attributes
        state_updates = {}
        for key, value in prediction.__dict__.items():
            if not key.startswith('_'):
                full_key = f"{state_key_prefix}{key}"
                state_updates[full_key] = value
    
    return state_updates


__all__ = ["dspy_node"]