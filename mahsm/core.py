from langfuse import get_client
from langfuse.langchain import CallbackHandler
from openinference.instrumentation.dspy import DSPyInstrumentor
import os
import dspy
import inspect

def init():
    """
    Initializes the mahsm environment for unified tracing.
    Reads LANGFUSE_* environment variables to set up the client.
    """
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        print("mahsm: Warning - LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY not found. Tracing will be disabled.")
        return None
    
    get_client()
    print("mahsm: Langfuse client initialized.")
    
    DSPyInstrumentor().instrument()
    print("mahsm: DSPy instrumented for automatic tracing.")
    return CallbackHandler()

def dspy_node(module_or_class):
    """
    A decorator/wrapper that transforms a dspy.Module (class or instance) into
    a LangGraph-compatible node function.
    
    Usage:
        # As a decorator on a class
        @ma.dspy_node
        class MyModule(ma.Module):
            ...
        
        # As a functional wrapper on an instance
        cot = ma.dspy.ChainOfThought("query -> answer")
        node = ma.dspy_node(cot)
    """
    # Check if we're decorating a class or wrapping an instance
    if inspect.isclass(module_or_class):
        # Class decorator path
        if not issubclass(module_or_class, dspy.Module):
            raise TypeError("dspy_node can only decorate dspy.Module subclasses.")
        
        # Return a wrapper class that will instantiate and convert
        class WrappedModule:
            def __init__(self, *args, **kwargs):
                self.dspy_instance = module_or_class(*args, **kwargs)
                # Introspect the forward method, excluding 'self'
                params = inspect.signature(self.dspy_instance.forward).parameters
                self.input_fields = [k for k in params.keys() if k != 'self']
            
            def __call__(self, state: dict) -> dict:
                # 1. Gather inputs from state by matching keys with forward() args
                dspy_inputs = {key: state[key] for key in self.input_fields if key in state}
                
                # 2. Execute the DSPy module (call forward directly)
                result = self.dspy_instance(**dspy_inputs)
                
                # 3. Prepare state updates by returning the module's output fields
                state_updates = {
                    key: value for key, value in result.__dict__.items() if not key.startswith('_')
                }
                return state_updates
        
        return WrappedModule
    
    elif isinstance(module_or_class, dspy.Module):
        # Instance wrapping path
        dspy_instance = module_or_class
        
        # Introspect the forward method, excluding 'self'
        params = inspect.signature(dspy_instance.forward).parameters
        input_fields = [k for k in params.keys() if k != 'self']
        
        def node_function(state: dict) -> dict:
            # 1. Gather inputs from state by matching keys with forward() args
            dspy_inputs = {key: state[key] for key in input_fields if key in state}
            
            # 2. Execute the DSPy module
            result = dspy_instance(**dspy_inputs)
            
            # 3. Prepare state updates by returning the module's output fields
            state_updates = {
                key: value for key, value in result.__dict__.items() if not key.startswith('_')
            }
            return state_updates
        
        return node_function
    
    else:
        raise TypeError("dspy_node requires either a dspy.Module class or instance.")