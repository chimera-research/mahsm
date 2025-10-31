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

class dspy_node:
    """
    A decorator that transforms a dspy.Module class into a factory for
    creating LangGraph-compatible node functions.
    """
    def __init__(self, cls):
        if not issubclass(cls, dspy.Module):
            raise TypeError("dspy_node can only decorate dspy.Module subclasses.")
        self.dspy_module_class = cls
        # Introspect the forward method to find the expected inputs
        self.input_fields = inspect.signature(cls.forward).parameters.keys()

    def __call__(self, *args, **kwargs):
        """
        When the decorated class is instantiated, this returns the actual
        LangGraph node function that LangGraph will execute.
        """
        dspy_instance = self.dspy_module_class(*args, **kwargs)

        def node_function(state: dict) -> dict:
            # 1. Gather inputs from state by matching keys with forward() args
            dspy_inputs = {key: state[key] for key in self.input_fields if key in state}

            # 2. Execute the DSPy module
            result = dspy_instance.with_retry(**dspy_inputs)

            # 3. Prepare state updates by returning the module's output fields
            state_updates = {
                key: value for key, value in result.__dict__.items() if not key.startswith('_')
            }
            return state_updates
            
        return node_function