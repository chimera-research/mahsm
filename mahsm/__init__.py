"""
mahsm: Multi-Agent Hyper-Scaling Methods

A unified framework for building, tracing, evaluating, and optimizing multi-agent systems.

Core modules:
- mahsm.core: DSPy-LangGraph integration (dspy_node helper function)
- mahsm.dspy: DSPy re-export
- mahsm.graph: LangGraph re-export
- mahsm.trace: Langfuse tracing integration (init, @observe)
- mahsm.test: EvalProtocol integration (PytestHarness, evaluation helpers)
- mahsm.tuning: Tuning utilities
"""

__version__ = "0.2.2"

# Core integration
from .core import dspy_node

# Tracing (Langfuse integration)
from . import trace

# Curated namespaces
from . import dspy
from . import graph
from . import tune

# Testing module (may not work on Windows due to eval-protocol dependency)
try:
    from . import test
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(
        f"mahsm.test module unavailable (likely Windows compatibility issue): {e}"
    )

# Top-level convenience re-exports
from .graph import END, START
from .dspy import Module, Signature, InputField, OutputField
from langchain_core.messages import HumanMessage, AIMessage

__all__ = [
    # Core
    "dspy_node",
    # Namespaces
    "trace",
    "dspy",
    "graph",
    "tune",
    "test",
    # Convenience exports
    "END",
    "START",
    "Module",
    "Signature",
    "InputField",
    "OutputField",
    "HumanMessage",
    "AIMessage",
]