"""
mahsm: Multi-Agent Hyper-Scaling Methods

A unified framework for building, tracing, evaluating, and optimizing multi-agent systems.

Core modules:
- mahsm.core: DSPy-LangGraph integration (@dspy_node decorator)
- mahsm.tracing: Langfuse tracing integration (init, @observe)
- mahsm.testing: EvalProtocol integration (PytestHarness, evaluation helpers)
"""

__version__ = "0.1.1"

# Core integration
from .core import dspy_node

# Tracing (Langfuse integration)
from . import tracing

# Curated namespaces
from . import dspy
from . import graph

# Testing module (may not work on Windows due to eval-protocol dependency)
try:
    from . import testing
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(
        f"mahsm.testing module unavailable (likely Windows compatibility issue): {e}"
    )

# Top-level convenience re-exports
from .graph import END, START
from .dspy import Module, Signature, InputField, OutputField
from langchain_core.messages import HumanMessage, AIMessage

__all__ = [
    # Core
    "dspy_node",
    # Namespaces
    "tracing",
    "dspy",
    "graph",
    "testing",
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