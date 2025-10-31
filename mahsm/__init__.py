"""
mahsm: Multi-Agent Hyper-Scaling Methods
A unified framework for building, tracing, evaluating, and optimizing multi-agent systems.
"""

from .core import init, dspy_node

# --- Curated Namespaces ---
from . import dspy
from . import graph

# Testing module may not work on Windows due to eval-protocol dependency
try:
    from . import testing
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(f"mahsm.testing module unavailable (likely Windows compatibility issue): {e}")

# --- Top-level convenience re-exports ---
from .graph import END, START
from .dspy import Module, Signature, InputField, OutputField
from langchain_core.messages import HumanMessage, AIMessage