"""
MAHSM (Multi-Agent Hyper-Scaling Methods) v0.1.0

A Python library that bridges DSPy prompt optimization with LangGraph runtime orchestration.

This library provides three core primitives:
- ma.prompt: Save and load optimized prompts with tool schemas
- ma.inference: Execute prompts in agentic loops with full message transparency
- ma.config: Global configuration and environment integration

Quick Start:
    >>> import mahsm as ma
    >>> 
    >>> # 1. Save optimized prompt from DSPy
    >>> ma.prompt.save(compiled_program, name="task", version="v1", tools=[tool1])
    >>> 
    >>> # 2. Load and validate
    >>> prompt = ma.prompt.load("task_v1", validate_tools=[tool1])
    >>> 
    >>> # 3. Execute with transparency
    >>> result, messages = ma.inference(
    ...     model="openai/gpt-4o-mini",
    ...     prompt=prompt,
    ...     tools=[tool1],
    ...     input="query"
    ... )

Key Features:
    - **DSPy Integration**: Extract optimized prompts from DSPy GEPA and other optimizers
    - **LangGraph Compatible**: Works seamlessly with LangGraph's MessagesState
    - **Full Traceability**: Automatic LangFuse integration for observability
    - **Tool Validation**: OpenAI function schema validation ensures runtime safety
    - **Minimal Dependencies**: Only uses LangChain Core message types

Documentation:
    - API Reference: docs/api/
    - User Guides: docs/guides/
    - Examples: docs/examples/
    - Concepts: docs/concepts/

GitHub: https://github.com/chimera-research/mahsm
"""

__version__ = "0.1.0"
__author__ = "MAHSM Contributors"
__license__ = "MIT"

# Import core modules
from mahsm import prompt
from mahsm import config
from mahsm.inference import inference

# Import exceptions for user access
from mahsm.exceptions import (
    MAHSMError,
    ValidationError,
    ToolExecutionError,
    MaxIterationsError,
    ConfigurationError,
    ModelError,
)

__all__ = [
    "__version__",
    "prompt",
    "inference",
    "config",
    # Exceptions
    "MAHSMError",
    "ValidationError",
    "ToolExecutionError",
    "MaxIterationsError",
    "ConfigurationError",
    "ModelError",
]
