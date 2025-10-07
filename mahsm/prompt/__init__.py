"""
Prompt management module for MAHSM.

Provides functions for saving and loading optimized prompts with tool schemas.
"""

from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
from mahsm.prompt.storage import (
    extract_prompt_from_dspy,
    save_artifact,
    load_artifact,
    get_prompts_dir,
)
from mahsm.prompt.validation import extract_tools_schemas, validate_tools_against_schemas


def save(
    compiled_agent: Any,
    name: str,
    version: str,
    tools: List[Callable],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Extract optimized prompt from DSPy compiled program and save with tool schemas.

    This function extracts the optimized prompt from a DSPy compiled program
    (typically from GEPA or other optimizers) and saves it along with the tool
    schemas in a JSON artifact for later use.

    Args:
        compiled_agent: DSPy compiled program with optimized signature
        name: Task identifier for the prompt (used in filename)
        version: Version identifier (e.g., "v1", "v2.1")
        tools: List of tool functions to extract schemas from
        metadata: Optional additional metadata (e.g., {"optimizer": "GEPA"})

    Returns:
        Path to the created JSON artifact file (~/.mahsm/prompts/{name}_{version}.json)

    Raises:
        ValueError: If compiled_agent has no optimized prompt
        ValidationError: If tool schemas are invalid
        IOError: If unable to write artifact file

    Example:
        >>> import dspy
        >>> import mahsm as ma
        >>> 
        >>> # After DSPy optimization
        >>> compiled_program = optimizer.compile(program, trainset=data)
        >>> 
        >>> # Save with tools
        >>> artifact_path = ma.prompt.save(
        ...     compiled_program,
        ...     name="data_analysis",
        ...     version="v1",
        ...     tools=[calculate_stats, generate_chart],
        ...     metadata={"optimizer": "GEPA"}
        ... )
        >>> print(f"Saved to: {artifact_path}")
    """
    # Extract prompt from DSPy program
    prompt = extract_prompt_from_dspy(compiled_agent)

    # Extract tool schemas
    tool_schemas = extract_tools_schemas(tools)

    # Save artifact
    return save_artifact(
        prompt=prompt,
        tools_schemas=tool_schemas,
        task_name=name,
        task_version=version,
        metadata=metadata,
    )


def load(name_version: str, validate_tools: Optional[List[Callable]] = None) -> str:
    """
    Load saved prompt and optionally validate tool compatibility.

    This function loads a previously saved prompt artifact and optionally
    validates that the provided tools match the saved tool schemas.

    Args:
        name_version: Combined name and version (e.g., "task_name_v1")
        validate_tools: Optional list of tools to validate against saved schemas

    Returns:
        The prompt string from the artifact

    Raises:
        FileNotFoundError: If artifact file doesn't exist
        ValidationError: If validate_tools don't match saved schemas
        json.JSONDecodeError: If artifact file is corrupted

    Example:
        >>> import mahsm as ma
        >>> 
        >>> # Load prompt with tool validation
        >>> prompt = ma.prompt.load(
        ...     "data_analysis_v1",
        ...     validate_tools=[calculate_stats, generate_chart]
        ... )
        >>> 
        >>> # Use in LangGraph node
        >>> def my_node(state):
        ...     result, messages = ma.inference(
        ...         model="openai/gpt-4o-mini",
        ...         prompt=prompt,
        ...         tools=[calculate_stats, generate_chart],
        ...         input=state["messages"][-1].content
        ...     )
        ...     return {"messages": messages}
    """
    # Load artifact
    artifact = load_artifact(name_version)

    # Validate tools if provided
    if validate_tools:
        validate_tools_against_schemas(validate_tools, artifact["tools"])

    # Return prompt string
    return artifact["prompt"]


__all__ = ["save", "load"]
