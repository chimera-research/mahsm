"""
API Contract: Prompt Management
MAHSM v0.1.0 - ma.prompt module

This file defines the expected API surface for prompt save/load operations.
Contract tests will validate these signatures and behaviors.
"""

from typing import List, Dict, Any, Optional, Callable
from pathlib import Path


def save(
    compiled_agent: Any,  # DSPy compiled program
    name: str,
    version: str,
    tools: List[Callable],
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Extract optimized prompt from DSPy compiled program and save with tool schemas.
    
    Args:
        compiled_agent: DSPy compiled program with optimized signature
        name: Task identifier for the prompt
        version: Version identifier (e.g., "v1", "v2.1")
        tools: List of tool functions to extract schemas from
        metadata: Optional additional metadata to store
        
    Returns:
        Path to the created JSON artifact file
        
    Raises:
        ValueError: If compiled_agent has no optimized prompt
        ValidationError: If tool schemas are invalid
        IOError: If unable to write artifact file
        
    Contract Requirements:
        - MUST extract prompt from compiled_agent.predict.signature.instructions
        - MUST generate OpenAI function schemas for all tools
        - MUST create ~/.mahsm/prompts/{name}_{version}.json
        - MUST include creation timestamp and optimizer metadata
        - MUST validate tool schemas before saving
    """
    pass


def load(
    name_version: str,
    validate_tools: Optional[List[Callable]] = None
) -> str:
    """
    Load saved prompt and optionally validate tool compatibility.
    
    Args:
        name_version: Combined name and version (e.g., "task_name_v1")
        validate_tools: Optional list of tools to validate against saved schemas
        
    Returns:
        The prompt string from the artifact
        
    Raises:
        FileNotFoundError: If artifact file doesn't exist
        ValidationError: If validate_tools don't match saved schemas
        JSONDecodeError: If artifact file is corrupted
        
    Contract Requirements:
        - MUST load from ~/.mahsm/prompts/{name_version}.json
        - MUST validate tool schemas if validate_tools provided
        - MUST compare tool names and parameter structures exactly
        - MUST return only the prompt string, not full artifact
    """
    pass


# Expected artifact schema for contract validation
ARTIFACT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "minLength": 1},
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "type": {"const": "object"},
                            "properties": {"type": "object"},
                            "required": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["type", "properties"]
                    }
                },
                "required": ["name", "description", "parameters"]
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "created_at": {"type": "string", "format": "date-time"},
                "optimizer": {"type": "string"},
                "version": {"type": "string"}
            },
            "required": ["created_at", "optimizer", "version"]
        },
        "task_name": {"type": "string"},
        "task_version": {"type": "string"}
    },
    "required": ["prompt", "tools", "metadata", "task_name", "task_version"]
}
