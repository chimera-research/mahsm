"""
Prompt Artifact Storage
MAHSM v0.1.0

Handles JSON artifact save/load operations for optimized prompts.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from mahsm.exceptions import ValidationError


def get_prompts_dir() -> Path:
    """
    Get the prompts directory, creating it if it doesn't exist.

    Returns:
        Path to ~/.mahsm/prompts/ directory
    """
    mahsm_home = Path(os.environ.get("MAHSM_HOME", Path.home() / ".mahsm"))
    prompts_dir = mahsm_home / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    return prompts_dir


def extract_prompt_from_dspy(compiled_agent: Any) -> str:
    """
    Extract optimized prompt from DSPy compiled program.

    Args:
        compiled_agent: DSPy compiled program with optimized signature

    Returns:
        The optimized prompt string

    Raises:
        ValueError: If prompt cannot be extracted
    """
    try:
        # Try to access compiled_agent.predict.signature.instructions
        if hasattr(compiled_agent, "predict"):
            if hasattr(compiled_agent.predict, "signature"):
                if hasattr(compiled_agent.predict.signature, "instructions"):
                    instructions = compiled_agent.predict.signature.instructions
                    if instructions and isinstance(instructions, str):
                        return instructions

        # Alternative: Check if compiled_agent itself has signature
        if hasattr(compiled_agent, "signature"):
            if hasattr(compiled_agent.signature, "instructions"):
                instructions = compiled_agent.signature.instructions
                if instructions and isinstance(instructions, str):
                    return instructions

        raise ValueError("Could not find signature.instructions in compiled program")

    except AttributeError as e:
        raise ValueError(
            f"DSPy compiled program does not have expected structure: {e}\n"
            "Expected: compiled_agent.predict.signature.instructions"
        )


def save_artifact(
    prompt: str,
    tools_schemas: List[Dict[str, Any]],
    task_name: str,
    task_version: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save prompt artifact to JSON file.

    Args:
        prompt: The prompt string to save
        tools_schemas: List of tool schemas in OpenAI format
        task_name: Task identifier
        task_version: Version identifier
        metadata: Optional additional metadata

    Returns:
        Path to the created artifact file

    Raises:
        ValidationError: If artifact validation fails
        IOError: If file cannot be written
    """
    # Validate prompt
    if not prompt or not isinstance(prompt, str):
        raise ValidationError(
            "Prompt must be a non-empty string", validation_type="prompt", details={"prompt": prompt}
        )

    # Validate task_name and task_version for filename safety
    if not task_name or "/" in task_name or "\\" in task_name:
        raise ValidationError(
            f"Invalid task_name: '{task_name}'. Must not contain path separators.",
            validation_type="task_name",
        )

    # Create artifact structure
    artifact = {
        "prompt": prompt,
        "tools": tools_schemas,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "optimizer": metadata.get("optimizer", "unknown") if metadata else "unknown",
            "version": "1.0.0",
        },
        "task_name": task_name,
        "task_version": task_version,
    }

    # Merge additional metadata
    if metadata:
        for key, value in metadata.items():
            if key not in ["created_at", "version"]:
                artifact["metadata"][key] = value

    # Validate artifact structure
    validate_artifact_structure(artifact)

    # Determine file path
    prompts_dir = get_prompts_dir()
    filename = f"{task_name}_{task_version}.json"
    filepath = prompts_dir / filename

    # Atomic write: write to temp file, then rename
    temp_filepath = filepath.with_suffix(".json.tmp")

    try:
        with open(temp_filepath, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)

        # Atomic rename
        temp_filepath.replace(filepath)

        return filepath

    except Exception as e:
        # Clean up temp file on error
        if temp_filepath.exists():
            temp_filepath.unlink()
        raise IOError(f"Failed to write artifact to {filepath}: {e}")


def load_artifact(name_version: str) -> Dict[str, Any]:
    """
    Load prompt artifact from JSON file.

    Args:
        name_version: Combined name and version (e.g., "task_name_v1")

    Returns:
        The artifact dictionary

    Raises:
        FileNotFoundError: If artifact file doesn't exist
        json.JSONDecodeError: If artifact file is corrupted
    """
    prompts_dir = get_prompts_dir()
    filepath = prompts_dir / f"{name_version}.json"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Artifact '{name_version}' not found at {filepath}\n"
            f"Available artifacts: {list_artifacts()}"
        )

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            artifact = json.load(f)

        # Validate loaded artifact
        validate_artifact_structure(artifact)

        return artifact

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Artifact file '{name_version}.json' is corrupted: {e.msg}", e.doc, e.pos
        )


def validate_artifact_structure(artifact: Dict[str, Any]) -> None:
    """
    Validate artifact structure against expected schema.

    Args:
        artifact: The artifact dictionary to validate

    Raises:
        ValidationError: If artifact structure is invalid
    """
    required_fields = ["prompt", "tools", "metadata", "task_name", "task_version"]

    for field in required_fields:
        if field not in artifact:
            raise ValidationError(
                f"Artifact missing required field: '{field}'",
                validation_type="artifact_structure",
                details={"missing_field": field, "available_fields": list(artifact.keys())},
            )

    # Validate types
    if not isinstance(artifact["prompt"], str):
        raise ValidationError(
            "Artifact 'prompt' must be a string",
            validation_type="artifact_structure",
            details={"actual_type": type(artifact["prompt"]).__name__},
        )

    if not isinstance(artifact["tools"], list):
        raise ValidationError(
            "Artifact 'tools' must be a list",
            validation_type="artifact_structure",
            details={"actual_type": type(artifact["tools"]).__name__},
        )

    if not isinstance(artifact["metadata"], dict):
        raise ValidationError(
            "Artifact 'metadata' must be a dictionary",
            validation_type="artifact_structure",
            details={"actual_type": type(artifact["metadata"]).__name__},
        )

    # Validate metadata fields
    required_metadata_fields = ["created_at", "optimizer", "version"]
    for field in required_metadata_fields:
        if field not in artifact["metadata"]:
            raise ValidationError(
                f"Artifact metadata missing required field: '{field}'",
                validation_type="metadata_structure",
                details={"missing_field": field},
            )


def list_artifacts() -> List[str]:
    """
    List all available prompt artifacts.

    Returns:
        List of artifact names (without .json extension)
    """
    prompts_dir = get_prompts_dir()
    if not prompts_dir.exists():
        return []

    artifacts = []
    for filepath in prompts_dir.glob("*.json"):
        if not filepath.name.endswith(".tmp"):
            artifacts.append(filepath.stem)

    return sorted(artifacts)


__all__ = [
    "get_prompts_dir",
    "extract_prompt_from_dspy",
    "save_artifact",
    "load_artifact",
    "validate_artifact_structure",
    "list_artifacts",
]

