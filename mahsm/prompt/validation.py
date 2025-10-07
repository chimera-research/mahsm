"""
Tool Schema Validation
MAHSM v0.1.0

Handles OpenAI function schema extraction and validation for tools.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, get_type_hints
from mahsm.exceptions import ValidationError


def extract_tool_schema(tool_func: Callable) -> Dict[str, Any]:
    """
    Extract OpenAI function calling format schema from a Python function.

    Args:
        tool_func: Python function to extract schema from

    Returns:
        Tool schema in OpenAI function calling format

    Raises:
        ValidationError: If schema cannot be extracted
    """
    if not callable(tool_func):
        raise ValidationError(
            f"Tool must be callable, got {type(tool_func)}",
            validation_type="tool_callable",
            details={"tool": str(tool_func)},
        )

    # Get function name
    name = tool_func.__name__

    # Get docstring for description
    description = tool_func.__doc__ or f"Function {name}"
    description = description.strip().split("\n")[0]  # First line only

    # Get function signature
    try:
        sig = inspect.signature(tool_func)
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"Could not extract signature from tool '{name}': {e}",
            validation_type="tool_signature",
        )

    # Build parameters schema
    properties = {}
    required = []

    # Get type hints if available
    try:
        type_hints = get_type_hints(tool_func)
    except Exception:
        type_hints = {}

    for param_name, param in sig.parameters.items():
        # Skip self, cls, *args, **kwargs
        if param_name in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Determine parameter type
        param_type = "string"  # Default
        param_schema = {"type": param_type}

        if param_name in type_hints:
            hint = type_hints[param_name]
            param_schema = python_type_to_json_schema(hint)

        # Add parameter annotation if available
        if param.annotation != inspect.Parameter.empty:
            # Use annotation to refine schema
            anno_schema = python_type_to_json_schema(param.annotation)
            param_schema.update(anno_schema)

        properties[param_name] = param_schema

        # Check if parameter is required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Build OpenAI function schema
    schema = {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }

    return schema


def python_type_to_json_schema(python_type: Any) -> Dict[str, Any]:
    """
    Convert Python type annotation to JSON Schema type.

    Args:
        python_type: Python type annotation

    Returns:
        JSON Schema type definition
    """
    # Handle basic types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    # Get origin type (for generics like List[str])
    origin = getattr(python_type, "__origin__", None)

    if origin is not None:
        if origin is list:
            # Handle List[T]
            args = getattr(python_type, "__args__", ())
            if args:
                item_schema = python_type_to_json_schema(args[0])
                return {"type": "array", "items": item_schema}
            return {"type": "array"}

        elif origin is dict:
            return {"type": "object"}

    # Check direct type mapping
    for py_type, json_schema in type_mapping.items():
        if python_type is py_type or python_type == py_type:
            return json_schema

    # Default to string for unknown types
    return {"type": "string"}


def extract_tools_schemas(tools: List[Callable]) -> List[Dict[str, Any]]:
    """
    Extract schemas from multiple tools.

    Args:
        tools: List of tool functions

    Returns:
        List of tool schemas in OpenAI format

    Raises:
        ValidationError: If any tool schema extraction fails
    """
    schemas = []
    errors = []

    for tool in tools:
        try:
            schema = extract_tool_schema(tool)
            schemas.append(schema)
        except ValidationError as e:
            errors.append(f"Tool '{getattr(tool, '__name__', str(tool))}': {e}")

    if errors:
        raise ValidationError(
            "Failed to extract schemas from tools:\n" + "\n".join(errors),
            validation_type="tools_extraction",
            details={"error_count": len(errors)},
        )

    return schemas


def validate_tool_against_schema(tool_func: Callable, saved_schema: Dict[str, Any]) -> bool:
    """
    Validate that a tool function matches a saved schema.

    Args:
        tool_func: Python function to validate
        saved_schema: Saved tool schema in OpenAI format

    Returns:
        True if tool matches schema

    Raises:
        ValidationError: If tool doesn't match schema
    """
    # Extract current schema
    current_schema = extract_tool_schema(tool_func)

    # Compare name
    if current_schema["name"] != saved_schema["name"]:
        raise ValidationError(
            f"Tool name mismatch: expected '{saved_schema['name']}', got '{current_schema['name']}'",
            validation_type="tool_name_mismatch",
            details={"expected": saved_schema["name"], "actual": current_schema["name"]},
        )

    # Compare parameters structure
    current_params = current_schema["parameters"]
    saved_params = saved_schema["parameters"]

    # Compare required parameters
    current_required = set(current_params.get("required", []))
    saved_required = set(saved_params.get("required", []))

    if current_required != saved_required:
        raise ValidationError(
            f"Tool '{current_schema['name']}' required parameters mismatch",
            validation_type="tool_parameters_mismatch",
            details={
                "expected_required": list(saved_required),
                "actual_required": list(current_required),
                "missing": list(saved_required - current_required),
                "extra": list(current_required - saved_required),
            },
        )

    # Compare parameter names (properties)
    current_props = set(current_params.get("properties", {}).keys())
    saved_props = set(saved_params.get("properties", {}).keys())

    if current_props != saved_props:
        raise ValidationError(
            f"Tool '{current_schema['name']}' parameter names mismatch",
            validation_type="tool_parameters_mismatch",
            details={
                "expected_params": list(saved_props),
                "actual_params": list(current_props),
                "missing": list(saved_props - current_props),
                "extra": list(current_props - saved_props),
            },
        )

    # Note: We don't strictly validate parameter types because Python's
    # dynamic typing and inference can lead to minor differences that
    # don't affect functionality. We only validate structure.

    return True


def validate_tools_against_schemas(
    tools: List[Callable], saved_schemas: List[Dict[str, Any]]
) -> bool:
    """
    Validate multiple tools against saved schemas.

    Args:
        tools: List of tool functions
        saved_schemas: List of saved tool schemas

    Returns:
        True if all tools match their schemas

    Raises:
        ValidationError: If validation fails
    """
    # Check count match
    if len(tools) != len(saved_schemas):
        raise ValidationError(
            f"Tool count mismatch: expected {len(saved_schemas)} tools, got {len(tools)}",
            validation_type="tool_count_mismatch",
            details={"expected_count": len(saved_schemas), "actual_count": len(tools)},
        )

    # Create mapping of tool names from saved schemas
    saved_by_name = {schema["name"]: schema for schema in saved_schemas}

    # Validate each tool
    for tool in tools:
        tool_name = tool.__name__

        if tool_name not in saved_by_name:
            available_tools = list(saved_by_name.keys())
            raise ValidationError(
                f"Tool '{tool_name}' not found in saved schemas",
                validation_type="tool_not_found",
                details={"missing_tool": tool_name, "available_tools": available_tools},
            )

        # Validate tool against its saved schema
        validate_tool_against_schema(tool, saved_by_name[tool_name])

    return True


__all__ = [
    "extract_tool_schema",
    "extract_tools_schemas",
    "validate_tool_against_schema",
    "validate_tools_against_schemas",
    "python_type_to_json_schema",
]

