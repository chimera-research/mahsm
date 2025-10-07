"""
MAHSM Exception Hierarchy
v0.1.0

Custom exceptions for MAHSM with detailed error messages and context preservation.
"""

from typing import Dict, List, Any, Optional


class MAHSMError(Exception):
    """
    Base exception for all MAHSM errors.

    All MAHSM-specific exceptions inherit from this base class,
    allowing users to catch all MAHSM errors with a single except clause.
    """

    pass


class ValidationError(MAHSMError):
    """
    Raised when validation fails (tool schemas, configuration, etc.).

    This exception includes specific validation failure details to help
    users understand and fix the validation issue.

    Attributes:
        message: Detailed validation failure message
        validation_type: Type of validation that failed (e.g., 'tool_schema', 'config')
        details: Additional context about the validation failure
    """

    def __init__(
        self, message: str, validation_type: Optional[str] = None, details: Optional[Dict] = None
    ):
        self.validation_type = validation_type
        self.details = details or {}
        super().__init__(message)


class ToolExecutionError(MAHSMError):
    """
    Raised when tool execution fails during inference.

    This exception wraps the original tool execution error and includes
    the tool name and call details for debugging.

    Attributes:
        tool_name: Name of the tool that failed
        tool_call: The tool call that caused the failure
        original_error: The original exception that was raised
    """

    def __init__(self, tool_name: str, tool_call: Dict[str, Any], original_error: Exception):
        self.tool_name = tool_name
        self.tool_call = tool_call
        self.original_error = original_error

        # Construct detailed error message
        error_msg = (
            f"Tool '{tool_name}' execution failed: {original_error}\n"
            f"Tool call details: {tool_call}\n"
            f"Original error type: {type(original_error).__name__}"
        )

        super().__init__(error_msg)

    def __str__(self) -> str:
        """Return detailed string representation."""
        return (
            f"ToolExecutionError: Tool '{self.tool_name}' failed\n"
            f"  Original error: {self.original_error}\n"
            f"  Tool call: {self.tool_call}"
        )


class MaxIterationsError(MAHSMError):
    """
    Raised when inference loop exceeds maximum iterations.

    This exception includes the current iteration count and partial results
    to help debug infinite loops or set appropriate iteration limits.

    Attributes:
        max_iterations: The maximum iteration limit that was exceeded
        current_messages: The message history at the point of failure
        iteration_count: Number of iterations completed before failure
    """

    def __init__(
        self, max_iterations: int, current_messages: List[Any], iteration_count: Optional[int] = None
    ):
        self.max_iterations = max_iterations
        self.current_messages = current_messages
        self.iteration_count = iteration_count or len(current_messages)

        message = (
            f"Inference exceeded maximum iterations ({max_iterations})\n"
            f"Completed {self.iteration_count} iterations\n"
            f"Partial conversation contains {len(current_messages)} messages"
        )

        super().__init__(message)

    def get_partial_result(self) -> List[Any]:
        """
        Get the partial conversation history up to the point of failure.

        Returns:
            List of messages that were collected before hitting the limit
        """
        return self.current_messages


class ConfigurationError(MAHSMError):
    """
    Raised when configuration is invalid or missing.

    This exception includes specific configuration issue details to help
    users identify and fix configuration problems.

    Attributes:
        config_key: The configuration key that has an issue
        issue: Description of the configuration issue
        suggestion: Optional suggestion for fixing the issue
    """

    def __init__(
        self, message: str, config_key: Optional[str] = None, suggestion: Optional[str] = None
    ):
        self.config_key = config_key
        self.suggestion = suggestion

        # Enhance message with suggestion if provided
        if suggestion:
            message = f"{message}\nSuggestion: {suggestion}"

        super().__init__(message)


class ModelError(MAHSMError):
    """
    Raised when model invocation fails.

    This exception wraps the original model API error and includes the
    model identifier and request details for debugging.

    Attributes:
        model: The model identifier that was used
        original_error: The original exception from the model API
        request_details: Optional details about the request that failed
    """

    def __init__(
        self,
        model: str,
        original_error: Exception,
        request_details: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.original_error = original_error
        self.request_details = request_details or {}

        # Construct detailed error message
        error_msg = (
            f"Model '{model}' invocation failed: {original_error}\n"
            f"Original error type: {type(original_error).__name__}"
        )

        if request_details:
            error_msg += f"\nRequest details: {request_details}"

        super().__init__(error_msg)

    def __str__(self) -> str:
        """Return detailed string representation."""
        msg = f"ModelError: Model '{self.model}' failed\n  Original error: {self.original_error}"
        if self.request_details:
            msg += f"\n  Request: {self.request_details}"
        return msg


# Exception hierarchy for validation and documentation
EXCEPTION_HIERARCHY = {
    "MAHSMError": {
        "ValidationError": [],
        "ToolExecutionError": [],
        "MaxIterationsError": [],
        "ConfigurationError": [],
        "ModelError": [],
    }
}


def get_exception_hierarchy() -> Dict[str, Any]:
    """
    Get the complete exception hierarchy for MAHSM.

    Returns:
        Dictionary representing the exception hierarchy
    """
    return EXCEPTION_HIERARCHY


__all__ = [
    "MAHSMError",
    "ValidationError",
    "ToolExecutionError",
    "MaxIterationsError",
    "ConfigurationError",
    "ModelError",
    "get_exception_hierarchy",
]


