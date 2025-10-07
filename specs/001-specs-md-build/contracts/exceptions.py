"""
API Contract: Exception Hierarchy
MAHSM v0.1.0 - Exception definitions

This file defines the expected exception hierarchy for MAHSM.
Contract tests will validate these exception types and behaviors.
"""


class MAHSMError(Exception):
    """Base exception for all MAHSM errors."""
    pass


class ValidationError(MAHSMError):
    """
    Raised when validation fails (tool schemas, configuration, etc.).
    
    Contract Requirements:
        - MUST include specific validation failure details
        - MUST be raised during ma.prompt.load() tool validation
        - MUST be raised during ma.prompt.save() schema validation
    """
    pass


class ToolExecutionError(MAHSMError):
    """
    Raised when tool execution fails during inference.
    
    Contract Requirements:
        - MUST wrap original tool execution exceptions
        - MUST include tool name and call details
        - MUST preserve original traceback information
    """
    
    def __init__(self, tool_name: str, tool_call: dict, original_error: Exception):
        self.tool_name = tool_name
        self.tool_call = tool_call
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' execution failed: {original_error}")


class MaxIterationsError(MAHSMError):
    """
    Raised when inference loop exceeds maximum iterations.
    
    Contract Requirements:
        - MUST include current iteration count
        - MUST include partial results if available
        - MUST be raised by ma.inference() when max_iterations exceeded
    """
    
    def __init__(self, max_iterations: int, current_messages: list):
        self.max_iterations = max_iterations
        self.current_messages = current_messages
        super().__init__(f"Inference exceeded maximum iterations ({max_iterations})")


class ConfigurationError(MAHSMError):
    """
    Raised when configuration is invalid or missing.
    
    Contract Requirements:
        - MUST include specific configuration issue details
        - MUST be raised during config initialization
        - MUST be raised by get_checkpointer() for invalid types
    """
    pass


class ModelError(MAHSMError):
    """
    Raised when model invocation fails.
    
    Contract Requirements:
        - MUST wrap original model API exceptions
        - MUST include model identifier and request details
        - MUST preserve original error information
    """
    
    def __init__(self, model: str, original_error: Exception):
        self.model = model
        self.original_error = original_error
        super().__init__(f"Model '{model}' invocation failed: {original_error}")


# Exception hierarchy for contract validation
EXCEPTION_HIERARCHY = {
    "MAHSMError": {
        "ValidationError": [],
        "ToolExecutionError": [],
        "MaxIterationsError": [],
        "ConfigurationError": [],
        "ModelError": []
    }
}

# Expected exception scenarios for contract tests
EXCEPTION_SCENARIOS = [
    {
        "scenario": "Tool schema mismatch",
        "function": "ma.prompt.load",
        "exception": "ValidationError",
        "trigger": "Provide tools that don't match saved schemas"
    },
    {
        "scenario": "Tool execution failure",
        "function": "ma.inference",
        "exception": "ToolExecutionError",
        "trigger": "Tool function raises exception"
    },
    {
        "scenario": "Max iterations exceeded",
        "function": "ma.inference",
        "exception": "MaxIterationsError",
        "trigger": "Agentic loop exceeds max_iterations"
    },
    {
        "scenario": "Invalid checkpointer type",
        "function": "ma.config.get_checkpointer",
        "exception": "ConfigurationError",
        "trigger": "Request unsupported checkpointer type"
    },
    {
        "scenario": "Model API failure",
        "function": "ma.inference",
        "exception": "ModelError",
        "trigger": "Model API returns error"
    }
]
