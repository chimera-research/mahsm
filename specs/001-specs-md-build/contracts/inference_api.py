"""
API Contract: Inference Execution
MAHSM v0.1.0 - ma.inference module

This file defines the expected API surface for agentic inference execution.
Contract tests will validate these signatures and behaviors.
"""

from typing import List, Dict, Any, Union, Tuple, Callable, Optional
from langchain_core.messages import BaseMessage


def inference(
    model: str,
    prompt: str,
    tools: List[Callable],
    input: Union[str, Dict[str, Any]],
    state: Optional[List[BaseMessage]] = None,
    max_iterations: int = 10,
    **kwargs
) -> Tuple[Any, List[BaseMessage]]:
    """
    Execute prompt with tools in an agentic loop, maintaining message transparency.
    
    Args:
        model: Model identifier (e.g., "openai/gpt-4o-mini")
        prompt: The prompt string (typically from ma.prompt.load())
        tools: List of tool functions to make available
        input: User input (string, dict, or JSON-serializable)
        state: Optional existing message state to continue
        max_iterations: Maximum number of agentic loop iterations
        **kwargs: Additional model parameters
        
    Returns:
        Tuple of (final_result, complete_message_list)
        
    Raises:
        MaxIterationsError: If loop exceeds max_iterations
        ToolExecutionError: If tool execution fails
        ModelError: If model invocation fails
        
    Contract Requirements:
        - MUST create SystemMessage from prompt
        - MUST create HumanMessage from input
        - MUST execute agentic loop until no tool_calls or max_iterations
        - MUST call tools as Python functions, wrap results in ToolMessage
        - MUST append ALL messages to state in correct chronological order
        - MUST handle tool execution errors gracefully
        - MUST return both final result and complete message history
    """
    pass


# Expected message flow for contract validation
EXPECTED_MESSAGE_FLOW = [
    "SystemMessage",      # Prompt
    "HumanMessage",       # User input
    "AIMessage",          # Model response (may have tool_calls)
    "ToolMessage",        # Tool execution result (if tool_calls exist)
    "AIMessage",          # Model response to tool results
    # ... continues until no tool_calls
]

# Tool execution contract
def execute_tool(
    tool_function: Callable,
    tool_call: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a single tool function call.
    
    Args:
        tool_function: The Python function to execute
        tool_call: OpenAI tool call format with name, arguments
        
    Returns:
        Dict with tool execution result
        
    Raises:
        ToolExecutionError: If tool execution fails
        
    Contract Requirements:
        - MUST call tool_function(**tool_call.arguments)
        - MUST wrap exceptions in ToolExecutionError
        - MUST return JSON-serializable result
    """
    pass
