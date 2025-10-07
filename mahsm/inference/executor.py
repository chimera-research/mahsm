"""
Inference Executor
MAHSM v0.1.0

Implements agentic loop execution with tool calling and message transparency.
Includes LangFuse tracing integration and model abstraction.
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from mahsm.exceptions import ToolExecutionError, MaxIterationsError, ModelError
from mahsm.inference.messages import (
    create_system_message,
    create_human_message,
    create_tool_message,
    extract_messages_from_state,
)


def _get_langfuse_tracer():
    """
    Get LangFuse tracer if configured, otherwise return None.
    
    Returns:
        Langfuse client or None if not configured
    """
    try:
        from mahsm.config import config
        
        if config.langfuse_enabled:
            from langfuse import Langfuse
            
            return Langfuse(
                public_key=config.langfuse_public_key,
                secret_key=config.langfuse_secret_key,
            )
    except Exception:
        # Silently fail - tracing is optional
        pass
    
    return None


def execute_tool(tool_func: Callable, tool_call: Dict[str, Any]) -> Any:
    """
    Execute a single tool function call.

    Args:
        tool_func: The Python function to execute
        tool_call: Tool call dict with 'function' containing 'name' and 'arguments'

    Returns:
        Tool execution result

    Raises:
        ToolExecutionError: If tool execution fails
    """
    tool_name = tool_func.__name__

    try:
        # Extract arguments from tool_call
        if "function" in tool_call:
            # OpenAI format: tool_call.function.arguments
            arguments_str = tool_call["function"].get("arguments", "{}")
        elif "args" in tool_call:
            # Alternative format
            arguments_str = json.dumps(tool_call["args"])
        else:
            arguments_str = "{}"

        # Parse arguments
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            arguments = {}

        # Call the tool function
        if isinstance(arguments, dict):
            result = tool_func(**arguments)
        else:
            result = tool_func(arguments)

        return result

    except Exception as e:
        raise ToolExecutionError(tool_name=tool_name, tool_call=tool_call, original_error=e)


def parse_model_response(response: Any) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Parse model response to extract content and tool calls.

    Args:
        response: Model response object

    Returns:
        Tuple of (content, tool_calls) where tool_calls may be None
    """
    # Handle different response formats
    if hasattr(response, "content"):
        content = response.content
    elif isinstance(response, dict):
        content = response.get("content", "")
    else:
        content = str(response)

    # Extract tool calls if present
    tool_calls = None
    if hasattr(response, "tool_calls"):
        tool_calls = response.tool_calls
    elif isinstance(response, dict) and "tool_calls" in response:
        tool_calls = response["tool_calls"]

    return content, tool_calls


def invoke_model(
    model: str,
    messages: List[BaseMessage],
    tools: Optional[List[Callable]] = None,
    **kwargs
) -> AIMessage:
    """
    Invoke the model with messages and optional tools.

    Args:
        model: Model identifier (e.g., "openai/gpt-4o-mini")
        messages: List of messages for the model
        tools: Optional list of tool functions
        **kwargs: Additional model parameters

    Returns:
        AIMessage with the model's response

    Raises:
        ModelError: If model invocation fails
    """
    try:
        # Parse model identifier
        if "/" in model:
            provider, model_name = model.split("/", 1)
        else:
            provider = "openai"
            model_name = model

        # Import and configure model based on provider
        if provider == "openai":
            from openai import OpenAI

            client = OpenAI()

            # Convert messages to OpenAI format
            openai_messages = []
            for msg in messages:
                msg_dict = {"role": _get_role_for_message(msg), "content": msg.content}
                openai_messages.append(msg_dict)

            # Prepare request
            request_params = {
                "model": model_name,
                "messages": openai_messages,
            }

            # Add tools if provided
            if tools:
                from mahsm.prompt.validation import extract_tools_schemas

                tool_schemas = extract_tools_schemas(tools)
                request_params["tools"] = [
                    {"type": "function", "function": schema} for schema in tool_schemas
                ]

            # Add any additional kwargs
            request_params.update(kwargs)

            # Invoke model
            response = client.chat.completions.create(**request_params)

            # Extract response
            choice = response.choices[0]
            message = choice.message

            # Create AIMessage
            ai_message = AIMessage(content=message.content or "")

            # Add tool calls if present
            if message.tool_calls:
                ai_message.tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]

            return ai_message

        else:
            raise ModelError(
                model=model,
                original_error=ValueError(f"Unsupported model provider: {provider}"),
                request_details={"provider": provider, "model_name": model_name},
            )

    except Exception as e:
        if isinstance(e, ModelError):
            raise
        raise ModelError(
            model=model,
            original_error=e,
            request_details={"messages_count": len(messages), "tools_count": len(tools) if tools else 0},
        )


def _get_role_for_message(msg: BaseMessage) -> str:
    """Get OpenAI role for a message type."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

    if isinstance(msg, SystemMessage):
        return "system"
    elif isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    elif isinstance(msg, ToolMessage):
        return "tool"
    else:
        return "user"


def inference(
    model: str,
    prompt: str,
    tools: List[Callable],
    input: Union[str, Dict[str, Any]],
    state: Optional[Union[List[BaseMessage], Dict[str, Any]]] = None,
    max_iterations: int = 10,
    **kwargs
) -> Tuple[Any, List[BaseMessage]]:
    """
    Execute prompt with tools in an agentic loop with full message transparency.

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
    """
    # Initialize LangFuse tracing (T024)
    tracer = _get_langfuse_tracer()
    trace = None
    start_time = time.time()
    
    if tracer:
        try:
            trace = tracer.trace(
                name="mahsm_inference",
                metadata={
                    "model": model,
                    "tools": [t.__name__ for t in tools],
                    "max_iterations": max_iterations,
                }
            )
        except Exception:
            # Silently fail - don't break inference for tracing issues
            trace = None
    
    # Initialize message list
    messages = extract_messages_from_state(state) if state else []

    # Add SystemMessage with prompt (if not already present)
    if not messages or not any(isinstance(msg, type(create_system_message(""))) for msg in messages):
        messages.insert(0, create_system_message(prompt))

    # Add HumanMessage with input
    messages.append(create_human_message(input))

    # Create tool name to function mapping
    tools_by_name = {tool.__name__: tool for tool in tools}

    # Agentic loop
    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # Invoke model
        ai_message = invoke_model(model, messages, tools, **kwargs)
        messages.append(ai_message)

        # Check for tool calls
        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
            # No tool calls - we're done
            break

        # Execute tools
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["function"]["name"]

            if tool_name not in tools_by_name:
                # Tool not found - create error message
                error_content = f"Error: Tool '{tool_name}' not found"
                tool_message = create_tool_message(
                    content=error_content,
                    tool_call_id=tool_call["id"],
                    tool_name=tool_name,
                )
                messages.append(tool_message)
                continue

            # Execute tool
            tool_func = tools_by_name[tool_name]
            try:
                result = execute_tool(tool_func, tool_call)

                # Create ToolMessage with result
                tool_message = create_tool_message(
                    content=result, tool_call_id=tool_call["id"], tool_name=tool_name
                )
                messages.append(tool_message)

            except ToolExecutionError:
                # Re-raise tool execution errors
                raise

        # Continue loop to get model's response to tool results

    # Check if we exceeded max iterations
    if iteration >= max_iterations:
        # Check if last message still has tool calls
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            raise MaxIterationsError(
                max_iterations=max_iterations,
                current_messages=messages,
                iteration_count=iteration,
            )

    # Extract final result from last AIMessage
    final_result = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            final_result = msg.content
            break

    # Finalize trace with performance metrics (T024)
    if trace:
        try:
            execution_time = time.time() - start_time
            trace.update(
                output=final_result,
                metadata={
                    "execution_time_seconds": execution_time,
                    "total_messages": len(messages),
                    "iterations": iteration,
                }
            )
        except Exception:
            # Silently fail - don't break inference for tracing issues
            pass

    return final_result, messages


__all__ = [
    "execute_tool",
    "invoke_model",
    "inference",
    "parse_model_response",
]

