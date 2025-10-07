"""
Inference execution module for MAHSM.

Provides the inference function for executing prompts in agentic loops.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from langchain_core.messages import BaseMessage
from mahsm.inference.executor import inference as _inference


def inference(
    model: str,
    prompt: str,
    tools: List[Callable],
    input: Union[str, Dict[str, Any]],
    state: Optional[Union[List[BaseMessage], Dict[str, Any]]] = None,
    max_iterations: int = 10,
    **kwargs,
) -> Tuple[Any, List[BaseMessage]]:
    """
    Execute prompt with tools in an agentic loop with full message transparency.

    This is the core inference function that executes a prompt with tools in an
    agentic loop, maintaining complete transparency through LangGraph-compatible
    message states. All interactions (system prompt, user input, AI responses,
    tool calls, tool results) are captured as messages in chronological order.

    Args:
        model: Model identifier (e.g., "openai/gpt-4o-mini", "gpt-4")
        prompt: The prompt string (typically from ma.prompt.load())
        tools: List of tool functions to make available to the model
        input: User input (string, dict, or JSON-serializable object)
        state: Optional existing message state to continue conversation
        max_iterations: Maximum number of agentic loop iterations (default: 10)
        **kwargs: Additional model parameters (temperature, max_tokens, etc.)

    Returns:
        Tuple of (final_result, complete_message_list) where:
        - final_result: The final response content from the model
        - complete_message_list: List of all messages in chronological order

    Raises:
        MaxIterationsError: If loop exceeds max_iterations without completing
        ToolExecutionError: If a tool execution fails
        ModelError: If model invocation fails

    Example:
        >>> import mahsm as ma
        >>> 
        >>> # Define tools
        >>> def calculate_stats(data: list) -> dict:
        ...     return {"mean": sum(data) / len(data)}
        >>> 
        >>> # Load prompt
        >>> prompt = ma.prompt.load("task_v1", validate_tools=[calculate_stats])
        >>> 
        >>> # Execute inference
        >>> result, messages = ma.inference(
        ...     model="openai/gpt-4o-mini",
        ...     prompt=prompt,
        ...     tools=[calculate_stats],
        ...     input="Analyze data: [1, 2, 3, 4, 5]",
        ...     max_iterations=5
        ... )
        >>> 
        >>> # Inspect complete conversation
        >>> for i, msg in enumerate(messages):
        ...     print(f"{i+1}. {type(msg).__name__}: {msg.content[:50]}...")

    Message Transparency:
        The returned message list contains a complete record of the conversation:
        1. SystemMessage - The prompt
        2. HumanMessage - User input
        3. AIMessage - Model response (may include tool_calls)
        4. ToolMessage(s) - Tool execution results (if tools were called)
        5. AIMessage - Model's response to tool results
        ... (repeats steps 3-5 until no more tool calls or max_iterations)

    Integration with LangGraph:
        >>> from langgraph.graph import StateGraph, MessagesState
        >>> 
        >>> def my_node(state: MessagesState):
        ...     prompt = ma.prompt.load("task_v1")
        ...     result, messages = ma.inference(
        ...         model="openai/gpt-4o-mini",
        ...         prompt=prompt,
        ...         tools=[my_tool],
        ...         input=state["messages"][-1].content,
        ...         state=state.get("messages", [])
        ...     )
        ...     return {"messages": messages}
    """
    return _inference(
        model=model,
        prompt=prompt,
        tools=tools,
        input=input,
        state=state,
        max_iterations=max_iterations,
        **kwargs,
    )


__all__ = ["inference"]
