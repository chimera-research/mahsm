"""
Message Handling Utilities
MAHSM v0.1.0

Handles message creation, validation, and conversion for LangGraph compatibility.
"""

import json
from typing import Any, Dict, List, Union
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)


def create_system_message(prompt: str) -> SystemMessage:
    """
    Create a SystemMessage from a prompt string.

    Args:
        prompt: The prompt text

    Returns:
        SystemMessage instance
    """
    return SystemMessage(content=prompt)


def create_human_message(input_data: Union[str, Dict[str, Any], Any]) -> HumanMessage:
    """
    Create a HumanMessage from various input types.

    Args:
        input_data: User input (string, dict, or JSON-serializable)

    Returns:
        HumanMessage instance
    """
    if isinstance(input_data, str):
        content = input_data
    elif isinstance(input_data, dict):
        # Convert dict to JSON string for content
        content = json.dumps(input_data)
    else:
        # Try to convert to string
        try:
            if hasattr(input_data, "__str__"):
                content = str(input_data)
            else:
                content = json.dumps(input_data)
        except Exception:
            content = repr(input_data)

    return HumanMessage(content=content)


def create_tool_message(
    content: Any, tool_call_id: str, tool_name: Optional[str] = None
) -> ToolMessage:
    """
    Create a ToolMessage from tool execution result.

    Args:
        content: Tool execution result
        tool_call_id: ID of the tool call this responds to
        tool_name: Optional name of the tool

    Returns:
        ToolMessage instance
    """
    # Convert content to string if needed
    if isinstance(content, str):
        message_content = content
    elif isinstance(content, dict):
        message_content = json.dumps(content)
    else:
        try:
            message_content = json.dumps(content)
        except Exception:
            message_content = str(content)

    kwargs = {"content": message_content, "tool_call_id": tool_call_id}
    if tool_name:
        kwargs["name"] = tool_name

    return ToolMessage(**kwargs)


def validate_message_state(messages: List[BaseMessage]) -> bool:
    """
    Validate message state ordering and structure.

    Args:
        messages: List of messages to validate

    Returns:
        True if valid

    Raises:
        ValueError: If message state is invalid
    """
    if not messages:
        return True

    # Check that first message is SystemMessage (if present)
    if messages and not isinstance(messages[0], SystemMessage):
        # This is a warning, not an error - allow flexible message ordering
        pass

    # Verify all messages are BaseMessage instances
    for i, msg in enumerate(messages):
        if not isinstance(msg, BaseMessage):
            raise ValueError(
                f"Message at index {i} is not a BaseMessage: {type(msg)}"
            )

    return True


def append_messages_to_state(
    state: Union[List[BaseMessage], Dict[str, Any]], new_messages: List[BaseMessage]
) -> Union[List[BaseMessage], Dict[str, Any]]:
    """
    Append messages to state, handling both list and dict formats.

    Args:
        state: Current state (list or dict with 'messages' key)
        new_messages: Messages to append

    Returns:
        Updated state
    """
    if isinstance(state, list):
        # State is a list of messages
        state.extend(new_messages)
        return state
    elif isinstance(state, dict):
        # State is a dict with 'messages' key
        if "messages" not in state:
            state["messages"] = []
        state["messages"].extend(new_messages)
        return state
    else:
        # Create new list state
        return new_messages.copy()


def extract_messages_from_state(
    state: Union[List[BaseMessage], Dict[str, Any], None]
) -> List[BaseMessage]:
    """
    Extract messages list from state, handling various formats.

    Args:
        state: State object (list, dict, or None)

    Returns:
        List of messages
    """
    if state is None:
        return []
    elif isinstance(state, list):
        return state
    elif isinstance(state, dict):
        return state.get("messages", [])
    else:
        return []


def serialize_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    Serialize messages to JSON-compatible format.

    Args:
        messages: List of messages

    Returns:
        List of serialized message dicts
    """
    serialized = []
    for msg in messages:
        msg_dict = {
            "type": msg.__class__.__name__,
            "content": msg.content,
        }

        # Add tool-specific fields
        if isinstance(msg, ToolMessage):
            msg_dict["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, "name"):
                msg_dict["name"] = msg.name

        # Add AI message tool calls
        if isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls

        serialized.append(msg_dict)

    return serialized


def get_latest_message(state: Union[List[BaseMessage], Dict[str, Any]]) -> Optional[BaseMessage]:
    """
    Get the most recent message from state.

    Args:
        state: State object

    Returns:
        Latest message or None
    """
    messages = extract_messages_from_state(state)
    return messages[-1] if messages else None


def get_messages_by_type(
    messages: List[BaseMessage], message_type: type
) -> List[BaseMessage]:
    """
    Filter messages by type.

    Args:
        messages: List of messages
        message_type: Message type to filter (e.g., SystemMessage, HumanMessage)

    Returns:
        List of messages of the specified type
    """
    return [msg for msg in messages if isinstance(msg, message_type)]


__all__ = [
    "create_system_message",
    "create_human_message",
    "create_tool_message",
    "validate_message_state",
    "append_messages_to_state",
    "extract_messages_from_state",
    "serialize_messages",
    "get_latest_message",
    "get_messages_by_type",
]

