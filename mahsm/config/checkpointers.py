"""
Checkpointer Integration
MAHSM v0.1.0

Provides LangGraph-compatible checkpointer integration for state persistence.
"""

from typing import Any, Optional
from mahsm.exceptions import ConfigurationError


def get_checkpointer(checkpoint_type: str = "memory", **kwargs) -> Any:
    """
    Get a LangGraph-compatible checkpointer.

    Args:
        checkpoint_type: Type of checkpointer ("memory", "sqlite", "postgres")
        **kwargs: Additional checkpointer configuration

    Returns:
        LangGraph-compatible checkpointer instance

    Raises:
        ConfigurationError: If checkpoint_type is unsupported or configuration is invalid

    Examples:
        >>> # Memory checkpointer (default)
        >>> checkpointer = get_checkpointer("memory")
        
        >>> # SQLite checkpointer
        >>> checkpointer = get_checkpointer("sqlite", conn_string="checkpoints.db")
        
        >>> # Postgres checkpointer
        >>> checkpointer = get_checkpointer("postgres", connection_string="postgresql://...")
    """
    if checkpoint_type == "memory":
        return _get_memory_checkpointer(**kwargs)
    elif checkpoint_type == "sqlite":
        return _get_sqlite_checkpointer(**kwargs)
    elif checkpoint_type == "postgres":
        return _get_postgres_checkpointer(**kwargs)
    else:
        raise ConfigurationError(
            f"Unsupported checkpointer type: '{checkpoint_type}'",
            config_key="checkpoint_type",
            suggestion="Supported types: 'memory', 'sqlite', 'postgres'",
        )


def _get_memory_checkpointer(**kwargs) -> Any:
    """
    Get an in-memory checkpointer.

    Returns:
        MemorySaver instance from LangGraph
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    except ImportError as e:
        raise ConfigurationError(
            f"Failed to import MemorySaver from langgraph: {e}",
            config_key="checkpoint_type",
            suggestion="Install langgraph: pip install langgraph",
        )


def _get_sqlite_checkpointer(**kwargs) -> Any:
    """
    Get a SQLite checkpointer.

    Args:
        **kwargs: Configuration options
            - conn_string: SQLite connection string (default: ":memory:")
            - connection_string: Alternative name for conn_string

    Returns:
        SqliteSaver instance from LangGraph
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
    except ImportError as e:
        raise ConfigurationError(
            f"Failed to import SQLite checkpointer: {e}",
            config_key="checkpoint_type",
            suggestion="Install langgraph with SQLite support: pip install langgraph",
        )

    # Get connection string from kwargs
    conn_string = kwargs.get("conn_string") or kwargs.get("connection_string", ":memory:")

    try:
        # Create SQLite connection
        conn = sqlite3.connect(conn_string, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to create SQLite checkpointer: {e}",
            config_key="conn_string",
            suggestion=f"Check SQLite connection string: {conn_string}",
        )


def _get_postgres_checkpointer(**kwargs) -> Any:
    """
    Get a Postgres checkpointer.

    Args:
        **kwargs: Configuration options
            - connection_string: Postgres connection string (required)
            - conn_string: Alternative name for connection_string

    Returns:
        PostgresSaver instance from LangGraph
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as e:
        raise ConfigurationError(
            f"Failed to import Postgres checkpointer: {e}",
            config_key="checkpoint_type",
            suggestion="Install langgraph with Postgres support: pip install langgraph[postgres]",
        )

    # Get connection string from kwargs
    connection_string = kwargs.get("connection_string") or kwargs.get("conn_string")

    if not connection_string:
        raise ConfigurationError(
            "Postgres checkpointer requires 'connection_string' parameter",
            config_key="connection_string",
            suggestion="Provide a Postgres connection string: postgresql://user:pass@host:port/db",
        )

    try:
        return PostgresSaver.from_conn_string(connection_string)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to create Postgres checkpointer: {e}",
            config_key="connection_string",
            suggestion=f"Check Postgres connection string and server availability",
        )


def validate_checkpointer(checkpointer: Any) -> bool:
    """
    Validate that an object is a LangGraph-compatible checkpointer.

    Args:
        checkpointer: Object to validate

    Returns:
        True if valid checkpointer

    Raises:
        ConfigurationError: If checkpointer is invalid
    """
    # Check for required methods/attributes
    required_methods = ["put", "get", "list"]

    for method in required_methods:
        if not hasattr(checkpointer, method):
            raise ConfigurationError(
                f"Checkpointer missing required method: '{method}'",
                config_key="checkpointer",
                suggestion="Ensure checkpointer implements LangGraph checkpointer interface",
            )

    return True


__all__ = [
    "get_checkpointer",
    "validate_checkpointer",
]

