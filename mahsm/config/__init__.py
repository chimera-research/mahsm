"""
Configuration module for MAHSM.

Provides global configuration object and environment variable integration.
"""

from mahsm.config.settings import Config, config
from mahsm.config.checkpointers import get_checkpointer


# Update config with get_checkpointer method
def _add_get_checkpointer_method():
    """Add get_checkpointer as a method to config instance."""
    config.get_checkpointer = get_checkpointer


_add_get_checkpointer_method()


__all__ = ["Config", "config", "get_checkpointer"]
