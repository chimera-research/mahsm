"""
API Contract: Configuration Management
MAHSM v0.1.0 - ma.config module

This file defines the expected API surface for global configuration.
Contract tests will validate these signatures and behaviors.
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path


class Config:
    """
    Global configuration object for MAHSM.
    
    Contract Requirements:
        - MUST auto-load from environment variables on import
        - MUST provide get_checkpointer() method
        - MUST support LangFuse integration when keys present
        - MUST be thread-safe for read operations
    """
    
    def __init__(self):
        """
        Initialize configuration from environment variables.
        
        Environment Variables:
            LANGFUSE_PUBLIC_KEY: LangFuse tracing public key
            LANGFUSE_SECRET_KEY: LangFuse tracing secret key
            MAHSM_HOME: Base directory for artifacts (default: ~/.mahsm)
            MAHSM_MAX_ITERATIONS: Default inference loop limit (default: 10)
        """
        pass
    
    @property
    def langfuse_enabled(self) -> bool:
        """Check if LangFuse tracing is configured."""
        pass
    
    @property
    def mahsm_home(self) -> Path:
        """Get the base directory for MAHSM artifacts."""
        pass
    
    @property
    def prompts_dir(self) -> Path:
        """Get the directory for prompt artifacts."""
        pass
    
    @property
    def default_max_iterations(self) -> int:
        """Get the default maximum iterations for inference loops."""
        pass
    
    def get_checkpointer(
        self,
        checkpoint_type: str = "memory",
        **kwargs
    ) -> Any:
        """
        Get a LangGraph-compatible checkpointer.
        
        Args:
            checkpoint_type: Type of checkpointer ("memory", "sqlite", "postgres")
            **kwargs: Additional checkpointer configuration
            
        Returns:
            LangGraph-compatible checkpointer instance
            
        Raises:
            ValueError: If checkpoint_type is unsupported
            ConfigurationError: If required configuration is missing
            
        Contract Requirements:
            - MUST return LangGraph-compatible checkpointer
            - MUST support "memory", "sqlite", "postgres" types
            - MUST handle configuration validation
        """
        pass
    
    def update(self, **kwargs) -> None:
        """
        Update configuration values at runtime.
        
        Args:
            **kwargs: Configuration key-value pairs to update
            
        Contract Requirements:
            - MUST validate configuration values
            - MUST be thread-safe
        """
        pass


# Global configuration instance
config: Config = None  # Will be initialized on module import


# Configuration validation schema
CONFIG_SCHEMA = {
    "langfuse_public_key": {"type": "string", "required": False},
    "langfuse_secret_key": {"type": "string", "required": False},
    "mahsm_home": {"type": "string", "required": False, "default": "~/.mahsm"},
    "default_max_iterations": {"type": "integer", "required": False, "default": 10, "minimum": 1}
}

# Supported checkpointer types
SUPPORTED_CHECKPOINTERS = ["memory", "sqlite", "postgres"]
