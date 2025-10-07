"""
Configuration Management
MAHSM v0.1.0

Handles environment variable auto-loading and global configuration.
"""

import os
import threading
from pathlib import Path
from typing import Optional
from mahsm.exceptions import ConfigurationError


class Config:
    """
    Global configuration object for MAHSM.

    Automatically loads configuration from environment variables on initialization.
    Thread-safe for read operations.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure only one config instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Config, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize configuration from environment variables."""
        if self._initialized:
            return

        self._lock = threading.RLock()
        self._load_from_environment()
        self._initialized = True

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # LangFuse keys
        self._langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        self._langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

        # MAHSM home directory
        default_home = str(Path.home() / ".mahsm")
        self._mahsm_home = Path(os.environ.get("MAHSM_HOME", default_home))

        # Default max iterations
        try:
            self._default_max_iterations = int(os.environ.get("MAHSM_MAX_ITERATIONS", "10"))
            if self._default_max_iterations < 1:
                raise ValueError("MAHSM_MAX_ITERATIONS must be >= 1")
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid MAHSM_MAX_ITERATIONS value: {e}",
                config_key="MAHSM_MAX_ITERATIONS",
                suggestion="Set to a positive integer (default: 10)",
            )

    @property
    def langfuse_enabled(self) -> bool:
        """Check if LangFuse tracing is configured."""
        return bool(self._langfuse_public_key and self._langfuse_secret_key)

    @property
    def langfuse_public_key(self) -> Optional[str]:
        """Get LangFuse public key."""
        return self._langfuse_public_key

    @property
    def langfuse_secret_key(self) -> Optional[str]:
        """Get LangFuse secret key."""
        return self._langfuse_secret_key

    @property
    def mahsm_home(self) -> Path:
        """Get the base directory for MAHSM artifacts."""
        return self._mahsm_home

    @property
    def prompts_dir(self) -> Path:
        """Get the directory for prompt artifacts."""
        prompts_dir = self._mahsm_home / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        return prompts_dir

    @property
    def default_max_iterations(self) -> int:
        """Get the default maximum iterations for inference loops."""
        return self._default_max_iterations

    def update(self, **kwargs) -> None:
        """
        Update configuration values at runtime.

        Args:
            **kwargs: Configuration key-value pairs to update

        Raises:
            ConfigurationError: If validation fails
        """
        with self._lock:
            for key, value in kwargs.items():
                if key == "default_max_iterations":
                    if not isinstance(value, int) or value < 1:
                        raise ConfigurationError(
                            f"default_max_iterations must be a positive integer, got {value}",
                            config_key="default_max_iterations",
                        )
                    self._default_max_iterations = value

                elif key == "mahsm_home":
                    self._mahsm_home = Path(value)

                elif key == "langfuse_public_key":
                    self._langfuse_public_key = value

                elif key == "langfuse_secret_key":
                    self._langfuse_secret_key = value

                else:
                    raise ConfigurationError(
                        f"Unknown configuration key: '{key}'",
                        config_key=key,
                        suggestion=f"Valid keys: default_max_iterations, mahsm_home, langfuse_public_key, langfuse_secret_key",
                    )


# Global config instance
config = Config()


__all__ = ["Config", "config"]

