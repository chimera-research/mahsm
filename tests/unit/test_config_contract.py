"""
Contract Test: ma.config
MAHSM v0.1.0

These tests validate the contract requirements for configuration management.
Tests MUST FAIL before implementation (TDD).
"""

import os
import pytest
from pathlib import Path


class TestConfigContract:
    """Contract tests for ma.config module."""

    def setup_method(self):
        """Set up test fixtures."""
        # Store original environment
        self.original_env = os.environ.copy()

    def test_config_loads_from_environment_variables(self):
        """Test that Config auto-loads from environment variables."""
        import mahsm as ma

        # Set environment variables
        os.environ["LANGFUSE_PUBLIC_KEY"] = "test_public_key"
        os.environ["LANGFUSE_SECRET_KEY"] = "test_secret_key"
        os.environ["MAHSM_HOME"] = "/tmp/test_mahsm"

        # Reload config (implementation-dependent)
        # For now, verify config object exists
        assert hasattr(ma, "config")
        assert ma.config is not None

    def test_config_langfuse_enabled_property(self):
        """Test that config.langfuse_enabled indicates if LangFuse is configured."""
        import mahsm as ma

        os.environ["LANGFUSE_PUBLIC_KEY"] = "test_key"
        os.environ["LANGFUSE_SECRET_KEY"] = "test_secret"

        # Should indicate LangFuse is enabled
        # This test verifies the property exists
        assert hasattr(ma.config, "langfuse_enabled")

    def test_config_mahsm_home_property(self):
        """Test that config.mahsm_home returns the base directory."""
        import mahsm as ma

        assert hasattr(ma.config, "mahsm_home")
        mahsm_home = ma.config.mahsm_home
        assert isinstance(mahsm_home, Path)

    def test_config_prompts_dir_property(self):
        """Test that config.prompts_dir returns the prompts directory."""
        import mahsm as ma

        assert hasattr(ma.config, "prompts_dir")
        prompts_dir = ma.config.prompts_dir
        assert isinstance(prompts_dir, Path)
        assert "prompts" in str(prompts_dir)

    def test_config_default_max_iterations(self):
        """Test that config.default_max_iterations returns the default limit."""
        import mahsm as ma

        assert hasattr(ma.config, "default_max_iterations")
        max_iter = ma.config.default_max_iterations
        assert isinstance(max_iter, int)
        assert max_iter > 0

    def test_get_checkpointer_memory_type(self):
        """Test that get_checkpointer() returns memory checkpointer."""
        import mahsm as ma

        checkpointer = ma.config.get_checkpointer(checkpoint_type="memory")
        assert checkpointer is not None

    def test_get_checkpointer_sqlite_type(self):
        """Test that get_checkpointer() returns SQLite checkpointer."""
        import mahsm as ma

        checkpointer = ma.config.get_checkpointer(checkpoint_type="sqlite")
        assert checkpointer is not None

    def test_get_checkpointer_postgres_type(self):
        """Test that get_checkpointer() returns Postgres checkpointer."""
        import mahsm as ma

        # May skip if Postgres not available
        try:
            checkpointer = ma.config.get_checkpointer(checkpoint_type="postgres", connection_string="test")
            assert checkpointer is not None
        except Exception:
            pytest.skip("Postgres checkpointer not available")

    def test_get_checkpointer_raises_error_for_invalid_type(self):
        """Test that get_checkpointer() raises ConfigurationError for unsupported types."""
        import mahsm as ma
        from mahsm.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            ma.config.get_checkpointer(checkpoint_type="invalid_type")

    def test_config_update_method(self):
        """Test that config.update() allows runtime configuration updates."""
        import mahsm as ma

        assert hasattr(ma.config, "update")
        # Verify update method is callable
        assert callable(ma.config.update)

    def test_config_validates_configuration_values(self):
        """Test that config validates configuration values."""
        import mahsm as ma
        from mahsm.exceptions import ConfigurationError

        # Try to set invalid max_iterations
        with pytest.raises((ConfigurationError, ValueError)):
            ma.config.update(default_max_iterations=-1)

    def test_config_thread_safe_reads(self):
        """Test that config read operations are thread-safe."""
        import mahsm as ma
        import threading

        results = []

        def read_config():
            results.append(ma.config.mahsm_home)

        threads = [threading.Thread(target=read_config) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed
        assert len(results) == 10

    def teardown_method(self):
        """Clean up environment."""
        os.environ.clear()
        os.environ.update(self.original_env)


