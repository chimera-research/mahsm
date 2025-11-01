"""
Pytest configuration and shared fixtures for mahsm tests.
"""
import pytest
import os
from typing import TypedDict
from unittest.mock import MagicMock, patch
import mahsm as ma


# ============================================================================
# Mock DSPy Configuration
# ============================================================================

@pytest.fixture(autouse=True)
def mock_dspy_lm():
    """
    Auto-use fixture that mocks DSPy LM to avoid actual API calls.
    
    This is applied to ALL tests automatically. Individual tests can
    override if they need to test with a real LM.
    """
    mock_lm = MagicMock()
    mock_lm.__call__ = MagicMock(return_value={"answer": "mocked response"})
    
    with patch('dspy.settings.configure') as mock_configure:
        yield mock_configure


@pytest.fixture
def mock_dspy_module_result():
    """
    Returns a factory for creating mock DSPy module results.
    
    Usage:
        result = mock_dspy_module_result(output="test", answer="42")
    """
    def _create_result(**kwargs):
        class MockResult:
            def __init__(self):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        return MockResult()
    
    return _create_result


# ============================================================================
# Mock Langfuse Configuration
# ============================================================================

@pytest.fixture
def mock_langfuse_env(monkeypatch):
    """
    Sets mock Langfuse environment variables.
    
    Usage:
        def test_something(mock_langfuse_env):
            # LANGFUSE_* env vars are now set
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-key")
    monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")


@pytest.fixture
def clear_langfuse_env(monkeypatch):
    """
    Clears Langfuse environment variables.
    
    Usage:
        def test_missing_creds(clear_langfuse_env):
            # No LANGFUSE_* env vars are set
    """
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)


# ============================================================================
# Common State Type Fixtures
# ============================================================================

@pytest.fixture
def simple_state_type():
    """Simple state type for basic testing."""
    class SimpleState(TypedDict):
        input: str
        output: str
    return SimpleState


@pytest.fixture
def multi_field_state_type():
    """Multi-field state type for complex testing."""
    class MultiFieldState(TypedDict):
        query: str
        processed: str
        result: str
        metadata: dict
    return MultiFieldState


@pytest.fixture
def numeric_state_type():
    """Numeric state type for mathematical operations."""
    class NumericState(TypedDict):
        value: int
        doubled: int
        tripled: int
    return NumericState


# ============================================================================
# Sample Module Definitions
# ============================================================================

@pytest.fixture
def echo_module():
    """
    Simple module that echoes input with a prefix.
    
    Usage:
        def test_something(echo_module):
            module = echo_module()
            result = module(input="test")
            assert result.output == "Echo: test"
    """
    class EchoModule(ma.Module):
        def forward(self, input):
            class Result:
                def __init__(self):
                    self.output = f"Echo: {input}"
            return Result()
    
    return EchoModule


@pytest.fixture
def doubler_module():
    """
    Module that doubles a numeric value.
    
    Usage:
        def test_something(doubler_module):
            module = doubler_module()
            result = module(value=5)
            assert result.doubled == 10
    """
    class DoublerModule(ma.Module):
        def forward(self, value):
            class Result:
                def __init__(self):
                    self.doubled = value * 2
            return Result()
    
    return DoublerModule


@pytest.fixture
def multi_param_module():
    """
    Module with multiple parameters for testing state mapping.
    
    Usage:
        def test_something(multi_param_module):
            module = multi_param_module()
            result = module(a=1, b=2, c=3)
            assert result.sum == 6
    """
    class MultiParamModule(ma.Module):
        def forward(self, a, b, c=0):
            class Result:
                def __init__(self):
                    self.sum = a + b + c
                    self.product = a * b * (c if c else 1)
            return Result()
    
    return MultiParamModule


# ============================================================================
# Test Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end (requires API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring actual LLM API"
    )
