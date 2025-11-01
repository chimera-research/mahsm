"""
Unit tests for mahsm/tracing.py - Testing Langfuse integration.
"""
import pytest
import mahsm as ma
from unittest.mock import Mock, patch, MagicMock


class TestTracingInit:
    """Test tracing.init() functionality."""
    
    def test_init_function_exists(self):
        """Test that init function is accessible."""
        assert hasattr(ma.tracing, "init")
        assert callable(ma.tracing.init)
    
    def test_init_with_all_params(self):
        """Test init with all parameters provided."""
        with patch('mahsm.tracing.Langfuse') as mock_langfuse:
            ma.tracing.init(
                public_key="test_public_key",
                secret_key="test_secret_key",
                host="https://test.langfuse.com"
            )
            
            # Verify Langfuse was instantiated
            mock_langfuse.assert_called_once_with(
                public_key="test_public_key",
                secret_key="test_secret_key",
                host="https://test.langfuse.com"
            )
    
    def test_init_with_minimal_params(self):
        """Test init with only required parameters."""
        with patch('mahsm.tracing.Langfuse') as mock_langfuse:
            ma.tracing.init(
                public_key="test_public_key",
                secret_key="test_secret_key"
            )
            
            # Should use default host
            mock_langfuse.assert_called_once()
    
    def test_init_with_env_variables(self):
        """Test that init works with environment variables."""
        with patch.dict('os.environ', {
            'LANGFUSE_PUBLIC_KEY': 'env_public_key',
            'LANGFUSE_SECRET_KEY': 'env_secret_key',
            'LANGFUSE_HOST': 'https://env.langfuse.com'
        }):
            with patch('mahsm.tracing.Langfuse') as mock_langfuse:
                # Should pick up env vars
                ma.tracing.init()
                
                # Verify called (may use env vars internally)
                assert mock_langfuse.called
    
    def test_init_returns_client(self):
        """Test that init returns a Langfuse client."""
        with patch('mahsm.tracing.Langfuse') as mock_langfuse:
            mock_client = Mock()
            mock_langfuse.return_value = mock_client
            
            client = ma.tracing.init(
                public_key="test_public_key",
                secret_key="test_secret_key"
            )
            
            assert client == mock_client


class TestObserveDecorator:
    """Test @observe decorator functionality."""
    
    def test_observe_function_exists(self):
        """Test that observe decorator is accessible."""
        assert hasattr(ma.tracing, "observe")
        assert callable(ma.tracing.observe)
    
    def test_observe_decorator_basic(self):
        """Test basic @observe decorator usage."""
        @ma.tracing.observe()
        def test_function(x):
            return x * 2
        
        # Should still be callable
        assert callable(test_function)
        
        # Should work correctly
        result = test_function(5)
        assert result == 10
    
    def test_observe_with_function_name(self):
        """Test @observe with explicit name."""
        @ma.tracing.observe(name="custom_function")
        def test_function(x):
            return x + 1
        
        result = test_function(10)
        assert result == 11
    
    def test_observe_preserves_function_signature(self):
        """Test that @observe preserves function signature."""
        @ma.tracing.observe()
        def test_function(a, b, c=None):
            return a + b + (c or 0)
        
        # Should work with all argument styles
        assert test_function(1, 2) == 3
        assert test_function(1, 2, 3) == 6
        assert test_function(1, 2, c=4) == 7
    
    def test_observe_with_return_values(self):
        """Test that @observe properly passes through return values."""
        @ma.tracing.observe()
        def complex_return():
            return {"key": "value", "number": 42}
        
        result = complex_return()
        assert result["key"] == "value"
        assert result["number"] == 42
    
    def test_observe_with_exceptions(self):
        """Test that @observe handles exceptions correctly."""
        @ma.tracing.observe()
        def failing_function():
            raise ValueError("Test error")
        
        # Should propagate exception
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
    
    def test_observe_multiple_decorators(self):
        """Test stacking multiple observe decorators."""
        @ma.tracing.observe(name="outer")
        @ma.tracing.observe(name="inner")
        def nested_function(x):
            return x * 3
        
        result = nested_function(4)
        assert result == 12


class TestTracingIntegration:
    """Test integration between tracing and other mahsm components."""
    
    def test_observe_with_dspy_module(self):
        """Test @observe works with DSPy modules."""
        @ma.tracing.observe()
        def helper_function(value):
            return value * 2
        
        class TracedModule(ma.Module):
            def forward(self, input):
                processed = helper_function(input)
                class Result:
                    def __init__(self):
                        self.output = processed
                return Result()
        
        module = TracedModule()
        result = module(input=5)
        assert result.output == 10
    
    def test_observe_in_dspy_node(self):
        """Test @observe inside @dspy_node decorated modules."""
        class NodeModule(ma.Module):
            @ma.tracing.observe()
            def helper(self, x):
                return x + 10
            
            def forward(self, value):
                result_val = self.helper(value)
                class Result:
                    def __init__(self):
                        self.value = result_val
                return Result()
        
        node = ma.dspy_node(NodeModule())
        result = node({"value": 5})
        assert result["value"] == 15
    
    def test_observe_with_graph_workflow(self):
        """Test @observe works within LangGraph workflows."""
        from typing import TypedDict
        
        class State(TypedDict):
            value: int
        
        @ma.tracing.observe()
        def traced_node(state):
            return {"value": state["value"] * 2}
        
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("traced", traced_node)
        workflow.add_edge(ma.START, "traced")
        workflow.add_edge("traced", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 3})
        assert result["value"] == 6


class TestTracingConfiguration:
    """Test tracing configuration and setup."""
    
    def test_tracing_disabled_by_default(self):
        """Test that tracing can work without init."""
        # Should not crash even if not initialized
        @ma.tracing.observe()
        def test_func():
            return "works"
        
        result = test_func()
        assert result == "works"
    
    def test_multiple_init_calls(self):
        """Test handling of multiple init calls."""
        with patch('mahsm.tracing.Langfuse') as mock_langfuse:
            mock_client = Mock()
            mock_langfuse.return_value = mock_client
            
            # First init
            client1 = ma.tracing.init(
                public_key="key1",
                secret_key="secret1"
            )
            
            # Second init should create new client
            client2 = ma.tracing.init(
                public_key="key2",
                secret_key="secret2"
            )
            
            # Should have been called twice
            assert mock_langfuse.call_count == 2


class TestTracingUtilities:
    """Test tracing utility functions."""
    
    def test_observe_as_context_manager(self):
        """Test if observe can be used as context manager."""
        # Note: This depends on Langfuse implementation
        # Just verify the decorator exists and works
        @ma.tracing.observe()
        def test_func():
            return "context"
        
        assert test_func() == "context"
    
    def test_observe_with_metadata(self):
        """Test observe with custom metadata."""
        # Test that decorator accepts metadata kwargs
        @ma.tracing.observe(metadata={"custom": "data"})
        def test_func(x):
            return x * 2
        
        result = test_func(5)
        assert result == 10


class TestTracingEdgeCases:
    """Test edge cases and error handling."""
    
    def test_observe_on_async_function(self):
        """Test that observe works with async functions."""
        @ma.tracing.observe()
        async def async_func(x):
            return x + 1
        
        # Should be callable
        assert callable(async_func)
    
    def test_observe_on_generator(self):
        """Test observe with generator functions."""
        @ma.tracing.observe()
        def generator_func(n):
            for i in range(n):
                yield i * 2
        
        result = list(generator_func(3))
        assert result == [0, 2, 4]
    
    def test_observe_on_class_method(self):
        """Test observe on class methods."""
        class TestClass:
            @ma.tracing.observe()
            def method(self, x):
                return x * 3
        
        obj = TestClass()
        result = obj.method(4)
        assert result == 12
    
    def test_observe_preserves_docstring(self):
        """Test that observe preserves function docstrings."""
        @ma.tracing.observe()
        def documented_func():
            """This is a test docstring."""
            return "test"
        
        assert documented_func.__doc__ is not None
        # Note: Langfuse's observe might modify this


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
