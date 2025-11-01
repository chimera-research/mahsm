"""
Smoke tests for mahsm - Quick sanity checks to verify nothing is fundamentally broken.

These tests should be extremely fast (<1s total) and catch major breakage.
"""
import pytest


def test_package_imports():
    """Test that mahsm package can be imported successfully."""
    try:
        import mahsm
        assert mahsm is not None
    except ImportError as e:
        pytest.fail(f"Failed to import mahsm: {e}")


def test_core_module_accessible():
    """Test that core functionality is accessible."""
    import mahsm as ma
    
    assert hasattr(ma, "dspy_node")
    assert callable(ma.dspy_node)


def test_tracing_module_accessible():
    """Test that tracing module is accessible."""
    import mahsm as ma
    
    assert hasattr(ma, "tracing")
    assert hasattr(ma.tracing, "init")
    assert hasattr(ma.tracing, "observe")


def test_dspy_namespace_accessible():
    """Test that DSPy namespace is accessible."""
    import mahsm as ma
    
    assert hasattr(ma, "dspy")
    assert hasattr(ma.dspy, "Module")
    assert hasattr(ma.dspy, "Signature")
    assert hasattr(ma.dspy, "Predict")
    assert hasattr(ma.dspy, "ChainOfThought")


def test_graph_namespace_accessible():
    """Test that LangGraph namespace is accessible."""
    import mahsm as ma
    
    assert hasattr(ma, "graph")
    assert hasattr(ma.graph, "StateGraph")
    assert hasattr(ma, "START")
    assert hasattr(ma, "END")


def test_convenience_exports():
    """Test that convenience exports work."""
    import mahsm as ma
    
    # DSPy convenience exports
    assert hasattr(ma, "Module")
    assert hasattr(ma, "Signature")
    assert hasattr(ma, "InputField")
    assert hasattr(ma, "OutputField")
    
    # LangGraph convenience exports
    assert hasattr(ma, "START")
    assert hasattr(ma, "END")
    
    # LangChain message types
    assert hasattr(ma, "HumanMessage")
    assert hasattr(ma, "AIMessage")


def test_version_string_exists():
    """Test that version string is defined."""
    import mahsm
    
    assert hasattr(mahsm, "__version__")
    assert isinstance(mahsm.__version__, str)
    assert len(mahsm.__version__) > 0


def test_basic_decorator_syntax():
    """Test that @dspy_node decorator can be applied without errors."""
    import mahsm as ma
    
    # Should not raise any errors
    @ma.dspy_node
    class TestModule(ma.Module):
        def forward(self, input):
            class Result:
                def __init__(self):
                    self.output = "test"
            return Result()
    
    # Verify it returns a wrapper
    assert TestModule is not None


def test_graph_construction_doesnt_crash():
    """Test that basic graph construction doesn't crash."""
    import mahsm as ma
    from typing import TypedDict
    
    class State(TypedDict):
        value: int
    
    # Should not raise any errors
    workflow = ma.graph.StateGraph(State)
    
    def dummy_node(state):
        return {"value": state.get("value", 0) + 1}
    
    workflow.add_node("node", dummy_node)
    workflow.add_edge(ma.START, "node")
    workflow.add_edge("node", ma.END)
    
    # Compile should not crash
    graph = workflow.compile()
    assert graph is not None


def test_testing_module_optional():
    """Test that testing module is optional and doesn't break imports."""
    import mahsm as ma
    
    # Testing module may or may not be available (Windows compatibility)
    # Just verify the import didn't fail
    testing_available = hasattr(ma, "testing")
    
    if testing_available:
        assert hasattr(ma.testing, "PytestHarness")
    else:
        # If not available, that's okay - it's optional
        pass


def test_tuning_module_temporarily_disabled():
    """Test that tuning module is temporarily disabled."""
    import mahsm as ma
    
    # Tuning module should NOT be available (temporarily disabled due to syntax error)
    assert not hasattr(ma, "tuning")
    # TODO: Re-enable test once tuning.py is fixed


def test_all_exports_list():
    """Test that __all__ is properly defined."""
    import mahsm
    
    assert hasattr(mahsm, "__all__")
    assert isinstance(mahsm.__all__, list)
    assert len(mahsm.__all__) > 0
    
    # Key exports should be in __all__
    assert "dspy_node" in mahsm.__all__
    assert "tracing" in mahsm.__all__
    assert "dspy" in mahsm.__all__
    assert "graph" in mahsm.__all__


def test_no_syntax_errors_in_modules():
    """Test that all main modules can be imported without syntax errors."""
    # These imports should not raise SyntaxError
    try:
        import mahsm.core
        import mahsm.tracing
        import mahsm.dspy
        import mahsm.graph
        import mahsm.testing  # May fail on Windows, that's ok
    except ImportError:
        # ImportError is ok (missing dependencies), SyntaxError is not
        pass
    except SyntaxError as e:
        pytest.fail(f"Syntax error in module: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
