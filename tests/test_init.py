"""
Unit tests for mahsm/__init__.py - Testing package initialization and structure.
"""
import pytest
import importlib
import sys


def test_package_can_be_imported():
    """Test that mahsm package can be imported."""
    import mahsm
    assert mahsm is not None


def test_version_attribute():
    """Test that __version__ is defined and is a string."""
    import mahsm
    
    assert hasattr(mahsm, "__version__")
    assert isinstance(mahsm.__version__, str)
    assert len(mahsm.__version__) > 0
    
    # Check semantic versioning format
    parts = mahsm.__version__.split(".")
    assert len(parts) >= 2  # At least major.minor


def test_all_attribute_defined():
    """Test that __all__ is defined and contains expected exports."""
    import mahsm
    
    assert hasattr(mahsm, "__all__")
    assert isinstance(mahsm.__all__, list)
    assert len(mahsm.__all__) > 0


def test_core_exports_in_all():
    """Test that core functionality is listed in __all__."""
    import mahsm
    
    # Core decorator
    assert "dspy_node" in mahsm.__all__
    
    # Namespaces
    assert "tracing" in mahsm.__all__
    assert "dspy" in mahsm.__all__
    assert "graph" in mahsm.__all__
    assert "testing" in mahsm.__all__
    
    # Convenience exports
    assert "START" in mahsm.__all__
    assert "END" in mahsm.__all__
    assert "Module" in mahsm.__all__
    assert "Signature" in mahsm.__all__


def test_all_exports_are_accessible():
    """Test that everything in __all__ is actually accessible."""
    import mahsm
    
    for name in mahsm.__all__:
        if name.startswith("#"):  # Skip commented out items
            continue
        assert hasattr(mahsm, name), f"{name} is in __all__ but not accessible"


def test_dspy_node_export():
    """Test that dspy_node is exported and callable."""
    import mahsm
    
    assert hasattr(mahsm, "dspy_node")
    assert callable(mahsm.dspy_node)


def test_tracing_namespace():
    """Test that tracing namespace is properly exported."""
    import mahsm
    
    assert hasattr(mahsm, "tracing")
    assert hasattr(mahsm.tracing, "init")
    assert hasattr(mahsm.tracing, "observe")
    assert callable(mahsm.tracing.init)
    assert callable(mahsm.tracing.observe)


def test_dspy_namespace():
    """Test that dspy namespace is properly exported."""
    import mahsm
    
    assert hasattr(mahsm, "dspy")
    
    # Check key DSPy classes
    assert hasattr(mahsm.dspy, "Module")
    assert hasattr(mahsm.dspy, "Signature")
    assert hasattr(mahsm.dspy, "Predict")
    assert hasattr(mahsm.dspy, "ChainOfThought")
    assert hasattr(mahsm.dspy, "InputField")
    assert hasattr(mahsm.dspy, "OutputField")


def test_graph_namespace():
    """Test that graph namespace is properly exported."""
    import mahsm
    
    assert hasattr(mahsm, "graph")
    
    # Check key LangGraph classes
    assert hasattr(mahsm.graph, "StateGraph")
    # MessageGraph deprecated in favor of StateGraph
    assert hasattr(mahsm.graph, "END")
    assert hasattr(mahsm.graph, "START")


def test_convenience_exports():
    """Test that convenience re-exports work at top level."""
    import mahsm
    
    # DSPy convenience exports
    assert hasattr(mahsm, "Module")
    assert hasattr(mahsm, "Signature")
    assert hasattr(mahsm, "InputField")
    assert hasattr(mahsm, "OutputField")
    
    # Should be same as dspy namespace versions
    assert mahsm.Module is mahsm.dspy.Module
    assert mahsm.Signature is mahsm.dspy.Signature
    assert mahsm.InputField is mahsm.dspy.InputField
    assert mahsm.OutputField is mahsm.dspy.OutputField


def test_langgraph_convenience_exports():
    """Test that LangGraph convenience exports work."""
    import mahsm
    
    assert hasattr(mahsm, "START")
    assert hasattr(mahsm, "END")
    
    # Should be same as graph namespace versions
    assert mahsm.START is mahsm.graph.START
    assert mahsm.END is mahsm.graph.END


def test_langchain_message_exports():
    """Test that LangChain message types are exported."""
    import mahsm
    
    assert hasattr(mahsm, "HumanMessage")
    assert hasattr(mahsm, "AIMessage")
    
    # Should be usable
    human_msg = mahsm.HumanMessage(content="Hello")
    assert human_msg.content == "Hello"
    
    ai_msg = mahsm.AIMessage(content="Hi there")
    assert ai_msg.content == "Hi there"


def test_testing_namespace_optional():
    """Test that testing namespace is optional (Windows compatibility)."""
    import mahsm
    
    # May or may not be available
    testing_available = hasattr(mahsm, "testing")
    
    if testing_available:
        assert hasattr(mahsm.testing, "PytestHarness")
    # If not available, that's okay - it's expected on Windows


def test_tuning_namespace_temporarily_disabled():
    """Test that tuning namespace is temporarily disabled."""
    import mahsm
    
    # Should NOT be available (temporarily disabled)
    assert not hasattr(mahsm, "tuning")


def test_module_docstring():
    """Test that package has a docstring."""
    import mahsm
    
    assert mahsm.__doc__ is not None
    assert len(mahsm.__doc__) > 0
    assert "mahsm" in mahsm.__doc__.lower()


def test_namespace_isolation():
    """Test that namespaces are properly isolated."""
    import mahsm
    
    # DSPy and graph should be separate
    assert mahsm.dspy is not mahsm.graph
    
    # Tracing should be its own module
    assert mahsm.tracing is not mahsm.dspy
    assert mahsm.tracing is not mahsm.graph


def test_import_variants():
    """Test different import patterns work correctly."""
    # Test: import mahsm
    import mahsm as ma1
    assert hasattr(ma1, "dspy_node")
    
    # Test: from mahsm import dspy_node
    from mahsm import dspy_node as dn
    assert callable(dn)
    
    # Test: from mahsm import tracing
    from mahsm import tracing as tr
    assert hasattr(tr, "init")
    
    # Test: from mahsm.dspy import Module
    from mahsm.dspy import Module as DspyModule
    assert DspyModule is not None
    
    # Test: from mahsm.graph import StateGraph
    from mahsm.graph import StateGraph as SG
    assert SG is not None


def test_no_unexpected_exports():
    """Test that we're not polluting the namespace with internal imports."""
    import mahsm
    
    # These should NOT be exported
    unexpected_exports = [
        "sys",
        "os",
        "importlib",
        "warnings",  # Used internally but shouldn't be exported
    ]
    
    for name in unexpected_exports:
        if name in mahsm.__all__:
            pytest.fail(f"Unexpected export in __all__: {name}")


def test_reimport_is_idempotent():
    """Test that reimporting mahsm doesn't cause issues."""
    import mahsm as ma1
    import mahsm as ma2
    
    # Should be the same module
    assert ma1 is ma2
    
    # Reimport with importlib
    import importlib
    ma3 = importlib.reload(mahsm)
    
    # Core functionality should still work
    assert hasattr(ma3, "dspy_node")
    assert hasattr(ma3, "tracing")


def test_submodule_imports():
    """Test that submodules can be imported directly."""
    # Test direct submodule imports
    import mahsm.core
    assert hasattr(mahsm.core, "dspy_node")
    
    import mahsm.tracing
    assert hasattr(mahsm.tracing, "init")
    
    import mahsm.dspy
    assert hasattr(mahsm.dspy, "Module")
    
    import mahsm.graph
    assert hasattr(mahsm.graph, "StateGraph")


def test_import_order_independence():
    """Test that import order doesn't matter."""
    # Remove mahsm from sys.modules to force fresh import
    modules_to_remove = [key for key in sys.modules if key.startswith("mahsm")]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    # Import in different order
    from mahsm.dspy import Module
    from mahsm.graph import StateGraph
    from mahsm import dspy_node
    import mahsm
    
    # Everything should still work
    assert callable(dspy_node)
    assert Module is mahsm.Module
    assert StateGraph is mahsm.graph.StateGraph


def test_circular_import_safety():
    """Test that there are no circular import issues."""
    # This test passes if imports succeed without circular dependency errors
    try:
        import mahsm
        import mahsm.core
        import mahsm.tracing
        import mahsm.dspy
        import mahsm.graph
    except ImportError as e:
        if "circular" in str(e).lower():
            pytest.fail(f"Circular import detected: {e}")
        else:
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
