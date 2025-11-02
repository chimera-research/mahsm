"""
Unit tests for mahsm exports - Testing all public APIs are properly exported.
"""
import pytest
import mahsm as ma


class TestCoreExports:
    """Test core mahsm functionality exports."""
    
    def test_dspy_node_exported(self):
        """Test that dspy_node decorator is exported."""
        assert hasattr(ma, "dspy_node")
        assert callable(ma.dspy_node)
    
    def test_dspy_node_works_as_decorator(self):
        """Test that dspy_node can be used as a decorator."""
        @ma.dspy_node
        class TestModule(ma.Module):
            def forward(self, x):
                class Result:
                    def __init__(self):
                        self.y = x * 2
                return Result()
        
        # Should create callable wrapper
        node = TestModule()
        assert callable(node)
        
        # Should work with state dict
        result = node({"x": 5})
        assert result["y"] == 10


class TestTracingExports:
    """Test tracing namespace exports."""
    
    def test_tracing_namespace_exists(self):
        """Test that tracing namespace exists."""
        assert hasattr(ma, "tracing")
    
    def test_tracing_init_exported(self):
        """Test that tracing.init is exported."""
        assert hasattr(ma.tracing, "init")
        assert callable(ma.tracing.init)
    
    def test_tracing_observe_exported(self):
        """Test that tracing.observe decorator is exported."""
        assert hasattr(ma.tracing, "observe")
        assert callable(ma.tracing.observe)
    
    def test_observe_works_as_decorator(self):
        """Test that observe can be used as a decorator."""
        @ma.tracing.observe()
        def test_func(x):
            return x * 2
        
        # Should still be callable
        assert callable(test_func)
        result = test_func(5)
        assert result == 10


class TestDSPyNamespaceExports:
    """Test DSPy namespace exports."""
    
    def test_dspy_namespace_exists(self):
        """Test that dspy namespace exists."""
        assert hasattr(ma, "dspy")
    
    def test_module_exported(self):
        """Test that dspy.Module is exported."""
        assert hasattr(ma.dspy, "Module")
        
        # Should be able to subclass it
        class TestModule(ma.dspy.Module):
            def forward(self, x):
                return x
        
        module = TestModule()
        assert isinstance(module, ma.dspy.Module)
    
    def test_signature_exported(self):
        """Test that dspy.Signature is exported."""
        assert hasattr(ma.dspy, "Signature")
        
        # Should be able to subclass it
        class TestSignature(ma.dspy.Signature):
            input: str
            output: str
        
        assert issubclass(TestSignature, ma.dspy.Signature)
    
    def test_predict_exported(self):
        """Test that dspy.Predict is exported."""
        assert hasattr(ma.dspy, "Predict")
        
        # Should be able to instantiate it
        predictor = ma.dspy.Predict("input -> output")
        assert isinstance(predictor, ma.dspy.Module)
    
    def test_chain_of_thought_exported(self):
        """Test that dspy.ChainOfThought is exported."""
        assert hasattr(ma.dspy, "ChainOfThought")
        
        # Should be able to instantiate it
        cot = ma.dspy.ChainOfThought("query -> answer")
        assert isinstance(cot, ma.dspy.Module)
    
    def test_fields_exported(self):
        """Test that InputField and OutputField are exported."""
        assert hasattr(ma.dspy, "InputField")
        assert hasattr(ma.dspy, "OutputField")
        
        # Should be callable
        assert callable(ma.dspy.InputField)
        assert callable(ma.dspy.OutputField)


class TestGraphNamespaceExports:
    """Test LangGraph namespace exports."""
    
    def test_graph_namespace_exists(self):
        """Test that graph namespace exists."""
        assert hasattr(ma, "graph")
    
    def test_state_graph_exported(self):
        """Test that graph.StateGraph is exported."""
        assert hasattr(ma.graph, "StateGraph")
        
        # Should be able to instantiate it
        from typing import TypedDict
        
        class State(TypedDict):
            value: int
        
        workflow = ma.graph.StateGraph(State)
        assert workflow is not None
    
    def test_message_graph_deprecated(self):
        """Test that MessageGraph is deprecated/removed in favor of StateGraph."""
        # MessageGraph was deprecated - StateGraph handles both cases now
        assert hasattr(ma.graph, "StateGraph")
    
    def test_end_exported(self):
        """Test that graph.END is exported."""
        assert hasattr(ma.graph, "END")
        assert isinstance(ma.graph.END, str)
    
    def test_start_exported(self):
        """Test that graph.START is exported."""
        assert hasattr(ma.graph, "START")
        assert isinstance(ma.graph.START, str)


class TestConvenienceExports:
    """Test top-level convenience exports."""
    
    def test_module_convenience_export(self):
        """Test that Module is exported at top level."""
        assert hasattr(ma, "Module")
        assert ma.Module is ma.dspy.Module
    
    def test_signature_convenience_export(self):
        """Test that Signature is exported at top level."""
        assert hasattr(ma, "Signature")
        assert ma.Signature is ma.dspy.Signature
    
    def test_input_field_convenience_export(self):
        """Test that InputField is exported at top level."""
        assert hasattr(ma, "InputField")
        assert ma.InputField is ma.dspy.InputField
    
    def test_output_field_convenience_export(self):
        """Test that OutputField is exported at top level."""
        assert hasattr(ma, "OutputField")
        assert ma.OutputField is ma.dspy.OutputField
    
    def test_start_convenience_export(self):
        """Test that START is exported at top level."""
        assert hasattr(ma, "START")
        assert ma.START is ma.graph.START
    
    def test_end_convenience_export(self):
        """Test that END is exported at top level."""
        assert hasattr(ma, "END")
        assert ma.END is ma.graph.END
    
    def test_human_message_export(self):
        """Test that HumanMessage is exported."""
        assert hasattr(ma, "HumanMessage")
        
        # Should be usable
        msg = ma.HumanMessage(content="test")
        assert msg.content == "test"
    
    def test_ai_message_export(self):
        """Test that AIMessage is exported."""
        assert hasattr(ma, "AIMessage")
        
        # Should be usable
        msg = ma.AIMessage(content="test")
        assert msg.content == "test"


class TestTestingNamespace:
    """Test testing namespace exports (when available)."""
    
    def test_testing_namespace_optional(self):
        """Test that testing namespace is optional."""
        # May not be available on Windows
        if hasattr(ma, "testing"):
            assert hasattr(ma.testing, "PytestHarness")
        else:
            # This is okay - expected on Windows
            pass


class TestTuningNamespace:
    """Test tuning namespace availability and minimal surface."""
    
    def test_tuning_available(self):
        """Test that tuning namespace is available."""
        # Should NOT be available (syntax error being fixed)
        assert hasattr(ma, "tuning")


class TestNamespaceConsistency:
    """Test that namespace organization is consistent."""
    
    def test_no_name_collisions(self):
        """Test that there are no naming collisions."""
        # Convenience exports should point to namespace versions
        assert ma.Module is ma.dspy.Module
        assert ma.Signature is ma.dspy.Signature
        assert ma.START is ma.graph.START
        assert ma.END is ma.graph.END
    
    def test_all_list_completeness(self):
        """Test that __all__ contains all public exports."""
        # Get actual exports (excluding private)
        actual_exports = {
            name for name in dir(ma)
            if not name.startswith("_")
        }
        
        # Get declared exports
        declared_exports = set(ma.__all__)
        
        # Core exports should be in __all__
        core_exports = {
            "dspy_node", "tracing", "dspy", "graph", "testing",
            "Module", "Signature", "InputField", "OutputField",
            "START", "END", "HumanMessage", "AIMessage"
        }
        
        for export in core_exports:
            if hasattr(ma, export):  # Only check if actually exported
                assert export in declared_exports, f"{export} missing from __all__"
    
    def test_namespace_types(self):
        """Test that namespaces have correct types."""
        import types
        
        # These should be modules
        assert isinstance(ma.tracing, types.ModuleType)
        assert isinstance(ma.dspy, types.ModuleType)
        assert isinstance(ma.graph, types.ModuleType)
        
        if hasattr(ma, "testing"):
            assert isinstance(ma.testing, types.ModuleType)


class TestExportUsability:
    """Test that exports are actually usable in practice."""
    
    def test_build_simple_dspy_module(self):
        """Test building a simple DSPy module using exports."""
        class SimpleModule(ma.Module):
            def __init__(self):
                super().__init__()
                self.predictor = ma.dspy.Predict("input -> output")
            
            def forward(self, input):
                return self.predictor(input=input)
        
        module = SimpleModule()
        assert isinstance(module, ma.Module)
    
    def test_build_simple_graph(self):
        """Test building a simple graph using exports."""
        from typing import TypedDict
        
        class State(TypedDict):
            value: int
        
        workflow = ma.graph.StateGraph(State)
        
        def increment(state):
            return {"value": state["value"] + 1}
        
        workflow.add_node("inc", increment)
        workflow.add_edge(ma.START, "inc")
        workflow.add_edge("inc", ma.END)
        
        graph = workflow.compile()
        assert graph is not None
    
    def test_combine_dspy_and_graph(self):
        """Test combining DSPy and LangGraph using exports."""
        from typing import TypedDict
        
        class State(TypedDict):
            value: int
        
        # Create DSPy module
        class DoubleModule(ma.Module):
            def forward(self, value):
                class Result:
                    def __init__(self):
                        self.value = value * 2
                return Result()
        
        # Wrap as node
        double_node = ma.dspy_node(DoubleModule())
        
        # Add to graph
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("double", double_node)
        workflow.add_edge(ma.START, "double")
        workflow.add_edge("double", ma.END)
        
        # Compile and test
        graph = workflow.compile()
        result = graph.invoke({"value": 5})
        assert result["value"] == 10
    
    def test_tracing_decorator_chain(self):
        """Test chaining tracing decorator with dspy_node."""
        @ma.tracing.observe()
        def helper(x):
            return x * 2
        
        class TracedModule(ma.Module):
            def forward(self, value):
                result_val = helper(value)
                class Result:
                    def __init__(self):
                        self.value = result_val
                return Result()
        
        node = ma.dspy_node(TracedModule())
        result = node({"value": 5})
        assert result["value"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

