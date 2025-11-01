"""
Unit tests for mahsm/dspy.py - Testing DSPy namespace and re-exports.
"""
import pytest
import mahsm as ma


class TestDSPyNamespace:
    """Test DSPy namespace structure and exports."""
    
    def test_dspy_namespace_exists(self):
        """Test that dspy namespace is accessible."""
        assert hasattr(ma, "dspy")
        assert ma.dspy is not None
    
    def test_dspy_is_module(self):
        """Test that dspy is a proper module."""
        import types
        assert isinstance(ma.dspy, types.ModuleType)


class TestDSPyCoreClasses:
    """Test core DSPy classes are properly exported."""
    
    def test_module_class_exported(self):
        """Test that Module class is exported."""
        assert hasattr(ma.dspy, "Module")
        
        # Should be usable as base class
        class TestModule(ma.dspy.Module):
            def forward(self, x):
                return x
        
        module = TestModule()
        assert isinstance(module, ma.dspy.Module)
    
    def test_signature_exported(self):
        """Test that Signature is exported."""
        assert hasattr(ma.dspy, "Signature")
        # Signature might be a class or module in DSPy
        assert ma.dspy.Signature is not None
    
    def test_predict_exported(self):
        """Test that Predict class is exported."""
        assert hasattr(ma.dspy, "Predict")
        
        # Should be instantiable
        predictor = ma.dspy.Predict("input -> output")
        assert isinstance(predictor, ma.dspy.Module)
    
    def test_chain_of_thought_exported(self):
        """Test that ChainOfThought is exported."""
        assert hasattr(ma.dspy, "ChainOfThought")
        
        # Should be instantiable
        cot = ma.dspy.ChainOfThought("query -> answer")
        assert isinstance(cot, ma.dspy.Module)
    
    def test_react_exported(self):
        """Test that ReAct is exported."""
        assert hasattr(ma.dspy, "ReAct")
        
        # Should be instantiable
        react = ma.dspy.ReAct("query -> answer")
        assert isinstance(react, ma.dspy.Module)


class TestDSPyFields:
    """Test DSPy field definitions."""
    
    def test_input_field_exported(self):
        """Test that InputField is exported."""
        assert hasattr(ma.dspy, "InputField")
        assert callable(ma.dspy.InputField)
    
    def test_output_field_exported(self):
        """Test that OutputField is exported."""
        assert hasattr(ma.dspy, "OutputField")
        assert callable(ma.dspy.OutputField)
    
    def test_field_usage_in_signature(self):
        """Test using fields in custom signatures."""
        class CustomSignature(ma.dspy.Signature):
            """Test signature with fields."""
            query: str = ma.dspy.InputField()
            answer: str = ma.dspy.OutputField()
        
        assert CustomSignature is not None


class TestDSPyModuleUsage:
    """Test practical usage of DSPy modules."""
    
    def test_create_simple_module(self):
        """Test creating a simple DSPy module."""
        class SimpleModule(ma.dspy.Module):
            def __init__(self):
                super().__init__()
                self.predictor = ma.dspy.Predict("input -> output")
            
            def forward(self, input):
                return self.predictor(input=input)
        
        module = SimpleModule()
        assert isinstance(module, ma.dspy.Module)
    
    def test_module_with_multiple_predictors(self):
        """Test module with multiple predictors."""
        class MultiPredictorModule(ma.dspy.Module):
            def __init__(self):
                super().__init__()
                self.first = ma.dspy.Predict("query -> context")
                self.second = ma.dspy.Predict("context -> answer")
            
            def forward(self, query):
                context_result = self.first(query=query)
                return self.second(context=context_result.context)
        
        module = MultiPredictorModule()
        assert hasattr(module, "first")
        assert hasattr(module, "second")
    
    def test_module_composition(self):
        """Test composing multiple modules."""
        class SubModule(ma.dspy.Module):
            def forward(self, x):
                class Result:
                    def __init__(self):
                        self.value = x * 2
                return Result()
        
        class ParentModule(ma.dspy.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModule()
            
            def forward(self, x):
                result = self.sub(x=x)
                class FinalResult:
                    def __init__(self):
                        self.value = result.value + 1
                return FinalResult()
        
        module = ParentModule()
        result = module(x=5)
        assert result.value == 11  # (5 * 2) + 1


class TestDSPyPredictors:
    """Test different predictor types."""
    
    def test_predict_basic(self):
        """Test basic Predict module."""
        predictor = ma.dspy.Predict("input -> output")
        assert isinstance(predictor, ma.dspy.Module)
    
    def test_chain_of_thought_basic(self):
        """Test ChainOfThought module."""
        cot = ma.dspy.ChainOfThought("question -> answer")
        assert isinstance(cot, ma.dspy.Module)
    
    def test_react_basic(self):
        """Test ReAct module."""
        react = ma.dspy.ReAct("task -> result")
        assert isinstance(react, ma.dspy.Module)
    
    def test_predictor_with_custom_signature(self):
        """Test predictor with custom signature class."""
        class QASignature(ma.dspy.Signature):
            """Question answering signature."""
            question: str = ma.dspy.InputField()
            answer: str = ma.dspy.OutputField()
        
        predictor = ma.dspy.Predict(QASignature)
        assert isinstance(predictor, ma.dspy.Module)


class TestDSPyWithMahsm:
    """Test DSPy integration with mahsm features."""
    
    def test_dspy_module_with_dspy_node(self):
        """Test using dspy modules with @dspy_node decorator."""
        class TestModule(ma.dspy.Module):
            def forward(self, value):
                class Result:
                    def __init__(self):
                        self.doubled = value * 2
                return Result()
        
        # Wrap with @dspy_node
        wrapped = ma.dspy_node(TestModule())
        
        # Should work with state dict
        result = wrapped({"value": 5})
        assert result["doubled"] == 10
    
    def test_dspy_predictor_with_dspy_node(self):
        """Test wrapping DSPy predictors with @dspy_node."""
        predictor = ma.dspy.Predict("input -> output")
        
        # Should be wrappable
        wrapped = ma.dspy_node(predictor)
        assert callable(wrapped)
    
    def test_dspy_module_in_graph(self):
        """Test using DSPy modules in LangGraph."""
        from typing import TypedDict
        
        class State(TypedDict):
            value: int
        
        class DoubleModule(ma.dspy.Module):
            def forward(self, value):
                class Result:
                    def __init__(self):
                        self.value = value * 2
                return Result()
        
        # Wrap as node
        node = ma.dspy_node(DoubleModule())
        
        # Add to graph
        workflow = ma.graph.StateGraph(State)
        workflow.add_node("double", node)
        workflow.add_edge(ma.START, "double")
        workflow.add_edge("double", ma.END)
        
        graph = workflow.compile()
        result = graph.invoke({"value": 7})
        assert result["value"] == 14


class TestDSPyModuleFeatures:
    """Test advanced DSPy module features."""
    
    def test_module_with_parameters(self):
        """Test module with trainable parameters."""
        class ParameterizedModule(ma.dspy.Module):
            def __init__(self, multiplier):
                super().__init__()
                self.multiplier = multiplier
            
            def forward(self, x):
                class Result:
                    def __init__(self, val):
                        self.output = val
                return Result(x * self.multiplier)
        
        module = ParameterizedModule(multiplier=3)
        result = module(x=4)
        assert result.output == 12
    
    def test_module_state_persistence(self):
        """Test that module state persists across calls."""
        class StatefulModule(ma.dspy.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0
            
            def forward(self, x):
                self.counter += 1
                class Result:
                    def __init__(self, val, count):
                        self.value = val
                        self.count = count
                return Result(x, self.counter)
        
        module = StatefulModule()
        
        result1 = module(x=1)
        assert result1.count == 1
        
        result2 = module(x=2)
        assert result2.count == 2
        
        assert module.counter == 2
    
    def test_module_with_nested_modules(self):
        """Test modules containing other modules."""
        class InnerModule(ma.dspy.Module):
            def forward(self, x):
                class Result:
                    def __init__(self):
                        self.value = x + 1
                return Result()
        
        class OuterModule(ma.dspy.Module):
            def __init__(self):
                super().__init__()
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()
            
            def forward(self, x):
                r1 = self.inner1(x=x)
                r2 = self.inner2(x=r1.value)
                return r2
        
        module = OuterModule()
        result = module(x=5)
        assert result.value == 7  # 5 + 1 + 1


class TestDSPyStringSignatures:
    """Test DSPy string-based signatures."""
    
    def test_simple_string_signature(self):
        """Test simple string signature."""
        predictor = ma.dspy.Predict("input -> output")
        assert isinstance(predictor, ma.dspy.Module)
    
    def test_multi_input_signature(self):
        """Test signature with multiple inputs."""
        predictor = ma.dspy.Predict("context, question -> answer")
        assert isinstance(predictor, ma.dspy.Module)
    
    def test_multi_output_signature(self):
        """Test signature with multiple outputs."""
        predictor = ma.dspy.Predict("text -> summary, keywords")
        assert isinstance(predictor, ma.dspy.Module)
    
    def test_complex_string_signature(self):
        """Test complex string signature."""
        predictor = ma.dspy.Predict("context, question, examples -> answer, reasoning")
        assert isinstance(predictor, ma.dspy.Module)


class TestDSPyConvenienceExports:
    """Test that DSPy exports are available at top level."""
    
    def test_module_at_top_level(self):
        """Test that Module is available at ma.Module."""
        assert hasattr(ma, "Module")
        assert ma.Module is ma.dspy.Module
    
    def test_signature_at_top_level(self):
        """Test that Signature is available at ma.Signature."""
        assert hasattr(ma, "Signature")
        assert ma.Signature is ma.dspy.Signature
    
    def test_input_field_at_top_level(self):
        """Test that InputField is available at ma.InputField."""
        assert hasattr(ma, "InputField")
        assert ma.InputField is ma.dspy.InputField
    
    def test_output_field_at_top_level(self):
        """Test that OutputField is available at ma.OutputField."""
        assert hasattr(ma, "OutputField")
        assert ma.OutputField is ma.dspy.OutputField


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
