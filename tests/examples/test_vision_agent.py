"""
Tests for the vision_agent.py example.

These tests verify that the vision agent can be constructed and executed
without errors, using mocked DSPy responses to avoid API calls.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))

import dspy
from vision_agent import (
    VisionAgentState,
    VisualObserver,
    VisualReasoner,
    build_vision_agent
)


class TestVisionAgentState:
    """Test the state definition for vision agents."""
    
    def test_state_structure(self):
        """Test that VisionAgentState has the expected fields."""
        # This is a TypedDict, so we check annotations
        annotations = VisionAgentState.__annotations__
        
        assert "image" in annotations
        assert "question" in annotations
        assert "observations" in annotations
        assert "reasoning" in annotations
        assert "answer" in annotations
        assert "confidence" in annotations


class TestVisualObserver:
    """Test the VisualObserver module."""
    
    @patch('dspy.ChainOfThought')
    def test_init(self, mock_cot):
        """Test VisualObserver initialization."""
        observer = VisualObserver()
        
        # Verify ChainOfThought was called with correct signature
        mock_cot.assert_called_once()
        call_args = mock_cot.call_args[0][0]
        assert "image" in call_args
        assert "observations" in call_args
    
    @patch.object(dspy.Module, '__init__')
    @patch('dspy.ChainOfThought')
    def test_forward(self, mock_cot, mock_init):
        """Test VisualObserver.forward() method."""
        # Setup mock
        mock_init.return_value = None
        mock_observe = Mock()
        mock_observe.return_value = Mock(observations="A cat sitting on a mat")
        mock_cot.return_value = mock_observe
        
        # Create observer
        observer = VisualObserver()
        observer.observe = mock_observe
        
        # Test forward
        mock_image = Mock(spec=dspy.Image)
        result = observer.forward(image=mock_image)
        
        # Verify
        mock_observe.assert_called_once_with(image=mock_image)
        assert hasattr(result, 'observations')


class TestVisualReasoner:
    """Test the VisualReasoner module."""
    
    @patch('dspy.ChainOfThought')
    def test_init(self, mock_cot):
        """Test VisualReasoner initialization."""
        reasoner = VisualReasoner()
        
        # Verify ChainOfThought was called
        mock_cot.assert_called_once()
        call_args = mock_cot.call_args[0][0]
        assert "observations" in call_args
        assert "question" in call_args
        assert "reasoning" in call_args
        assert "answer" in call_args
        assert "confidence" in call_args
    
    @patch.object(dspy.Module, '__init__')
    @patch('dspy.ChainOfThought')
    def test_forward(self, mock_cot, mock_init):
        """Test VisualReasoner.forward() method."""
        # Setup mock
        mock_init.return_value = None
        mock_reason = Mock()
        mock_reason.return_value = Mock(
            reasoning="The cat is clearly visible",
            answer="Orange",
            confidence=0.95
        )
        mock_cot.return_value = mock_reason
        
        # Create reasoner
        reasoner = VisualReasoner()
        reasoner.reason = mock_reason
        
        # Test forward
        result = reasoner.forward(
            observations="A cat sitting on a mat",
            question="What color is the cat?"
        )
        
        # Verify
        mock_reason.assert_called_once_with(
            observations="A cat sitting on a mat",
            question="What color is the cat?"
        )
        assert hasattr(result, 'reasoning')
        assert hasattr(result, 'answer')
        assert hasattr(result, 'confidence')


class TestVisionAgentGraph:
    """Test the vision agent graph construction."""
    
    @patch('vision_agent.VisualObserver')
    @patch('vision_agent.VisualReasoner')
    def test_build_vision_agent(self, mock_reasoner_class, mock_observer_class):
        """Test that build_vision_agent creates a valid graph."""
        # Setup mocks
        mock_observer = Mock()
        mock_reasoner = Mock()
        mock_observer_class.return_value = mock_observer
        mock_reasoner_class.return_value = mock_reasoner
        
        # Build graph
        graph = build_vision_agent()
        
        # Verify graph structure
        assert graph is not None
        
        # Verify nodes were instantiated
        mock_observer_class.assert_called_once()
        mock_reasoner_class.assert_called_once()
    
    @patch('vision_agent.VisualObserver')
    @patch('vision_agent.VisualReasoner')
    @patch('dspy.configure')
    def test_graph_execution_flow(
        self,
        mock_configure,
        mock_reasoner_class,
        mock_observer_class
    ):
        """Test that the graph executes the expected workflow."""
        # Setup mocks for modules
        mock_observer_instance = MagicMock()
        mock_reasoner_instance = MagicMock()
        
        # Mock the __call__ methods to return state updates
        mock_observer_instance.return_value = {
            "observations": "A cat sitting on a mat"
        }
        mock_reasoner_instance.return_value = {
            "reasoning": "The cat is clearly visible",
            "answer": "Orange",
            "confidence": 0.95
        }
        
        mock_observer_class.return_value = mock_observer_instance
        mock_reasoner_class.return_value = mock_reasoner_instance
        
        # Build and execute graph
        graph = build_vision_agent()
        
        # Create mock image
        mock_image = Mock(spec=dspy.Image)
        
        # Execute graph
        result = graph.invoke({
            "image": mock_image,
            "question": "What color is the cat?"
        })
        
        # Verify both nodes were called
        assert mock_observer_instance.called
        assert mock_reasoner_instance.called
        
        # Verify final state contains expected fields
        assert "observations" in result
        assert "answer" in result
        assert "confidence" in result


class TestIntegration:
    """Integration tests for the vision agent example."""
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration"),
        reason="Integration tests require --run-integration flag"
    )
    def test_full_workflow_with_mock_api(self):
        """
        Test the full workflow with mocked API responses.
        
        This test requires the --run-integration flag:
            pytest tests/examples/test_vision_agent.py --run-integration
        """
        # This would test the full workflow with actual DSPy calls
        # but using a mock LM to avoid API costs
        pass


def pytest_addoption(parser):
    """Add custom pytest command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require API calls"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
