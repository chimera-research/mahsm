"""
Vision Agent Example - Visual Question Answering with mahsm

This example demonstrates how to build a multimodal agent using mahsm
that can analyze images and answer questions about their content.

It showcases:
- Using dspy.Image for vision inputs
- @ma.dspy_node decorator with multimodal data
- LangGraph orchestration for vision workflows
- Multi-step visual reasoning

Requirements:
    pip install dspy pillow

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (for GPT-4o-mini)
"""

import dspy
import mahsm as ma
from typing import TypedDict
from langgraph.graph import StateGraph, END


# ============================================================================
# State Definition
# ============================================================================

class VisionAgentState(TypedDict):
    """State for visual question answering workflow."""
    image: dspy.Image  # Input image
    question: str  # Question about the image
    
    # Intermediate outputs
    observations: str  # Initial visual observations
    reasoning: str  # Step-by-step reasoning
    
    # Final output
    answer: str  # Final answer
    confidence: float  # Confidence score (0-1)


# ============================================================================
# DSPy Modules for Vision Processing
# ============================================================================

@ma.dspy_node
class VisualObserver(dspy.Module):
    """
    Extracts initial observations from an image.
    
    This node performs the first pass over the image to identify
    key visual elements without answering the specific question yet.
    """
    
    def __init__(self):
        super().__init__()
        self.observe = dspy.ChainOfThought(
            "image: dspy.Image -> observations: str"
        )
    
    def forward(self, image):
        """Extract visual observations from the image."""
        result = self.observe(image=image)
        return result


@ma.dspy_node
class VisualReasoner(dspy.Module):
    """
    Performs step-by-step visual reasoning to answer the question.
    
    Takes the initial observations and the question, then generates
    detailed reasoning before producing the final answer.
    """
    
    def __init__(self):
        super().__init__()
        self.reason = dspy.ChainOfThought(
            """observations: str, question: str -> 
               reasoning: str, answer: str, confidence: float"""
        )
    
    def forward(self, observations, question):
        """Reason about the question using the observations."""
        result = self.reason(
            observations=observations,
            question=question
        )
        return result


# ============================================================================
# Graph Construction
# ============================================================================

def build_vision_agent() -> StateGraph:
    """
    Builds the vision agent workflow graph.
    
    Workflow:
        1. Observe: Extract visual observations from image
        2. Reason: Use observations to answer the question
    
    Returns:
        Compiled LangGraph StateGraph
    """
    # Initialize graph with state schema
    graph = StateGraph(VisionAgentState)
    
    # Add nodes
    observer = VisualObserver()
    reasoner = VisualReasoner()
    
    graph.add_node("observe", observer)
    graph.add_node("reason", reasoner)
    
    # Define workflow
    graph.set_entry_point("observe")
    graph.add_edge("observe", "reason")
    graph.add_edge("reason", END)
    
    # Compile
    return graph.compile()


# ============================================================================
# Example Usage
# ============================================================================

def run_example():
    """
    Run a simple vision agent example.
    
    This demonstrates the complete workflow:
    1. Load an image
    2. Ask a question about it
    3. Get a reasoned answer with confidence
    """
    import os
    from pathlib import Path
    
    print("🔍 mahsm Vision Agent Example\n")
    print("=" * 60)
    
    # Configure DSPy with a vision-capable model
    print("\n📝 Configuring DSPy with GPT-5-mini...")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Please set it with: export OPENAI_API_KEY=your-key")
        return
    
    # Configure with GPT-5-mini (newest, smartest, cheapest, fastest vision model)
    dspy.configure(
        lm=dspy.LM(
            model="openai/gpt-5-mini",  # Latest vision-capable model from OpenAI
            api_key=api_key
        )
    )
    
    print("✅ DSPy configured with gpt-5-mini\n")
    
    # Build the agent
    print("🏗️  Building vision agent graph...")
    agent = build_vision_agent()
    print("✅ Agent built successfully\n")
    
    # Example 1: Using an image URL
    print("=" * 60)
    print("Example 1: Image from URL")
    print("=" * 60)
    
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"
    question = "What color is the cat in this image?"
    
    print(f"\n🖼️  Image: {image_url}")
    print(f"❓ Question: {question}\n")
    
    # Load image from URL
    image = dspy.Image.from_url(image_url)
    
    # Run the agent
    print("🤔 Processing...\n")
    result = agent.invoke({
        "image": image,
        "question": question
    })
    
    # Display results
    print("📊 Results:")
    print("-" * 60)
    print(f"Observations: {result['observations'][:200]}...")
    print(f"\nReasoning: {result['reasoning'][:200]}...")
    print(f"\n✅ Answer: {result['answer']}")
    print(f"📈 Confidence: {result['confidence']:.2%}\n")
    
    # Example 2: Using a local file (if exists)
    print("=" * 60)
    print("Example 2: Local Image File")
    print("=" * 60)
    
    # Check if example image exists
    example_image_path = Path("examples/test_image.jpg")
    
    if example_image_path.exists():
        print(f"\n🖼️  Image: {example_image_path}")
        question2 = "Describe the main subject of this image."
        print(f"❓ Question: {question2}\n")
        
        # Load image from file
        image2 = dspy.Image.from_file(str(example_image_path))
        
        # Run the agent
        print("🤔 Processing...\n")
        result2 = agent.invoke({
            "image": image2,
            "question": question2
        })
        
        # Display results
        print("📊 Results:")
        print("-" * 60)
        print(f"Answer: {result2['answer']}")
        print(f"Confidence: {result2['confidence']:.2%}\n")
    else:
        print(f"\n💡 Tip: Place an image at '{example_image_path}' to test with local files\n")
    
    print("=" * 60)
    print("✨ Vision Agent Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_example()
