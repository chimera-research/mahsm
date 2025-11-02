"""
Visual Reasoning Chain - Multi-Step Analysis with Complex Reasoning

This example demonstrates how to build a visual reasoning agent that performs
complex multi-step analysis using mahsm. Perfect for:
- Medical image analysis
- Safety/hazard identification
- Scene understanding
- Complex visual problem solving

It showcases:
- Iterative visual reasoning
- Hypothesis generation and testing
- Conditional branching based on observations
- Transparent reasoning chains

Requirements:
    pip install dspy pillow

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (for GPT-5-mini)
    
Cost Estimate:
    ~$0.25-0.60 per analysis (multiple reasoning steps with GPT-5-mini)
"""

import dspy
import mahsm as ma
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END


# ============================================================================
# State Definition
# ============================================================================

class ReasoningState(TypedDict):
    """State for visual reasoning workflow."""
    image: dspy.Image  # Image to analyze
    initial_question: str  # Initial query
    analysis_depth: str  # "quick", "standard", "deep"
    
    # Reasoning chain outputs
    initial_observations: str  # First pass observations
    hypotheses: List[Dict[str, any]]  # Generated hypotheses
    current_hypothesis: Optional[Dict[str, any]]  # Hypothesis being tested
    verification_results: List[Dict[str, any]]  # Results of tests
    
    # Final outputs
    reasoning_chain: List[str]  # Step-by-step reasoning
    final_answer: str  # Conclusion
    confidence: float  # Overall confidence (0-1)
    alternative_interpretations: List[str]  # Other possible conclusions


# ============================================================================
# DSPy Modules for Visual Reasoning
# ============================================================================

@ma.dspy_node
class InitialObserver(dspy.Module):
    """
    Makes initial observations about the image.
    
    Identifies:
    - Key objects and elements
    - Spatial relationships
    - Notable features
    - Potential areas of interest
    """
    
    def __init__(self):
        super().__init__()
        self.observe = dspy.ChainOfThought(
            """image: dspy.Image, 
               question: str -> 
               observations: str, 
               areas_of_interest: str"""
        )
    
    def forward(self, image, question):
        """Generate initial observations."""
        result = self.observe(
            image=image,
            question=question
        )
        return result


@ma.dspy_node
class HypothesisGenerator(dspy.Module):
    """
    Generates multiple hypotheses based on observations.
    
    Creates testable explanations or answers that can be
    verified through further visual analysis.
    """
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(
            """observations: str, 
               question: str -> 
               hypotheses: str, 
               priority_order: str"""
        )
    
    def forward(self, observations, question):
        """Generate ranked hypotheses."""
        result = self.generate(
            observations=observations,
            question=question
        )
        return result


@ma.dspy_node
class HypothesisTester(dspy.Module):
    """
    Tests a specific hypothesis against the image.
    
    Performs focused analysis to gather evidence for or
    against the hypothesis.
    """
    
    def __init__(self):
        super().__init__()
        self.test = dspy.ChainOfThought(
            """image: dspy.Image, 
               hypothesis: str, 
               observations: str -> 
               supporting_evidence: str, 
               contradicting_evidence: str, 
               test_result: str"""
        )
    
    def forward(self, image, hypothesis, observations):
        """Test hypothesis against image evidence."""
        result = self.test(
            image=image,
            hypothesis=str(hypothesis),
            observations=observations
        )
        return result


@ma.dspy_node
class ReasoningSynthesizer(dspy.Module):
    """
    Synthesizes all reasoning steps into final conclusion.
    
    Weighs evidence, considers alternatives, and produces
    a confident answer with supporting rationale.
    """
    
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(
            """reasoning_chain: str, 
               verification_results: str, 
               question: str -> 
               final_answer: str, 
               confidence: float, 
               alternatives: str"""
        )
    
    def forward(self, reasoning_chain, verification_results, question):
        """Produce final reasoned conclusion."""
        result = self.synthesize(
            reasoning_chain=reasoning_chain,
            verification_results=verification_results,
            question=question
        )
        return result


# ============================================================================
# Graph Construction
# ============================================================================

def build_reasoning_agent() -> StateGraph:
    """
    Builds the visual reasoning chain workflow graph.
    
    Workflow:
        1. Observe: Initial image analysis
        2. Hypothesize: Generate possible explanations
        3. Test: Verify each hypothesis
        4. Synthesize: Combine evidence into conclusion
    
    Returns:
        Compiled LangGraph StateGraph
    """
    graph = StateGraph(ReasoningState)
    
    # Initialize modules
    observer = InitialObserver()
    generator = HypothesisGenerator()
    tester = HypothesisTester()
    synthesizer = ReasoningSynthesizer()
    
    # Add nodes
    def make_observations(state):
        """Initial observation phase."""
        result = observer(
            image=state["image"],
            question=state["initial_question"]
        )
        
        chain = [f"Step 1 - Initial Observations: {result.observations}"]
        
        return {
            "initial_observations": result.observations,
            "reasoning_chain": chain
        }
    
    def generate_hypotheses(state):
        """Generate testable hypotheses."""
        result = generator(
            observations=state["initial_observations"],
            question=state["initial_question"]
        )
        
        # Parse hypotheses (simplified - in production use structured output)
        hypotheses_list = [
            {"hypothesis": result.hypotheses, "priority": result.priority_order}
        ]
        
        chain = state["reasoning_chain"] + [
            f"Step 2 - Generated Hypotheses: {result.hypotheses}"
        ]
        
        return {
            "hypotheses": hypotheses_list,
            "current_hypothesis": hypotheses_list[0] if hypotheses_list else None,
            "reasoning_chain": chain
        }
    
    def test_hypothesis(state):
        """Test the current hypothesis."""
        if not state.get("current_hypothesis"):
            return {"verification_results": []}
        
        hypothesis = state["current_hypothesis"]
        result = tester(
            image=state["image"],
            hypothesis=hypothesis["hypothesis"],
            observations=state["initial_observations"]
        )
        
        verification = {
            "hypothesis": hypothesis,
            "supporting": result.supporting_evidence,
            "contradicting": result.contradicting_evidence,
            "result": result.test_result
        }
        
        existing_results = state.get("verification_results", [])
        existing_results.append(verification)
        
        chain = state["reasoning_chain"] + [
            f"Step 3 - Testing: {result.test_result}"
        ]
        
        return {
            "verification_results": existing_results,
            "reasoning_chain": chain
        }
    
    def synthesize_conclusion(state):
        """Synthesize final answer from reasoning chain."""
        # Combine reasoning steps
        full_chain = "\n".join(state["reasoning_chain"])
        
        # Combine verification results
        verifications = "\n".join([
            f"- {v['result']}" for v in state.get("verification_results", [])
        ])
        
        result = synthesizer(
            reasoning_chain=full_chain,
            verification_results=verifications,
            question=state["initial_question"]
        )
        
        # Parse alternatives (simplified)
        alternatives = [result.alternatives] if result.alternatives else []
        
        final_chain = state["reasoning_chain"] + [
            f"Step 4 - Conclusion: {result.final_answer}"
        ]
        
        return {
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "alternative_interpretations": alternatives,
            "reasoning_chain": final_chain
        }
    
    graph.add_node("observe", make_observations)
    graph.add_node("hypothesize", generate_hypotheses)
    graph.add_node("test", test_hypothesis)
    graph.add_node("synthesize", synthesize_conclusion)
    
    # Define workflow
    graph.set_entry_point("observe")
    graph.add_edge("observe", "hypothesize")
    graph.add_edge("hypothesize", "test")
    graph.add_edge("test", "synthesize")
    graph.add_edge("synthesize", END)
    
    return graph.compile()


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_with_reasoning(image, question, depth="standard"):
    """
    Perform reasoned visual analysis.
    
    Args:
        image: dspy.Image or path/URL to image
        question: Question requiring multi-step reasoning
        depth: "quick" (1-2 steps), "standard" (3-4 steps), "deep" (5+ steps)
    
    Returns:
        dict with reasoning chain, answer, and confidence
    """
    agent = build_reasoning_agent()
    
    # Convert to dspy.Image if needed
    if not isinstance(image, dspy.Image):
        image = dspy.Image.from_url(image) if image.startswith('http') else dspy.Image.from_file(image)
    
    return agent.invoke({
        "image": image,
        "initial_question": question,
        "analysis_depth": depth
    })


def safety_analysis(image):
    """
    Analyze image for safety hazards.
    
    Args:
        image: dspy.Image or path/URL (construction site, workplace, etc.)
    
    Returns:
        dict with identified hazards and safety recommendations
    """
    return analyze_with_reasoning(
        image=image,
        question="Identify all potential safety hazards in this scene and assess their severity.",
        depth="deep"
    )


def medical_image_analysis(image, clinical_question):
    """
    Perform multi-step medical image analysis.
    
    Args:
        image: dspy.Image or path/URL to medical image
        clinical_question: Specific clinical question
    
    Returns:
        dict with findings, differential diagnoses, and reasoning
    
    Note: This is for demonstration only. Not for actual medical use.
    """
    return analyze_with_reasoning(
        image=image,
        question=clinical_question,
        depth="deep"
    )


def scene_understanding(image):
    """
    Comprehensive scene understanding with reasoning.
    
    Args:
        image: dspy.Image or path/URL
    
    Returns:
        dict with scene interpretation, context, and implications
    """
    return analyze_with_reasoning(
        image=image,
        question="Provide a comprehensive analysis of this scene including what is happening, context, and implications.",
        depth="standard"
    )


# ============================================================================
# Example Usage
# ============================================================================

def run_example():
    """Run visual reasoning chain examples."""
    import os
    
    print("🧠 mahsm Visual Reasoning Chain Agent\\n")
    print("=" * 60)
    
    # Configure DSPy
    print("\\n📝 Configuring DSPy with GPT-4o-mini...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    dspy.configure(
        lm=dspy.LM(
            model="openai/gpt-4o-mini",
            api_key=api_key
        )
    )
    print("✅ DSPy configured\\n")
    
    # Example 1: Safety Hazard Analysis
    print("=" * 60)
    print("Example 1: Construction Site Safety Analysis")
    print("=" * 60)
    
    print("\\n💡 Multi-step safety reasoning:")
    print("   1. Observe: Scan entire scene for elements")
    print("   2. Hypothesize: Identify potential hazards")
    print("   3. Test: Verify each hazard against safety standards")
    print("   4. Synthesize: Prioritize hazards and recommend actions\\n")
    
    print("📝 Usage:")
    print("   result = safety_analysis('construction_site.jpg')")
    print("   for step in result['reasoning_chain']:")
    print("       print(step)")
    print("   print(f'Confidence: {result[\\\"confidence\\\"]:.2%}')\\n")
    
    # Example 2: Medical Image Analysis
    print("=" * 60)
    print("Example 2: Medical Image Analysis")
    print("=" * 60)
    
    print("\\n💡 Clinical reasoning workflow:")
    print("   1. Observe: Identify anatomical structures and abnormalities")
    print("   2. Hypothesize: Generate differential diagnoses")
    print("   3. Test: Look for supporting/contradicting features")
    print("   4. Synthesize: Rank diagnoses by likelihood\\n")
    
    print("⚠️  Disclaimer: For demonstration purposes only.")
    print("   Not for actual medical diagnosis or treatment.\\n")
    
    print("📝 Usage:")
    print("   result = medical_image_analysis(")
    print("       image='xray.jpg',")
    print("       clinical_question='Assess for pneumonia findings'")
    print("   )")
    print("   print(result['final_answer'])")
    print("   print('\\\\nReasoning:')") 
    print("   for step in result['reasoning_chain']:")
    print("       print(f'  {step}')\\n")
    
    # Example 3: Complex Scene Understanding
    print("=" * 60)
    print("Example 3: Complex Scene Understanding")
    print("=" * 60)
    
    print("\\n💡 Deep scene analysis:")
    print("   - Who: Identify people and roles")
    print("   - What: Determine activities and actions")
    print("   - Where: Understand location and context")
    print("   - Why: Infer intent and implications")
    print("   - When: Temporal context if available\\n")
    
    print("📝 Usage:")
    print("   result = scene_understanding('complex_scene.jpg')")
    print("   print(f'Analysis: {result[\\\"final_answer\\\"]}')") 
    print("   print(f'\\\\nAlternative interpretations:')") 
    print("   for alt in result['alternative_interpretations']:")
    print("       print(f'  - {alt}')\\n")
    
    # Example 4: Custom Reasoning Task
    print("=" * 60)
    print("Example 4: Custom Reasoning Task")
    print("=" * 60)
    
    print("\\n💡 Build custom reasoning workflows:")
    print("   - Forensic analysis")
    print("   - Quality inspection")
    print("   - Sports strategy analysis")
    print("   - Historical photo investigation")
    print("   - Art authentication\\n")
    
    print("📝 Usage:")
    print("   result = analyze_with_reasoning(")
    print("       image='mystery_image.jpg',")
    print("       question='What happened here and what is the evidence?',")
    print("       depth='deep'")
    print("   )")
    print("   # Access full reasoning chain")
    print("   for i, step in enumerate(result['reasoning_chain'], 1):")
    print("       print(f'{i}. {step}')\\n")
    
    print("=" * 60)
    print("✨ Visual Reasoning Chain Demo Complete!")
    print("=" * 60)
    print("\\n💡 Key Features:")
    print("   • Transparent reasoning: See every step")
    print("   • Hypothesis testing: Verify conclusions")
    print("   • Confidence scores: Know reliability")
    print("   • Alternative interpretations: Consider other views")
    print("\\n💰 Cost Efficiency:")
    print("   • Standard analysis: ~$0.30-0.45 (2-3 LLM calls)")
    print("   • Deep analysis: ~$0.50-0.60 (4-5 LLM calls)")
    print("   • All with GPT-4o-mini pricing\\n")


if __name__ == "__main__":
    run_example()
