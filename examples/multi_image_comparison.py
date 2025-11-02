"""
Multi-Image Comparison Agent - Compare and Analyze Multiple Images

This example demonstrates how to build a multi-image comparison agent using mahsm.
Perfect for:
- Before/after analysis
- Product comparison
- Quality control & defect detection
- Visual A/B testing

It showcases:
- Comparing 2+ images simultaneously
- Structured difference analysis
- Confidence scoring for findings
- Production-ready error handling

Requirements:
    pip install dspy pillow

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (for GPT-5-mini)
    
Cost Estimate:
    ~$0.25 per comparison (2 images with GPT-5-mini)
"""

import dspy
import mahsm as ma
from typing import TypedDict, List
from langgraph.graph import StateGraph, END


# ============================================================================
# State Definition
# ============================================================================

class ComparisonState(TypedDict):
    """State for multi-image comparison workflow."""
    images: List[dspy.Image]  # Images to compare
    comparison_type: str  # Type: "before_after", "product", "quality_control"
    
    # Analysis outputs
    individual_analyses: List[str]  # Analysis of each image
    differences: List[dict]  # Structured differences found
    summary: str  # Overall comparison summary
    confidence: float  # Confidence in findings (0-1)


# ============================================================================
# DSPy Modules for Image Comparison
# ============================================================================

@ma.dspy_node
class ImageAnalyzer(dspy.Module):
    """
    Analyzes individual images to extract key features.
    
    This node processes each image independently to identify:
    - Objects and elements
    - Colors and composition
    - Quality indicators
    - Notable features
    """
    
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(
            "image: dspy.Image, purpose: str -> analysis: str, key_features: str"
        )
    
    def forward(self, image, purpose):
        """Extract features from a single image."""
        result = self.analyze(
            image=image,
            purpose=purpose
        )
        return result


@ma.dspy_node
class DifferenceDetector(dspy.Module):
    """
    Compares analyses to identify specific differences.
    
    Takes individual image analyses and produces structured
    difference descriptions with locations and severity.
    """
    
    def __init__(self):
        super().__init__()
        self.compare = dspy.ChainOfThought(
            """analyses: List[str], comparison_type: str -> 
               differences: str, confidence: float"""
        )
    
    def forward(self, analyses, comparison_type):
        """Detect differences between analyzed images."""
        # Join analyses for comparison
        combined_analyses = "\n\n".join([
            f"Image {i+1}: {analysis}" 
            for i, analysis in enumerate(analyses)
        ])
        
        result = self.compare(
            analyses=combined_analyses,
            comparison_type=comparison_type
        )
        return result


@ma.dspy_node
class ComparisonSummarizer(dspy.Module):
    """
    Generates executive summary of comparison findings.
    
    Produces actionable summary highlighting:
    - Most significant differences
    - Recommendations
    - Key takeaways
    """
    
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(
            """differences: str, comparison_type: str -> 
               summary: str, recommendations: str"""
        )
    
    def forward(self, differences, comparison_type):
        """Create executive summary of comparison."""
        result = self.summarize(
            differences=differences,
            comparison_type=comparison_type
        )
        return result


# ============================================================================
# Graph Construction
# ============================================================================

def build_comparison_agent() -> StateGraph:
    """
    Builds the multi-image comparison workflow graph.
    
    Workflow:
        1. Analyze: Process each image individually
        2. Compare: Identify differences between analyses
        3. Summarize: Generate executive summary
    
    Returns:
        Compiled LangGraph StateGraph
    """
    graph = StateGraph(ComparisonState)
    
    # Initialize modules
    analyzer = ImageAnalyzer()
    detector = DifferenceDetector()
    summarizer = ComparisonSummarizer()
    
    # Add nodes
    def analyze_all(state):
        """Analyze all input images."""
        analyses = []
        for img in state["images"]:
            result = analyzer(
                image=img,
                purpose=state["comparison_type"]
            )
            analyses.append(result.analysis)
        return {"individual_analyses": analyses}
    
    def detect_differences(state):
        """Compare analyses to find differences."""
        result = detector(
            analyses=state["individual_analyses"],
            comparison_type=state["comparison_type"]
        )
        # Parse differences string into structured format
        differences = [{
            "description": result.differences,
            "confidence": result.confidence
        }]
        return {
            "differences": differences,
            "confidence": result.confidence
        }
    
    def summarize_findings(state):
        """Generate summary of comparison."""
        # Combine all differences
        all_diffs = "\n".join([
            d["description"] for d in state["differences"]
        ])
        
        result = summarizer(
            differences=all_diffs,
            comparison_type=state["comparison_type"]
        )
        return {"summary": result.summary}
    
    graph.add_node("analyze", analyze_all)
    graph.add_node("compare", detect_differences)
    graph.add_node("summarize", summarize_findings)
    
    # Define workflow
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "compare")
    graph.add_edge("compare", "summarize")
    graph.add_edge("summarize", END)
    
    return graph.compile()


# ============================================================================
# Convenience Functions
# ============================================================================

def compare_before_after(before_image, after_image):
    """
    Compare before/after images to identify changes.
    
    Args:
        before_image: dspy.Image or path/URL to before image
        after_image: dspy.Image or path/URL to after image
    
    Returns:
        dict with analysis, differences, and summary
    """
    agent = build_comparison_agent()
    
    # Convert to dspy.Image if needed
    if not isinstance(before_image, dspy.Image):
        before_image = dspy.Image.from_url(before_image) if before_image.startswith('http') else dspy.Image.from_file(before_image)
    if not isinstance(after_image, dspy.Image):
        after_image = dspy.Image.from_url(after_image) if after_image.startswith('http') else dspy.Image.from_file(after_image)
    
    return agent.invoke({
        "images": [before_image, after_image],
        "comparison_type": "before_after"
    })


def compare_products(product_images):
    """
    Compare multiple product images for e-commerce.
    
    Args:
        product_images: List of dspy.Image or paths/URLs
    
    Returns:
        dict with comparative analysis
    """
    agent = build_comparison_agent()
    
    # Convert all to dspy.Image
    images = []
    for img in product_images:
        if not isinstance(img, dspy.Image):
            images.append(
                dspy.Image.from_url(img) if img.startswith('http') 
                else dspy.Image.from_file(img)
            )
        else:
            images.append(img)
    
    return agent.invoke({
        "images": images,
        "comparison_type": "product"
    })


def quality_control_check(reference_image, test_images):
    """
    Compare test images against reference for QC.
    
    Args:
        reference_image: dspy.Image or path/URL (expected/correct)
        test_images: List of images to check against reference
    
    Returns:
        dict with QC results and defects found
    """
    agent = build_comparison_agent()
    
    # Convert to dspy.Image
    if not isinstance(reference_image, dspy.Image):
        reference_image = dspy.Image.from_url(reference_image) if reference_image.startswith('http') else dspy.Image.from_file(reference_image)
    
    images = [reference_image]
    for img in test_images:
        if not isinstance(img, dspy.Image):
            images.append(
                dspy.Image.from_url(img) if img.startswith('http')
                else dspy.Image.from_file(img)
            )
        else:
            images.append(img)
    
    return agent.invoke({
        "images": images,
        "comparison_type": "quality_control"
    })


# ============================================================================
# Example Usage
# ============================================================================

def run_example():
    """Run multi-image comparison examples."""
    import os
    
    print("🔍 mahsm Multi-Image Comparison Agent\\n")
    print("=" * 60)
    
    # Configure DSPy
    print("\\n📝 Configuring DSPy with GPT-5-mini...")
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
    
    # Example: Before/After Comparison
    print("=" * 60)
    print("Example: Before/After Room Renovation")
    print("=" * 60)
    
    # Using example images (you can replace with your own)
    before_url = "https://example.com/room-before.jpg"  # Replace
    after_url = "https://example.com/room-after.jpg"    # Replace
    
    print(f"\\n🖼️  Comparing renovation results...")
    print(f"   Before: {before_url}")
    print(f"   After: {after_url}\\n")
    
    # Note: Replace URLs with actual images to run
    # result = compare_before_after(before_url, after_url)
    # 
    # print("📊 Results:")
    # print("-" * 60)
    # print(f"Summary: {result['summary']}")
    # print(f"Confidence: {result['confidence']:.2%}")
    # print(f"\\nKey Differences:")
    # for i, diff in enumerate(result['differences'], 1):
    #     print(f"  {i}. {diff['description']}")
    
    print("💡 Replace example URLs with real images to run comparison\\n")
    
    print("=" * 60)
    print("✨ Comparison Agent Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_example()
