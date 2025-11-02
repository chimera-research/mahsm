# mahsm Examples

This directory contains practical examples demonstrating mahsm's capabilities.

## Available Examples

### 1. Vision Agent (`vision_agent.py`)

A complete multimodal agent that performs visual question answering using GPT-4 Vision.

**Features:**
- Uses `dspy.Image` for vision inputs
- Multi-step visual reasoning workflow
- Works with both URLs and local image files
- Demonstrates `@ma.dspy_node` with multimodal data

**Usage:**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your-key-here

# Run the example
python examples/vision_agent.py
```

**Requirements:**
- Python 3.10+
- OpenAI API key with GPT-4 Vision access
- `pip install dspy pillow`

**What it does:**
1. Loads an image (from URL or file)
2. Extracts visual observations
3. Performs step-by-step reasoning
4. Answers your question with confidence score

**Example Output:**
```
🔍 mahsm Vision Agent Example
============================================================

📝 Configuring DSPy with GPT-4 Vision...
✅ DSPy configured with gpt-4o

🏗️  Building vision agent graph...
✅ Agent built successfully

============================================================
Example 1: Image from URL
============================================================

🖼️  Image: https://...cat.jpg
❓ Question: What color is the cat in this image?

🤔 Processing...

📊 Results:
------------------------------------------------------------
Observations: The image shows a domestic cat with distinctive...
Reasoning: Based on the visual observations, the cat has...
✅ Answer: The cat is orange/ginger colored
📈 Confidence: 95.00%
```

## Running Examples

All examples are self-contained and can be run independently:

```bash
# From project root
python examples/vision_agent.py
```

## Testing Examples

Examples have accompanying tests in `tests/examples/`:

```bash
pytest tests/examples/test_vision_agent.py
```

## Creating Your Own Examples

When creating new examples:

1. **Follow the pattern:**
   - Define state with TypedDict
   - Create DSPy modules with `@ma.dspy_node`
   - Build graph with LangGraph
   - Include clear docstrings

2. **Make them runnable:**
   - Include a `if __name__ == "__main__"` block
   - Add environment variable checks
   - Provide clear error messages

3. **Document thoroughly:**
   - Explain what the example demonstrates
   - List requirements
   - Show expected output

4. **Add tests:**
   - Create corresponding test file
   - Test both success and error cases
   - Mock external APIs when possible

## Example Template

```python
\"\"\"
Your Example - Brief Description

This example demonstrates:
- Feature 1
- Feature 2
- Feature 3
\"\"\"

import dspy
import mahsm as ma
from typing import TypedDict
from langgraph.graph import StateGraph, END

# State definition
class YourState(TypedDict):
    # Define your state fields
    pass

# DSPy modules
@ma.dspy_node
class YourModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Initialize
    
    def forward(self, **inputs):
        # Implement
        pass

# Graph construction
def build_your_agent():
    graph = StateGraph(YourState)
    # Build graph
    return graph.compile()

# Example usage
if __name__ == "__main__":
    agent = build_your_agent()
    result = agent.invoke({...})
    print(result)
```
