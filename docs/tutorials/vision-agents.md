# Building Vision Agents with mahsm

This tutorial shows you how to build multimodal agents that can process images using mahsm's declarative API.

## Overview

Vision agents combine computer vision capabilities with language models to:
- Answer questions about images
- Extract information from visual content
- Make decisions based on visual observations
- Perform multi-step reasoning about scenes

## Prerequisites

- Python 3.10+
- OpenAI API key with GPT-4 Vision access
- Basic understanding of mahsm concepts (DSPy modules, LangGraph)

Install required packages:
```bash
pip install dspy pillow
```

## Quick Start

### 1. Configure Vision Model

```python
import dspy
import os

# Set up DSPy with a vision-capable model
dspy.configure(
    lm=dspy.LM(
        model="openai/gpt-4o",  # GPT-4 with vision
        api_key=os.getenv("OPENAI_API_KEY")
    )
)
```

### 2. Define Your State

Vision workflows need to track both images and text:

```python
from typing import TypedDict
import dspy

class VisionState(TypedDict):
    # Input
    image: dspy.Image
    question: str
    
    # Output
    answer: str
```

### 3. Create Vision Modules

Use `dspy.Image` in your signatures:

```python
import mahsm as ma

@ma.dspy_node
class ImageAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(
            "image: dspy.Image, question: str -> answer: str"
        )
    
    def forward(self, image, question):
        return self.analyze(image=image, question=question)
```

### 4. Build Your Graph

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(VisionState)
analyzer = ImageAnalyzer()

graph.add_node("analyze", analyzer)
graph.set_entry_point("analyze")
graph.add_edge("analyze", END)

app = graph.compile()
```

### 5. Run It!

```python
# Load an image
image = dspy.Image.from_url("https://example.com/photo.jpg")

# Or from a file
image = dspy.Image.from_file("photo.jpg")

# Run the agent
result = app.invoke({
    "image": image,
    "question": "What's in this image?"
})

print(result["answer"])
```

## Working with Images

### Loading Images

DSPy supports three ways to load images:

```python
# From URL
image = dspy.Image.from_url("https://example.com/image.jpg")

# From local file
image = dspy.Image.from_file("path/to/image.jpg")

# From PIL Image
from PIL import Image
pil_img = Image.open("photo.jpg")
image = dspy.Image.from_PIL(pil_img)
```

### Supported Formats

- **Image types:** PNG, JPG, JPEG, WEBP
- **Max size:** Depends on model (typically ~20MB)
- **Automatic encoding:** DSPy handles base64 encoding

## Multi-Step Vision Workflows

For complex visual reasoning, break tasks into multiple steps:

```python
@ma.dspy_node
class VisualObserver(dspy.Module):
    """Extract visual features"""
    def __init__(self):
        super().__init__()
        self.observe = dspy.ChainOfThought(
            "image: dspy.Image -> features: str, objects: list[str]"
        )
    
    def forward(self, image):
        return self.observe(image=image)

@ma.dspy_node
class VisualReasoner(dspy.Module):
    """Reason about observations"""
    def __init__(self):
        super().__init__()
        self.reason = dspy.ChainOfThought(
            "features: str, objects: list[str], question: str -> answer: str"
        )
    
    def forward(self, features, objects, question):
        return self.reason(
            features=features,
            objects=objects,
            question=question
        )

# Build multi-step graph
graph = StateGraph(VisionState)
graph.add_node("observe", VisualObserver())
graph.add_node("reason", VisualReasoner())
graph.add_edge("observe", "reason")
graph.add_edge("reason", END)
```

## Best Practices

### 1. Structured Outputs

Request specific output formats:

```python
self.analyze = dspy.ChainOfThought(
    """image: dspy.Image -> 
       objects: list[str], 
       colors: list[str], 
       description: str"""
)
```

### 2. Confidence Scores

Include confidence in your outputs:

```python
self.predict = dspy.Predict(
    "image: dspy.Image, question: str -> answer: str, confidence: float"
)
```

### 3. Error Handling

Handle missing images gracefully:

```python
def load_image_safely(path):
    try:
        return dspy.Image.from_file(path)
    except FileNotFoundError:
        print(f"Image not found: {path}")
        return None
```

### 4. Cost Management

Vision APIs can be expensive. Optimize by:
- Resizing large images before processing
- Caching results for repeated queries
- Using batch processing when possible

## Example: Visual QA Agent

Here's a complete example from `examples/vision_agent.py`:

```python
import dspy
import mahsm as ma
from typing import TypedDict
from langgraph.graph import StateGraph, END

class VQAState(TypedDict):
    image: dspy.Image
    question: str
    observations: str
    answer: str
    confidence: float

@ma.dspy_node
class VisualObserver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.observe = dspy.ChainOfThought(
            "image: dspy.Image -> observations: str"
        )
    
    def forward(self, image):
        return self.observe(image=image)

@ma.dspy_node
class VisualReasoner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reason = dspy.ChainOfThought(
            """observations: str, question: str -> 
               answer: str, confidence: float"""
        )
    
    def forward(self, observations, question):
        return self.reason(
            observations=observations,
            question=question
        )

def build_agent():
    graph = StateGraph(VQAState)
    graph.add_node("observe", VisualObserver())
    graph.add_node("reason", VisualReasoner())
    graph.set_entry_point("observe")
    graph.add_edge("observe", "reason")
    graph.add_edge("reason", END)
    return graph.compile()

# Usage
agent = build_agent()
result = agent.invoke({
    "image": dspy.Image.from_url("https://example.com/cat.jpg"),
    "question": "What color is the cat?"
})
```

## Troubleshooting

### "Image is not JSON serializable"

Make sure you're using `dspy.Image` in your state definition, not raw PIL images.

### Vision model not responding

Check that you're using a vision-capable model:
- OpenAI: `gpt-4o`, `gpt-4-vision-preview`
- Anthropic: `claude-3-opus`, `claude-3-sonnet`

### High API costs

- Use smaller images when possible
- Cache repeated queries
- Consider using Claude 3 Haiku for simpler tasks

## Next Steps

- Explore `examples/vision_agent.py` for a complete working example
- Learn about [LangFuse tracing](../concepts/observability.md) for vision agents
- Check out advanced patterns in the [multimodal spec](../multimodal-capabilities-spec.md)

## Additional Resources

- [DSPy Image Documentation](https://dspy.ai/api/primitives/Image)
- [GPT-4 Vision API](https://platform.openai.com/docs/guides/vision)
- [mahsm Examples](../../examples/)
