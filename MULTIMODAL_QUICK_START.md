# Multimodal Vision Support - Quick Start

🎉 **mahsm now supports multimodal vision agents!**

This document provides a quick introduction to the new multimodal capabilities added in the `feature/multimodal-vision-support` branch.

## What's New

### 1. Vision Agent Example (`examples/vision_agent.py`)

A complete, runnable example demonstrating visual question answering with GPT-4 Vision:

```bash
export OPENAI_API_KEY=your-key
python examples/vision_agent.py
```

**Features:**
- Uses `dspy.Image` for vision inputs
- Multi-step visual reasoning workflow (observe → reason)
- Works with both URLs and local files
- Fully integrated with mahsm's `@ma.dspy_node` decorator

### 2. Tutorial Documentation (`docs/tutorials/vision-agents.md`)

Comprehensive tutorial covering:
- Quick start guide
- Image loading patterns
- Multi-step vision workflows
- Best practices
- Troubleshooting

### 3. Test Suite (`tests/examples/test_vision_agent.py`)

Complete test coverage for the vision agent with mocked API calls.

### 4. Multimodal Capabilities Specification (`docs/multimodal-capabilities-spec.md`)

70+ page research document analyzing:
- DSPy multimodal primitives
- LangGraph state handling
- LangFuse tracing capabilities
- Evaluation frameworks
- 12-month implementation roadmap

## Quick Example

```python
import dspy
import mahsm as ma
from typing import TypedDict
from langgraph.graph import StateGraph, END

# Configure vision model
dspy.configure(lm=dspy.LM(model="openai/gpt-4o", api_key="..."))

# Define state
class VisionState(TypedDict):
    image: dspy.Image
    question: str
    answer: str

# Create vision module
@ma.dspy_node
class ImageAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(
            "image: dspy.Image, question: str -> answer: str"
        )
    
    def forward(self, image, question):
        return self.analyze(image=image, question=question)

# Build graph
graph = StateGraph(VisionState)
graph.add_node("analyze", ImageAnalyzer())
graph.set_entry_point("analyze")
graph.add_edge("analyze", END)
app = graph.compile()

# Run it!
image = dspy.Image.from_url("https://example.com/cat.jpg")
result = app.invoke({
    "image": image,
    "question": "What color is the cat?"
})
print(result["answer"])
```

## File Structure

```
mahsm/
├── examples/
│   ├── README.md                    # Examples documentation
│   └── vision_agent.py               # Vision QA agent example
├── docs/
│   ├── multimodal-capabilities-spec.md   # Comprehensive spec
│   └── tutorials/
│       └── vision-agents.md          # Tutorial guide
└── tests/
    └── examples/
        └── test_vision_agent.py      # Test suite
```

## Architecture Highlights

### State Handling

LangGraph automatically handles `dspy.Image` objects in state:
- JSON serialization for checkpointing
- Efficient image caching
- Multi-modal state tracking

### Integration Points

The vision agent demonstrates three key integration points:

1. **@ma.dspy_node with Images**: DSPy modules accept `dspy.Image` inputs
2. **LangGraph State**: TypedDict with `dspy.Image` fields
3. **Multi-step Reasoning**: Observe → Reason workflow pattern

## Implementation Status

✅ **Completed (Phase 0: Foundation)**
- Vision agent example
- Tutorial documentation  
- Test suite
- Comprehensive research spec

🔄 **Next Steps (Phase 1)**
- Enhanced `@ma.dspy_node` with automatic `dspy.Image` handling
- LangFuse integration for vision traces
- Audio/video primitives
- Optimizers for multimodal workflows

## Testing

```bash
# Verify imports work
python -c "import sys; sys.path.insert(0, 'examples'); from vision_agent import *; print('OK')"

# Run tests (requires --run-integration for actual API calls)
pytest tests/examples/test_vision_agent.py -v
```

## Requirements

- Python 3.10+
- OpenAI API key with GPT-4 Vision access
- `pip install dspy pillow` (for image handling)

## Learn More

- **Tutorial**: `docs/tutorials/vision-agents.md`
- **Full Spec**: `docs/multimodal-capabilities-spec.md`
- **Example Code**: `examples/vision_agent.py`

## Design Principles

This implementation follows mahsm's core principles:

1. **Isolated**: All additions are self-contained; no disruption to existing code
2. **Declarative**: Uses `@ma.dspy_node` and DSPy signatures naturally
3. **Observable**: Ready for LangFuse integration
4. **Testable**: Complete test coverage with mocked APIs
5. **Documented**: Tutorial, examples, and comprehensive spec

## Contributing

The multimodal roadmap is in `docs/multimodal-capabilities-spec.md`. 

Key areas for contribution:
- Audio support (speech recognition, TTS)
- Video analysis workflows
- Document understanding (PDF, images with OCR)
- Multimodal optimizers
- Cross-modal retrieval

## Branch Information

- **Branch**: `feature/multimodal-vision-support`
- **Status**: Ready for review
- **PR**: Create PR to merge into `main`

## Questions?

See the tutorial at `docs/tutorials/vision-agents.md` or the comprehensive spec at `docs/multimodal-capabilities-spec.md`.
