# MAHSM (Multi-Agent Hyper-Scaling Methods) v0.1.0

A Python library that bridges DSPy prompt optimization with LangGraph runtime orchestration without lock-in to the LangChain ecosystem.

## Overview

MAHSM provides three core primitives that maintain a clean message-based state architecture:

- **`ma.prompt.save()`**: Extract and store optimized prompts from DSPy with tool schemas
- **`ma.prompt.load()`**: Retrieve and validate prompts at runtime
- **`ma.inference()`**: Execute prompts in agentic loops with full message transparency

## Key Features

✨ **DSPy Integration**: Seamlessly extract optimized prompts from DSPy GEPA and other optimizers

🔄 **LangGraph Compatible**: Works with LangGraph's MessagesState for complete conversation transparency

📊 **Full Traceability**: Automatic LangFuse integration for observability

🛠️ **Tool Validation**: OpenAI function schema validation ensures runtime safety

🎯 **Minimal Dependencies**: Only uses LangChain Core message types, no ecosystem lock-in

## Installation

```bash
pip install mahsm
```

### Development Installation

```bash
git clone https://github.com/chimera-research/mahsm.git
cd mahsm
pip install -e ".[dev]"
```

## Quick Start

### 1. Optimize with DSPy

```python
import dspy
from dspy.teleprompt import GEPA

# Configure and optimize
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4o-mini"))
program = dspy.Predict(YourSignature)
compiled_program = optimizer.compile(program, trainset=your_data)
```

### 2. Save Optimized Prompt

```python
import mahsm as ma

def my_tool(param: str) -> str:
    """A tool function."""
    return f"Result: {param}"

# Save with tool schemas
ma.prompt.save(
    compiled_program,
    name="my_task",
    version="v1",
    tools=[my_tool]
)
```

### 3. Load and Execute

```python
from langgraph.graph import StateGraph, MessagesState

def my_node(state: MessagesState):
    # Load with validation
    prompt = ma.prompt.load("my_task_v1", validate_tools=[my_tool])
    
    # Execute with transparency
    result, messages = ma.inference(
        model="openai/gpt-4o-mini",
        prompt=prompt,
        tools=[my_tool],
        input=state["messages"][-1].content,
        state=state.get("messages", [])
    )
    
    return {"messages": messages}
```

## Configuration

MAHSM supports configuration via environment variables:

```bash
# LangFuse tracing (optional)
export LANGFUSE_PUBLIC_KEY="your_public_key"
export LANGFUSE_SECRET_KEY="your_secret_key"

# Custom MAHSM home directory (optional)
export MAHSM_HOME="~/.mahsm"

# Default max iterations (optional)
export MAHSM_MAX_ITERATIONS="10"
```

## Message Transparency

MAHSM maintains complete conversation transparency with proper message ordering:

1. `SystemMessage` - The optimized prompt
2. `HumanMessage` - User input
3. `AIMessage` - Model responses with tool_calls
4. `ToolMessage` - Tool execution results
5. (Repeat 3-4 until completion)

Inspect the full conversation history in LangFuse traces or directly from the returned messages list.

## Requirements

- Python 3.11+
- DSPy
- LangGraph
- LangChain Core (message types only)
- LangFuse (optional, for tracing)
- OpenAI SDK

## Documentation

- [API Reference](docs/api/)
- [User Guides](docs/guides/)
- [Examples](docs/examples/)
- [Concepts](docs/concepts/)

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/ -m unit

# Integration tests
pytest tests/integration/ -m integration

# With coverage
pytest --cov=mahsm --cov-report=html
```

### Code Quality

```bash
# Format code
black mahsm tests

# Sort imports
isort mahsm tests

# Type checking
mypy mahsm

# Linting
flake8 mahsm tests
```

## Contributing

Contributions are welcome! Please see our contributing guidelines for details.

## License

MIT License - see LICENSE file for details.

## Citation

If you use MAHSM in your research, please cite:

```bibtex
@software{mahsm2025,
  title = {MAHSM: Multi-Agent Hyper-Scaling Methods},
  author = {MAHSM Contributors},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/chimera-research/mahsm}
}
```

## Support

- 📖 [Documentation](https://mahsm.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/chimera-research/mahsm/issues)
- 💬 [Discussions](https://github.com/chimera-research/mahsm/discussions)
