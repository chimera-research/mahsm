# mahsm: Multi-Agent Hyper-Scaling Methods

<p align="center">
  <strong>A declarative framework for building, tracing, and evaluating multi-agent LLM systems</strong>
</p>

<p align="center">
  <a href="https://github.com/chimera-research/mahsm/actions"><img alt="CI Status" src="https://github.com/chimera-research/mahsm/workflows/CI/badge.svg"></a>
  <a href="https://pypi.org/project/mahsm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mahsm"></a>
  <a href="https://pypi.org/project/mahsm/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/mahsm"></a>
  <a href="https://github.com/chimera-research/mahsm/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/chimera-research/mahsm"></a>
</p>

---

## What is mahsm?

**mahsm** unifies four best-in-class libraries into a single, zero-boilerplate framework:

| Component | Purpose | What mahsm Adds |
|-----------|---------|----------------|
| **[DSPy](https://dspy.ai/)** | Programming LLMs with optimizable modules | Automatic LangGraph integration via `@ma.dspy_node` |
| **[LangGraph](https://www.langchain.com/langgraph)** | Stateful, cyclical multi-agent workflows | Zero-boilerplate DSPy module wrapping |
| **[Langfuse](https://langfuse.com/)** | Deep LLM observability & tracing | Automatic instrumentation with `ma.init()` |
| **[EvalProtocol](https://pypi.org/project/eval-protocol/)** | Pytest-based LLM evaluation | Pre-configured harness for LangGraph apps |

### The mahsm Advantage

**Before mahsm** (Traditional approach):
```python
# Manual state mapping, verbose boilerplate
def my_node(state: dict) -> dict:
    module = MyDSPyModule()
    result = module(input=state["input"])
    return {"output": result.output}  # Manual extraction

# Separate tracing setup
from langfuse import Langfuse
langfuse = Langfuse()
# ... complex instrumentation code ...
```

**With mahsm** (Declarative approach):
```python
import mahsm as ma

ma.init()  # One-line tracing setup

@ma.dspy_node  # Automatic state mapping
class MyModule(ma.Module):
    def forward(self, input):
        return self.predictor(input=input)
```

---

## ⚡ Quick Start

### Installation

```bash
pip install mahsm
```

### Your First Agent (60 seconds)

```python
import mahsm as ma
from typing import TypedDict
import dspy
import os

# 1. Configure DSPy
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)

# 2. Enable tracing
handler = ma.tracing.init()

# 3. Define agent state
class State(TypedDict):
    query: str
    answer: str

@ma.dspy_node
class Researcher(ma.Module):
    def __init__(self):
        super().__init__()
        self.cot = ma.dspy.ChainOfThought("query -> answer")
    
    def forward(self, query):
        return self.cot(query=query)

# 3. Build graph
workflow = ma.graph.StateGraph(State)
workflow.add_node("researcher", Researcher())
workflow.add_edge(ma.START, "researcher")
workflow.add_edge("researcher", ma.END)

graph = workflow.compile()

# 4. Run
result = graph.invoke({"query": "What is DSPy?"})
print(result["answer"])
```

**That's it!** Your agent is now:
- ✅ Running with proper state management
- ✅ Automatically traced in Langfuse
- ✅ Ready for evaluation with EvalProtocol

---

## 📖 Complete Tutorial

Want to build a production-ready agent with full observability and evaluation?

**👉 [Read the Complete Quickstart Guide](./QUICKSTART.md)**

You'll learn:
- 🎯 Building complex multi-node agents
- 📊 Setting up Langfuse for tracing
- 🧪 Running systematic evaluations
- 📈 Viewing results in Langfuse & EvalProtocol UIs
- ⚡ Optimizing agents with DSPy compilers

---

## 🏗️ Core Features

### 1. `@ma.dspy_node` Decorator

The heart of mahsm: converts DSPy modules to LangGraph nodes **automatically**.

**Supports two patterns:**

#### Class Decorator
```python
@ma.dspy_node
class MyModule(ma.Module):
    def forward(self, input1, input2):
        # Your logic here
        return self.predictor(input1=input1, input2=input2)

# Use in graph
workflow.add_node("my_node", MyModule())
```

#### Instance Wrapper
```python
# Wrap any DSPy module instance
cot = ma.dspy.ChainOfThought("question -> answer")
node = ma.dspy_node(cot)

# Use directly
workflow.add_node("cot", node)
```

**What it does:**
- ✅ Introspects `forward()` parameters (excludes `self`)
- ✅ Extracts matching fields from LangGraph state
- ✅ Returns result as state updates (non-private fields)
- ✅ No manual state mapping needed

### 2. `ma.tracing.init()` - One-Line Tracing

```python
handler = ma.tracing.init()
```

**Automatically:**
- ✅ Initializes Langfuse client from environment variables
- ✅ Instruments DSPy for automatic trace capture
- ✅ Returns LangChain `CallbackHandler` for LangGraph tracing
- ⚠️ Gracefully warns if credentials missing

### 3. `ma.testing.PytestHarness`

Bridge from LangGraph apps to EvalProtocol evaluations:

```python
from my_agent import graph
import mahsm as ma

harness = ma.testing.PytestHarness(graph=graph)

@ma.testing.evaluation_test(
    data_loaders=harness.data_loaders,
    rollout_processor=harness.rollout_processor,
    completion_params=[{"model": "openai/gpt-4o-mini"}],
)
async def test_quality(row):
    return await ma.testing.aha_judge(row, judge_model="openai/gpt-4o", rubric="...")
```

---

## 🛠️ Development

### Setup

```bash
git clone https://github.com/chimera-research/mahsm.git
cd mahsm
pip install -e .
```

### Run Tests

```bash
python tests/test_core.py              # Unit tests
python tests/test_graph_integration.py  # Integration tests
```

### CI/CD

- **GitHub Actions**: Runs tests on push/PR (Python 3.10-3.12, Linux/Mac/Windows)
- **PyPI Publishing**: Automatic via GitHub Releases or manual workflow dispatch

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        mahsm Framework                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │   DSPy       │─────▶│  @dspy_node  │                   │
│  │   Modules    │      │  Decorator   │                   │
│  └──────────────┘      └──────┬───────┘                   │
│                                │                            │
│                                ▼                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │          LangGraph StateGraph                    │     │
│  │  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐         │     │
│  │  │Node1│──▶│Node2│──▶│Node3│──▶│End  │         │     │
│  │  └─────┘   └─────┘   └─────┘   └─────┘         │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│         ┌───────────────┴───────────────┐                 │
│         │                                │                 │
│         ▼                                ▼                 │
│  ┌──────────────┐              ┌──────────────┐           │
│  │   Langfuse   │              │ EvalProtocol │           │
│  │   Tracing    │              │  Evaluation  │           │
│  └──────────────┘              └──────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. **Check existing issues** or create one
2. **Fork the repo** and create a feature branch
3. **Write tests** for new functionality
4. **Submit a PR** with clear description

---

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

---

## 🌟 Acknowledgments

mahsm stands on the shoulders of giants:

- **[DSPy](https://github.com/stanfordnlp/dspy)** by Stanford NLP
- **[LangGraph](https://github.com/langchain-ai/langgraph)** by LangChain
- **[Langfuse](https://github.com/langfuse/langfuse)** by Langfuse Team
- **[EvalProtocol](https://github.com/areibman/eval-protocol)** by Fireworks AI

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/chimera-research/mahsm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chimera-research/mahsm/discussions)
- **Twitter**: [@chimera_research](https://twitter.com/chimera_research) _(if applicable)_

---

<p align="center">
  <strong>Built with ❤️ by Chimera Research</strong>
</p>

<p align="center">
  <a href="https://github.com/chimera-research/mahsm">⭐ Star us on GitHub</a>
</p>

