# mahsm Documentation

> **Build production-grade AI systems with declarative simplicity.**

mahsm is a Python framework that combines the best tools for building, tracing, and evaluating LLM-powered applications—wrapped in a simple, declarative API.

---

## What is mahsm?

mahsm integrates four powerful frameworks into a unified development experience:

- **[DSPy](building-blocks/dspy/overview.md)** → Prompt engineering through programming
- **[LangGraph](building-blocks/langgraph/overview.md)** → Stateful, cyclical agent workflows  
- **[Langfuse](building-blocks/langfuse/overview.md)** → Production-grade observability
- **[EvalProtocol](building-blocks/evalprotocol/overview.md)** → Systematic evaluation & testing

Instead of learning four different APIs, you learn one: mahsm's declarative interface.

---

## Why mahsm?

### The Problem

Building production LLM applications requires:
1. **Smart prompting** (DSPy's modules & optimizers)
2. **Complex workflows** (LangGraph's state machines)
3. **Deep observability** (Langfuse's tracing)
4. **Rigorous testing** (EvalProtocol's evaluations)

Each framework has its own API, patterns, and integration challenges.

### The Solution

mahsm provides:

```python
import mahsm as ma
import dspy
import os

# 1. Configure once
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
ma.tracing.init()  # Automatic tracing for everything

# 2. Define agents declaratively
@ma.dspy_node
class Researcher(ma.Module):
    def __init__(self):
        super().__init__()
        self.researcher = dspy.ChainOfThought("question -> findings")
    
    def forward(self, question):
        return self.researcher(question=question)

# 3. Build workflows visually
workflow = ma.graph.StateGraph(MyState)
workflow.add_node("research", Researcher())
workflow.add_edge(ma.START, "research")
graph = workflow.compile()

# 4. Run & automatically trace
result = graph.invoke({"question": "..."})
# ✅ All LLM calls traced to Langfuse
# ✅ Full execution graph visible
# ✅ Costs & latencies tracked

# 5. Evaluate systematically
@ma.testing.evaluation_test(...)
async def test_quality(row):
    return await ma.testing.aha_judge(row, rubric="...")
# ✅ Results synced to Langfuse
# ✅ Model comparisons automated
```

**Result**: You write less code, iterate faster, and ship with confidence.

---

## Key Features

### 🎯 Declarative API

Define what you want, not how to build it:

```python
# Instead of manually chaining prompts...
@ma.dspy_node
class MyAgent(ma.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("input -> output")
```

### 🔄 Automatic Tracing

One line enables observability for all frameworks:

```python
ma.tracing.init()
# ✅ DSPy modules traced
# ✅ LangGraph nodes traced
# ✅ Custom @observe functions traced
```

### 📊 Unified Testing

Test across models, prompts, and configurations:

```python
@ma.testing.evaluation_test(
    completion_params=[
        {"model": "openai/gpt-4o-mini"},
        {"model": "openai/gpt-4o"},
    ]
)
async def test_agent(row):
    # Runs on both models, compares results
    pass
```

### 🚀 Production-Ready

- Type-safe state management (TypedDict)
- Structured logging with Langfuse
- Automated evaluation pipelines
- Cost & latency tracking

---

## Quick Start

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

# Configure
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
ma.tracing.init()

# Define state
class State(TypedDict):
    question: str
    answer: str

# Define agent
@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

# Build graph
workflow = ma.graph.StateGraph(State)
workflow.add_node("qa", QA())
workflow.add_edge(ma.START, "qa")
workflow.add_edge("qa", ma.END)
graph = workflow.compile()

# Run
result = graph.invoke({"question": "What is DSPy?"})
print(result["answer"])
# Output visible in Langfuse UI automatically!
```

**Next**: Follow the [Quick Start Guide](getting-started/quickstart.md) for a complete walkthrough.

---

## Learning Path

### 🎓 New to LLM Development?

Start here to learn the fundamentals:

1. **[Installation](getting-started/installation.md)** - Set up your environment
2. **[Core Concepts](getting-started/core-concepts.md)** - Understanding the mahsm philosophy
3. **[Your First Agent](guides/first-agent.md)** - Build a complete agent step-by-step

### 🔧 Want to Understand the Building Blocks?

Deep dive into each framework:

- **[DSPy Basics](building-blocks/dspy/overview.md)** - Signatures, modules, optimizers
- **[LangGraph Basics](building-blocks/langgraph/overview.md)** - State, nodes, edges, routing
- **[Langfuse Basics](building-blocks/langfuse/overview.md)** - Tracing, observability, scoring
- **[EvalProtocol Basics](building-blocks/evalprotocol/overview.md)** - Testing, evaluation, metrics

### 🚀 Ready to Build?

Check out complete examples:

- **[Research Agent](examples/research-agent.md)** - Multi-step reasoning pipeline
- **[Multi-Agent System](guides/multi-agent-systems.md)** - Coordinated agent teams
- **[Evaluation Pipeline](examples/evaluation-pipeline.md)** - Comprehensive testing setup

Or jump straight into the Tutorials:
- **[Tutorials Overview](tutorials/overview.md)** — End-to-end, copy‑pasteable guides

---

## Architecture

mahsm is built on four pillars:

```
┌─────────────────────────────────────────────┐
│             Your Application                │
│   (Agents, Workflows, Evaluations)          │
└─────────────────┬───────────────────────────┘
                  │
                  │ mahsm API
                  │
┌─────────────────▼───────────────────────────┐
│              mahsm Core                      │
│  ┌──────────┬──────────┬──────────────┐    │
│  │  @dspy   │ .tracing │   .testing   │    │
│  │  _node   │  .init() │ .evaluation  │    │
│  └────┬─────┴─────┬────┴───────┬──────┘    │
└───────┼───────────┼────────────┼───────────┘
        │           │            │
   ┌────▼────┐ ┌───▼─────┐ ┌────▼────────┐
   │  DSPy   │ │Langfuse │ │EvalProtocol │
   │ Modules │ │ Tracing │ │    Tests    │
   └─────────┘ └─────────┘ └─────────────┘
        │           │            │
   ┌────▼────────────▼────────────▼─────┐
   │        LangGraph Workflows          │
   │   (StateGraph, compile, invoke)     │
   └─────────────────────────────────────┘
```

**Key Points**:
- **DSPy** powers intelligent prompting
- **LangGraph** orchestrates execution
- **Langfuse** traces everything automatically
- **EvalProtocol** validates quality

mahsm's `@dspy_node` decorator bridges DSPy modules and LangGraph nodes, while `ma.tracing.init()` instruments the entire stack.

---

## Community & Support

- **📖 Documentation**: You're reading it! Explore the sidebar →
- **💬 GitHub Discussions**: [Ask questions](https://github.com/chimera-research/mahsm/discussions)
- **🐛 Issues**: [Report bugs](https://github.com/chimera-research/mahsm/issues)
- **⭐ Star the repo**: [Show your support](https://github.com/chimera-research/mahsm)

---

## What's Next?

- **[Installation Guide](getting-started/installation.md)** → Set up mahsm
- **[Quick Start](getting-started/quickstart.md)** → Build your first agent  
- **[DSPy Overview](building-blocks/dspy/overview.md)** → Learn prompt engineering
- **[LangGraph Overview](building-blocks/langgraph/overview.md)** → Learn workflows

---

**Ready to build? Let's go!** 🚀
