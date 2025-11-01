# Quickstart: Building Your First mahsm Agent

Let's build a simple research agent in 60 seconds to see how mahsm works. This example demonstrates the declarative nature of the framework.

---

## Complete Example

```python
import mahsm as ma
from typing import TypedDict, Optional
import dspy
import os

# 1. Configure DSPy
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)

# 2. Initialize tracing (do this once at the start of your app)
ma.tracing.init()

# 3. Define the shared state
class AgentState(TypedDict):
    question: str
    research_result: Optional[str]

# 4. Create a reasoning node with @ma.dspy_node
@ma.dspy_node
class Researcher(ma.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("question -> research_result")

    def forward(self, question):
        return self.predictor(question=question)

# 5. Build and compile the graph
workflow = ma.graph.StateGraph(AgentState)
workflow.add_node("researcher", Researcher())
workflow.add_edge(ma.START, "researcher")
workflow.add_edge("researcher", ma.END)
graph = workflow.compile()

# 6. Run your agent
result = graph.invoke({"question": "What is the future of multi-agent AI systems?"})
print(result['research_result'])
# ✅ Automatically traced in Langfuse!
```

That's it! You've built a fully observable and testable agent with minimal boilerplate.

---

## Breaking It Down

### 1. Configure DSPy

```python
import dspy
import os

lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
```

**What's happening:**
- Configure the language model that DSPy will use
- `dspy.LM()` supports OpenAI, Anthropic, local models, and more
- Use environment variables for API keys (never hardcode!)

### 2. Initialize Tracing

```python
ma.tracing.init()
```

**What's happening:**
- One line enables automatic tracing for all LLM calls
- Traces are sent to Langfuse for observability
- Make sure you have `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` in your environment

### 3. Define State

```python
from typing import TypedDict, Optional

class AgentState(TypedDict):
    question: str
    research_result: Optional[str]
```

**What's happening:**
- State is a TypedDict that flows through your workflow
- `question` is required (input)
- `research_result` is optional (populated by nodes)
- Type-safe and IDE-friendly

### 4. Create a DSPy Node

```python
@ma.dspy_node
class Researcher(ma.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("question -> research_result")

    def forward(self, question):
        return self.predictor(question=question)
```

**What's happening:**
- `@ma.dspy_node` makes your DSPy module work with LangGraph
- `ChainOfThought` adds reasoning before answering
- Signature `"question -> research_result"` matches state keys
- `forward()` method contains your logic

### 5. Build the Workflow

```python
workflow = ma.graph.StateGraph(AgentState)
workflow.add_node("researcher", Researcher())
workflow.add_edge(ma.START, "researcher")
workflow.add_edge("researcher", ma.END)
graph = workflow.compile()
```

**What's happening:**
- `StateGraph(AgentState)` creates a workflow with your state schema
- `add_node()` adds your Researcher to the graph
- `add_edge()` defines the flow: START → researcher → END
- `compile()` turns the workflow into an executable graph

### 6. Run Your Agent

```python
result = graph.invoke({"question": "What is the future of multi-agent AI systems?"})
print(result['research_result'])
```

**What's happening:**
- `invoke()` runs the workflow with initial state
- State flows through nodes, getting updated along the way
- Returns the final state with all populated fields
- All LLM calls are automatically traced to Langfuse!

---

## Next Steps

- **[Core Concepts](core-concepts.md)** → Understand the mahsm philosophy
- **[DSPy Overview](../building-blocks/dspy/overview.md)** → Learn about DSPy modules
- **[LangGraph Overview](../building-blocks/langgraph/overview.md)** → Learn about workflows
- **[Your First Agent](../guides/first-agent.md)** → Build a complete multi-step agent

---

**Ready to build more? Explore the [Building Blocks](../building-blocks/dspy/overview.md)!** 🚀