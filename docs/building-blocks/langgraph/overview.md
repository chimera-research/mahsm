# LangGraph Overview

> **TL;DR**: LangGraph builds stateful, cyclical workflows for LLM agents—think state machines for AI.

## What is LangGraph?

**LangGraph** is a framework for building **stateful, multi-step workflows** with LLMs. Unlike simple chains (input → LLM → output), LangGraph enables:

- **Cycles**: Agents can loop, retry, and refine
- **State**: Persistent memory across steps
- **Branching**: Conditional routing based on outputs
- **Parallelism**: Run multiple nodes concurrently

Think of it as a **state machine** where each node is an AI agent or tool.

---

## Why LangGraph?

### The Problem with Chains

Traditional LLM chains are linear:

```python
# ❌ Linear chain - can't loop or branch
query → retrieve_docs → generate_answer → done
```

Real agents need to:
- Loop until a condition is met
- Branch based on intermediate results
- Maintain state across steps

### The LangGraph Solution

```python
# ✅ Cyclic workflow with branching
       ┌─────────────┐
       │  generate   │
       │   query     │
       └──────┬──────┘
              │
              ▼
       ┌─────────────┐
       │   search    │◄─────┐
       └──────┬──────┘      │
              │              │
              ▼              │
       ┌─────────────┐      │
       │ synthesize  │      │
       └──────┬──────┘      │
              │              │
              ▼              │
       ┌─────────────┐      │
       │   check     │──────┘
       │  quality    │ if poor, retry
       └──────┬──────┘
              │ if good
              ▼
            END
```

---

## Core Concepts

### 1. **State**

State is a `TypedDict` that flows through your workflow:

```python
from typing import TypedDict, Optional

class ResearchState(TypedDict):
    question: str
    search_query: Optional[str]
    findings: Optional[str]
    answer: Optional[str]
```

**[Learn more about State →](state.md)**

### 2. **Nodes**

Nodes are functions or agents that process state:

```python
import mahsm as ma

@ma.dspy_node
class Researcher(ma.Module):
    def __init__(self):
        super().__init__()\n        self.research = dspy.ChainOfThought("question -> findings")
    
    def forward(self, question):
        return self.research(question=question)
```

**[Learn more about Nodes & Edges →](nodes-edges.md)**

### 3. **Edges**

Edges connect nodes:

```python
# Simple edge
workflow.add_edge("node_a", "node_b")

# Conditional edge
workflow.add_conditional_edges(
    "checker",
    lambda state: "retry" if state["quality"] < 0.7 else END
)
```

**[Learn more about Conditional Routing →](conditional-routing.md)**

### 4. **Graph Compilation**

Compile the workflow into an executable graph:

```python
workflow = ma.graph.StateGraph(MyState)
workflow.add_node("agent", my_agent)
workflow.add_edge(ma.START, "agent")
workflow.add_edge("agent", ma.END)

graph = workflow.compile()  # ✅ Ready to run
```

**[Learn more about Compilation →](compilation.md)**

---

## Quick Example

Let's build a self-correcting Q&A agent:

```python
import mahsm as ma
from typing import TypedDict, Optional
import dspy
import os

# Configure
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
ma.tracing.init()

# 1. Define state
class QAState(TypedDict):
    question: str
    answer: Optional[str]
    quality_score: Optional[float]
    iteration: int

# 2. Define nodes
@ma.dspy_node
class Answerer(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

@ma.dspy_node
class QualityChecker(ma.Module):
    def __init__(self):
        super().__init__()
        self.checker = dspy.Predict("question, answer -> quality_score: float 0-1")
    
    def forward(self, question, answer):
        return self.checker(question=question, answer=answer)

def increment_iteration(state: QAState) -> QAState:
    """Increment iteration counter."""
    return {"iteration": state.get("iteration", 0) + 1}

# 3. Define routing
def should_retry(state: QAState):
    """Retry if quality is low and we haven't exceeded max iterations."""
    if state.get("iteration", 0) >= 3:
        return ma.END  # Give up after 3 tries
    
    quality = float(state.get("quality_score", 0))
    if quality < 0.7:
        return "answer"  # Retry
    return ma.END  # Good enough!

# 4. Build graph
workflow = ma.graph.StateGraph(QAState)

workflow.add_node("answer", Answerer())
workflow.add_node("check", QualityChecker())
workflow.add_node("increment", increment_iteration)

workflow.add_edge(ma.START, "increment")
workflow.add_edge("increment", "answer")
workflow.add_edge("answer", "check")
workflow.add_conditional_edges("check", should_retry)

graph = workflow.compile()

# 5. Run
result = graph.invoke({
    "question": "Explain quantum entanglement simply.",
    "iteration": 0
})

print(f"Answer: {result['answer']}")
print(f"Quality: {result['quality_score']}")
print(f"Iterations: {result['iteration']}")
# ✅ Agent retries until quality threshold is met!
```

---

## LangGraph in mahsm

mahsm enhances LangGraph with:

### 1. **Simplified Node Creation**

```python
# Without mahsm
def my_node(state):
    # Manual state extraction
    question = state["question"]
    # Call LLM
    response = llm.complete(question)
    # Manual state update
    return {"answer": response}

# With mahsm
@ma.dspy_node
class MyNode(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)
# ✅ State extraction/merging handled automatically
```

### 2. **Automatic Tracing**

```python
ma.tracing.init()  # One line
# ✅ All LangGraph nodes traced to Langfuse
# ✅ All DSPy calls traced
# ✅ Custom functions with @observe traced
```

### 3. **Type-Safe State**

```python
class MyState(TypedDict):
    question: str
    answer: str

# ✅ IDE autocomplete
# ✅ Static type checking
# ✅ Runtime validation
```

---

## When to Use LangGraph

### ✅ Great For:
- **Multi-step agents** that need memory
- **Cyclical workflows** (retry, refine, iterate)
- **Conditional branching** based on outputs
- **Complex orchestration** of multiple agents
- **Human-in-the-loop** systems

### ❌ Not Ideal For:
- **Simple one-shot completions** (use DSPy directly)
- **Purely stateless operations** (no need for state management)
- **Real-time streaming** (LangGraph is batch-oriented)

---

## Common Patterns

### 1. **Linear Pipeline**

```python
START → agent1 → agent2 → agent3 → END
```

```python
workflow.add_edge(ma.START, "agent1")
workflow.add_edge("agent1", "agent2")
workflow.add_edge("agent2", "agent3")
workflow.add_edge("agent3", ma.END)
```

### 2. **Conditional Branching**

```python
                  ┌─────────┐
      START ─────►│ router  │
                  └────┬────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
         ┌────────┐       ┌────────┐
         │ path_a │       │ path_b │
         └───┬────┘       └───┬────┘
             └────────┬────────┘
                      ▼
                    END
```

```python
workflow.add_conditional_edges(
    "router",
    lambda state: "path_a" if condition(state) else "path_b"
)
```

### 3. **Retry Loop**

```python
        ┌──────────────┐
        │              │
        ▼              │
    ┌───────┐    ┌─────┴─────┐
    │  try  │───►│   check   │
    └───────┘    └───────────┘
                      │
                      ▼
                    END (if success)
```

```python
workflow.add_conditional_edges(
    "check",
    lambda state: "try" if not success(state) else ma.END
)
```

### 4. **Multi-Agent Collaboration**

```python
        ┌───────────┐
    ┌──►│ researcher│─────┐
    │   └───────────┘     │
    │                     ▼
    │   ┌───────────┐   ┌──────────┐
    └───│coordinator│◄──│synthesizer│
        └───────────┘   └──────────┘
```

---

## Best Practices

### ✅ Do:

1. **Use TypedDict for state**
   ```python
   class State(TypedDict):
       field: str
   ```

2. **Keep nodes focused**
   ```python
   # ✅ Single responsibility
   @ma.dspy_node
   class QueryGenerator(ma.Module):
       # Only generates queries
       pass
   ```

3. **Handle missing state gracefully**
   ```python
   def my_router(state):
       value = state.get("key", default_value)
       # ...
   ```

4. **Use conditional edges for routing**
   ```python
   workflow.add_conditional_edges("checker", route_function)
   ```

### ❌ Don't:

1. **Mutate state directly**
   ```python
   # ❌ Don't do this
   def node(state):
       state["key"] = "value"  # Mutates input!
       return state
   
   # ✅ Do this
   def node(state):
       return {"key": "value"}  # Returns update
   ```

2. **Create infinite loops without exit conditions**
   ```python
   # ❌ No way to exit
   workflow.add_conditional_edges("node", lambda s: "node")
   
   # ✅ Add exit condition
   def router(state):
       if state["count"] > 10:
           return ma.END
       return "node"
   ```

3. **Over-complicate the graph**
   ```python
   # ❌ Too many branches
   # Keep it simple and readable
   ```

---

## Next Steps

- **[State Management](state.md)** → Learn about TypedDict and state updates
- **[Nodes & Edges](nodes-edges.md)** → Build your graph components
- **[Conditional Routing](conditional-routing.md)** → Add branching logic
- **[Compilation & Execution](compilation.md)** → Run your workflows
- **[Visualization](visualization.md)** → Visualize your graphs

---

## External Resources

- **[Official LangGraph Docs](https://langchain-ai.github.io/langgraph/)** - Comprehensive guide
- **[LangGraph GitHub](https://github.com/langchain-ai/langgraph)** - Source code and examples
- **[LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)** - Step-by-step guides

---

**Ready to dive deeper? Start with [State Management →](state.md)**
