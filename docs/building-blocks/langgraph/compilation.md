# LangGraph Compilation & Execution

> **TL;DR**: Compile your workflow into an executable graph, then run it with `invoke()` or `stream()`.

## Workflow Lifecycle

Building and running a LangGraph workflow has three stages:

1. **Build**: Create workflow with nodes and edges
2. **Compile**: Convert to executable graph
3. **Execute**: Run with input state

```python
import mahsm as ma

# 1. Build
workflow = ma.graph.StateGraph(State)
workflow.add_node("step1", step1_func)
workflow.add_edge(ma.START, "step1")
workflow.add_edge("step1", ma.END)

# 2. Compile
graph = workflow.compile()

# 3. Execute
result = graph.invoke({"input": "Hello"})
```

---

## Compilation

### Basic Compilation

Call `compile()` on your workflow:

```python
graph = workflow.compile()
```

**What happens during compilation:**
- Validates workflow structure
- Checks for unreachable nodes
- Optimizes execution order
- Creates execution engine

### Compilation Errors

Common issues:

```python
# ❌ Missing START edge
workflow = ma.graph.StateGraph(State)
workflow.add_node("node1", func1)
# ERROR: No entry point!

# ✅ Fixed
workflow.add_edge(ma.START, "node1")

# ❌ Unreachable node
workflow.add_edge(ma.START, "node1")
workflow.add_edge("node1", ma.END)
workflow.add_node("orphan", orphan_func)  # Never reached!
# WARNING: Node "orphan" is unreachable

# ❌ No exit
workflow.add_edge(ma.START, "loop1")
workflow.add_edge("loop1", "loop2")
workflow.add_edge("loop2", "loop1")  # Infinite loop!
# ERROR: No path to END
```

### Best Practices

✅ **Compile once, reuse many times**

```python
# ✅ Good: Compile once
graph = workflow.compile()
for input in inputs:
    result = graph.invoke(input)

# ❌ Bad: Recompiling every time
for input in inputs:
    graph = workflow.compile()  # Wasteful!
    result = graph.invoke(input)
```

---

## Execution Methods

### invoke() - Synchronous Execution

Run workflow and get final state:

```python
result = graph.invoke({"question": "What is Python?"})
print(result)  # Final state
```

**Characteristics:**
- Blocks until complete
- Returns final state only
- Simple and straightforward

### stream() - Streaming Execution

Stream state updates as they happen:

```python
for state_update in graph.stream({"question": "What is Python?"}):
    print(f"Update: {state_update}")
```

**Characteristics:**
- Yields state after each node
- Great for progress tracking
- See intermediate results

### Example: Stream vs Invoke

```python
import mahsm as ma
from typing import TypedDict

class State(TypedDict):
    count: int

def increment(state):
    print(f"  Node: Incrementing {state['count']}")
    return {"count": state["count"] + 1}

workflow = ma.graph.StateGraph(State)
workflow.add_node("inc1", increment)
workflow.add_node("inc2", increment)
workflow.add_node("inc3", increment)
workflow.add_edge(ma.START, "inc1")
workflow.add_edge("inc1", "inc2")
workflow.add_edge("inc2", "inc3")
workflow.add_edge("inc3", ma.END)

graph = workflow.compile()

print("=== With invoke() ===")
result = graph.invoke({"count": 0})
print(f"Final result: {result}")

print("\n=== With stream() ===")
for state in graph.stream({"count": 0}):
    print(f"Intermediate: {state}")
```

Output:
```
=== With invoke() ===
  Node: Incrementing 0
  Node: Incrementing 1
  Node: Incrementing 2
Final result: {'count': 3}

=== With stream() ===
  Node: Incrementing 0
Intermediate: {'count': 1}
  Node: Incrementing 1
Intermediate: {'count': 2}
  Node: Incrementing 2
Intermediate: {'count': 3}
```

---

## Input & Output

### Input Format

Pass initial state as a dict:

```python
result = graph.invoke({
    "question": "What is Python?",
    "context": "Programming",
    "max_length": 100
})
```

**Rules:**
- Must be a dictionary
- Keys must match state TypedDict fields
- Optional fields can be omitted

### Output Format

Returns final state as a dict:

```python
result = graph.invoke({"input": "Hello"})
# result = {"input": "Hello", "output": "World", ...}

# Access fields
print(result["output"])
```

### Partial Inputs

You don't need to provide all state fields:

```python
class State(TypedDict):
    question: str
    answer: str
    confidence: float
    metadata: dict

# Only provide required fields
result = graph.invoke({"question": "What is AI?"})
# Other fields are added by nodes
```

---

## Execution Control

### Setting Configuration

Configure execution behavior:

```python
from langgraph.checkpoint.memory import MemorySaver

# Add checkpointing
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Run with config
result = graph.invoke(
    {"question": "Hello"},
    config={"configurable": {"thread_id": "conversation-1"}}
)
```

### Interrupting Execution

For long-running workflows:

```python
import asyncio

# Async version
async def run_with_timeout():
    try:
        result = await asyncio.wait_for(
            graph.ainvoke({"input": "data"}),
            timeout=30.0
        )
        return result
    except asyncio.TimeoutError:
        print("Workflow timed out!")
        return None
```

---

## Complete Example

```python
import mahsm as ma
from typing import TypedDict, Optional
import dspy
import os

# Configure
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
ma.tracing.init()

# State
class PipelineState(TypedDict):
    question: str
    category: Optional[str]
    answer: Optional[str]
    sources: Optional[list]

# Nodes
def categorize(state: PipelineState) -> dict:
    question = state["question"]
    if "code" in question.lower():
        return {"category": "programming"}
    elif "math" in question.lower():
        return {"category": "mathematics"}
    else:
        return {"category": "general"}

@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question, category -> answer")
    
    def forward(self, question, category):
        return self.qa(question=question, category=category)

def add_sources(state: PipelineState) -> dict:
    # Mock sources
    return {"sources": ["Source 1", "Source 2"]}

# Router
def route_by_category(state: PipelineState):
    """Skip sources for simple questions."""
    if state["category"] == "general":
        return ma.END
    return "add_sources"

# Build workflow
workflow = ma.graph.StateGraph(PipelineState)
workflow.add_node("categorize", categorize)
workflow.add_node("qa", QA())
workflow.add_node("add_sources", add_sources)

workflow.add_edge(ma.START, "categorize")
workflow.add_edge("categorize", "qa")
workflow.add_conditional_edges("qa", route_by_category)
workflow.add_edge("add_sources", ma.END)

# Compile once
graph = workflow.compile()

# Execute multiple times
print("=== Example 1: General Question ===")
result1 = graph.invoke({"question": "What is Python?"})
print(f"Category: {result1['category']}")
print(f"Answer: {result1['answer']}")
print(f"Sources: {result1.get('sources', 'None')}")

print("\n=== Example 2: Code Question ===")
result2 = graph.invoke({"question": "How do I write a Python loop?"})
print(f"Category: {result2['category']}")
print(f"Answer: {result2['answer']}")
print(f"Sources: {result2.get('sources', 'None')}")

print("\n=== Example 3: Streaming ===")
for state in graph.stream({"question": "What is 2+2?"}):
    print(f"State update: category={state.get('category')}, "
          f"has_answer={bool(state.get('answer'))}")
```

---

## Error Handling

### Handling Node Errors

Wrap nodes in try/except:

```python
def safe_node(state):
    try:
        result = risky_operation(state)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}

# Route based on error
def error_router(state):
    if state.get("error"):
        return "error_handler"
    return "next_step"
```

### Validation

Validate state before execution:

```python
def validate_input(state):
    """Validate input before processing."""
    if not state.get("question"):
        raise ValueError("Question is required")
    
    if len(state["question"]) > 1000:
        raise ValueError("Question too long")
    
    return {}

# Add as first node
workflow.add_edge(ma.START, "validate")
workflow.add_node("validate", validate_input)
workflow.add_edge("validate", "main_flow")
```

---

## Performance Optimization

### Reuse Compiled Graphs

```python
# ✅ Good: Global compiled graph
GRAPH = workflow.compile()

def process_request(input_data):
    return GRAPH.invoke(input_data)

# ❌ Bad: Recompile each time
def process_request(input_data):
    graph = workflow.compile()  # Slow!
    return graph.invoke(input_data)
```

### Batch Processing

Process multiple inputs efficiently:

```python
inputs = [
    {"question": "Q1"},
    {"question": "Q2"},
    {"question": "Q3"}
]

# Sequential
results = [graph.invoke(inp) for inp in inputs]

# Parallel (if graph is thread-safe)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(graph.invoke, inputs))
```

### Caching

Cache compiled graphs:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_graph():
    workflow = build_workflow()  # Your build logic
    return workflow.compile()

# Always returns same compiled graph
graph = get_graph()
```

---

## Debugging Execution

### Print State at Each Node

```python
def debug_node(state):
    import json
    print(f"=== State at debug point ===")
    print(json.dumps(state, indent=2))
    return {}

workflow.add_node("debug", debug_node)
workflow.add_edge("some_node", "debug")
workflow.add_edge("debug", "next_node")
```

### Use Langfuse Tracing

```python
ma.tracing.init()

# All execution is automatically traced!
result = graph.invoke({"input": "data"})

# View in Langfuse UI:
# - Full execution trace
# - State at each node
# - LLM calls
# - Timing information
```

### Stream for Visibility

```python
for state in graph.stream({"input": "data"}):
    print(f"After node: {state}")
    if "error" in state:
        print(f"ERROR: {state['error']}")
        break
```

---

## Testing Workflows

### Unit Testing Nodes

Test individual nodes:

```python
import unittest

class TestNodes(unittest.TestCase):
    def test_categorize(self):
        state = {"question": "What is code?"}
        result = categorize(state)
        self.assertEqual(result["category"], "programming")
```

### Integration Testing

Test full workflow:

```python
def test_workflow():
    graph = workflow.compile()
    
    result = graph.invoke({"question": "Test"})
    
    assert "answer" in result
    assert result["answer"] is not None
```

### Property Testing

Test workflow properties:

```python
def test_workflow_always_produces_answer():
    graph = workflow.compile()
    
    test_questions = [
        "What is AI?",
        "How does Python work?",
        "Explain quantum physics"
    ]
    
    for question in test_questions:
        result = graph.invoke({"question": question})
        assert result["answer"], f"No answer for: {question}"
```

---

## Async Execution

For async operations:

```python
import asyncio

# Use async nodes
async def async_node(state):
    result = await async_api_call(state["input"])
    return {"result": result}

# Use ainvoke
result = await graph.ainvoke({"input": "data"})

# Use astream
async for state in graph.astream({"input": "data"}):
    print(state)
```

---

## Best Practices Summary

### ✅ Do:

1. **Compile once, execute many**
2. **Use streaming for long workflows**
3. **Handle errors gracefully**
4. **Validate inputs**
5. **Cache compiled graphs**
6. **Use tracing for debugging**

### ❌ Don't:

1. **Recompile unnecessarily**
2. **Ignore errors**
3. **Mutate state directly**
4. **Block without timeout**
5. **Skip input validation**

---

## Next Steps

- **[Visualization](visualization.md)** → Visualize your workflows
- **[LangGraph Overview](overview.md)** → Review core concepts
- **[Best Practices](../dspy/best-practices.md)** → Production patterns

---

## External Resources

- **[LangGraph Execution](https://langchain-ai.github.io/langgraph/how-tos/)** - Official how-tos
- **[LangGraph API](https://langchain-ai.github.io/langgraph/reference/)** - API reference

---

**Next: Visualize workflows with [Visualization →](visualization.md)**
