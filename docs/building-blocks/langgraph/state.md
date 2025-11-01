# LangGraph State Management

> **TL;DR**: State in LangGraph is a TypedDict that flows through your workflow, carrying data between nodes.

## What is State?

In LangGraph, **state** is a dictionary that:
- Flows through your workflow
- Is read by nodes
- Is updated by nodes
- Maintains type safety with `TypedDict`

Think of it as **shared memory** for your agent workflow.

---

## Defining State

Use Python's `TypedDict` to define your state schema:

```python
from typing import TypedDict, Optional

class MyState(TypedDict):
    question: str
    answer: Optional[str]
    confidence: Optional[float]
```

**Benefits:**
- ✅ IDE autocomplete
- ✅ Type checking
- ✅ Clear documentation
- ✅ Runtime validation

---

## State Flow

State flows through nodes in your workflow:

```python
import mahsm as ma
from typing import TypedDict

class State(TypedDict):
    input: str
    intermediate: str
    output: str

# Node 1: Reads 'input', writes 'intermediate'
def node1(state: State) -> dict:
    result = process(state["input"])
    return {"intermediate": result}

# Node 2: Reads 'intermediate', writes 'output'
def node2(state: State) -> dict:
    result = finalize(state["intermediate"])
    return {"output": result}

# Build workflow
workflow = ma.graph.StateGraph(State)
workflow.add_node("node1", node1)
workflow.add_node("node2", node2)
workflow.add_edge(ma.START, "node1")
workflow.add_edge("node1", "node2")
workflow.add_edge("node2", ma.END)
graph = workflow.compile()

# Run
result = graph.invoke({"input": "Hello"})
# State flows: {"input": "Hello"} → {"input": "Hello", "intermediate": "..."} → {"input": "Hello", "intermediate": "...", "output": "..."}
print(result["output"])
```

---

## State Updates

### Immutable Updates

Nodes return **updates**, not full state:

```python
# ❌ DON'T: Mutate state directly
def bad_node(state):
    state["answer"] = "Paris"  # Mutates input!
    return state

# ✅ DO: Return updates
def good_node(state):
    return {"answer": "Paris"}  # Returns update
```

LangGraph merges your update into the state automatically:

```python
# Before node
state = {"question": "What is the capital of France?"}

# Node returns
update = {"answer": "Paris"}

# After node (automatic merge)
state = {
    "question": "What is the capital of France?",
    "answer": "Paris"
}
```

---

## Optional vs Required Fields

Use `Optional` for fields that may not exist initially:

```python
from typing import TypedDict, Optional

class State(TypedDict):
    # Required fields (must be in initial input)
    question: str
    
    # Optional fields (nodes will populate)
    answer: Optional[str]
    reasoning: Optional[str]
    confidence: Optional[float]
```

---

## Complex State Types

### Lists

```python
from typing import List

class State(TypedDict):
    messages: List[str]
    findings: List[dict]
```

**Appending to lists:**

```python
def add_message(state: State) -> dict:
    # Option 1: Replace entire list
    new_messages = state["messages"] + ["New message"]
    return {"messages": new_messages}
    
    # Option 2: Use Annotated for automatic appending (advanced)
    # See LangGraph docs for details
```

### Nested Dicts

```python
class State(TypedDict):
    user: dict  # {"name": str, "email": str}
    config: dict
```

### Custom Classes

```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str

class State(TypedDict):
    user: User
    active: bool
```

---

## State in mahsm

### With @dspy_node

`@dspy_node` automatically extracts and updates state:

```python
import mahsm as ma
from typing import TypedDict, Optional

class QAState(TypedDict):
    question: str
    reasoning: Optional[str]
    answer: Optional[str]

@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> reasoning, answer")
    
    def forward(self, question):
        return self.qa(question=question)

# In workflow
workflow = ma.graph.StateGraph(QAState)
workflow.add_node("qa", QA())

# When QA runs:
# 1. Extracts 'question' from state
# 2. Calls forward(question=state["question"])
# 3. Merges {"reasoning": "...", "answer": "..."} into state
```

### With Regular Functions

```python
def my_node(state: QAState) -> dict:
    """Regular function node."""
    question = state["question"]
    # Process...
    return {"answer": "Paris"}

workflow.add_node("my_node", my_node)
```

---

## State Initialization

### Basic Initialization

```python
# Invoke with initial state
result = graph.invoke({
    "question": "What is DSPy?",
    "confidence": 0.0
})
```

### From User Input

```python
def create_initial_state(user_input: str) -> dict:
    """Create initial state from user input."""
    return {
        "question": user_input,
        "iteration": 0,
        "history": []
    }

state = create_initial_state("What is LangGraph?")
result = graph.invoke(state)
```

---

## State Persistence

State is immutable within a single execution:

```python
# Single execution
result = graph.invoke({"question": "Hello"})
# State flows through workflow and is returned

# New execution (fresh state)
result2 = graph.invoke({"question": "Goodbye"})
# Independent from result
```

For persistent state across executions, use LangGraph's checkpointing (advanced):

```python
from langgraph.checkpoint.memory import MemorySaver

# Compile with memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Run with thread ID
config = {"configurable": {"thread_id": "user_123"}}
result1 = graph.invoke({"question": "Hello"}, config=config)
result2 = graph.invoke({"question": "Continue..."}, config=config)
# result2 has access to result1's state!
```

---

## Best Practices

### ✅ Do:

1. **Use TypedDict**
   ```python
   # ✅ Type-safe
   class State(TypedDict):
       field: str
   ```

2. **Make node outputs Optional**
   ```python
   # ✅ Clear that these are populated later
   answer: Optional[str]
   ```

3. **Return only updates**
   ```python
   # ✅ Clean updates
   def node(state):
       return {"answer": "Paris"}
   ```

4. **Use descriptive field names**
   ```python
   # ✅ Clear purpose
   user_question: str
   generated_answer: str
   quality_score: float
   ```

5. **Match DSPy signatures to state keys**
   ```python
   class State(TypedDict):
       question: str
       answer: str
   
   # ✅ Signature matches
   dspy.ChainOfThought("question -> answer")
   ```

### ❌ Don't:

1. **Mutate state directly**
   ```python
   # ❌ Never do this
   def node(state):
       state["answer"] = "Paris"
       return state
   ```

2. **Use vague names**
   ```python
   # ❌ What is this?
   result: str
   data: dict
   ```

3. **Make everything required**
   ```python
   # ❌ Will error if not in initial input
   class State(TypedDict):
       question: str
       answer: str  # Should be Optional[str]
   ```

4. **Return full state unnecessarily**
   ```python
   # ❌ Redundant
   def node(state):
       return {**state, "answer": "Paris"}
   
   # ✅ Just return update
   def node(state):
       return {"answer": "Paris"}
   ```

---

## Debugging State

### Print State in Nodes

```python
def debug_node(state):
    print(f"Current state: {state}")
    return {}

workflow.add_node("debug", debug_node)
```

### Trace State Flow

```python
# Run with verbose output
result = graph.invoke({"question": "Hello"})
print(f"Final state: {result}")
```

### Use Langfuse

With `ma.tracing.init()`, state is automatically logged:

```python
ma.tracing.init()
result = graph.invoke({"question": "Hello"})
# Check Langfuse UI to see state at each node!
```

---

## Advanced: State Reducers

For complex state updates (like appending to lists), use reducers:

```python
from typing import Annotated
from langgraph.graph import add

class State(TypedDict):
    # Normal field
    question: str
    
    # Auto-appending list (uses 'add' reducer)
    messages: Annotated[List[str], add]

# Now nodes can just return new messages
def add_message(state):
    return {"messages": ["New message"]}
# LangGraph automatically appends to existing messages!
```

---

## State Schema Evolution

As your workflow grows, extend your state:

```python
# v1
class StateV1(TypedDict):
    question: str
    answer: Optional[str]

# v2 (add new fields)
class StateV2(TypedDict):
    question: str
    answer: Optional[str]
    confidence: Optional[float]  # New field
    sources: Optional[List[str]]  # New field
```

Existing nodes continue working (they ignore new fields).

---

## Example: Multi-Step Research State

```python
from typing import TypedDict, Optional, List

class ResearchState(TypedDict):
    # Input
    question: str
    
    # Intermediate
    search_queries: Optional[List[str]]
    raw_findings: Optional[List[dict]]
    
    # Output
    synthesized_answer: Optional[str]
    sources: Optional[List[str]]
    confidence: Optional[float]
    
    # Metadata
    iteration: int
    total_tokens: int

# Use in workflow
@ma.dspy_node
class QueryGenerator(ma.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict("question -> search_queries: list of queries")
    
    def forward(self, question):
        return self.gen(question=question)

# Workflow automatically manages all state fields!
```

---

## Next Steps

- **[Nodes & Edges](nodes-edges.md)** → Learn how nodes interact with state
- **[Conditional Routing](conditional-routing.md)** → Route based on state
- **[Your First Agent](../../guides/first-agent.md)** → Build a stateful agent

---

## External Resources

- **[LangGraph State Docs](https://langchain-ai.github.io/langgraph/concepts/#state)** - Official guide
- **[TypedDict Documentation](https://docs.python.org/3/library/typing.html#typing.TypedDict)** - Python docs

---

**Next: Learn about [Nodes & Edges →](nodes-edges.md)**
