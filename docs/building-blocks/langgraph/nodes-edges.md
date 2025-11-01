# LangGraph Nodes & Edges

> **TL;DR**: Nodes are functions or agents that process state; edges define how state flows between them.

## What are Nodes?

**Nodes** are the processing units in your workflow. Each node:
- Receives the current state
- Performs some operation (LLM call, tool use, logic)
- Returns an update to merge into state

Think of nodes as **steps in your agent's workflow**.

---

## Creating Nodes

### Function Nodes

The simplest type of node is a regular Python function:

```python
import mahsm as ma
from typing import TypedDict, Optional

class State(TypedDict):
    question: str
    answer: Optional[str]

def answer_question(state: State) -> dict:
    """A simple function node."""
    question = state["question"]
    answer = f"The answer to '{question}' is..."
    return {"answer": answer}

# Add to workflow
workflow = ma.graph.StateGraph(State)
workflow.add_node("answerer", answer_question)
```

**Key points:**
- Takes `state` as input
- Returns a dict with updates
- Updates are automatically merged into state

### DSPy Nodes

Use `@ma.dspy_node` for DSPy modules:

```python
@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

workflow.add_node("qa", QA())
```

**Advantages:**
- Automatic state extraction
- Signature-based field mapping
- Built-in Langfuse tracing
- Clean, declarative code

### Class-Based Nodes

Create reusable node classes:

```python
class CustomNode:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, state: State) -> dict:
        """Make the class callable."""
        # Your logic here
        return {"answer": "Result"}

node = CustomNode(config={"param": "value"})
workflow.add_node("custom", node)
```

---

## Node Patterns

### 1. **Data Transformation**

Transform state data:

```python
def clean_input(state):
    """Clean and validate input."""
    question = state["question"].strip().lower()
    return {"question": question}
```

### 2. **LLM Calls**

Call language models:

```python
@ma.dspy_node
class Summarizer(ma.Module):
    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought("text -> summary")
    
    def forward(self, text):
        return self.summarizer(text=text)
```

### 3. **Tool Use**

Execute external tools:

```python
def search_web(state):
    """Search the web for information."""
    query = state["search_query"]
    results = search_api(query)  # Your API call
    return {"search_results": results}
```

### 4. **Conditional Logic**

Add branching logic:

```python
def classifier(state):
    """Classify input type."""
    question = state["question"]
    
    if "calculate" in question.lower():
        return {"category": "math"}
    elif "search" in question.lower():
        return {"category": "research"}
    else:
        return {"category": "general"}
```

### 5. **State Aggregation**

Combine multiple state fields:

```python
def aggregate_results(state):
    """Combine results from multiple sources."""
    findings = state.get("findings", [])
    sources = state.get("sources", [])
    
    combined = {
        "final_answer": "\n".join(findings),
        "total_sources": len(sources)
    }
    return combined
```

---

## What are Edges?

**Edges** define how state flows between nodes. There are three types:

1. **Normal edges**: Direct connections
2. **Conditional edges**: Branch based on state
3. **Special edges**: START and END

---

## Normal Edges

Connect nodes directly:

```python
workflow.add_edge("node_a", "node_b")
# State flows: node_a → node_b
```

### Example: Linear Pipeline

```python
workflow.add_edge(ma.START, "step1")
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", "step3")
workflow.add_edge("step3", ma.END)

# Flow: START → step1 → step2 → step3 → END
```

---

## Conditional Edges

Branch based on state:

```python
def router(state):
    """Decide which node to go to next."""
    if state["category"] == "math":
        return "calculator"
    elif state["category"] == "research":
        return "researcher"
    else:
        return "general_qa"

workflow.add_conditional_edges("classifier", router)
```

### Example: Retry Logic

```python
def should_retry(state):
    """Retry if quality is low."""
    if state.get("iteration", 0) >= 3:
        return ma.END  # Max retries reached
    
    quality = state.get("quality_score", 0)
    if quality < 0.7:
        return "generate_answer"  # Retry
    return ma.END  # Success

workflow.add_conditional_edges("quality_check", should_retry)
```

### Example: Multi-Path Routing

```python
def route_by_complexity(state):
    """Route based on question complexity."""
    question = state["question"]
    
    if len(question.split()) < 5:
        return "simple_qa"
    elif "research" in question.lower():
        return "research_agent"
    else:
        return "advanced_qa"

workflow.add_conditional_edges("router", route_by_complexity)
```

---

## Special Edges

### START

Entry point of the workflow:

```python
workflow.add_edge(ma.START, "first_node")
```

### END

Exit point of the workflow:

```python
workflow.add_edge("last_node", ma.END)

# Or conditional
def maybe_end(state):
    if state["done"]:
        return ma.END
    return "continue"

workflow.add_conditional_edges("checker", maybe_end)
```

---

## Complete Example

Here's a full workflow with multiple node types and edges:

```python
import mahsm as ma
from typing import TypedDict, Optional
import dspy

# Configure
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
ma.tracing.init()

# State
class QAState(TypedDict):
    question: str
    category: Optional[str]
    search_results: Optional[str]
    answer: Optional[str]
    quality_score: Optional[float]
    iteration: int

# Nodes
def classifier(state: QAState) -> dict:
    """Classify question type."""
    question = state["question"]
    if "search" in question.lower():
        return {"category": "research"}
    return {"category": "direct"}

@ma.dspy_node
class Researcher(ma.Module):
    def __init__(self):
        super().__init__()
        self.research = dspy.ChainOfThought("question -> search_results")
    
    def forward(self, question):
        return self.research(question=question)

@ma.dspy_node
class DirectQA(ma.Module):
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

def increment_iteration(state: QAState) -> dict:
    return {"iteration": state.get("iteration", 0) + 1}

# Routing functions
def route_by_category(state: QAState):
    """Route based on classification."""
    if state["category"] == "research":
        return "researcher"
    return "direct_qa"

def should_retry(state: QAState):
    """Check if we should retry."""
    if state.get("iteration", 0) >= 2:
        return ma.END
    
    quality = float(state.get("quality_score", 0))
    if quality < 0.7:
        return "increment"  # Retry
    return ma.END

# Build workflow
workflow = ma.graph.StateGraph(QAState)

# Add nodes
workflow.add_node("classifier", classifier)
workflow.add_node("researcher", Researcher())
workflow.add_node("direct_qa", DirectQA())
workflow.add_node("quality_check", QualityChecker())
workflow.add_node("increment", increment_iteration)

# Add edges
workflow.add_edge(ma.START, "classifier")
workflow.add_conditional_edges("classifier", route_by_category)
workflow.add_edge("researcher", "quality_check")
workflow.add_edge("direct_qa", "quality_check")
workflow.add_conditional_edges("quality_check", should_retry)
workflow.add_edge("increment", "classifier")  # Loop back

# Compile
graph = workflow.compile()

# Run
result = graph.invoke({
    "question": "Search for information about LangGraph",
    "iteration": 0
})
print(f"Answer: {result['answer']}")
print(f"Quality: {result['quality_score']}")
print(f"Iterations: {result['iteration']}")
```

---

## Best Practices

### ✅ Do:

1. **Keep nodes focused**
   ```python
   # ✅ Single responsibility
   def extract_entities(state):
       # Only extracts entities
       pass
   
   def classify_entities(state):
       # Only classifies
       pass
   ```

2. **Return only updates**
   ```python
   # ✅ Clean update
   def node(state):
       return {"answer": "Paris"}
   
   # ❌ Redundant
   def node(state):
       return {**state, "answer": "Paris"}
   ```

3. **Use descriptive node names**
   ```python
   # ✅ Clear purpose
   workflow.add_node("extract_entities", extract_entities)
   workflow.add_node("classify_sentiment", classify_sentiment)
   
   # ❌ Vague
   workflow.add_node("node1", func1)
   ```

4. **Handle missing state gracefully**
   ```python
   # ✅ Safe access
   def node(state):
       value = state.get("field", default_value)
       return {"result": process(value)}
   ```

5. **Use conditional edges for branching**
   ```python
   # ✅ Explicit routing
   def router(state):
       if condition:
           return "path_a"
       return "path_b"
   
   workflow.add_conditional_edges("router", router)
   ```

### ❌ Don't:

1. **Mutate state directly**
   ```python
   # ❌ Never do this
   def node(state):
       state["answer"] = "Paris"
       return state
   ```

2. **Create side effects**
   ```python
   # ❌ Side effects make debugging hard
   global_var = None
   
   def node(state):
       global global_var
       global_var = state["value"]  # Bad!
       return {}
   ```

3. **Use long, complex nodes**
   ```python
   # ❌ Too much in one node
   def mega_node(state):
       # 100 lines of code...
       pass
   
   # ✅ Split into smaller nodes
   def step1(state): pass
   def step2(state): pass
   def step3(state): pass
   ```

---

## Debugging Nodes

### Print State

```python
def debug_node(state):
    print(f"State at debug point: {state}")
    return {}

workflow.add_node("debug", debug_node)
workflow.add_edge("some_node", "debug")
workflow.add_edge("debug", "next_node")
```

### Use Langfuse Tracing

```python
ma.tracing.init()

# All nodes are automatically traced!
# Check Langfuse UI to see:
# - State at each node
# - Node execution times
# - LLM calls
```

### Test Nodes Independently

```python
import unittest

class TestNodes(unittest.TestCase):
    def test_classifier(self):
        state = {"question": "Search for cats"}
        result = classifier(state)
        self.assertEqual(result["category"], "research")
```

---

## Performance Tips

### 1. **Parallelize Independent Nodes**

LangGraph can run independent nodes in parallel (advanced feature):

```python
# These can run in parallel
workflow.add_node("fetch_data_a", fetch_a)
workflow.add_node("fetch_data_b", fetch_b)
workflow.add_edge(ma.START, "fetch_data_a")
workflow.add_edge(ma.START, "fetch_data_b")
```

### 2. **Minimize State Size**

Only include necessary fields:

```python
# ✅ Minimal state
class State(TypedDict):
    question: str
    answer: str

# ❌ Too much
class State(TypedDict):
    question: str
    answer: str
    intermediate_result_1: str
    intermediate_result_2: str
    # ... many more fields
```

### 3. **Cache Expensive Operations**

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_lookup(key: str):
    # Expensive operation
    return result

def node(state):
    result = expensive_lookup(state["key"])
    return {"result": result}
```

---

## Next Steps

- **[Conditional Routing](conditional-routing.md)** → Advanced routing patterns
- **[Compilation & Execution](compilation.md)** → Run your workflows
- **[Visualization](visualization.md)** → Visualize your graphs
- **[Your First Agent](../../guides/first-agent.md)** → Build a complete agent

---

## External Resources

- **[LangGraph Nodes Docs](https://langchain-ai.github.io/langgraph/concepts/#nodes)** - Official guide
- **[LangGraph Edges Docs](https://langchain-ai.github.io/langgraph/concepts/#edges)** - Official guide

---

**Next: Master [Conditional Routing →](conditional-routing.md)**
