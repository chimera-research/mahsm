# LangGraph Conditional Routing

> **TL;DR**: Conditional routing dynamically chooses the next node based on state—enabling branching, loops, and adaptive workflows.

## What is Conditional Routing?

**Conditional routing** allows your workflow to make decisions based on the current state. Instead of a fixed path, your agent can:

- **Branch**: Choose between multiple paths
- **Loop**: Repeat steps until a condition is met
- **Exit**: End early based on criteria
- **Adapt**: Change behavior dynamically

Think of it as **if/else logic for workflows**.

---

## Basic Conditional Edges

Use `add_conditional_edges()` to route based on state:

```python
def router_function(state):
    """Decide which node to go to next."""
    if state["condition"]:
        return "node_a"
    else:
        return "node_b"

workflow.add_conditional_edges("source_node", router_function)
```

**How it works:**
1. `source_node` executes and updates state
2. `router_function` receives the updated state
3. Returns the name of the next node to execute
4. LangGraph routes to that node

---

## Routing Patterns

### 1. **Binary Branch**

Choose between two paths:

```python
def binary_router(state):
    """Simple yes/no routing."""
    if state["needs_research"]:
        return "research_agent"
    else:
        return "direct_answer"

workflow.add_conditional_edges("classifier", binary_router)
```

### 2. **Multi-Way Branch**

Choose from multiple paths:

```python
def multi_router(state):
    """Route to different specialists."""
    category = state["category"]
    
    if category == "math":
        return "calculator"
    elif category == "research":
        return "researcher"
    elif category == "code":
        return "code_generator"
    else:
        return "general_qa"

workflow.add_conditional_edges("categorizer", multi_router)
```

### 3. **Retry Loop**

Loop until success:

```python
def retry_logic(state):
    """Retry if not good enough."""
    # Check if we've tried too many times
    if state.get("attempts", 0) >= 3:
        return ma.END  # Give up
    
    # Check quality
    if state.get("quality_score", 0) < 0.8:
        return "generate"  # Try again
    
    return ma.END  # Success!

workflow.add_conditional_edges("quality_check", retry_logic)
```

### 4. **Early Exit**

Skip remaining steps if done:

```python
def early_exit(state):
    """Exit early if we have an answer."""
    if state.get("answer") and state.get("confidence", 0) > 0.9:
        return ma.END  # Skip remaining steps
    
    return "next_step"  # Continue processing

workflow.add_conditional_edges("initial_check", early_exit)
```

### 5. **Dynamic Path Selection**

Choose path based on multiple factors:

```python
def smart_router(state):
    \"\"\"Route based on complexity and urgency.\"\"\"
    complexity = len(state["question"].split())
    urgent = state.get("urgent", False)
    
    if urgent and complexity < 10:
        return "fast_qa"  # Quick answer
    elif complexity > 50:
        return "detailed_research"  # Deep dive
    elif state.get("has_code"):
        return "code_analyzer"  # Code-specific
    else:
        return "standard_qa"  # Normal processing

workflow.add_conditional_edges("triage", smart_router)
```

---

## Advanced Patterns

### Conditional Looping with State Tracking

Loop with iteration counter:

```python
from typing import TypedDict, Optional

class State(TypedDict):
    question: str
    answer: Optional[str]
    iteration: int
    max_iterations: int

def increment_counter(state):
    \"\"\"Increment iteration count.\"\"\"
    return {"iteration": state.get("iteration", 0) + 1}

def should_continue(state):
    \"\"\"Continue if not done and under max iterations.\"\"\"
    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        return ma.END
    
    if state.get("answer_quality", 0) >= 0.8:
        return ma.END
    
    return "generate_answer"

# Build loop
workflow.add_node("increment", increment_counter)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("check_quality", check_quality)

workflow.add_edge(ma.START, "increment")
workflow.add_edge("increment", "generate_answer")
workflow.add_edge("generate_answer", "check_quality")
workflow.add_conditional_edges("check_quality", should_continue)
```

### Fallback Chains

Try multiple strategies until one works:

```python
def fallback_router(state):
    \"\"\"Try progressively more expensive strategies.\"\"\"
    attempt = state.get("attempt", 1)
    
    if attempt == 1:
        return "cheap_model"
    elif attempt == 2:
        return "medium_model"
    elif attempt == 3:
        return "expensive_model"
    else:
        return ma.END  # All strategies failed

def check_success(state):
    \"\"\"Route based on success.\"\"\"
    if state.get("success"):
        return ma.END
    else:
        # Increment attempt and try next strategy
        return "increment_attempt"

workflow.add_conditional_edges("evaluator", check_success)
```

### State-Machine Routing

Implement a full state machine:

```python
def state_machine_router(state):
    \"\"\"Route based on current phase.\"\"\"
    phase = state.get("phase", "init")
    
    if phase == "init":
        return "initialize"
    elif phase == "collect":
        return "collect_data"
    elif phase == "process":
        return "process_data"
    elif phase == "validate":
        return "validate_results"
    elif phase == "done":
        return ma.END
    else:
        # Unknown phase, go to error handler
        return "error_handler"

# Each node updates the phase
def initialize(state):
    # ... initialization logic
    return {"phase": "collect"}

def collect_data(state):
    # ... collection logic
    return {"phase": "process"}

# etc.
```

---

## Router Functions in Detail

### Return Values

Router functions must return:

1. **Node name (string)**: Go to that node
   ```python
   return "next_node"
   ```

2. **END**: Terminate workflow
   ```python
   return ma.END
   ```

### Best Practices

#### ✅ Keep Routers Pure

```python
# ✅ Good: Pure function
def router(state):
    return "next" if state["value"] > 0 else "other"

# ❌ Bad: Side effects
def router(state):
    global counter
    counter += 1  # Side effect!
    return "next"
```

#### ✅ Handle Missing Keys

```python
# ✅ Good: Safe access
def router(state):
    value = state.get("key", default_value)
    if value > threshold:
        return "high_path"
    return "low_path"

# ❌ Bad: Can crash
def router(state):
    if state["key"] > threshold:  # KeyError if missing!
        return "high_path"
    return "low_path"
```

#### ✅ Use Type Hints

```python
# ✅ Good: Clear types
def router(state: MyState) -> str:
    \"\"\"
    Route based on category.
    
    Returns:
        Name of next node
    \"\"\"
    return "next_node"
```

---

## Complete Example: Self-Improving Agent

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
class AgentState(TypedDict):
    question: str
    attempt: int
    strategy: str
    answer: Optional[str]
    quality_score: Optional[float]
    done: bool

# Nodes
@ma.dspy_node
class SimpleQA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

@ma.dspy_node
class ComplexQA(ma.Module):
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
        result = self.checker(question=question, answer=answer)
        return {
            "quality_score": float(result.quality_score),
            "done": float(result.quality_score) >= 0.7
        }

def increment_attempt(state: AgentState) -> dict:
    return {"attempt": state.get("attempt", 0) + 1}

# Router functions
def strategy_selector(state: AgentState) -> str:
    \"\"\"Select strategy based on attempt.\"\"\"
    attempt = state.get("attempt", 1)
    
    if attempt == 1:
        return "simple_qa"
    elif attempt >= 2:
        return "complex_qa"
    else:
        return ma.END

def should_retry(state: AgentState) -> str:
    \"\"\"Decide if we should retry.\"\"\"
    # Check if done
    if state.get("done", False):
        return ma.END
    
    # Check max attempts
    if state.get("attempt", 0) >= 3:
        return ma.END  # Give up
    
    # Retry with different strategy
    return "increment"

# Build workflow
workflow = ma.graph.StateGraph(AgentState)

# Add nodes
workflow.add_node("increment", increment_attempt)
workflow.add_node("simple_qa", SimpleQA())
workflow.add_node("complex_qa", ComplexQA())
workflow.add_node("quality_check", QualityChecker())

# Build flow with conditional routing
workflow.add_edge(ma.START, "increment")
workflow.add_conditional_edges("increment", strategy_selector)
workflow.add_edge("simple_qa", "quality_check")
workflow.add_edge("complex_qa", "quality_check")
workflow.add_conditional_edges("quality_check", should_retry)

# Compile
graph = workflow.compile()

# Run
result = graph.invoke({
    "question": "Explain quantum entanglement simply.",
    "attempt": 0,
    "done": False
})

print(f"Final answer: {result['answer']}")
print(f"Quality: {result['quality_score']}")
print(f"Attempts: {result['attempt']}")
print(f"Strategy: Simple QA\" if result['attempt'] == 1 else \"Complex QA\")")
```

---

## Common Routing Scenarios

### 1. **Quality Gate**

Only proceed if quality threshold is met:

```python
def quality_gate(state):
    quality = state.get("quality_score", 0)
    
    if quality >= 0.9:
        return "finalize"  # High quality, finish
    elif quality >= 0.7:
        return "polish"  # Good, needs polish
    else:
        return "regenerate"  # Low quality, try again
```

### 2. **Resource-Based Routing**

Route based on available resources:

```python
def resource_router(state):
    budget_remaining = state.get("budget", 0)
    
    if budget_remaining > 100:
        return "expensive_model"  # Can afford best
    elif budget_remaining > 10:
        return "medium_model"  # Mid-tier
    else:
        return "cheap_model"  # Low budget
```

### 3. **Time-Based Routing**

Route based on urgency:

```python
import time

def time_router(state):
    deadline = state.get("deadline", float('inf'))
    time_left = deadline - time.time()
    
    if time_left < 60:  # Less than 1 minute
        return "fast_path"
    elif time_left < 300:  # Less than 5 minutes
        return "balanced_path"
    else:
        return "thorough_path"
```

### 4. **Error Recovery**

Handle errors gracefully:

```python
def error_recovery(state):
    error = state.get("error")
    retry_count = state.get("retry_count", 0)
    
    if error is None:
        return "success"  # No error
    elif retry_count < 3:
        return "retry"  # Try again
    else:
        return "fallback"  # Use fallback strategy
```

---

## Debugging Conditional Routes

### Log Routing Decisions

```python
import logging

logger = logging.getLogger(__name__)

def logged_router(state):
    \"\"\"Router with logging.\"\"\"
    decision = make_routing_decision(state)
    logger.info(f"Routing to {decision} based on state: {state}")
    return decision
```

### Test Routers Independently

```python
import unittest

class TestRouters(unittest.TestCase):
    def test_quality_router(self):
        # High quality → END
        state = {"quality_score": 0.9}
        self.assertEqual(quality_router(state), ma.END)
        
        # Low quality → retry
        state = {"quality_score": 0.3, "attempt": 1}
        self.assertEqual(quality_router(state), "retry")
```

### Visualize Routes

Use visualization (covered in next section) to see all possible paths.

---

## Performance Considerations

### Avoid Expensive Operations in Routers

```python
# ❌ Bad: Expensive operation
def slow_router(state):
    # Don't do this!
    expensive_result = call_llm_to_decide(state)
    return expensive_result

# ✅ Good: Use state values
def fast_router(state):
    # State already has what we need
    return "next" if state["ready"] else "wait"
```

### Minimize Router Complexity

```python
# ❌ Bad: Too complex
def complex_router(state):
    # 50 lines of nested if/else...
    pass

# ✅ Good: Extract logic to nodes
def simple_router(state):
    # Router just checks a flag
    return state["next_node_name"]

# Let a node compute the routing decision
def decision_node(state):
    # Complex logic here
    next_node = complex_decision_logic(state)
    return {"next_node_name": next_node}
```

---

## Next Steps

- **[Compilation & Execution](compilation.md)** → Run your workflows
- **[Visualization](visualization.md)** → See your routing paths
- **[State Management](state.md)** → Review state patterns

---

## External Resources

- **[LangGraph Conditional Edges](https://langchain-ai.github.io/langgraph/how-tos/branching/)** - Official guide
- **[LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)** - Routing examples

---

**Next: Learn about [Compilation & Execution →](compilation.md)**
