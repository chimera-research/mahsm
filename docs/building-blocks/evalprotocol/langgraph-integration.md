# LangGraph Integration

> **TL;DR**: Evaluate entire LangGraph workflows end-to-end by testing multi-step agent behavior, routing decisions, and state transformations—all with automatic tracing in Langfuse.

## Why Evaluate Workflows?

DSPy modules are single-step operations, but real agents are multi-step workflows:

```
User Input → Node 1 → Node 2 → Conditional Route → Node 3 → Output
```

You need to test:
- ✅ **End-to-end behavior** - Does the full workflow produce correct results?
- ✅ **Routing decisions** - Does the agent choose the right path?
- ✅ **State management** - Is state properly maintained across nodes?
- ✅ **Error handling** - Does the workflow handle failures gracefully?

---

## Quick Start

### Basic Workflow Evaluation

```python
import mahsm as ma
from langgraph.graph import StateGraph, END
from eval_protocol import EvalTest, Grader

# Define workflow
class State(TypedDict):
    question: str
    answer: str

def process_node(state: State) -> State:
    # Your logic
    return {**state, "answer": "..."}

# Build graph
builder = StateGraph(State)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

app = builder.compile()

# Test the workflow
test = EvalTest(
    name="workflow_test",
    input={"question": "What is 2+2?"},
    grader=Grader.llm_as_judge(
        criteria="Output contains correct answer (4)"
    )
)

# Run test on the workflow
result = test.run_workflow(app)

print(f"Passed: {result.passed}")
print(f"Output: {result.output}")
```

---

## Testing Patterns

### 1. End-to-End Testing

Test the complete workflow from input to output:

```python
from eval_protocol import EvalTest, EvalSuite, Grader

# Create multi-node workflow
builder = StateGraph(State)
builder.add_node("analyze", analyze_query)
builder.add_node("retrieve", retrieve_info)
builder.add_node("respond", generate_response)

builder.add_edge(START, "analyze")
builder.add_edge("analyze", "retrieve")
builder.add_edge("retrieve", "respond")
builder.add_edge("respond", END)

app = builder.compile()

# Test end-to-end
suite = EvalSuite(
    name="workflow_quality",
    tests=[
        EvalTest(
            name="simple_query",
            input={"question": "What is AI?"},
            grader=Grader.llm_as_judge(
                criteria="Response is accurate and well-explained"
            )
        ),
        EvalTest(
            name="complex_query",
            input={"question": "Compare machine learning and deep learning"},
            grader=Grader.llm_as_judge(
                criteria="Response covers both topics and explains differences"
            )
        )
    ]
)

# Run evaluation
ma.tracing.init()
results = suite.run_workflow(app)
```

### 2. Routing Decision Testing

Verify that conditional routing works correctly:

```python
# Workflow with routing
def router(state: State) -> str:
    \"\"\"Route based on question type.\"\"\"
    if "math" in state["question"].lower():
        return "math_solver"
    elif "code" in state["question"].lower():
        return "code_helper"
    else:
        return "general_qa"

builder = StateGraph(State)
builder.add_node("router", router_node)
builder.add_node("math_solver", math_node)
builder.add_node("code_helper", code_node)
builder.add_node("general_qa", qa_node)

builder.add_conditional_edges(
    "router",
    router,
    {
        "math_solver": "math_solver",
        "code_helper": "code_helper",
        "general_qa": "general_qa"
    }
)

app = builder.compile()

# Test routing
routing_suite = EvalSuite(
    name="routing_tests",
    tests=[
        EvalTest(
            name="math_question_routed_correctly",
            input={"question": "What is 25 * 4?"},
            grader=Grader.custom(lambda output, _: (
                10.0 if output.get("route") == "math_solver" else 0.0
            ))
        ),
        EvalTest(
            name="code_question_routed_correctly",
            input={"question": "Write a Python function to sort a list"},
            grader=Grader.custom(lambda output, _: (
                10.0 if output.get("route") == "code_helper" else 0.0
            ))
        )
    ]
)

results = routing_suite.run_workflow(app)
```

### 3. State Transformation Testing

Verify state is correctly updated across nodes:

```python
# Test state changes
test = EvalTest(
    name="state_accumulation",
    input={"messages": [], "user_query": "Hello"},
    grader=Grader.custom(lambda output, _: (
        10.0 if len(output.get("messages", [])) >= 2 else 0.0
    ))
)

result = test.run_workflow(app)

# Check intermediate states in Langfuse trace
```

### 4. Error Handling Testing

Ensure workflow handles errors gracefully:

```python
suite = EvalSuite(
    name="error_handling",
    tests=[
        # Empty input
        EvalTest(
            name="empty_input",
            input={"question": ""},
            grader=Grader.llm_as_judge(
                criteria="Handles empty input without crashing"
            )
        ),
        
        # Invalid input
        EvalTest(
            name="invalid_input",
            input={"invalid_key": "value"},
            grader=Grader.custom(lambda output, _: (
                10.0 if "error" not in output.lower() else 5.0
            ))
        ),
        
        # Edge case
        EvalTest(
            name="very_long_input",
            input={"question": "word " * 5000},
            grader=Grader.llm_as_judge(
                criteria="Handles long input gracefully"
            )
        )
    ]
)

results = suite.run_workflow(app)
```

---

## Complete Example

### Multi-Agent Research Assistant

```python
import dspy
import mahsm as ma
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from eval_protocol import EvalTest, EvalSuite, Grader

# Configure DSPy
lm = dspy.LM(model="openai/gpt-4o-mini", api_key="...")
dspy.configure(lm=lm)

# Define state
class ResearchState(TypedDict):
    query: str
    search_results: list[str]
    summary: str
    final_answer: str

# Create nodes
@ma.dspy_node
class SearchAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.search = dspy.ChainOfThought("query -> search_query")
    
    def forward(self, query: str) -> str:
        return self.search(query=query).search_query

@ma.dspy_node
class SummaryAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought("results -> summary")
    
    def forward(self, results: str) -> str:
        return self.summarize(results=results).summary

@ma.dspy_node
class ResponseAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought("summary, query -> answer")
    
    def forward(self, summary: str, query: str) -> str:
        return self.respond(summary=summary, query=query).answer

# Build workflow
def search_node(state: ResearchState) -> ResearchState:
    agent = SearchAgent()
    search_query = agent(query=state["query"])
    # Simulate search
    results = [f"Result for: {search_query}"]
    return {**state, "search_results": results}

def summarize_node(state: ResearchState) -> ResearchState:
    agent = SummaryAgent()
    results_str = "\\n".join(state["search_results"])
    summary = agent(results=results_str)
    return {**state, "summary": summary}

def respond_node(state: ResearchState) -> ResearchState:
    agent = ResponseAgent()
    answer = agent(summary=state["summary"], query=state["query"])
    return {**state, "final_answer": answer}

# Assemble graph
builder = StateGraph(ResearchState)
builder.add_node("search", search_node)
builder.add_node("summarize", summarize_node)
builder.add_node("respond", respond_node)

builder.add_edge(START, "search")
builder.add_edge("search", "summarize")
builder.add_edge("summarize", "respond")
builder.add_edge("respond", END)

app = builder.compile()

# Create evaluation suite
suite = EvalSuite(
    name="research_assistant",
    tests=[
        EvalTest(
            name="factual_question",
            input={"query": "What is photosynthesis?"},
            grader=Grader.llm_as_judge(
                criteria=\"\"\"
                - Answer is scientifically accurate
                - Includes key concepts (light, chlorophyll, glucose)
                - Well-organized and clear
                \"\"\"
            )
        ),
        EvalTest(
            name="comparison_question",
            input={"query": "Compare photosynthesis and respiration"},
            grader=Grader.llm_as_judge(
                criteria=\"\"\"
                - Covers both processes
                - Explains key differences
                - Mentions inputs and outputs
                \"\"\"
            )
        ),
        EvalTest(
            name="recent_topic",
            input={"query": "What are the latest developments in quantum computing?"},
            grader=Grader.llm_as_judge(
                criteria=\"\"\"
                - Answer acknowledges it's based on available information
                - Provides relevant context
                - Well-structured response
                \"\"\"
            )
        )
    ]
)

# Run evaluation with tracing
ma.tracing.init(
    tags=["research-assistant", "evaluation"],
    release="v1.0.0"
)

results = suite.run_workflow(app)

print(f"\\nResults:")
print(f"Pass rate: {results.pass_rate:.1%}")
print(f"Average score: {results.average_score}/10")
print(f"\\nView detailed traces in Langfuse")
```

---

## Inspecting Workflow Execution

### View in Langfuse

When you run workflow evaluations with tracing enabled:

```python
ma.tracing.init()
results = suite.run_workflow(app)
```

Langfuse captures:

1. **Complete workflow trace**
   - All nodes executed
   - State at each step
   - Routing decisions

2. **Individual node spans**
   - Input to each node
   - Output from each node
   - Duration per node

3. **LLM calls within nodes**
   - Prompts sent
   - Responses received
   - Tokens and costs

4. **Test result metadata**
   - Pass/fail status
   - Score
   - Grading rationale

### Example Trace Structure

```
Trace: "research_assistant_evaluation"
└─ Test: "factual_question"
   ├─ Input: {\"query\": \"What is photosynthesis?\"}
   ├─ Workflow Execution:
   │  ├─ Node: "search"
   │  │  └─ DSPy Module: SearchAgent
   │  │     └─ LLM Call: "generate search query"
   │  ├─ Node: "summarize"
   │  │  └─ DSPy Module: SummaryAgent
   │  │     └─ LLM Call: "summarize results"
   │  └─ Node: "respond"
   │     └─ DSPy Module: ResponseAgent
   │        └─ LLM Call: "generate response"
   ├─ Output: {\"final_answer\": \"...\"}
   ├─ Grader LLM Call: "evaluate quality"
   ├─ Score: 8.5/10
   └─ Passed: true
```

---

## Advanced Patterns

### 1. Multi-Path Workflows

Test different execution paths:

```python
def conditional_router(state: State) -> str:
    if state["complexity"] == "simple":
        return "simple_path"
    else:
        return "complex_path"

builder.add_conditional_edges(
    "router",
    conditional_router,
    {
        "simple_path": "simple_node",
        "complex_path": "complex_node"
    }
)

# Test both paths
suite = EvalSuite(
    name="multi_path_test",
    tests=[
        EvalTest(
            name="simple_path",
            input={"query": "Hi", "complexity": "simple"},
            grader=Grader.contains("simple")
        ),
        EvalTest(
            name="complex_path",
            input={"query": "Explain...", "complexity": "complex"},
            grader=Grader.contains("complex")
        )
    ]
)
```

### 2. Loop Testing

Verify loops execute correctly:

```python
def should_continue(state: State) -> str:
    if state["iterations"] < 3:
        return "continue"
    return "end"

builder.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue": "process",  # Loop back
        "end": END
    }
)

# Test loop behavior
test = EvalTest(
    name="loop_execution",
    input={"iterations": 0},
    grader=Grader.custom(lambda output, _: (
        10.0 if output.get("iterations") == 3 else 0.0
    ))
)
```

### 3. Human-in-the-Loop Simulation

Test workflows with simulated human input:

```python
def human_approval(state: State) -> State:
    # Simulate human approval
    return {**state, "approved": True}

builder.add_node("human_review", human_approval)

# Test with automated approval
test = EvalTest(
    name="approval_flow",
    input={"request": "Deploy to production"},
    grader=Grader.custom(lambda output, _: (
        10.0 if output.get("approved") else 0.0
    ))
)
```

---

## Testing Best Practices

### ✅ Do:

1. **Test end-to-end first**
   ```python
   # Start with full workflow tests
   suite = EvalSuite(tests=[
       EvalTest(name="e2e_test", input={...}, grader=...)
   ])
   ```

2. **Then test individual paths**
   ```python
   # Once e2e works, test edge cases and specific paths
   ```

3. **Verify state transformations**
   ```python
   # Check that state changes are correct
   test = EvalTest(
       input={"initial": "value"},
       grader=Grader.custom(lambda output, _: (
           10.0 if output.get("transformed") == "expected" else 0.0
       ))
   )
   ```

4. **Test error recovery**
   ```python
   # Ensure workflow handles failures
   test = EvalTest(
       input={"cause_error": True},
       grader=Grader.llm_as_judge(
           "Workflow handles error gracefully"
       )
   )
   ```

### ❌ Don't:

1. **Don't only test individual nodes**
   ```python
   # ❌ Bad - misses integration issues
   test_node_1()
   test_node_2()
   test_node_3()
   
   # ✅ Good - tests full workflow
   test_workflow()
   ```

2. **Don't ignore routing logic**
   ```python
   # ❌ Bad - assumes routing works
   
   # ✅ Good - explicitly tests routing
   suite = EvalSuite(tests=[
       EvalTest(name="route_to_a", ...),
       EvalTest(name="route_to_b", ...)
   ])
   ```

3. **Don't skip edge cases**
   ```python
   # ✅ Good - comprehensive coverage
   tests = [
       EvalTest(name="normal", ...),
       EvalTest(name="empty_input", ...),
       EvalTest(name="max_length", ...),
       EvalTest(name="invalid_format", ...)
   ]
   ```

---

## Debugging Workflows

### Failed Workflow Test

When a workflow test fails:

```python
results = suite.run_workflow(app)

for test_result in results.test_results:
    if not test_result.passed:
        print(f"\\n❌ Failed: {test_result.test_name}")
        print(f"Input: {test_result.input}")
        print(f"Output: {test_result.output}")
        print(f"Trace ID: {test_result.trace_id}")
        print("\\nOpen Langfuse to see full workflow execution")
```

### View in Langfuse

1. Open Langfuse dashboard
2. Search for trace_id
3. Expand workflow execution
4. See which node failed or produced unexpected output
5. Review LLM calls within that node

---

## Performance Testing

### Latency Testing

```python
import time

test = EvalTest(
    name="latency_test",
    input={"query": "Quick question"},
    grader=Grader.custom(lambda output, _: (
        10.0 if output.get("duration", 999) < 5.0 else 0.0
    ))
)

# Track duration
start = time.time()
result = test.run_workflow(app)
duration = time.time() - start

print(f"Duration: {duration:.2f}s")
```

### Cost Testing

```python
# Run evaluation with cost tracking
ma.tracing.init()
results = suite.run_workflow(app)

# View costs in Langfuse:
# - Total tokens used
# - Cost per test
# - Cost per node
```

---

## CI/CD Integration

### Automated Workflow Testing

```python
# tests/test_workflow.py
import pytest
from eval_protocol import EvalSuite

@pytest.fixture
def workflow():
    # Build and return compiled workflow
    return app

@pytest.fixture
def eval_suite():
    return EvalSuite(name="workflow_tests", tests=[...])

def test_workflow_quality(workflow, eval_suite):
    results = eval_suite.run_workflow(workflow)
    assert results.pass_rate >= 0.9

def test_workflow_latency(workflow):
    # Test that workflow completes quickly
    start = time.time()
    workflow.invoke({"query": "test"})
    duration = time.time() - start
    assert duration < 10.0, f"Workflow too slow: {duration}s"
```

---

## Next Steps

1. **[Langfuse Dashboard](https://cloud.langfuse.com)** - View your workflow traces
2. **[DSPy Modules](../dspy/modules.md)** - Build better nodes for your workflows
3. **[LangGraph Compilation](../langgraph/compilation.md)** - Optimize workflow execution

---

## External Resources

- **[LangGraph Testing](https://langchain-ai.github.io/langgraph/how-tos/testing/)** - Official testing guide
- **[Workflow Patterns](https://langchain-ai.github.io/langgraph/concepts/patterns/)** - Common workflow patterns
- **[EvalProtocol](https://pypi.org/project/eval-protocol/)** - Evaluation framework docs

---

**Congratulations! You've completed the Building Blocks documentation.** 🎉

You now know how to use all 4 frameworks:
- ✅ **DSPy** - Build LLM modules
- ✅ **LangGraph** - Orchestrate multi-step workflows
- ✅ **Langfuse** - Trace and monitor everything
- ✅ **EvalProtocol** - Test and evaluate systematically

**Next: Explore [Guides →](../../guides/index.md) for end-to-end tutorials!**
