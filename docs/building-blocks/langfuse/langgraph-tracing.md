# LangGraph Tracing with Langfuse

> **TL;DR**: LangGraph workflows are automatically traced—every node, state transition, and routing decision is captured in Langfuse.

## Automatic Workflow Tracing

mahsm automatically traces entire LangGraph workflows:

```python
import mahsm as ma
from typing import TypedDict, Optional
import dspy
import os

# Configure
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
ma.tracing.init()

# Define state
class State(TypedDict):
    question: str
    answer: Optional[str]

# Create nodes
@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

# Build workflow
workflow = ma.graph.StateGraph(State)
workflow.add_node("qa", QA())
workflow.add_edge(ma.START, "qa")
workflow.add_edge("qa", ma.END)

graph = workflow.compile()

# Run - automatically traced!
result = graph.invoke({"question": "What is Python?"})

# ✅ Complete workflow traced in Langfuse:
# - Workflow execution
# - Each node execution
# - State at each step
# - All LLM calls
# - Routing decisions
```

---

## What Gets Traced?

### Simple Workflow

```python
workflow = ma.graph.StateGraph(State)
workflow.add_node("step1", step1_func)
workflow.add_node("step2", step2_func)
workflow.add_edge(ma.START, "step1")
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", ma.END)

graph = workflow.compile()
result = graph.invoke({"input": "data"})
```

**In Langfuse UI:**
```
Trace: Workflow Execution
├─ Span: step1
│   ├─ Input: {input: "data"}
│   └─ Output: {intermediate: "result1"}
└─ Span: step2
    ├─ Input: {intermediate: "result1"}
    └─ Output: {final: "result2"}
```

### Conditional Routing

```python
def router(state):
    if state["category"] == "simple":
        return "fast_path"
    return "complex_path"

workflow.add_conditional_edges("classifier", router)
```

**In Langfuse UI:**
```
Trace: Conditional Workflow
├─ Span: classifier
│   └─ Output: {category: "simple"}
├─ Decision: router → fast_path
└─ Span: fast_path
    └─ Output: {answer: "Quick result"}
```

---

## Tracing Patterns

### Linear Pipeline

```python
class PipelineState(TypedDict):
    input: str
    stage1_output: Optional[str]
    stage2_output: Optional[str]
    final: Optional[str]

def stage1(state):
    return {"stage1_output": f"Processed: {state['input']}"}

def stage2(state):
    return {"stage2_output": f"Enhanced: {state['stage1_output']}"}

def stage3(state):
    return {"final": f"Final: {state['stage2_output']}"}

workflow = ma.graph.StateGraph(PipelineState)
workflow.add_node("stage1", stage1)
workflow.add_node("stage2", stage2)
workflow.add_node("stage3", stage3)

workflow.add_edge(ma.START, "stage1")
workflow.add_edge("stage1", "stage2")
workflow.add_edge("stage2", "stage3")
workflow.add_edge("stage3", ma.END)

graph = workflow.compile()
result = graph.invoke({"input": "test"})

# Each stage traced in sequence
```

### Branching Workflow

```python
class BranchState(TypedDict):
    question: str
    category: Optional[str]
    answer: Optional[str]

def classify(state):
    if "code" in state["question"].lower():
        return {"category": "programming"}
    return {"category": "general"}

@ma.dspy_node
class ProgrammingQA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

@ma.dspy_node
class GeneralQA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

def route(state):
    if state["category"] == "programming":
        return "programming_qa"
    return "general_qa"

workflow = ma.graph.StateGraph(BranchState)
workflow.add_node("classify", classify)
workflow.add_node("programming_qa", ProgrammingQA())
workflow.add_node("general_qa", GeneralQA())

workflow.add_edge(ma.START, "classify")
workflow.add_conditional_edges("classify", route)
workflow.add_edge("programming_qa", ma.END)
workflow.add_edge("general_qa", ma.END)

graph = workflow.compile()
result = graph.invoke({"question": "How do I write a Python loop?"})

# Traces show which branch was taken
```

**In Langfuse UI:**
```
Trace: Branching Workflow
├─ Span: classify
│   ├─ Input: {question: "How do I write a Python loop?"}
│   └─ Output: {category: "programming"}
├─ Decision: route → programming_qa
└─ Span: programming_qa (DSPy Module)
    └─ Generation: ChainOfThought
        ├─ Input: question="..."
        ├─ Output: answer="..."
        └─ Cost: $0.0015
```

### Loop Workflow

```python
class LoopState(TypedDict):
    question: str
    answer: Optional[str]
    quality: Optional[float]
    iteration: int

@ma.dspy_node
class Generator(ma.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.gen(question=question)

@ma.dspy_node
class QualityChecker(ma.Module):
    def __init__(self):
        super().__init__()
        self.check = dspy.Predict("answer -> quality: float 0-1")
    
    def forward(self, answer):
        return self.check(answer=answer)

def increment(state):
    return {"iteration": state.get("iteration", 0) + 1}

def should_retry(state):
    if state.get("iteration", 0) >= 3:
        return ma.END
    if float(state.get("quality", 0)) < 0.7:
        return "increment"
    return ma.END

workflow = ma.graph.StateGraph(LoopState)
workflow.add_node("generator", Generator())
workflow.add_node("checker", QualityChecker())
workflow.add_node("increment", increment)

workflow.add_edge(ma.START, "generator")
workflow.add_edge("generator", "checker")
workflow.add_conditional_edges("checker", should_retry)
workflow.add_edge("increment", "generator")

graph = workflow.compile()
result = graph.invoke({"question": "Explain AI", "iteration": 0})

# Traces show loop iterations
```

**In Langfuse UI:**
```
Trace: Loop Workflow
├─ Iteration 1
│   ├─ Span: generator
│   ├─ Span: checker (quality: 0.5)
│   └─ Decision: should_retry → increment
├─ Iteration 2
│   ├─ Span: generator
│   ├─ Span: checker (quality: 0.8)
│   └─ Decision: should_retry → END
└─ Final: {answer: "...", quality: 0.8, iteration: 2}
```

---

## Debugging Workflows

### Find Slow Nodes

```python
# Langfuse shows duration for each node
# Identify bottlenecks:
# - Node A: 0.2s
# - Node B: 3.5s ← Slow!
# - Node C: 0.3s
```

### Track Routing Decisions

```python
# See which paths are taken
def router(state):
    # Decision logged in Langfuse
    if condition:
        return "path_a"  # Logged
    return "path_b"  # Logged

# Review in Langfuse:
# - 80% go to path_a
# - 20% go to path_b
```

### Analyze State Changes

```python
# State at each node is logged
# See how state evolves:
# 
# After classifier:
#   {question: "...", category: "tech"}
#
# After researcher:
#   {question: "...", category: "tech", findings: [...]}
#
# After answerer:
#   {question: "...", category: "tech", findings: [...], answer: "..."}
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
class ResearchState(TypedDict):
    question: str
    needs_research: bool
    search_query: Optional[str]
    findings: Optional[str]
    answer: Optional[str]

# Nodes
def classifier(state: ResearchState) -> dict:
    \"\"\"Classify if research is needed.\"\"\"
    question = state["question"]
    needs_research = len(question.split()) > 10
    return {"needs_research": needs_research}

@ma.dspy_node
class QueryGenerator(ma.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict("question -> search_query")
    
    def forward(self, question):
        return self.gen(question=question)

@ma.dspy_node
class Researcher(ma.Module):
    def __init__(self):
        super().__init__()
        self.research = dspy.ChainOfThought("search_query -> findings")
    
    def forward(self, search_query):
        return self.research(search_query=search_query)

@ma.dspy_node
class DirectQA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

@ma.dspy_node
class SynthesizeAnswer(ma.Module):
    def __init__(self):
        super().__init__()
        self.synth = dspy.ChainOfThought("question, findings -> answer")
    
    def forward(self, question, findings):
        return self.synth(question=question, findings=findings)

# Router
def route_by_research_need(state: ResearchState):
    if state["needs_research"]:
        return "query_gen"
    return "direct_qa"

def route_to_synthesize(state: ResearchState):
    return "synthesize"

# Build workflow
workflow = ma.graph.StateGraph(ResearchState)

workflow.add_node("classifier", classifier)
workflow.add_node("query_gen", QueryGenerator())
workflow.add_node("researcher", Researcher())
workflow.add_node("direct_qa", DirectQA())
workflow.add_node("synthesize", SynthesizeAnswer())

workflow.add_edge(ma.START, "classifier")
workflow.add_conditional_edges("classifier", route_by_research_need)
workflow.add_edge("query_gen", "researcher")
workflow.add_conditional_edges("researcher", route_to_synthesize)
workflow.add_edge("synthesize", ma.END)
workflow.add_edge("direct_qa", ma.END)

# Compile and run
graph = workflow.compile()

# Test both paths
print("=== Simple Question ===\")
result1 = graph.invoke({"question": "What is Python?"})
print(f"Answer: {result1['answer']}")

print("\n=== Complex Question ===\")
result2 = graph.invoke({"question": "What are the latest developments in quantum computing and their implications?\"})
print(f"Query: {result2.get('search_query', 'N/A')}")
print(f"Answer: {result2['answer']}")

print("\n✨ Check Langfuse UI to compare both execution paths!")
```

**In Langfuse UI you'll see two different traces:**

**Trace 1 (Simple Question):**
```
Trace: Simple Path
├─ Span: classifier (needs_research: false)
├─ Decision: route_by_research_need → direct_qa
└─ Span: direct_qa
    └─ Generation: Predict
        └─ Cost: $0.0003
```

**Trace 2 (Complex Question):**
```
Trace: Research Path
├─ Span: classifier (needs_research: true)
├─ Decision: route_by_research_need → query_gen
├─ Span: query_gen
│   └─ Generation: Predict
│       └─ Cost: $0.0003
├─ Span: researcher
│   └─ Generation: ChainOfThought
│       └─ Cost: $0.0012
├─ Decision: route_to_synthesize → synthesize
└─ Span: synthesize
    └─ Generation: ChainOfThought
        └─ Cost: $0.0015
```

---

## Best Practices

### ✅ Do:

1. **Use descriptive node names**
   ```python
   # ✅ Good - clear in traces
   workflow.add_node("extract_entities", extract_entities)
   workflow.add_node("classify_sentiment", classify_sentiment)
   
   # ❌ Bad - unclear
   workflow.add_node("node1", func1)
   workflow.add_node("node2", func2)
   ```

2. **Review workflow traces regularly**
   - Check execution paths
   - Identify bottlenecks
   - Monitor routing decisions

3. **Compare different inputs**
   - See which paths are taken
   - Identify edge cases
   - Optimize common paths

4. **Use state effectively**
   - Include relevant context
   - Track progress through workflow
   - Debug state transformations

### ❌ Don't:

1. **Don't put sensitive data in state**
   ```python
   # ❌ Bad - PII in trace
   result = graph.invoke({
       "user_ssn": "123-45-6789",  # Will be in Langfuse!
       "credit_card": "1234-5678-..."
   })
   ```

2. **Don't ignore failed paths**
   - Review failures in Langfuse
   - Identify error patterns
   - Fix routing issues

3. **Don't create overly complex workflows**
   - If traces are confusing, workflow is too complex
   - Simplify routing logic
   - Break into smaller sub-workflows

---

## Next Steps

- **[Manual Tracing](manual-tracing.md)** → Add custom spans
- **[LangGraph Visualization](../langgraph/visualization.md)** → Visualize workflow structure
- **[Building Your First Agent](../../guides/first-agent.md)** → Put it all together

---

## External Resources

- **[Langfuse Tracing](https://langfuse.com/docs/tracing)** - Official tracing guide
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - LangGraph docs

---

**Next: Add custom spans with [Manual Tracing →](manual-tracing.md)**
