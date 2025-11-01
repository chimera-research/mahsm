# Langfuse Overview

> **TL;DR**: Langfuse provides observability for AI applications—trace LLM calls, debug workflows, analyze costs, and improve performance.

## What is Langfuse?

**Langfuse** is an open-source observability platform for LLM applications. It helps you:

- **Trace** every LLM call and agent step
- **Debug** issues in production
- **Analyze** performance and costs
- **Monitor** quality over time
- **Optimize** prompts based on real data

Think of it as **your eyes into what your AI agent is actually doing**.

---

## Why Use Langfuse?

### Without Langfuse

```python
# You only see the final result
result = agent.run("Build a web scraper")
print(result)  # ✅ or ❌ ?

# Questions you can't answer:
# - Which LLM calls were made?
# - How much did it cost?
# - Where did it fail?
# - How long did each step take?
# - What were the exact prompts?
```

### With Langfuse

```python
ma.tracing.init()  # One line!

result = agent.run("Build a web scraper")

# Now you can see:
# ✅ Every LLM call with prompts and responses
# ✅ Total cost: $0.0042
# ✅ Latency: 2.3s (breakdown by step)
# ✅ Full execution trace in beautiful UI
# ✅ Token usage per call
```

---

## Key Concepts

### Traces

A **trace** represents one complete execution:

```
Trace: "Build a web scraper"
├─ Generation: Plan creation (0.8s, $0.001)
├─ Generation: Code generation (1.2s, $0.002)
└─ Generation: Documentation (0.3s, $0.001)
```

Every agent execution gets one trace.

### Spans

**Spans** are steps within a trace:

```
Trace: QA Pipeline
├─ Span: Classify question
│  └─ Generation: LLM classification
├─ Span: Research
│  ├─ Generation: Query generation
│  └─ Generation: Summarization
└─ Span: Final answer
   └─ Generation: Answer synthesis
```

Spans represent logical steps (nodes in your workflow).

### Generations

**Generations** are individual LLM calls:

```
Generation
├─ Model: gpt-4o-mini
├─ Prompt: "Classify: What is Python?"
├─ Response: "Category: programming"
├─ Tokens: 20 input, 5 output
├─ Cost: $0.0001
└─ Latency: 234ms
```

Every time you call an LLM, Langfuse captures it.

---

## Hierarchy

```
Trace (Execution)
  └─ Span (Workflow Step / Node)
      └─ Generation (LLM Call)
          └─ Prompt & Response
```

**Example:**
```
Trace: "Research quantum physics"
  └─ Span: Research Agent
      ├─ Generation: Generate search query
      ├─ Generation: Summarize results
      └─ Generation: Create final answer
```

---

## mahsm Integration

mahsm makes Langfuse integration **automatic and seamless**.

### Automatic Tracing

```python
import mahsm as ma
import dspy

# 1. Initialize once
ma.tracing.init()

# 2. Build your agent
@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

# 3. Run - automatic tracing!
qa = QA()
result = qa(question="What is Python?")

# 🎉 Everything is automatically traced in Langfuse!
```

**No manual instrumentation needed!**

---

## What Gets Traced?

### DSPy Modules

```python
@ma.dspy_node
class MyModule(ma.Module):
    def forward(self, input):
        # All LLM calls here are traced
        return result
```

**Automatically traced:**
- Module name
- Input parameters
- Output values
- All internal LLM calls
- Token usage and costs
- Latency

### LangGraph Workflows

```python
workflow = ma.graph.StateGraph(State)
workflow.add_node("step1", step1)
workflow.add_node("step2", step2)
graph = workflow.compile()

# All nodes and state transitions traced
result = graph.invoke({"input": "data"})
```

**Automatically traced:**
- Node execution order
- State at each step
- Conditional routing decisions
- Total workflow time
- All LLM calls within nodes

### Manual Spans

```python
@ma.tracing.observe(name="Custom Logic")
def my_function(data):
    # Custom span for non-LLM work
    return process(data)
```

---

## The Langfuse UI

### Dashboard

See all your traces:

```
Recent Traces
┌─────────────────────────────────────────────────┐
│ Time    │ Name           │ Status │ Cost    │ Latency │
├─────────────────────────────────────────────────┤
│ 14:32   │ Research Task  │ ✅     │ $0.034  │ 3.2s    │
│ 14:30   │ QA Request     │ ✅     │ $0.002  │ 0.8s    │
│ 14:28   │ Code Gen       │ ❌     │ $0.015  │ 5.1s    │
└─────────────────────────────────────────────────┘
```

### Trace Details

Click any trace to see:

1. **Timeline** - Visual execution flow
2. **Generations** - Every LLM call
3. **Metadata** - Costs, tokens, latency
4. **I/O** - Inputs and outputs at each step

### Analytics

Track over time:
- Total costs
- Average latency
- Success rates
- Token usage
- Model distribution

---

## When to Use Langfuse

### Development ✅

```python
# Debug locally
ma.tracing.init()

# Run your agent
result = agent.run(input)

# Check Langfuse UI to see what happened
```

### Production ✅

```python
# Monitor in production
ma.tracing.init(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY")
)

# All production traffic is traced
```

### Optimization ✅

```python
# Trace before and after optimization
ma.tracing.init()

# Compare costs, latency, quality
# Make data-driven decisions
```

---

## Setup Options

### Langfuse Cloud (Easiest)

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Get API keys
3. Done!

```python
ma.tracing.init(
    public_key="YOUR_LANGFUSE_PUBLIC_KEY",
    secret_key="YOUR_LANGFUSE_SECRET_KEY"
)
```

### Self-Hosted

Run Langfuse locally:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker-compose up
```

Then:
```python
ma.tracing.init(host="http://localhost:3000")
```

---

## Configuration

### Basic

```python
ma.tracing.init()
```

Uses environment variables:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (optional)

### Explicit

```python
ma.tracing.init(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com"
)
```

### With Metadata

```python
ma.tracing.init(
    public_key="YOUR_PUBLIC_KEY",
    secret_key="YOUR_SECRET_KEY",
    session_id="user-123",
    tags=["production", "api-v2"]
)
```

---

## Complete Example

```python
import mahsm as ma
from typing import TypedDict, Optional
import dspy
import os

# Configure DSPy
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)

# Initialize Langfuse tracing
ma.tracing.init()

# Define state
class QAState(TypedDict):
    question: str
    answer: Optional[str]

# Create DSPy module
@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

# Build workflow
workflow = ma.graph.StateGraph(QAState)
workflow.add_node("qa", QA())
workflow.add_edge(ma.START, "qa")
workflow.add_edge("qa", ma.END)

graph = workflow.compile()

# Run - automatically traced!
result = graph.invoke({"question": "What is machine learning?"})

print(f"Answer: {result['answer']}")
print("\n✨ Check Langfuse UI to see the full trace!")
```

**In Langfuse UI you'll see:**
- Trace: "QA Pipeline"
  - Span: "qa" node
    - Generation: ChainOfThought call
      - Prompt: "question -> answer"
      - Input: "What is machine learning?"
      - Output: [Complete answer]
      - Cost: $0.0015
      - Latency: 456ms

---

## Benefits

### 🐛 Debugging

Find issues fast:
- See exact prompts that failed
- Identify slow steps
- Track error patterns

### 💰 Cost Management

Control spending:
- Track costs per execution
- Identify expensive operations
- Optimize high-cost paths

### ⚡ Performance

Improve speed:
- Find bottlenecks
- Measure optimization impact
- Monitor latency trends

### 📊 Analytics

Make data-driven decisions:
- Compare prompt versions
- Track quality over time
- Understand user patterns

---

## Best Practices

### ✅ Do:

1. **Initialize once at startup**
   ```python
   ma.tracing.init()
   # Then run your app
   ```

2. **Use descriptive names**
   ```python
   @ma.tracing.observe(name="Extract Entities")
   def extract_entities(text):
       pass
   ```

3. **Tag important traces**
   ```python
   ma.tracing.init(tags=["production", "customer-tier-1"])
   ```

4. **Check Langfuse regularly**
   - Review traces daily
   - Monitor costs
   - Identify patterns

### ❌ Don't:

1. **Initialize multiple times**
   ```python
   # ❌ Don't do this
   ma.tracing.init()
   ma.tracing.init()  # Redundant!
   ```

2. **Log sensitive data**
   ```python
   # ❌ Don't trace PII
   result = process(user_ssn)  # Will be in trace!
   ```

3. **Ignore the UI**
   - Tracing is useless if you don't review traces
   - Set up dashboards
   - Create alerts

---

## Next Steps

Now that you understand Langfuse concepts:

1. **[Initialization](initialization.md)** → Set up Langfuse
2. **[DSPy Tracing](dspy-tracing.md)** → Trace DSPy modules
3. **[LangGraph Tracing](langgraph-tracing.md)** → Trace workflows
4. **[Manual Tracing](manual-tracing.md)** → Add custom spans

---

## External Resources

- **[Langfuse Docs](https://langfuse.com/docs)** - Official documentation
- **[Langfuse Cloud](https://cloud.langfuse.com)** - Hosted platform
- **[Langfuse GitHub](https://github.com/langfuse/langfuse)** - Self-hosting

---

**Next: Set up Langfuse with [Initialization →](initialization.md)**
