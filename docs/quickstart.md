# MAHSM Quickstart Guide

## Installation

```bash
pip install mahsm
```

Or from source:
```bash
git clone https://github.com/chimera-research/mahsm.git
cd mahsm
pip install -e .
```

## Complete Workflow

### Step 1: Optimize with DSPy

```python
import dspy
from dspy.teleprompt import GEPA

# Configure DSPy
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4o-mini"))

# Define your task
class DataAnalysis(dspy.Signature):
    """Analyze data and provide insights."""
    data = dspy.InputField()
    insights = dspy.OutputField()

# Optimize (or use simple program)
program = dspy.Predict(DataAnalysis)
```

### Step 2: Save with MAHSM

```python
import mahsm as ma

def calculate_stats(data: list, measures: list) -> dict:
    """Calculate statistical measures."""
    import statistics
    results = {}
    if "mean" in measures:
        results["mean"] = statistics.mean(data)
    return results

# Save optimized prompt
ma.prompt.save(
    program,
    name="data_analysis",
    version="v1",
    tools=[calculate_stats]
)
```

### Step 3: Use in LangGraph

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage

def analysis_node(state: MessagesState):
    # Load prompt with validation
    prompt = ma.prompt.load(
        "data_analysis_v1",
        validate_tools=[calculate_stats]
    )
    
    # Execute with full transparency
    result, messages = ma.inference(
        model="openai/gpt-4o-mini",
        prompt=prompt,
        tools=[calculate_stats],
        input=state["messages"][-1].content,
        state=state.get("messages", [])
    )
    
    return {"messages": messages}

# Build workflow
workflow = StateGraph(MessagesState)
workflow.add_node("analyze", analysis_node)
workflow.set_entry_point("analyze")
workflow.set_finish_point("analyze")

app = workflow.compile()

# Execute
result = app.invoke({
    "messages": [HumanMessage(content="Analyze: [1, 2, 3, 4, 5]")]
})
```

## Configuration

### LangFuse Tracing (Optional)

```bash
export LANGFUSE_PUBLIC_KEY="your_key"
export LANGFUSE_SECRET_KEY="your_secret"
```

### Custom Settings

```bash
export MAHSM_HOME="~/.mahsm"
export MAHSM_MAX_ITERATIONS="10"
```

## Message Transparency

MAHSM maintains complete conversation history:

```python
for i, msg in enumerate(result["messages"]):
    print(f"{i+1}. {type(msg).__name__}: {msg.content[:50]}...")
```

Output:
```
1. SystemMessage: You are an expert analyst...
2. HumanMessage: Analyze: [1, 2, 3, 4, 5]
3. AIMessage: I'll calculate statistics...
4. ToolMessage: {"mean": 3.0}
5. AIMessage: The mean is 3.0...
```

## Next Steps

- See [API Reference](api/) for complete documentation
- Check [Examples](examples/) for more use cases
- Read [Concepts](concepts/) for architecture details

