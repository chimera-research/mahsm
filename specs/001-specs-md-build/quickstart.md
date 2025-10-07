# Quickstart Guide: MAHSM v0.1.0

**Date**: October 7, 2025  
**Feature**: MAHSM (Multi-Agent Hyper-Scaling Methods) v0.1.0  
**Purpose**: End-to-end workflow validation and user onboarding

## Overview

This quickstart demonstrates the complete MAHSM workflow: optimizing prompts with DSPy, saving them with tool schemas, loading them in LangGraph nodes, and executing inference with full message transparency.

## Prerequisites

```bash
pip install mahsm dspy langgraph langfuse
```

## Step 1: DSPy Optimization (Offline)

```python
import dspy
from dspy.teleprompt import GEPA

# Configure DSPy
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4o-mini"))

# Define your task signature
class DataAnalysis(dspy.Signature):
    """Analyze data and provide statistical insights."""
    data = dspy.InputField(desc="Raw data to analyze")
    insights = dspy.OutputField(desc="Statistical insights and recommendations")

# Create and optimize the program
program = dspy.Predict(DataAnalysis)
optimizer = GEPA(metric=your_metric_function)
compiled_program = optimizer.compile(program, trainset=your_training_data)

print("Optimized prompt:", compiled_program.predict.signature.instructions)
```

## Step 2: Save Optimized Prompt with Tools

```python
import mahsm as ma

# Define your tools
def calculate_statistics(data: list[float], measures: list[str]) -> dict:
    """Calculate statistical measures for the given data."""
    import statistics
    results = {}
    if "mean" in measures:
        results["mean"] = statistics.mean(data)
    if "median" in measures:
        results["median"] = statistics.median(data)
    if "stdev" in measures:
        results["stdev"] = statistics.stdev(data)
    return results

def generate_chart(data: list[float], chart_type: str) -> str:
    """Generate a chart description for the data."""
    return f"Generated {chart_type} chart for {len(data)} data points"

# Save the optimized prompt with tools
artifact_path = ma.prompt.save(
    compiled_program,
    name="data_analysis",
    version="v1",
    tools=[calculate_statistics, generate_chart]
)

print(f"Prompt saved to: {artifact_path}")
```

## Step 3: Load Prompt in LangGraph Node

```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Define your LangGraph node
def analysis_node(state: MessagesState):
    # Load the optimized prompt with tool validation
    prompt = ma.prompt.load(
        "data_analysis_v1",
        validate_tools=[calculate_statistics, generate_chart]
    )
    
    # Execute inference with the loaded prompt
    result, messages = ma.inference(
        model="openai/gpt-4o-mini",
        prompt=prompt,
        tools=[calculate_statistics, generate_chart],
        input=state["messages"][-1].content,  # Latest user message
        state=state.get("messages", []),
        max_iterations=5
    )
    
    # Return updated state with all messages
    return {"messages": messages}

# Build your LangGraph
workflow = StateGraph(MessagesState)
workflow.add_node("analyze", analysis_node)
workflow.set_entry_point("analyze")
workflow.set_finish_point("analyze")

app = workflow.compile()
```

## Step 4: Execute with Full Traceability

```python
from langchain_core.messages import HumanMessage

# Configure LangFuse tracing (optional)
import os
os.environ["LANGFUSE_PUBLIC_KEY"] = "your_public_key"
os.environ["LANGFUSE_SECRET_KEY"] = "your_secret_key"

# Execute the workflow
user_input = "Please analyze this dataset: [1.2, 2.3, 3.1, 4.5, 2.8, 3.9, 1.7, 4.2]"
initial_state = {"messages": [HumanMessage(content=user_input)]}

result = app.invoke(initial_state)

# Inspect the complete message history
print("Complete conversation:")
for i, message in enumerate(result["messages"]):
    print(f"{i+1}. {type(message).__name__}: {message.content[:100]}...")
    if hasattr(message, 'tool_calls') and message.tool_calls:
        print(f"   Tool calls: {[tc['function']['name'] for tc in message.tool_calls]}")
```

## Step 5: Verify Message Transparency

```python
# The message history should contain:
# 1. SystemMessage (the DSPy optimized prompt)
# 2. HumanMessage (user input)
# 3. AIMessage (model response with tool_calls)
# 4. ToolMessage (calculate_statistics result)
# 5. ToolMessage (generate_chart result)
# 6. AIMessage (final analysis based on tool results)

messages = result["messages"]
assert isinstance(messages[0], SystemMessage), "First message should be system prompt"
assert isinstance(messages[1], HumanMessage), "Second message should be user input"
assert any(isinstance(msg, ToolMessage) for msg in messages), "Should contain tool results"

print("✅ Message transparency verified!")
```

## Expected Output

```
Optimized prompt: You are an expert data analyst with advanced statistical knowledge...
Prompt saved to: ~/.mahsm/prompts/data_analysis_v1.json
Complete conversation:
1. SystemMessage: You are an expert data analyst with advanced statistical knowledge...
2. HumanMessage: Please analyze this dataset: [1.2, 2.3, 3.1, 4.5, 2.8, 3.9, 1.7, 4.2]
3. AIMessage: I'll analyze this dataset for you. Let me calculate some key statistics...
   Tool calls: ['calculate_statistics', 'generate_chart']
4. ToolMessage: {"mean": 2.9625, "median": 2.95, "stdev": 1.1547}
5. ToolMessage: Generated histogram chart for 8 data points
6. AIMessage: Based on the statistical analysis, your dataset shows...
✅ Message transparency verified!
```

## Validation Checklist

This quickstart validates the following requirements:

- [ ] **FR-001**: ma.prompt.save() extracts optimized prompts from DSPy
- [ ] **FR-002**: Artifacts stored in ~/.mahsm/prompts/ with correct naming
- [ ] **FR-003**: Tool schemas included in OpenAI function format
- [ ] **FR-004**: ma.prompt.load() retrieves and validates prompts
- [ ] **FR-005**: Tool validation compares names and parameters
- [ ] **FR-007**: ma.inference() executes agentic loops
- [ ] **FR-008**: SystemMessage and HumanMessage creation
- [ ] **FR-009**: Tools executed as Python functions, wrapped in ToolMessage
- [ ] **FR-010**: Messages appended in correct chronological order
- [ ] **FR-011**: Loop continues until no tool_calls or max_iterations
- [ ] **FR-014**: Returns tuple of (result, messages)
- [ ] **FR-016**: LangFuse tracing when environment variables set
- [ ] **FR-018**: Only LangChain message types used

## Troubleshooting

### Common Issues

1. **ValidationError during load**: Tools don't match saved schemas
   - Solution: Ensure tool function signatures match exactly

2. **MaxIterationsError**: Inference loop exceeded limit
   - Solution: Increase max_iterations or check for infinite loops

3. **ToolExecutionError**: Tool function failed
   - Solution: Check tool function implementation and error handling

4. **Missing artifact file**: FileNotFoundError during load
   - Solution: Verify artifact was saved and path is correct

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Inspect artifact contents
import json
with open("~/.mahsm/prompts/data_analysis_v1.json") as f:
    artifact = json.load(f)
    print("Saved prompt:", artifact["prompt"][:200])
    print("Tool schemas:", [tool["name"] for tool in artifact["tools"]])
```

---

**Quickstart Status**: ✅ Complete - End-to-end workflow defined with validation checklist
