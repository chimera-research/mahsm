# Quickstart: Building Your First `mahsm` Agent

Let's build a simple research agent to see how the core components of `mahsm` work together. This example demonstrates the declarative nature of the framework.

### 1. The Single Import

All of `mahsm`'s fused functionality is available through the top-level `ma` import.

```python
import mahsm as ma
from typing import TypedDict, Optional
```

### 2. Define the Shared State

The State is a TypedDict that defines the data structure that flows through your graph. Every node can read from and write to this state.

```python
class AgentState(TypedDict):
    input_query: str
    research_result: Optional[str]
```

### 3. Create a Reasoning Node with @ma.dspy_node

Here, we define the "brain" of our agent using a `dspy.Module`. The `@ma.dspy_node` decorator is the magic that makes this module compatible with the LangGraph orchestrator, automatically handling data mapping from the AgentState.

```python
@ma.dspy_node
class Researcher(ma.Module):
    def __init__(self):
        super().__init__()
        self.signature = "input_query -> research_result"
        self.predictor = ma.dspy.ChainOfThought(self.signature)

    def forward(self, input_query):
        return self.predictor(input_query=input_query)
```

### 4. Build and Compile the Graph

Finally, we use `ma.graph.StateGraph` to define the workflow. We add our `Researcher` node and define the edges that control the flow of execution.

```python
# Initialize the graph with our state definition
workflow = ma.graph.StateGraph(AgentState)

# Add the DSPy-powered node
workflow.add_node("researcher", Researcher())

# Define the workflow structure
workflow.add_edge(ma.START, "researcher")
workflow.add_edge("researcher", ma.END)

# Compile the graph into a runnable application
graph = workflow.compile()
```

### 5. Run and Trace

To run your agent, simply invoke the graph. If you've called ma.init() beforehand, the entire execution will be traced in LangFuse automatically.

```python
# Initialize tracing (do this once at the start of your app)
ma.init()

# Run the graph
inputs = {"input_query": "What is the future of multi-agent AI systems?"}
result = graph.invoke(inputs)

print(result['research_result'])
```

That's it! You've built a fully observable and testable agent with minimal boilerplate, focusing only on the essential logic.