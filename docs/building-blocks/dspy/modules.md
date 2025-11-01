# DSPy Modules

> **TL;DR**: Modules are reusable components that combine signatures with prompting strategies like chain-of-thought or ReAct.

## What are DSPy Modules?

**DSPy Modules** are the building blocks of your LLM pipelines. Each module:
- Takes a **signature** (input → output specification)
- Applies a **prompting strategy** (e.g., chain-of-thought, few-shot)
- Returns structured outputs

Think of modules as **smart function wrappers** that automatically generate and execute prompts.

---

## Built-in Modules

### 1. **dspy.Predict**

The simplest module—direct prediction without reasoning.

```python
import dspy

predictor = dspy.Predict("question -> answer")

result = predictor(question="What is the capital of France?")
print(result.answer)  # "Paris"
```

**When to use:**
- Simple, straightforward tasks
- When you don't need reasoning traces
- Fast, low-token operations

---

### 2. **dspy.ChainOfThought**

Adds step-by-step reasoning before the final answer.

```python
cot = dspy.ChainOfThought("question -> answer")

result = cot(question="If a train travels 60 mph for 2.5 hours, how far does it go?")
print(result.reasoning)  # "Let me think step by step..."
print(result.answer)      # "150 miles"
```

**How it works:**
- Automatically adds a `reasoning` field to outputs
- Prompts the model to "think step by step"
- Better for complex reasoning tasks

**When to use:**
- Mathematical problems
- Multi-step reasoning
- When you want to see the model's thought process

**Example with mahsm:**

```python
import mahsm as ma
from typing import TypedDict

class MathState(TypedDict):
    problem: str
    reasoning: str
    solution: str

@ma.dspy_node
class MathSolver(ma.Module):
    def __init__(self):
        super().__init__()
        self.solver = dspy.ChainOfThought("problem -> reasoning, solution")
    
    def forward(self, problem):
        return self.solver(problem=problem)

# Use in workflow
workflow = ma.graph.StateGraph(MathState)
workflow.add_node("solve", MathSolver())
# Both reasoning and solution are written to state!
```

---

### 3. **dspy.ReAct**

Implements the ReAct pattern (Reasoning + Acting) for tool-using agents.

```python
from dspy import ReAct

# Define tools
def search_web(query: str) -> str:
    """Search the web for information."""
    # Your search implementation
    return f"Results for: {query}"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Create ReAct module
react = dspy.ReAct("question -> answer", tools=[search_web, calculate])

result = react(question="What is the population of Tokyo times 2?")
print(result.answer)
# Agent will:
# 1. search_web("population of Tokyo")
# 2. calculate("37400000 * 2")
# 3. Return final answer
```

**How it works:**
- Model alternates between **Reasoning** and **Acting** (tool calls)
- Automatically generates tool calls based on the question
- Continues until it has enough information

**When to use:**
- When your agent needs external tools (search, calculator, APIs)
- Multi-step tasks requiring information gathering
- Agentic workflows

**Example with mahsm:**

```python
@ma.dspy_node
class ResearchAgent(ma.Module):
    def __init__(self, tools):
        super().__init__()
        self.react = dspy.ReAct("question -> answer", tools=tools)
    
    def forward(self, question):
        return self.react(question=question)

# Define tools
def search_papers(query: str) -> str:
    """Search academic papers."""
    return "Paper results..."

# Use in workflow
agent = ResearchAgent(tools=[search_papers])
workflow.add_node("research", agent)
```

---

### 4. **dspy.ProgramOfThought**

Combines natural language reasoning with code execution.

```python
pot = dspy.ProgramOfThought("problem -> answer")

result = pot(problem="Calculate the compound interest on $1000 at 5% for 3 years")
# Generates Python code, executes it, returns answer
print(result.answer)  # "$1157.63"
```

**When to use:**
- Mathematical computations
- Tasks requiring precise calculations
- When you want guaranteed accuracy for arithmetic

---

### 5. **dspy.MultiChainComparison**

Generates multiple reasoning chains and selects the best one.

```python
mcc = dspy.MultiChainComparison("question -> answer", M=3)

result = mcc(question="What are the benefits of renewable energy?")
# Generates 3 different reasoning chains, picks the best
print(result.answer)
```

**When to use:**
- High-stakes decisions
- When you want diverse perspectives
- Quality over speed

---

## Custom Modules

Create your own modules by subclassing `dspy.Module`:

```python
import dspy

class CustomPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        # Initialize sub-modules
        self.classifier = dspy.Predict("text -> category")
        self.summarizer = dspy.ChainOfThought("text, category -> summary")
    
    def forward(self, text):
        # Step 1: Classify
        category_result = self.classifier(text=text)
        category = category_result.category
        
        # Step 2: Summarize based on category
        if category == "technical":
            # Use chain-of-thought for complex content
            return self.summarizer(text=text, category=category)
        else:
            # Simple summary for non-technical
            return {"summary": text[:100]}
```

**Key points:**
1. Always call `super().__init__()`
2. Initialize sub-modules in `__init__`
3. Implement `forward()` method
4. Return a dict or DSPy prediction

---

## Module Composition

Combine modules to build complex pipelines:

```python
class ResearchPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_query = dspy.ChainOfThought("question -> search_query")
        self.synthesize = dspy.ChainOfThought("question, results -> answer, sources")
        self.verify = dspy.Predict("answer, sources -> confidence: float 0-1")
    
    def forward(self, question):
        # Step 1: Generate search query
        query_result = self.generate_query(question=question)
        
        # Step 2: Search (simulated)
        results = self.search_api(query_result.search_query)
        
        # Step 3: Synthesize answer
        synthesis = self.synthesize(question=question, results=results)
        
        # Step 4: Verify confidence
        verification = self.verify(
            answer=synthesis.answer,
            sources=synthesis.sources
        )
        
        return {
            "answer": synthesis.answer,
            "sources": synthesis.sources,
            "confidence": verification.confidence
        }
    
    def search_api(self, query):
        # Your search implementation
        return f"Results for {query}"
```

---

## Modules in mahsm

The `@dspy_node` decorator makes DSPy modules work seamlessly with LangGraph:

### Basic Usage

```python
import mahsm as ma
from typing import TypedDict

class State(TypedDict):
    question: str
    answer: str

@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

# Add to workflow
workflow = ma.graph.StateGraph(State)
workflow.add_node("qa", QA())
```

### Multi-Module Pipeline

```python
class PipelineState(TypedDict):
    question: str
    search_query: str
    results: str
    answer: str

@ma.dspy_node
class QueryGenerator(ma.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.ChainOfThought("question -> search_query")
    
    def forward(self, question):
        return self.gen(question=question)

@ma.dspy_node
class Synthesizer(ma.Module):
    def __init__(self):
        super().__init__()
        self.synth = dspy.ChainOfThought("question, results -> answer")
    
    def forward(self, question, results):
        return self.synth(question=question, results=results)

# Build workflow
workflow = ma.graph.StateGraph(PipelineState)
workflow.add_node("generate_query", QueryGenerator())
workflow.add_node("search", search_function)  # Regular function
workflow.add_node("synthesize", Synthesizer())

workflow.add_edge(ma.START, "generate_query")
workflow.add_edge("generate_query", "search")
workflow.add_edge("search", "synthesize")
workflow.add_edge("synthesize", ma.END)
```

---

## Module Configuration

### Model Selection

Configure the model used by all modules:

```python
import dspy
import os

# Option 1: OpenAI
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)

# Option 2: Anthropic
lm = dspy.LM('anthropic/claude-3-5-sonnet-20241022', api_key=os.getenv("ANTHROPIC_API_KEY"))
dspy.configure(lm=lm)

# Option 3: Local model
lm = dspy.LM('ollama/llama3.1', api_base='http://localhost:11434')
dspy.configure(lm=lm)
```

### Per-Module Configuration

```python
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use different models for different tasks
        self.fast_predictor = dspy.Predict("input -> output")
        self.complex_reasoner = dspy.ChainOfThought("input -> output")
    
    def forward(self, input):
        # Configure different models per call
        with dspy.settings.context(lm=dspy.LM('openai/gpt-4o-mini')):
            quick = self.fast_predictor(input=input)
        
        with dspy.settings.context(lm=dspy.LM('openai/gpt-4o')):
            detailed = self.complex_reasoner(input=input)
        
        return {"quick": quick.output, "detailed": detailed.output}
```

---

## Best Practices

### ✅ Do:

1. **Use ChainOfThought for complex tasks**
   ```python
   # ✅ Better reasoning
   dspy.ChainOfThought("question -> answer")
   ```

2. **Compose modules for complex pipelines**
   ```python
   class Pipeline(dspy.Module):
       def __init__(self):
           super().__init__()
           self.step1 = dspy.ChainOfThought("...")
           self.step2 = dspy.Predict("...")
   ```

3. **Match module outputs to state fields**
   ```python
   class State(TypedDict):
       answer: str
   
   # Signature matches state
   dspy.ChainOfThought("question -> answer")
   ```

4. **Use ReAct for tool-using agents**
   ```python
   dspy.ReAct("question -> answer", tools=[...])
   ```

### ❌ Don't:

1. **Mix Predict and ChainOfThought unnecessarily**
   ```python
   # ❌ Inconsistent reasoning
   self.mod1 = dspy.Predict("q -> a")
   self.mod2 = dspy.ChainOfThought("q -> a")
   # Pick one strategy per pipeline
   ```

2. **Forget to initialize parent class**
   ```python
   class MyModule(dspy.Module):
       def __init__(self):
           # ❌ Missing super().__init__()
           self.predictor = dspy.Predict("...")
   ```

3. **Create deeply nested modules**
   ```python
   # ❌ Too complex
   class A(dspy.Module):
       def __init__(self):
           self.b = B()
           
   class B(dspy.Module):
       def __init__(self):
           self.c = C()
   # Keep it flat and readable
   ```

---

## Comparison Table

| Module | Use Case | Reasoning | Tool Use | Speed |
|--------|----------|-----------|----------|-------|
| **Predict** | Simple tasks | ❌ None | ❌ No | ⚡⚡⚡ Fast |
| **ChainOfThought** | Complex reasoning | ✅ Yes | ❌ No | ⚡⚡ Medium |
| **ReAct** | Tool-using agents | ✅ Yes | ✅ Yes | ⚡ Slow |
| **ProgramOfThought** | Math/code tasks | ✅ Yes (code) | ✅ Code exec | ⚡ Slow |
| **MultiChainComparison** | High-quality outputs | ✅ Multiple | ❌ No | 🐌 Very slow |

---

## Next Steps

- **[DSPy Optimizers](optimizers.md)** → Automatically improve your modules
- **[Best Practices](best-practices.md)** → Production tips
- **[Your First Agent](../../guides/first-agent.md)** → Build a complete agent

---

## External Resources

- **[DSPy Modules Docs](https://dspy.ai/docs/building-blocks/modules)** - Official guide
- **[DSPy Examples](https://github.com/stanfordnlp/dspy/tree/main/examples)** - Real-world module usage

---

**Next: Learn about [DSPy Optimizers →](optimizers.md)**
