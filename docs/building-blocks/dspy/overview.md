# DSPy Overview

> **TL;DR**: DSPy turns prompt engineering into programming—define what you want, not how to prompt for it.

## What is DSPy?

**DSPy** (Declarative Self-improving Language Programs in Python) is a framework for building LLM applications through **programming**, not manual prompting.

Instead of writing and tweaking prompts like this:

```python
# ❌ Traditional prompting
prompt = """
You are a helpful assistant. Given a question, provide a detailed answer.

Question: {question}
Think step by step and provide your reasoning.

Answer:
"""
response = llm.complete(prompt.format(question="What is DSPy?"))
```

You write code like this:

```python
# ✅ DSPy approach
import dspy

class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.cot(question=question)

qa = QA()
result = qa(question="What is DSPy?")
print(result.answer)
```

**Key Insight**: You declare the **structure** (chain-of-thought reasoning), and DSPy generates the actual prompts automatically.

---

## Why DSPy?

### 1. **Composability**

Build complex pipelines from simple components:

```python
class ResearchPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_query = dspy.ChainOfThought("question -> search_query")
        self.synthesize = dspy.ChainOfThought("question, context -> answer")
    
    def forward(self, question):
        # Step 1: Generate search query
        query_result = self.generate_query(question=question)
        
        # Step 2: Search (simulated)
        context = search_api(query_result.search_query)
        
        # Step 3: Synthesize answer
        return self.synthesize(question=question, context=context)
```

### 2. **Automatic Optimization**

DSPy can automatically improve your prompts:

```python
from dspy.teleprompt import BootstrapFewShot

# Define success metric
def validate_answer(example, prediction):
    return example.answer.lower() in prediction.answer.lower()

# Optimize the pipeline
optimizer = BootstrapFewShot(metric=validate_answer)
optimized_qa = optimizer.compile(QA(), trainset=examples)

# optimized_qa now has better prompts learned from examples!
```

### 3. **Model Agnostic**

Switch between models without changing code:

```python
# Use GPT-4o-mini
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)

# Later, switch to Claude
lm = dspy.LM('anthropic/claude-3-5-sonnet-20241022', api_key=os.getenv("ANTHROPIC_API_KEY"))
dspy.configure(lm=lm)
# Your code stays the same!
```

---

## Core Concepts

### 1. **Signatures**

Signatures define input → output specifications:

```python
# Simple signature
"question -> answer"

# Multi-input signature
"question, context -> answer"

# With hints
"question -> answer: a detailed, technical response"
```

**[Learn more about Signatures →](signatures.md)**

### 2. **Modules**

Modules are reusable components that use signatures:

```python
# Built-in modules
dspy.Predict("question -> answer")          # Basic prediction
dspy.ChainOfThought("question -> answer")   # With reasoning
dspy.ReAct("question -> answer")            # Tool-using agent

# Custom modules
class MyAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("input -> output")
    
    def forward(self, input):
        return self.predictor(input=input)
```

**[Learn more about Modules →](modules.md)**

### 3. **Optimizers (Teleprompts)**

Optimizers automatically improve your modules:

```python
from dspy.teleprompt import BootstrapFewShot, MIPRO

# Few-shot learning
optimizer = BootstrapFewShot(metric=my_metric)
optimized = optimizer.compile(my_module, trainset=data)

# Advanced optimization
optimizer = MIPRO(metric=my_metric)
optimized = optimizer.compile(my_module, trainset=train, valset=val)
```

**[Learn more about Optimizers →](optimizers.md)**

---

## DSPy in mahsm

mahsm makes DSPy even easier by integrating it with LangGraph workflows:

### The `@dspy_node` Decorator

Convert any DSPy module into a LangGraph node:

```python
import mahsm as ma

@ma.dspy_node
class Researcher(ma.Module):
    def __init__(self):
        super().__init__()
        self.research = dspy.ChainOfThought("question -> findings")
    
    def forward(self, question):
        return self.research(question=question)

# Use it in a workflow
workflow = ma.graph.StateGraph(MyState)
workflow.add_node("researcher", Researcher())  # ✅ Works seamlessly
```

**How it works:**
1. `@dspy_node` wraps your DSPy module
2. Automatically extracts inputs from state
3. Merges outputs back into state
4. Handles Langfuse tracing

**[Learn more about @dspy_node →](../../api/core.md)**

---

## Quick Example: Building a Q&A Agent

Let's build a complete Q&A agent using DSPy + mahsm:

```python
import mahsm as ma
from typing import TypedDict
import dspy
import os

# 1. Configure DSPy
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
ma.tracing.init()

# 2. Define state
class QAState(TypedDict):
    question: str
    reasoning: str
    answer: str

# 3. Create DSPy module
@ma.dspy_node
class QAAgent(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> reasoning, answer")
    
    def forward(self, question):
        return self.qa(question=question)

# 4. Build LangGraph workflow
workflow = ma.graph.StateGraph(QAState)
workflow.add_node("qa", QAAgent())
workflow.add_edge(ma.START, "qa")
workflow.add_edge("qa", ma.END)
graph = workflow.compile()

# 5. Run
result = graph.invoke({"question": "What are the benefits of using DSPy?"})
print(f"Answer: {result['answer']}")
print(f"Reasoning: {result['reasoning']}")
# ✅ Automatically traced in Langfuse!
```

---

## When to Use DSPy

### ✅ Great For:
- **Complex reasoning tasks** requiring chain-of-thought
- **Multi-step pipelines** with intermediate outputs
- **Optimizable systems** where you can measure success
- **Model-agnostic applications** that need portability

### ❌ Not Ideal For:
- **Simple one-shot prompts** (just use the LLM API directly)
- **When you need exact prompt control** (DSPy generates prompts)
- **Streaming responses** with partial updates (DSPy is batch-oriented)

---

## Common Patterns

### 1. **Sequential Processing**

```python
class Pipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step1 = dspy.ChainOfThought("input -> intermediate")
        self.step2 = dspy.ChainOfThought("intermediate -> output")
    
    def forward(self, input):
        intermediate = self.step1(input=input)
        return self.step2(intermediate=intermediate.intermediate)
```

### 2. **Conditional Logic**

```python
class ConditionalAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict("question -> category")
        self.tech_expert = dspy.ChainOfThought("question -> answer")
        self.general_expert = dspy.Predict("question -> answer")
    
    def forward(self, question):
        category = self.classifier(question=question).category
        
        if "technical" in category.lower():
            return self.tech_expert(question=question)
        else:
            return self.general_expert(question=question)
```

### 3. **Tool Use with ReAct**

```python
class ToolUser(dspy.Module):
    def __init__(self, tools):
        super().__init__()
        self.react = dspy.ReAct("question -> answer")
        self.react.tools = tools
    
    def forward(self, question):
        return self.react(question=question)
```

---

## Next Steps

- **[DSPy Signatures](signatures.md)** → Learn how to define inputs and outputs
- **[DSPy Modules](modules.md)** → Explore built-in modules like ChainOfThought, ReAct
- **[DSPy Optimizers](optimizers.md)** → Automatically improve your prompts
- **[Best Practices](best-practices.md)** → Tips for production DSPy code

---

## External Resources

- **[Official DSPy Docs](https://dspy.ai/)** - Comprehensive DSPy documentation
- **[DSPy GitHub](https://github.com/stanfordnlp/dspy)** - Source code and examples
- **[DSPy Paper](https://arxiv.org/abs/2310.03714)** - Research paper explaining DSPy

---

**Ready to dive deeper? Start with [Signatures →](signatures.md)**
