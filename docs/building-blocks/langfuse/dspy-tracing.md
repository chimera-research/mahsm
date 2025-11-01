# DSPy Tracing with Langfuse

> **TL;DR**: `@ma.dspy_node` automatically traces all DSPy modules—no manual instrumentation needed!

## Automatic Tracing

mahsm automatically traces all DSPy modules decorated with `@ma.dspy_node`:

```python
import mahsm as ma
import dspy
import os

# 1. Configure DSPy
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)

# 2. Initialize tracing
ma.tracing.init()

# 3. Create module with @ma.dspy_node
@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

# 4. Run - automatically traced!
qa = QA()
result = qa(question="What is Python?")

# ✅ Everything traced in Langfuse:
# - Module execution
# - Input: question="What is Python?"
# - Output: answer="..."
# - All LLM calls
# - Token usage and costs
```

---

## What Gets Traced?

### Module Execution

```python
@ma.dspy_node
class Summarizer(ma.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought("text -> summary")
    
    def forward(self, text):
        return self.summarize(text=text)
```

**In Langfuse UI:**
```
Span: Summarizer
├─ Input: {text: "Long article..."}
├─ Output: {summary: "Brief summary"}
├─ Duration: 1.2s
└─ Generation: ChainOfThought
    ├─ Model: gpt-4o-mini
    ├─ Prompt: [Full prompt with reasoning instructions]
    ├─ Response: [Reasoning + summary]
    ├─ Tokens: 150 input, 50 output
    └─ Cost: $0.0012
```

### Multiple LLM Calls

```python
@ma.dspy_node
class Researcher(ma.Module):
    def __init__(self):
        super().__init__()
        self.generate_query = dspy.Predict("topic -> search_query")
        self.summarize = dspy.ChainOfThought("results -> summary")
    
    def forward(self, topic):
        # Both calls are traced separately
        query = self.generate_query(topic=topic)
        # ... search with query ...
        summary = self.summarize(results="search results")
        return {"query": query, "summary": summary}
```

**In Langfuse UI:**
```
Span: Researcher
├─ Input: {topic: "quantum physics"}
├─ Generation 1: Predict (generate_query)
│   ├─ Input: topic="quantum physics"
│   ├─ Output: search_query="quantum entanglement explained"
│   └─ Cost: $0.0005
├─ Generation 2: ChainOfThought (summarize)
│   ├─ Input: results="..."
│   ├─ Output: summary="..."
│   └─ Cost: $0.0015
└─ Output: {query: "...", summary: "..."}
```

---

## Tracing Patterns

### Simple Module

```python
@ma.dspy_node
class Classifier(ma.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict("text -> category")
    
    def forward(self, text):
        return self.classify(text=text)

# Use it
classifier = Classifier()
result = classifier(text="This is a programming tutorial")

# Traced automatically
```

### Chain of Modules

```python
@ma.dspy_node
class ExtractEntities(ma.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict("text -> entities: list")
    
    def forward(self, text):
        return self.extract(text=text)

@ma.dspy_node
class ClassifyEntities(ma.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict("entities: list -> categories: list")
    
    def forward(self, entities):
        return self.classify(entities=entities)

# Chain them
extract = ExtractEntities()
classify = ClassifyEntities()

entities = extract(text="Apple and Google are tech companies")
categories = classify(entities=entities)

# Both modules traced separately in Langfuse
```

### Nested Modules

```python
@ma.dspy_node
class SubModule(ma.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("input -> output")
    
    def forward(self, input):
        return self.predict(input=input)

@ma.dspy_node
class ParentModule(ma.Module):
    def __init__(self):
        super().__init__()
        self.sub = SubModule()
        self.final = dspy.Predict("output -> final")
    
    def forward(self, input):
        intermediate = self.sub(input=input)
        final = self.final(output=intermediate.output)
        return final

# Use parent
parent = ParentModule()
result = parent(input="test")

# Both modules traced in hierarchy
```

**In Langfuse UI:**
```
Span: ParentModule
├─ Span: SubModule
│   └─ Generation: Predict
└─ Generation: Predict (final)
```

---

## Common Module Types

### 1. ChainOfThought

```python
@ma.dspy_node
class ThinkingQA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

qa = ThinkingQA()
result = qa(question="Explain relativity")

# Langfuse shows reasoning process
```

### 2. ReAct

```python
@ma.dspy_node
class ReActAgent(ma.Module):
    def __init__(self, tools):
        super().__init__()
        self.react = dspy.ReAct("question -> answer", tools=tools)
    
    def forward(self, question):
        return self.react(question=question)

agent = ReActAgent(tools=[search_tool, calculator])
result = agent(question="What's 15% of the GDP of France?")

# Traces show:
# - Reasoning steps
# - Tool calls
# - Final answer
```

### 3. Multi-Stage Pipeline

```python
@ma.dspy_node
class Pipeline(ma.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = dspy.Predict("input -> intermediate")
        self.stage2 = dspy.ChainOfThought("intermediate -> output")
    
    def forward(self, input):
        intermediate = self.stage1(input=input)
        output = self.stage2(intermediate=intermediate.intermediate)
        return output

pipeline = Pipeline()
result = pipeline(input="raw data")

# Each stage traced separately
```

---

## Debugging with Traces

### Find Failures

```python
@ma.dspy_node
class ValidationQA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        result = self.qa(question=question)
        
        # Validation (also traced)
        if len(result.answer) < 10:
            raise ValueError("Answer too short!")
        
        return result

# Failed executions show up in Langfuse with error details
```

### Compare Prompts

Run the same module multiple times:

```python
qa = QA()

# Version 1
result1 = qa(question="What is AI?")

# Version 2 (after changing signature)
result2 = qa(question="What is AI?")

# Compare in Langfuse UI:
# - Different prompts
# - Different responses
# - Cost differences
```

### Analyze Costs

```python
@ma.dspy_node
class ExpensiveModule(ma.Module):
    def __init__(self):
        super().__init__()
        # Multiple LLM calls
        self.step1 = dspy.ChainOfThought("a -> b")
        self.step2 = dspy.ChainOfThought("b -> c")
        self.step3 = dspy.ChainOfThought("c -> d")
    
    def forward(self, a):
        b = self.step1(a=a)
        c = self.step2(b=b.b)
        d = self.step3(c=c.c)
        return d

# Check Langfuse to see:
# - Total cost per execution
# - Cost per step
# - Token usage breakdown
```

---

## Optimization with Tracing

### Before Optimization

```python
@ma.dspy_node
class SlowQA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        # Always uses ChainOfThought (slow & expensive)
        return self.qa(question=question)
```

**Langfuse shows:**
- Average latency: 2.5s
- Average cost: $0.005

### After Optimization

```python
@ma.dspy_node
class FastQA(ma.Module):
    def __init__(self):
        super().__init__()
        self.simple = dspy.Predict("question -> answer")
        self.complex = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        # Use simple prediction for easy questions
        if len(question.split()) < 10:
            return self.simple(question=question)
        else:
            return self.complex(question=question)
```

**Langfuse shows:**
- Average latency: 1.2s (52% faster!)
- Average cost: $0.002 (60% cheaper!)

---

## Best Practices

### ✅ Do:

1. **Always use @ma.dspy_node**
   ```python
   # ✅ Good - automatically traced
   @ma.dspy_node
   class MyModule(ma.Module):
       pass
   
   # ❌ Bad - not traced
   class MyModule(dspy.Module):
       pass
   ```

2. **Use descriptive class names**
   ```python
   # ✅ Good - clear in Langfuse UI
   @ma.dspy_node
   class ExtractKeyPoints(ma.Module):
       pass
   
   # ❌ Bad - unclear
   @ma.dspy_node
   class Module1(ma.Module):
       pass
   ```

3. **Keep modules focused**
   ```python
   # ✅ Good - single responsibility
   @ma.dspy_node
   class Summarizer(ma.Module):
       def forward(self, text):
           return self.summarize(text=text)
   
   # ❌ Bad - too much in one module
   @ma.dspy_node
   class DoEverything(ma.Module):
       def forward(self, input):
           # 100 lines of code...
           pass
   ```

4. **Review traces regularly**
   - Check daily for issues
   - Monitor costs
   - Find optimization opportunities

### ❌ Don't:

1. **Don't forget @ma.dspy_node**
   ```python
   # ❌ Missing decorator - no tracing!
   class QA(ma.Module):
       pass
   ```

2. **Don't trace sensitive data**
   ```python
   # ❌ Bad - PII in trace
   @ma.dspy_node
   class UserProfile(ma.Module):
       def forward(self, user_ssn, user_email):
           # This will be in Langfuse!
           pass
   ```

3. **Don't ignore failed traces**
   - Failed executions are valuable data
   - Review and fix issues

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

# Define modules
@ma.dspy_node
class QueryGenerator(ma.Module):
    """Generate a search query from a question."""
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict("question -> search_query")
    
    def forward(self, question):
        return self.generate(question=question)

@ma.dspy_node
class ResultSummarizer(ma.Module):
    """Summarize search results."""
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought("results -> summary")
    
    def forward(self, results):
        return self.summarize(results=results)

@ma.dspy_node
class AnswerGenerator(ma.Module):
    """Generate final answer from summary."""
    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought("question, summary -> answer")
    
    def forward(self, question, summary):
        return self.answer(question=question, summary=summary)

# Build pipeline
class ResearchPipeline:
    def __init__(self):
        self.query_gen = QueryGenerator()
        self.summarizer = ResultSummarizer()
        self.answerer = AnswerGenerator()
    
    def run(self, question):
        # Generate query
        query_result = self.query_gen(question=question)
        query = query_result.search_query
        
        # Mock search
        results = f"Search results for: {query}"
        
        # Summarize
        summary_result = self.summarizer(results=results)
        summary = summary_result.summary
        
        # Generate answer
        answer_result = self.answerer(question=question, summary=summary)
        
        return {
            "query": query,
            "summary": summary,
            "answer": answer_result.answer
        }

# Run
pipeline = ResearchPipeline()
result = pipeline.run("What is quantum computing?")

print(f"Query: {result['query']}")
print(f"Summary: {result['summary']}")
print(f"Answer: {result['answer']}")
print("\n✨ Check Langfuse UI to see the full trace!")
```

**In Langfuse UI you'll see:**
```
Trace: Research Pipeline
├─ Span: QueryGenerator
│   └─ Generation: Predict
│       ├─ Input: question="What is quantum computing?"
│       ├─ Output: search_query="quantum computing basics explained"
│       └─ Cost: $0.0003
├─ Span: ResultSummarizer
│   └─ Generation: ChainOfThought
│       ├─ Input: results="..."
│       ├─ Output: summary="..."
│       └─ Cost: $0.0012
└─ Span: AnswerGenerator
    └─ Generation: ChainOfThought
        ├─ Input: question="...", summary="..."
        ├─ Output: answer="..."
        └─ Cost: $0.0015
```

**Total cost:** $0.003  
**Total duration:** ~2.5s

---

## Next Steps

- **[LangGraph Tracing](langgraph-tracing.md)** → Trace LangGraph workflows
- **[Manual Tracing](manual-tracing.md)** → Add custom spans
- **[DSPy Best Practices](../dspy/best-practices.md)** → Production patterns

---

## External Resources

- **[Langfuse DSPy Integration](https://langfuse.com/docs/integrations/dspy)** - Official guide
- **[DSPy Tracing](https://dspy-docs.vercel.app/)** - DSPy docs

---

**Next: Trace LangGraph with [LangGraph Tracing →](langgraph-tracing.md)**
