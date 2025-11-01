# DSPy Best Practices

> **TL;DR**: Production-ready patterns for building robust, maintainable DSPy applications with mahsm.

## Module Design

### ✅ Keep Modules Focused

Each module should have a single, clear responsibility:

```python
# ✅ Good: Focused modules
class QueryGenerator(dspy.Module):
    """Only generates search queries."""
    def __init__(self):
        super().__init__()
        self.gen = dspy.ChainOfThought("question -> search_query")

class ResultSynthesizer(dspy.Module):
    """Only synthesizes results."""
    def __init__(self):
        super().__init__()
        self.synth = dspy.ChainOfThought("question, results -> answer")

# ❌ Bad: Does too much
class MegaModule(dspy.Module):
    """Tries to do everything."""
    def __init__(self):
        super().__init__()
        self.do_everything = dspy.ChainOfThought("anything -> everything")
```

### ✅ Use Descriptive Signatures

Make your signatures self-documenting:

```python
# ✅ Good: Clear and descriptive
"user_question, search_results -> synthesized_answer, confidence_score"

# ❌ Bad: Vague
"input -> output"
```

### ✅ Match Signatures to State

When using `@ma.dspy_node`, align signature fields with state keys:

```python
from typing import TypedDict, Optional

class State(TypedDict):
    question: str
    answer: Optional[str]
    confidence: Optional[float]

# ✅ Signature matches state exactly
@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer, confidence")
```

---

## Error Handling

### ✅ Validate Outputs

Always validate LLM outputs before using them:

```python
import dspy

class SafeQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        result = self.qa(question=question)
        
        # Validate output
        if not result.answer or len(result.answer) < 10:
            # Fallback or retry logic
            return {"answer": "I don't have enough information to answer that."}
        
        return {"answer": result.answer}
```

### ✅ Handle API Failures

Wrap LLM calls with error handling:

```python
import dspy
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def forward(self, question):
        try:
            result = self.qa(question=question)
            return {"answer": result.answer}
        except Exception as e:
            print(f"Error: {e}")
            # Log to monitoring system
            raise
```

---

## Performance Optimization

### ✅ Use Appropriate Module Types

Choose the right module for the task:

```python
# ✅ Simple tasks → Predict (fast)
classifier = dspy.Predict("text -> category")

# ✅ Complex reasoning → ChainOfThought (better quality)
reasoner = dspy.ChainOfThought("problem -> solution")

# ✅ Tool use → ReAct (agentic)
agent = dspy.ReAct("question -> answer", tools=tools)
```

### ✅ Cache Expensive Operations

Cache module compilations and optimizations:

```python
import dspy
from functools import lru_cache

@lru_cache(maxsize=1)
def get_optimized_qa():
    """Cache the optimized module."""
    qa = QA()
    
    # Check if we have a saved version
    try:
        qa.load("optimized_qa.json")
        return qa
    except FileNotFoundError:
        # Optimize and save
        optimizer = BootstrapFewShot(metric=accuracy)
        optimized = optimizer.compile(qa, trainset=trainset)
        optimized.save("optimized_qa.json")
        return optimized

# Use cached version
qa_module = get_optimized_qa()
```

### ✅ Batch When Possible

Process multiple inputs together:

```python
class BatchQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward_batch(self, questions):
        """Process multiple questions efficiently."""
        # DSPy handles batching internally
        results = [self.qa(question=q) for q in questions]
        return [r.answer for r in results]
```

---

## Testing

### ✅ Unit Test Your Modules

Test modules independently:

```python
import unittest
import dspy

class TestQA(unittest.TestCase):
    def setUp(self):
        # Use a mock LM for testing
        lm = dspy.LM('openai/gpt-4o-mini', api_key="test")
        dspy.configure(lm=lm)
        self.qa = QA()
    
    def test_qa_returns_answer(self):
        result = self.qa(question="What is 2+2?")
        self.assertIn("answer", result)
        self.assertIsInstance(result["answer"], str)
    
    def test_qa_handles_empty_question(self):
        result = self.qa(question="")
        # Should handle gracefully
        self.assertIsNotNone(result["answer"])
```

### ✅ Test with Real Examples

Create a test dataset:

```python
import dspy

test_cases = [
    dspy.Example(
        question="What is the capital of France?",
        expected_answer="Paris"
    ).with_inputs("question"),
    
    dspy.Example(
        question="What is 2+2?",
        expected_answer="4"
    ).with_inputs("question"),
]

def test_accuracy():
    qa = QA()
    correct = 0
    
    for case in test_cases:
        result = qa(question=case.question)
        if case.expected_answer.lower() in result["answer"].lower():
            correct += 1
    
    accuracy = correct / len(test_cases)
    assert accuracy >= 0.8, f"Accuracy too low: {accuracy}"
```

---

## Configuration Management

### ✅ Use Environment Variables

Never hardcode API keys or configuration:

```python
import os
import dspy

# ✅ Good: Environment variables
lm = dspy.LM(
    'openai/gpt-4o-mini',
    api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
    temperature=float(os.getenv("TEMPERATURE", "0.7"))
)
dspy.configure(lm=lm)

# ❌ Bad: Hardcoded
# lm = dspy.LM('openai/gpt-4o-mini', api_key="sk-...")
```

### ✅ Create Configuration Classes

Organize configuration:

```python
from dataclasses import dataclass
import os

@dataclass
class DSPyConfig:
    model: str = "openai/gpt-4o-mini"
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    max_tokens: int = 2000
    temperature: float = 0.7
    
    @classmethod
    def from_env(cls):
        return cls(
            model=os.getenv("DSPY_MODEL", "openai/gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
            temperature=float(os.getenv("TEMPERATURE", "0.7"))
        )

# Use it
config = DSPyConfig.from_env()
lm = dspy.LM(config.model, api_key=config.api_key, max_tokens=config.max_tokens)
dspy.configure(lm=lm)
```

---

## Monitoring & Observability

### ✅ Always Enable Tracing

Use mahsm's tracing in production:

```python
import mahsm as ma

# Enable at application startup
ma.tracing.init()

# All DSPy calls are now traced to Langfuse!
```

### ✅ Add Custom Metrics

Track custom metrics with Langfuse:

```python
from langfuse.decorators import observe

@observe(name="qa_pipeline")
def run_qa(question: str):
    result = qa_module(question=question)
    
    # Log custom metrics
    from langfuse import Langfuse
    client = Langfuse()
    client.score(
        name="answer_length",
        value=len(result["answer"])
    )
    
    return result
```

### ✅ Log Important Events

Use structured logging:

```python
import logging
import dspy

logger = logging.getLogger(__name__)

class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        logger.info(f"Processing question: {question[:50]}...")
        
        try:
            result = self.qa(question=question)
            logger.info(f"Generated answer of length {len(result.answer)}")
            return {"answer": result.answer}
        except Exception as e:
            logger.error(f"Failed to answer: {e}", exc_info=True)
            raise
```

---

## Prompt Engineering

### ✅ Use ChainOfThought for Complex Tasks

Enable reasoning for better outputs:

```python
# ✅ Good for complex tasks
reasoner = dspy.ChainOfThought("problem -> solution")

# ❌ Bad for complex tasks (no reasoning)
predictor = dspy.Predict("problem -> solution")
```

### ✅ Add Context to Signatures

Guide the model with descriptions:

```python
# ✅ Good: Descriptive
signature = "question: a user's technical question -> answer: a detailed, accurate response with examples"

# ❌ Bad: Vague
signature = "question -> answer"
```

### ✅ Use Few-Shot Examples

Optimize with BootstrapFewShot:

```python
from dspy.teleprompt import BootstrapFewShot

qa = QA()
optimizer = BootstrapFewShot(metric=accuracy, max_bootstrapped_demos=4)
optimized_qa = optimizer.compile(qa, trainset=examples)

# optimized_qa now includes learned examples
```

---

## Code Organization

### ✅ Separate Concerns

Organize code into modules:

```
my_project/
├── config/
│   └── dspy_config.py       # Configuration
├── modules/
│   ├── __init__.py
│   ├── qa.py                # QA module
│   ├── summarizer.py        # Summarizer module
│   └── classifier.py        # Classifier module
├── workflows/
│   ├── __init__.py
│   └── research_pipeline.py # LangGraph workflows
├── utils/
│   ├── __init__.py
│   └── metrics.py           # Evaluation metrics
└── main.py                  # Application entry point
```

### ✅ Use Type Hints

Make code maintainable with types:

```python
from typing import Dict, Any, Optional
import dspy

class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question: str) -> Dict[str, Any]:
        """
        Answer a question.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with 'answer' key
        """
        result = self.qa(question=question)
        return {"answer": result.answer}
```

---

## Deployment

### ✅ Use Versioned Artifacts

Save and version optimized modules:

```python
import dspy
from datetime import datetime

def save_optimized_module(module, version: str = None):
    """Save module with timestamp."""
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"optimized_qa_{version}.json"
    module.save(filename)
    print(f"Saved to {filename}")
    return filename

# Usage
optimized_qa = optimizer.compile(qa, trainset=train)
save_optimized_module(optimized_qa, version="v1.0.0")
```

### ✅ Implement Health Checks

Monitor service health:

```python
from fastapi import FastAPI, HTTPException
import dspy

app = FastAPI()

@app.get("/health")
def health_check():
    """Check if DSPy is configured correctly."""
    try:
        # Test configuration
        lm = dspy.settings.lm
        if lm is None:
            raise Exception("DSPy not configured")
        
        return {"status": "healthy", "model": str(lm)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### ✅ Handle Rate Limits

Implement backoff strategies:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError

class RateLimitedQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60)
    )
    def forward(self, question):
        return self.qa(question=question)
```

---

## Common Pitfalls

### ❌ Forgetting super().__init__()

```python
# ❌ Will break
class MyModule(dspy.Module):
    def __init__(self):
        # Missing super().__init__()!
        self.predictor = dspy.Predict("input -> output")

# ✅ Correct
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()  # Always call this!
        self.predictor = dspy.Predict("input -> output")
```

### ❌ Not Configuring DSPy

```python
# ❌ Will error
predictor = dspy.Predict("question -> answer")
result = predictor(question="Hello")  # Error: DSPy not configured!

# ✅ Configure first
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
predictor = dspy.Predict("question -> answer")
result = predictor(question="Hello")  # Works!
```

### ❌ Ignoring Optimization

```python
# ❌ Using unoptimized modules in production
qa = QA()
# Likely suboptimal performance

# ✅ Optimize first
optimizer = BootstrapFewShot(metric=accuracy)
optimized_qa = optimizer.compile(qa, trainset=train)
# Much better performance!
```

---

## Checklist for Production

Before deploying to production:

- [ ] All modules have `super().__init__()` calls
- [ ] DSPy is configured with proper LM
- [ ] Tracing is enabled (`ma.tracing.init()`)
- [ ] API keys are in environment variables
- [ ] Modules are optimized with real data
- [ ] Error handling is implemented
- [ ] Logging is configured
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance is acceptable (latency, cost)
- [ ] Rate limiting/backoff is handled
- [ ] Health checks are implemented
- [ ] Monitoring/alerts are set up

---

## Next Steps

- **[LangGraph Overview](../langgraph/overview.md)** → Build stateful workflows
- **[Production Deployment Guide](../../guides/production-deployment.md)** → Deploy your app
- **[Examples](../../examples/research-agent.md)** → See complete applications

---

## External Resources

- **[DSPy Documentation](https://dspy.ai/)** - Official docs
- **[Langfuse Documentation](https://langfuse.com/docs)** - Tracing & monitoring
- **[DSPy GitHub](https://github.com/stanfordnlp/dspy)** - Source code & examples

---

**Ready for production? Check out [Langfuse →](../langfuse/overview.md)**
