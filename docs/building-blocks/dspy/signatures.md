# DSPy Signatures

> **TL;DR**: Signatures are type specifications that tell DSPy what inputs your module needs and what outputs it should produce.

## What is a Signature?

A **signature** in DSPy is like a function type hint—it specifies:
- What **inputs** the module receives
- What **outputs** it should produce
- Optional **descriptions** for each field

Think of it as a contract between your code and the LLM.

---

## Basic Syntax

### String Signatures

The simplest way to define a signature:

```python
import dspy

# Single input → single output
"question -> answer"

# Multiple inputs → single output
"question, context -> answer"

# Multiple inputs → multiple outputs
"question, context -> answer, confidence"
```

**Format**: `input1, input2 -> output1, output2`

### Example

```python
# Create a predictor with a signature
qa = dspy.ChainOfThought("question -> answer")

# Use it
result = qa(question="What is DSPy?")
print(result.answer)  # Access output by name
```

---

## Adding Descriptions

You can add hints to guide the LLM:

```python
# Add output description
"question -> answer: a concise, factual response"

# Add input descriptions
"question: a technical question -> answer: a detailed explanation"

# Multiple fields with descriptions
"question: user query, context: relevant docs -> answer: synthesized response, sources: list of citations"
```

### Example with Descriptions

```python
predictor = dspy.ChainOfThought(
    "question: a user's question about AI -> answer: a detailed, technical explanation"
)

result = predictor(question="How does attention work in transformers?")
print(result.answer)
# Output will be more detailed and technical due to the hint
```

---

## Class-Based Signatures

For complex signatures, use classes:

```python
import dspy

class QASignature(dspy.Signature):
    """Answer questions with detailed explanations."""
    
    question = dspy.InputField(desc="The user's question")
    context = dspy.InputField(desc="Relevant background information")
    answer = dspy.OutputField(desc="A comprehensive answer")
    confidence = dspy.OutputField(desc="Confidence score (0-100)")
```

### Using Class Signatures

```python
# Pass the class (not an instance!)
predictor = dspy.ChainOfThought(QASignature)

result = predictor(
    question="What is DSPy?",
    context="DSPy is a framework for prompt programming..."
)
print(result.answer)
print(result.confidence)
```

---

## Input and Output Fields

### InputField

Defines what the module receives:

```python
import dspy

class MySignature(dspy.Signature):
    # Basic input
    query = dspy.InputField()
    
    # With description
    context = dspy.InputField(desc="Background information")
    
    # With format hint
    examples = dspy.InputField(desc="Few-shot examples", format=list)
```

### OutputField

Defines what the module produces:

```python
class MySignature(dspy.Signature):
    # Basic output
    answer = dspy.OutputField()
    
    # With description
    reasoning = dspy.OutputField(desc="Step-by-step thought process")
    
    # With prefix (shown before the output in the prompt)
    summary = dspy.OutputField(prefix="SUMMARY:")
```

---

## Common Patterns

### 1. **Simple Q&A**

```python
"question -> answer"
```

### 2. **Context-Aware Q&A**

```python
"question, context -> answer"
```

### 3. **Multi-Output**

```python
"document -> summary, key_points, sentiment"
```

### 4. **Complex Reasoning**

```python
class ReasoningSignature(dspy.Signature):
    """Solve complex problems with step-by-step reasoning."""
    
    problem = dspy.InputField(desc="The problem to solve")
    constraints = dspy.InputField(desc="Any constraints or requirements")
    
    reasoning = dspy.OutputField(desc="Step-by-step thought process")
    solution = dspy.OutputField(desc="The final solution")
    confidence = dspy.OutputField(desc="Confidence level (low/medium/high)")
```

### 5. **Classification**

```python
class ClassificationSignature(dspy.Signature):
    """Classify text into categories."""
    
    text = dspy.InputField(desc="The text to classify")
    categories = dspy.InputField(desc="Valid categories (comma-separated)")
    
    category = dspy.OutputField(desc="The chosen category")
    reason = dspy.OutputField(desc="Brief explanation for the choice")
```

---

## Signatures in mahsm

When using `@dspy_node`, signatures determine how inputs are extracted from state:

```python
import mahsm as ma
from typing import TypedDict

# 1. Define state
class ResearchState(TypedDict):
    question: str
    context: str
    answer: str
    reasoning: str

# 2. Create module with signature
@ma.dspy_node
class Researcher(ma.Module):
    def __init__(self):
        super().__init__()
        # Signature matches state fields
        self.research = dspy.ChainOfThought("question, context -> answer, reasoning")
    
    def forward(self, question, context):
        return self.research(question=question, context=context)

# 3. Use in workflow
workflow = ma.graph.StateGraph(ResearchState)
workflow.add_node("researcher", Researcher())

# When the node runs:
# - "question" and "context" are extracted from state
# - "answer" and "reasoning" are written back to state
```

**Key Point**: Match your signature field names to your state keys for seamless integration!

---

## Advanced: Dynamic Signatures

Create signatures programmatically:

```python
def create_signature(input_fields, output_fields):
    inputs = ", ".join(input_fields)
    outputs = ", ".join(output_fields)
    return f"{inputs} -> {outputs}"

# Example: Dynamic fields
sig = create_signature(["question", "context"], ["answer", "score"])
# Result: "question, context -> answer, score"

predictor = dspy.Predict(sig)
```

---

## Best Practices

### ✅ Do:

1. **Use descriptive field names**
   ```python
   "user_question -> detailed_answer"  # ✅ Clear
   ```

2. **Add descriptions for ambiguous fields**
   ```python
   "query: the user's search query -> results: list of relevant items"
   ```

3. **Match state keys in mahsm**
   ```python
   class State(TypedDict):
       question: str
       answer: str
   
   # Signature matches state
   dspy.ChainOfThought("question -> answer")
   ```

4. **Use multi-output for intermediate reasoning**
   ```python
   "question -> reasoning, answer"  # ✅ Captures thought process
   ```

### ❌ Don't:

1. **Use vague names**
   ```python
   "input -> output"  # ❌ Not descriptive
   ```

2. **Mix concerns in one field**
   ```python
   "query -> answer_and_confidence"  # ❌ Split into two outputs
   ```

3. **Over-complicate**
   ```python
   # ❌ Too many fields
   "q, c1, c2, c3, c4 -> a, r1, r2, r3, conf, meta"
   ```

---

## Troubleshooting

### Issue: LLM not returning expected output

**Solution**: Add more specific descriptions

```python
# Before (vague)
"text -> category"

# After (specific)
"text: a customer review -> category: one of [positive, negative, neutral]"
```

### Issue: Output format is inconsistent

**Solution**: Use structured output hints

```python
class StructuredSignature(dspy.Signature):
    query = dspy.InputField()
    answer = dspy.OutputField(desc="Answer in JSON format with keys: summary, details")
```

### Issue: State keys don't match signature

**Solution**: Ensure field names align

```python
# State has "user_query"
class State(TypedDict):
    user_query: str

# ❌ Signature uses "question"
dspy.ChainOfThought("question -> answer")  # Won't find "question" in state!

# ✅ Match the key
dspy.ChainOfThought("user_query -> answer")
```

---

## Next Steps

- **[DSPy Modules](modules.md)** → Learn about Predict, ChainOfThought, ReAct
- **[Your First Agent](../../guides/first-agent.md)** → Build a complete agent
- **[API Reference: @dspy_node](../../api/core.md)** → mahsm integration details

---

## External Resources

- **[DSPy Signatures Documentation](https://dspy.ai/docs/building-blocks/signatures)** - Official guide
- **[DSPy Examples](https://github.com/stanfordnlp/dspy/tree/main/examples)** - Real-world signature usage

---

**Next: Explore [DSPy Modules →](modules.md)**
