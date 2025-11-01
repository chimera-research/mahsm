# Manual Tracing with Langfuse

> **TL;DR**: Use `@ma.tracing.observe()` to add custom spans for non-LLM operations—database queries, API calls, data processing, etc.

## When to Use Manual Tracing

Use manual tracing for operations that **aren't automatically traced**:

- ✅ Database queries
- ✅ API calls
- ✅ File I/O
- ✅ Data processing
- ✅ Validation logic
- ✅ Custom business logic

**Don't need manual tracing for:**
- ❌ DSPy modules (use `@ma.dspy_node`)
- ❌ LangGraph nodes (automatically traced)
- ❌ LLM calls (automatically captured)

---

## Basic Usage

### Simple Function

```python
import mahsm as ma

ma.tracing.init()

@ma.tracing.observe(name="fetch_user_data")
def fetch_user_data(user_id):
    # This function is now traced
    data = database.query(f"SELECT * FROM users WHERE id={user_id}")
    return data

# Call it
user = fetch_user_data(user_id=123)

# Appears in Langfuse as a span
```

**In Langfuse UI:**
```
Span: fetch_user_data
├─ Input: {user_id: 123}
├─ Output: {name: "John", email: "..."}
└─ Duration: 45ms
```

### With Parameters

```python
@ma.tracing.observe(name="process_data")
def process_data(data, options):
    # Process data
    result = transform(data, options)
    return result

result = process_data(data=[1,2,3], options={"mode": "fast"})
```

---

## Tracing Patterns

### 1. API Calls

```python
import requests

@ma.tracing.observe(name="fetch_weather")
def fetch_weather(city):
    response = requests.get(f"https://api.weather.com/{city}")
    return response.json()

weather = fetch_weather("San Francisco")
```

### 2. Database Queries

```python
import sqlite3

@ma.tracing.observe(name="query_products")
def query_products(category):
    conn = sqlite3.connect("products.db")
    cursor = conn.execute(
        "SELECT * FROM products WHERE category=?",
        (category,)
    )
    results = cursor.fetchall()
    conn.close()
    return results

products = query_products("electronics")
```

### 3. File Operations

```python
@ma.tracing.observe(name="load_config")
def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

config = load_config("config.json")
```

### 4. Data Processing

```python
@ma.tracing.observe(name="clean_data")
def clean_data(raw_data):
    # Clean and validate
    cleaned = [item for item in raw_data if item.get("valid")]
    return cleaned

@ma.tracing.observe(name="transform_data")
def transform_data(cleaned_data):
    # Transform
    transformed = [process(item) for item in cleaned_data]
    return transformed

# Both steps traced separately
raw = fetch_raw_data()
cleaned = clean_data(raw)
transformed = transform_data(cleaned)
```

---

## Nested Spans

### Hierarchical Operations

```python
@ma.tracing.observe(name="parent_operation")
def parent_operation(input_data):
    # This creates a parent span
    result1 = child_operation_1(input_data)
    result2 = child_operation_2(result1)
    return result2

@ma.tracing.observe(name="child_operation_1")
def child_operation_1(data):
    # Nested under parent
    return process(data)

@ma.tracing.observe(name="child_operation_2")
def child_operation_2(data):
    # Also nested under parent
    return finalize(data)

result = parent_operation(input)
```

**In Langfuse UI:**
```
Span: parent_operation
├─ Span: child_operation_1
│   └─ Duration: 100ms
└─ Span: child_operation_2
    └─ Duration: 150ms
```

---

## Combining with DSPy & LangGraph

### Pipeline with Mixed Operations

```python
import mahsm as ma
import dspy
import os

# Configure
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
ma.tracing.init()

# Manual span for data fetching
@ma.tracing.observe(name="fetch_context")
def fetch_context(topic):
    # API call or database query
    return database.get_context(topic)

# DSPy module (automatic tracing)
@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, context, question):
        return self.qa(context=context, question=question)

# Manual span for post-processing
@ma.tracing.observe(name="format_answer")
def format_answer(raw_answer):
    # Format and validate
    return {
        "answer": raw_answer,
        "formatted": True,
        "timestamp": time.time()
    }

# Run pipeline - all steps traced
context = fetch_context("Python programming")
qa = QA()
raw_result = qa(context=context, question="What is Python?")
final = format_answer(raw_result.answer)
```

**In Langfuse UI:**
```
Trace: Complete Pipeline
├─ Span: fetch_context
│   └─ Duration: 45ms
├─ Span: QA (DSPy Module)
│   └─ Generation: ChainOfThought
│       └─ Cost: $0.0012
└─ Span: format_answer
    └─ Duration: 2ms
```

---

## Advanced Usage

### Custom Metadata

```python
@ma.tracing.observe(
    name="process_image",
    metadata={"version": "2.0", "model": "resnet50"}
)
def process_image(image_path):
    # Process image
    result = model.predict(image_path)
    return result
```

### Error Handling

```python
@ma.tracing.observe(name="risky_operation")
def risky_operation(data):
    try:
        result = process(data)
        return result
    except Exception as e:
        # Error is automatically captured in trace
        raise

# Failed executions show up in Langfuse with error details
```

### Conditional Tracing

```python
def process_data(data, trace=True):
    if trace:
        return _process_data_traced(data)
    else:
        return _process_data_untraced(data)

@ma.tracing.observe(name="process_data")
def _process_data_traced(data):
    return transform(data)

def _process_data_untraced(data):
    return transform(data)
```

---

## Real-World Example

```python
import mahsm as ma
from typing import TypedDict, Optional
import dspy
import os
import requests
import json

# Configure
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)
ma.tracing.init()

# Manual tracing for data operations
@ma.tracing.observe(name="fetch_stock_data")
def fetch_stock_data(symbol):
    \"\"\"Fetch stock data from API.\"\"\"
    response = requests.get(f"https://api.example.com/stocks/{symbol}")
    return response.json()

@ma.tracing.observe(name="calculate_metrics")
def calculate_metrics(stock_data):
    \"\"\"Calculate financial metrics.\"\"\"
    return {
        "avg_price": sum(stock_data["prices"]) / len(stock_data["prices"]),
        "volatility": calculate_volatility(stock_data["prices"]),
        "trend": "up" if stock_data["prices"][-1] > stock_data["prices"][0] else "down"
    }

# DSPy module for analysis
@ma.dspy_node
class StockAnalyzer(ma.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(
            "stock_symbol, metrics -> analysis, recommendation"
        )
    
    def forward(self, stock_symbol, metrics):
        return self.analyze(
            stock_symbol=stock_symbol,
            metrics=json.dumps(metrics)
        )

# Manual tracing for saving results
@ma.tracing.observe(name="save_analysis")
def save_analysis(symbol, analysis, recommendation):
    \"\"\"Save analysis to database.\"\"\"
    database.insert({
        "symbol": symbol,
        "analysis": analysis,
        "recommendation": recommendation,
        "timestamp": time.time()
    })
    return True

# Complete pipeline
def analyze_stock(symbol):
    \"\"\"Full stock analysis pipeline.\"\"\"
    # Fetch data (traced)
    stock_data = fetch_stock_data(symbol)
    
    # Calculate metrics (traced)
    metrics = calculate_metrics(stock_data)
    
    # AI analysis (traced via DSPy)
    analyzer = StockAnalyzer()
    result = analyzer(stock_symbol=symbol, metrics=metrics)
    
    # Save results (traced)
    save_analysis(
        symbol=symbol,
        analysis=result.analysis,
        recommendation=result.recommendation
    )
    
    return {
        "symbol": symbol,
        "metrics": metrics,
        "analysis": result.analysis,
        "recommendation": result.recommendation
    }

# Run
result = analyze_stock("AAPL")
print(f"Analysis: {result['analysis']}")
print(f"Recommendation: {result['recommendation']}")
print("\n✨ Check Langfuse UI to see the complete trace!")
```

**In Langfuse UI:**
```
Trace: Stock Analysis Pipeline
├─ Span: fetch_stock_data
│   ├─ Input: {symbol: "AAPL"}
│   ├─ Output: {prices: [...], volume: [...]}
│   └─ Duration: 234ms
├─ Span: calculate_metrics
│   ├─ Input: {prices: [...]}
│   ├─ Output: {avg_price: 150.23, volatility: 0.15, trend: "up"}
│   └─ Duration: 5ms
├─ Span: StockAnalyzer (DSPy Module)
│   └─ Generation: ChainOfThought
│       ├─ Input: stock_symbol="AAPL", metrics="..."
│       ├─ Output: analysis="...", recommendation="..."
│       └─ Cost: $0.0025
└─ Span: save_analysis
    ├─ Input: {symbol: "AAPL", analysis: "...", recommendation: "..."}
    ├─ Output: true
    └─ Duration: 12ms
```

**Total duration:** ~250ms  
**Total cost:** $0.0025

---

## Best Practices

### ✅ Do:

1. **Use descriptive names**
   ```python
   # ✅ Good - clear purpose
   @ma.tracing.observe(name="validate_user_input")
   def validate_input(data):
       pass
   
   # ❌ Bad - vague
   @ma.tracing.observe(name="func1")
   def func1(data):
       pass
   ```

2. **Trace expensive operations**
   ```python
   # ✅ Trace slow operations
   @ma.tracing.observe(name="complex_calculation")
   def complex_calc(data):
       # Time-consuming operation
       pass
   ```

3. **Keep spans focused**
   ```python
   # ✅ Good - single responsibility
   @ma.tracing.observe(name="fetch_data")
   def fetch_data():
       return database.query(...)
   
   @ma.tracing.observe(name="process_data")
   def process_data(data):
       return transform(data)
   
   # ❌ Bad - too much in one span
   @ma.tracing.observe(name="do_everything")
   def do_everything():
       data = fetch()
       processed = transform(data)
       saved = save(processed)
       return saved
   ```

4. **Add metadata for context**
   ```python
   # ✅ Good - helpful metadata
   @ma.tracing.observe(
       name="process_image",
       metadata={"model": "v2.0", "size": "large"}
   )
   def process_image(img):
       pass
   ```

### ❌ Don't:

1. **Don't trace trivial operations**
   ```python
   # ❌ Bad - too simple to trace
   @ma.tracing.observe(name="add_numbers")
   def add(a, b):
       return a + b
   ```

2. **Don't trace in tight loops**
   ```python
   # ❌ Bad - creates too many spans
   for item in items:
       @ma.tracing.observe(name="process_item")
       def process_item(item):
           return item * 2
       process_item(item)
   
   # ✅ Good - trace the loop, not each iteration
   @ma.tracing.observe(name="process_all_items")
   def process_all_items(items):
       return [item * 2 for item in items]
   ```

3. **Don't trace sensitive operations**
   ```python
   # ❌ Bad - sensitive data in trace
   @ma.tracing.observe(name="process_payment")
   def process_payment(credit_card_number, cvv):
       # This will be in Langfuse!
       pass
   ```

---

## Next Steps

Congratulations! You've completed the Langfuse Building Blocks section.

- **[DSPy Overview](../dspy/overview.md)** → Review DSPy concepts
- **[LangGraph Overview](../langgraph/overview.md)** → Review LangGraph concepts
- **[Build Your First Agent](../../guides/first-agent.md)** → Put everything together

---

## External Resources

- **[Langfuse Python SDK](https://langfuse.com/docs/sdk/python)** - Full SDK docs
- **[Langfuse Decorators](https://langfuse.com/docs/sdk/python/decorators)** - Decorator guide
- **[Custom Metadata](https://langfuse.com/docs/tracing/metadata)** - Metadata options

---

**You've completed Langfuse!** 🎉 Ready to build? **[Start Here →](../../guides/first-agent.md)**
