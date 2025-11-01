# Langfuse Initialization

> **TL;DR**: Initialize Langfuse tracing with `ma.tracing.init()` once at startup to enable automatic observability.

## Quick Start

### Minimal Setup

```python
import mahsm as ma

# Initialize with environment variables
ma.tracing.init()

# That's it! Now all LLM calls are traced
```

Requires these environment variables:
```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
```

---

## Getting API Keys

### Option 1: Langfuse Cloud (Recommended)

1. **Sign up** at [cloud.langfuse.com](https://cloud.langfuse.com)
2. **Create a project**
3. **Copy API keys** from project settings

```python
ma.tracing.init(
    public_key="pk-lf-1234567890abcdef",
    secret_key="sk-lf-1234567890abcdef"
)
```

### Option 2: Self-Hosted

Run Langfuse locally:

```bash
# Clone and start
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker-compose up

# Access at http://localhost:3000
```

Then:
```python
ma.tracing.init(
    host="http://localhost:3000",
    public_key="...",
    secret_key="..."
)
```

---

## Configuration Methods

### 1. Environment Variables (Recommended)

```bash
# .env file
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # Optional
```

```python
from dotenv import load_dotenv
load_dotenv()

ma.tracing.init()  # Reads from environment
```

**Pros:**
- Secure (no hardcoded keys)
- Easy to change per environment
- Standard practice

### 2. Explicit Parameters

```python
ma.tracing.init(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com"
)
```

**Pros:**
- Explicit and clear
- No environment setup needed

**Cons:**
- Keys in code (security risk)
- Harder to manage multiple environments

### 3. Configuration File

```python
# config.py
LANGFUSE_CONFIG = {
    "public_key": "pk-lf-...",
    "secret_key": "sk-lf-...",
    "host": "https://cloud.langfuse.com"
}

# app.py
from config import LANGFUSE_CONFIG
ma.tracing.init(**LANGFUSE_CONFIG)
```

---

## Initialization Options

### Basic

```python
ma.tracing.init()
```

### With Custom Host

```python
ma.tracing.init(
    host="http://localhost:3000"
)
```

### With Session ID

```python
ma.tracing.init(
    session_id="user-12345"  # Group traces by session
)
```

### With Tags

```python
ma.tracing.init(
    tags=["production", "v2.0", "high-priority"]
)
```

### With Release Version

```python
ma.tracing.init(
    release="v1.2.3"  # Track by release
)
```

### Full Configuration

```python
ma.tracing.init(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
    session_id="session-123",
    user_id="user-456",
    tags=["production"],
    release="v1.2.3",
    enabled=True  # Can disable for testing
)
```

---

## Environment-Specific Setup

### Development

```python
# config/development.py
import mahsm as ma

ma.tracing.init(
    # Use local instance
    host="http://localhost:3000",
    tags=["development"],
    enabled=True
)
```

### Staging

```python
# config/staging.py
import mahsm as ma
import os

ma.tracing.init(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    tags=["staging"],
    release=os.getenv("APP_VERSION")
)
```

### Production

```python
# config/production.py
import mahsm as ma
import os

ma.tracing.init(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    tags=["production"],
    release=os.getenv("APP_VERSION"),
    session_id=os.getenv("REQUEST_ID")  # From request context
)
```

---

## Complete Examples

### Web Application

```python
from flask import Flask
import mahsm as ma
import os

app = Flask(__name__)

# Initialize once at startup
ma.tracing.init(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    tags=["web-api"]
)

@app.route("/ask")
def ask():
    # Tracing is already enabled
    result = agent.run(request.args.get("question"))
    return {"answer": result}

if __name__ == "__main__":
    app.run()
```

### CLI Application

```python
import mahsm as ma
import os
import sys

def main():
    # Initialize tracing
    ma.tracing.init(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        tags=["cli"],
        session_id=f"cli-{os.getpid()}"
    )
    
    # Run your agent
    question = sys.argv[1] if len(sys.argv) > 1 else "What is AI?"
    result = agent.run(question)
    print(result)

if __name__ == "__main__":
    main()
```

### Batch Processing

```python
import mahsm as ma
import os

# Initialize once
ma.tracing.init(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    tags=["batch-processing"]
)

def process_batch(items):
    results = []
    for idx, item in enumerate(items):
        # Each execution gets its own trace
        result = agent.run(item)
        results.append(result)
    return results

# All executions are traced
batch = ["Question 1", "Question 2", "Question 3"]
results = process_batch(batch)
```

---

## Verifying Setup

### Check Connection

```python
import mahsm as ma

# Initialize
ma.tracing.init()

# Test with a simple trace
@ma.tracing.observe(name="test")
def test_function():
    return "Hello, Langfuse!"

result = test_function()
print("✅ Tracing initialized!")
print("Check your Langfuse dashboard for the 'test' trace")
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

ma.tracing.init()

# You'll see connection status and trace uploads in logs
```

---

## Troubleshooting

### Issue: "Authentication failed"

**Cause:** Invalid API keys

**Solution:**
```python
# Check your keys
import os
print(f"Public key: {os.getenv('LANGFUSE_PUBLIC_KEY')}")
print(f"Secret key: {os.getenv('LANGFUSE_SECRET_KEY')[:10]}...")

# Verify they match your Langfuse project
```

### Issue: "Cannot connect to host"

**Cause:** Wrong host URL or network issue

**Solution:**
```python
# Check host
ma.tracing.init(
    host="https://cloud.langfuse.com",  # Correct URL
    # Not: http://cloud.langfuse.com (missing 's')
)

# For self-hosted, ensure Langfuse is running
```

### Issue: "No traces appearing"

**Possible causes:**
1. **Tracing disabled**
   ```python
   # Make sure enabled=True (default)
   ma.tracing.init(enabled=True)
   ```

2. **Async flush not completed**
   ```python
   # Traces are flushed asynchronously
   # In scripts, add a small delay before exit
   import time
   time.sleep(2)  # Allow flush to complete
   ```

3. **Network firewall**
   ```python
   # Check if you can reach Langfuse
   import requests
   response = requests.get("https://cloud.langfuse.com")
   print(response.status_code)  # Should be 200
   ```

---

## Best Practices

### ✅ Do:

1. **Initialize once at startup**
   ```python
   # ✅ Good
   def main():
       ma.tracing.init()
       run_application()
   
   if __name__ == "__main__":
       main()
   ```

2. **Use environment variables**
   ```python
   # ✅ Good - secure
   ma.tracing.init()  # Reads from env
   
   # ❌ Bad - keys in code
   ma.tracing.init(
       public_key="pk-lf-hardcoded",  # Security risk!
       secret_key="sk-lf-hardcoded"
   )
   ```

3. **Tag by environment**
   ```python
   # ✅ Good - easy to filter
   ma.tracing.init(
       tags=[os.getenv("ENVIRONMENT", "development")]
   )
   ```

4. **Include version info**
   ```python
   # ✅ Good - track changes over time
   ma.tracing.init(
       release=os.getenv("APP_VERSION", "dev")
   )
   ```

### ❌ Don't:

1. **Don't initialize multiple times**
   ```python
   # ❌ Bad
   ma.tracing.init()
   # ... later ...
   ma.tracing.init()  # Unnecessary!
   ```

2. **Don't commit API keys**
   ```python
   # ❌ Bad - never commit keys
   ma.tracing.init(
       public_key="pk-lf-real-key",
       secret_key="sk-lf-real-key"
   )
   ```

3. **Don't forget to enable in production**
   ```python
   # ❌ Bad - defeats the purpose
   if os.getenv("ENVIRONMENT") != "production":
       ma.tracing.init()
   # You WANT tracing in production!
   ```

---

## Advanced Configuration

### Conditional Initialization

```python
import os
import mahsm as ma

# Only trace in certain environments
if os.getenv("ENABLE_TRACING", "true").lower() == "true":
    ma.tracing.init()
else:
    print("Tracing disabled")
```

### Custom Trace Context

```python
from contextvars import ContextVar

# Create context for request ID
request_id_var: ContextVar[str] = ContextVar("request_id")

def init_tracing_for_request(request_id: str):
    """Initialize tracing with request-specific context."""
    request_id_var.set(request_id)
    ma.tracing.init(
        session_id=request_id,
        tags=["api-request"]
    )
```

### Multiple Projects

```python
# Initialize for different projects
if os.getenv("PROJECT") == "chatbot":
    ma.tracing.init(
        public_key=os.getenv("CHATBOT_LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("CHATBOT_LANGFUSE_SECRET_KEY"),
        tags=["chatbot"]
    )
else:
    ma.tracing.init(
        public_key=os.getenv("ANALYTICS_LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("ANALYTICS_LANGFUSE_SECRET_KEY"),
        tags=["analytics"]
    )
```

---

## Next Steps

Now that Langfuse is initialized:

1. **[DSPy Tracing](dspy-tracing.md)** → Trace DSPy modules
2. **[LangGraph Tracing](langgraph-tracing.md)** → Trace LangGraph workflows
3. **[Manual Tracing](manual-tracing.md)** → Add custom spans

---

## External Resources

- **[Langfuse Authentication](https://langfuse.com/docs/authentication)** - Official auth guide
- **[Environment Variables](https://langfuse.com/docs/sdk/python#environment-variables)** - Configuration options
- **[Self-Hosting](https://langfuse.com/docs/deployment/self-host)** - Self-hosting guide

---

**Next: Trace DSPy modules with [DSPy Tracing →](dspy-tracing.md)**
