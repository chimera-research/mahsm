# Installation

Getting `mahsm` installed is quick and easy. We recommend using a virtual environment to manage your project's dependencies.

## Prerequisites

- Python 3.9+
- `uv` (or `pip`) package installer

## Installing `mahsm`

To install the core `mahsm` library, run the following command:

```bash
uv pip install mahsm
```

This will install mahsm and its core dependencies, including DSPy, LangGraph, LangFuse, and EvalProtocol.

## Setting Up Observability (LangFuse)

One of the core features of mahsm is its deep integration with LangFuse for observability. To enable it, you need to set the following environment variables:
```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com" # Or your self-hosted instance
```

You can find your keys in your LangFuse project settings.