"""
mahsm.tracing - Langfuse integration for tracing LLM calls

This module handles all Langfuse-related tracing functionality:
- Langfuse client initialization
- DSPy instrumentation
- LangGraph callback handlers
- Manual @observe decorator for custom tracing
"""

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from openinference.instrumentation.dspy import DSPyInstrumentor
import os
import warnings

# Try to import observe decorator (may not be available in all langfuse versions)
try:
    from langfuse.decorators import observe
except ImportError:
    try:
        from langfuse import observe
    except ImportError:
        # Provide a no-op decorator if observe is not available
        def observe(func):
            """No-op decorator when langfuse.observe is not available."""
            return func


def init(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
) -> CallbackHandler | None:
    """
    Initialize Langfuse tracing for mahsm applications.
    
    Sets up:
    - Langfuse client (for trace viewing/management)
    - DSPy instrumentation (automatic DSPy module tracing)
    - LangGraph callback handler (for graph execution tracing)
    
    Args:
        public_key: Langfuse public key (defaults to LANGFUSE_PUBLIC_KEY env var)
        secret_key: Langfuse secret key (defaults to LANGFUSE_SECRET_KEY env var)
        host: Langfuse host URL (defaults to LANGFUSE_HOST env var or cloud)
    
    Returns:
        CallbackHandler for LangGraph integration, or None if credentials missing
    
    Example:
        ```python
        import mahsm as ma
        
        # Initialize tracing (reads from environment variables)
        handler = ma.tracing.init()
        
        # Use with LangGraph
        graph = workflow.compile()
        result = graph.invoke({"query": "..."}, config={"callbacks": [handler]})
        ```
    
    Environment Variables:
        LANGFUSE_PUBLIC_KEY: Your Langfuse public key
        LANGFUSE_SECRET_KEY: Your Langfuse secret key
        LANGFUSE_HOST: Langfuse host URL (optional, defaults to cloud)
    """
    # Check for credentials
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_HOST")
    
    if not (public_key and secret_key):
        warnings.warn(
            "mahsm.tracing: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY not found. "
            "Tracing will be disabled. Set these environment variables to enable tracing."
        )
        return None
    
    # Initialize Langfuse client
    try:
        if host:
            client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
        else:
            client = get_client()
        
        print("✓ mahsm.tracing: Langfuse client initialized")
    except Exception as e:
        warnings.warn(f"mahsm.tracing: Failed to initialize Langfuse client: {e}")
        return None
    
    # Instrument DSPy for automatic tracing
    try:
        DSPyInstrumentor().instrument()
        print("✓ mahsm.tracing: DSPy instrumented for automatic tracing")
    except Exception as e:
        warnings.warn(f"mahsm.tracing: Failed to instrument DSPy: {e}")
    
    # Return callback handler for LangGraph integration
    try:
        handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        print("✓ mahsm.tracing: LangGraph callback handler ready")
        return handler
    except Exception as e:
        warnings.warn(f"mahsm.tracing: Failed to create callback handler: {e}")
        return None


# Re-export the @observe decorator for convenience
__all__ = ["init", "observe"]
