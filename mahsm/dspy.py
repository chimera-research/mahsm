"""
mahsm.dspy - DSPy re-export

This module re-exports the DSPy library for convenience.
Use `ma.dspy.*` to access any DSPy functionality.
"""

# DSPy configure functionality
from dspy.dsp.utils.settings import settings
"""
Example Usage:

lm = dspy.lm("openai/gpt-5")
dspy.configure(lm=lm, adapter=BAMLAdapter())
"""

# All Modules (CoT, ReAct, etc.)
from dspy.predict import *

# Required primitives including Python Interpreter
from dspy.primitives import *

# Enable embedding and retrieval support
from dspy.retrievers import *

# All Signatures (InputField, OutputField, Signature, etc.)
from dspy.signatures import *

# DSPy built-in evaluators
from dspy.evaluate import Evaluate  # isort: skip

# Handle various models, clients, cache, etc.
from dspy.clients import *  # isort: skip

# Use/Modify/Make other adapters with DSPy Programs
from dspy.adapters import Adapter, BAMLAdapter, ChatAdapter, JSONAdapter, XMLAdapter, TwoStepAdapter, Image, Audio, File, History, Type, Tool, ToolCalls, Code, Reasoning # isort: skip

# Utils for saving/streaming/etc. DSPy Programs
from dspy.utils import asyncify, syncify, load, mcp, parallelizer
from dspy.streaming.streamify import streamify