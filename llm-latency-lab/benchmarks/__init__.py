"""
Benchmark modules for LLM latency testing.

Each submodule focuses on a specific optimization category.
"""

from . import streaming
from . import caching
from . import parallelism
from . import model_routing
from . import agent_topology

__all__ = [
    "streaming",
    "caching",
    "parallelism",
    "model_routing",
    "agent_topology",
]
