"""
LLM Latency Lab - A benchmark suite for LLM latency optimization testing.

This package provides comprehensive benchmarking tools for measuring and
comparing various latency optimization techniques for Large Language Models.

Key modules:
- benchmarks: Benchmark implementations for different optimization categories
- instrumentation: Timing utilities and tracing integration
- harness: Benchmark orchestration and reporting
- scenarios: Predefined benchmark scenarios
"""

__version__ = "0.1.0"
__author__ = "Finian O'Neill"

from . import benchmarks
from . import instrumentation
from . import harness
from . import scenarios

__all__ = [
    "benchmarks",
    "instrumentation",
    "harness",
    "scenarios",
]
