"""
Parallelism benchmarks - Parallel tool calls and async patterns.
"""

from .benchmark import (
    sequential_tool_execution,
    parallel_tool_execution,
    compare_parallel_vs_sequential,
    batch_processing_comparison,
    ParallelismBenchmarkSuite,
    TOOLS,
    execute_tool,
)

__all__ = [
    "sequential_tool_execution",
    "parallel_tool_execution",
    "compare_parallel_vs_sequential",
    "batch_processing_comparison",
    "ParallelismBenchmarkSuite",
    "TOOLS",
    "execute_tool",
]
