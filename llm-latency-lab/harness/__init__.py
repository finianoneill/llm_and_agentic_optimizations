"""
Benchmark harness for LLM latency experiments.

Provides orchestration and reporting capabilities.
"""

from .runner import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    ScenarioRunner,
    OptimizationFlag,
    benchmark,
)

from .reporter import (
    ComparisonReport,
    ConsoleReporter,
    ChartReporter,
    JSONReporter,
)

__all__ = [
    # Runner
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ScenarioRunner",
    "OptimizationFlag",
    "benchmark",
    # Reporter
    "ComparisonReport",
    "ConsoleReporter",
    "ChartReporter",
    "JSONReporter",
]
