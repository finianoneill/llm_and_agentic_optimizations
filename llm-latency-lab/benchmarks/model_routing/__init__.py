"""
Model routing benchmarks - Small model classifier to large model generator.
"""

from .benchmark import (
    TaskComplexity,
    RoutingDecision,
    classify_request,
    routed_request,
    direct_large_model_request,
    direct_small_model_request,
    compare_routing_strategies,
    RoutingBenchmarkSuite,
    SMALL_MODEL,
    LARGE_MODEL,
)

__all__ = [
    "TaskComplexity",
    "RoutingDecision",
    "classify_request",
    "routed_request",
    "direct_large_model_request",
    "direct_small_model_request",
    "compare_routing_strategies",
    "RoutingBenchmarkSuite",
    "SMALL_MODEL",
    "LARGE_MODEL",
]
