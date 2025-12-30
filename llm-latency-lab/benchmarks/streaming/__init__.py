"""
Streaming benchmarks - TTFT vs full response comparison.
"""

from .benchmark import (
    streaming_response,
    non_streaming_response,
    compare_streaming_vs_non_streaming,
    StreamingBenchmarkSuite,
)

__all__ = [
    "streaming_response",
    "non_streaming_response",
    "compare_streaming_vs_non_streaming",
    "StreamingBenchmarkSuite",
]
