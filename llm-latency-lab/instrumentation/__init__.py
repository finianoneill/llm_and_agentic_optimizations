"""
Instrumentation module for LLM latency benchmarking.

Provides timing utilities and tracing integrations.
"""

from .timing import (
    Timer,
    TimingResult,
    StreamingTimer,
    LatencyCollector,
    timed,
    async_timed,
    timing_decorator,
    async_timing_decorator,
)

from .traces import (
    Tracer,
    TracingConfig,
    LangfuseTracer,
    get_tracer,
    init_tracing,
    shutdown_tracing,
    OTEL_AVAILABLE,
    LANGFUSE_AVAILABLE,
)

__all__ = [
    # Timing
    "Timer",
    "TimingResult",
    "StreamingTimer",
    "LatencyCollector",
    "timed",
    "async_timed",
    "timing_decorator",
    "async_timing_decorator",
    # Tracing
    "Tracer",
    "TracingConfig",
    "LangfuseTracer",
    "get_tracer",
    "init_tracing",
    "shutdown_tracing",
    "OTEL_AVAILABLE",
    "LANGFUSE_AVAILABLE",
]
