"""
Timing utilities for LLM latency benchmarking.

Provides decorators, context managers, and utilities for capturing:
- TTFT (Time to First Token)
- Total latency
- Token throughput (tokens/sec)
- Cache hit rates
"""

import asyncio
import functools
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator, Optional


@dataclass
class TimingResult:
    """Container for timing measurements."""

    name: str
    start_time: float
    end_time: float = 0.0
    ttft: Optional[float] = None  # Time to first token
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        """Total latency in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.ttft is None:
            return None
        return (self.ttft - self.start_time) * 1000

    @property
    def tokens_per_second(self) -> float:
        """Token throughput (output tokens per second)."""
        duration = self.end_time - self.start_time
        if duration <= 0 or self.output_tokens <= 0:
            return 0.0
        return self.output_tokens / duration

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as a fraction of input tokens."""
        if self.input_tokens <= 0:
            return 0.0
        return self.cache_read_input_tokens / self.input_tokens

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "total_latency_ms": self.total_latency_ms,
            "ttft_ms": self.ttft_ms,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_per_second": self.tokens_per_second,
            "cache_hit": self.cache_hit,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "metadata": self.metadata,
        }


class Timer:
    """Simple timer for manual timing control."""

    def __init__(self, name: str = "timer"):
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.ttft: Optional[float] = None
        self._running = False

    def start(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        self._running = True
        return self

    def stop(self) -> "Timer":
        """Stop the timer."""
        self.end_time = time.perf_counter()
        self._running = False
        return self

    def mark_first_token(self) -> "Timer":
        """Mark the time of first token arrival."""
        if self.ttft is None:
            self.ttft = time.perf_counter()
        return self

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        end = self.end_time if not self._running else time.perf_counter()
        return (end - self.start_time) * 1000

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.ttft is None:
            return None
        return (self.ttft - self.start_time) * 1000

    def to_result(self, **kwargs) -> TimingResult:
        """Convert to TimingResult with optional additional data."""
        return TimingResult(
            name=self.name,
            start_time=self.start_time,
            end_time=self.end_time if self.end_time else time.perf_counter(),
            ttft=self.ttft,
            **kwargs
        )


@contextmanager
def timed(name: str = "operation") -> Iterator[Timer]:
    """Context manager for timing synchronous operations.

    Usage:
        with timed("my_operation") as timer:
            # do work
            timer.mark_first_token()  # optional
        print(f"Elapsed: {timer.elapsed_ms}ms")
    """
    timer = Timer(name).start()
    try:
        yield timer
    finally:
        timer.stop()


@asynccontextmanager
async def async_timed(name: str = "operation") -> AsyncIterator[Timer]:
    """Async context manager for timing async operations.

    Usage:
        async with async_timed("my_operation") as timer:
            # do async work
            timer.mark_first_token()  # optional
        print(f"Elapsed: {timer.elapsed_ms}ms")
    """
    timer = Timer(name).start()
    try:
        yield timer
    finally:
        timer.stop()


def timing_decorator(name: Optional[str] = None):
    """Decorator for timing synchronous functions.

    Usage:
        @timing_decorator("my_function")
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> tuple[Any, TimingResult]:
            with timed(func_name) as timer:
                result = func(*args, **kwargs)
            return result, timer.to_result()

        return wrapper
    return decorator


def async_timing_decorator(name: Optional[str] = None):
    """Decorator for timing async functions.

    Usage:
        @async_timing_decorator("my_async_function")
        async def my_async_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> tuple[Any, TimingResult]:
            async with async_timed(func_name) as timer:
                result = await func(*args, **kwargs)
            return result, timer.to_result()

        return wrapper
    return decorator


class StreamingTimer:
    """Timer specialized for streaming LLM responses.

    Tracks token-by-token timing for streaming responses.
    """

    def __init__(self, name: str = "streaming"):
        self.name = name
        self.start_time: float = 0.0
        self.ttft: Optional[float] = None
        self.end_time: float = 0.0
        self.token_times: list[float] = []
        self.chunk_count: int = 0
        self._running = False

    def start(self) -> "StreamingTimer":
        """Start the streaming timer."""
        self.start_time = time.perf_counter()
        self._running = True
        return self

    def record_chunk(self, token_count: int = 1) -> "StreamingTimer":
        """Record a chunk arrival."""
        current_time = time.perf_counter()
        if self.ttft is None:
            self.ttft = current_time
        self.token_times.append(current_time)
        self.chunk_count += 1
        return self

    def stop(self) -> "StreamingTimer":
        """Stop the streaming timer."""
        self.end_time = time.perf_counter()
        self._running = False
        return self

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.ttft is None:
            return None
        return (self.ttft - self.start_time) * 1000

    @property
    def total_latency_ms(self) -> float:
        """Total latency in milliseconds."""
        end = self.end_time if not self._running else time.perf_counter()
        return (end - self.start_time) * 1000

    @property
    def inter_token_latencies_ms(self) -> list[float]:
        """List of inter-token latencies in milliseconds."""
        if len(self.token_times) < 2:
            return []
        latencies = []
        prev = self.token_times[0]
        for t in self.token_times[1:]:
            latencies.append((t - prev) * 1000)
            prev = t
        return latencies

    @property
    def avg_inter_token_latency_ms(self) -> float:
        """Average inter-token latency in milliseconds."""
        latencies = self.inter_token_latencies_ms
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    def to_result(self, **kwargs) -> TimingResult:
        """Convert to TimingResult."""
        result = TimingResult(
            name=self.name,
            start_time=self.start_time,
            end_time=self.end_time if self.end_time else time.perf_counter(),
            ttft=self.ttft,
            **kwargs
        )
        result.metadata["chunk_count"] = self.chunk_count
        result.metadata["avg_inter_token_latency_ms"] = self.avg_inter_token_latency_ms
        return result


class LatencyCollector:
    """Collects and aggregates timing results across multiple runs."""

    def __init__(self):
        self.results: list[TimingResult] = []

    def add(self, result: TimingResult) -> None:
        """Add a timing result."""
        self.results.append(result)

    def clear(self) -> None:
        """Clear all collected results."""
        self.results.clear()

    @property
    def count(self) -> int:
        """Number of collected results."""
        return len(self.results)

    def latencies_ms(self) -> list[float]:
        """List of all total latencies in milliseconds."""
        return [r.total_latency_ms for r in self.results]

    def ttfts_ms(self) -> list[float]:
        """List of all TTFT values in milliseconds (excluding None)."""
        return [r.ttft_ms for r in self.results if r.ttft_ms is not None]

    def percentile(self, values: list[float], p: float) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    def stats(self) -> dict:
        """Calculate aggregate statistics."""
        latencies = self.latencies_ms()
        ttfts = self.ttfts_ms()

        return {
            "count": self.count,
            "latency_p50_ms": self.percentile(latencies, 50),
            "latency_p95_ms": self.percentile(latencies, 95),
            "latency_p99_ms": self.percentile(latencies, 99),
            "latency_mean_ms": sum(latencies) / len(latencies) if latencies else 0,
            "latency_min_ms": min(latencies) if latencies else 0,
            "latency_max_ms": max(latencies) if latencies else 0,
            "ttft_p50_ms": self.percentile(ttfts, 50) if ttfts else None,
            "ttft_p95_ms": self.percentile(ttfts, 95) if ttfts else None,
            "ttft_p99_ms": self.percentile(ttfts, 99) if ttfts else None,
            "avg_tokens_per_second": (
                sum(r.tokens_per_second for r in self.results) / len(self.results)
                if self.results else 0
            ),
            "avg_cache_hit_rate": (
                sum(r.cache_hit_rate for r in self.results) / len(self.results)
                if self.results else 0
            ),
        }
