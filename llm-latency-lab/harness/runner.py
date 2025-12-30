"""
Benchmark orchestrator for running latency experiments.

Provides a configurable framework for running benchmark scenarios
with different optimization toggles and collecting results.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from instrumentation.timing import LatencyCollector, TimingResult


class OptimizationFlag(Enum):
    """Available optimization toggles."""

    STREAMING = "streaming"
    PROMPT_CACHING = "prompt_caching"
    PARALLEL_TOOLS = "parallel_tools"
    MODEL_ROUTING = "model_routing"
    CONNECTION_POOLING = "connection_pooling"
    SPECULATIVE_EXECUTION = "speculative_execution"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    description: str = ""
    num_runs: int = 20
    warmup_runs: int = 2
    model: str = "claude-sonnet-4-20250514"
    optimizations: set[OptimizationFlag] = field(default_factory=set)
    timeout_seconds: float = 300.0
    metadata: dict = field(default_factory=dict)

    # Optimization-specific settings
    streaming: bool = False
    cache_control: bool = False
    parallel_tools: bool = False
    routing_classifier_model: Optional[str] = None
    max_concurrent_tools: int = 5

    def __post_init__(self):
        # Sync boolean flags with optimization set
        if self.streaming:
            self.optimizations.add(OptimizationFlag.STREAMING)
        if self.cache_control:
            self.optimizations.add(OptimizationFlag.PROMPT_CACHING)
        if self.parallel_tools:
            self.optimizations.add(OptimizationFlag.PARALLEL_TOOLS)

    def has_optimization(self, opt: OptimizationFlag) -> bool:
        """Check if an optimization is enabled."""
        return opt in self.optimizations

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "num_runs": self.num_runs,
            "warmup_runs": self.warmup_runs,
            "model": self.model,
            "optimizations": [o.value for o in self.optimizations],
            "timeout_seconds": self.timeout_seconds,
            "streaming": self.streaming,
            "cache_control": self.cache_control,
            "parallel_tools": self.parallel_tools,
            "routing_classifier_model": self.routing_classifier_model,
            "max_concurrent_tools": self.max_concurrent_tools,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    config: BenchmarkConfig
    timing_results: list[TimingResult]
    stats: dict
    start_time: datetime
    end_time: datetime
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Fraction of runs that completed without errors."""
        total = len(self.timing_results) + len(self.errors)
        if total == 0:
            return 0.0
        return len(self.timing_results) / total

    def to_dict(self) -> dict:
        """Convert result to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "timing_results": [r.to_dict() for r in self.timing_results],
            "stats": self.stats,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "success_rate": self.success_rate,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Type alias for benchmark functions
BenchmarkFn = Callable[[BenchmarkConfig], TimingResult]
AsyncBenchmarkFn = Callable[[BenchmarkConfig], TimingResult]


def benchmark(
    streaming: bool = False,
    cache_control: bool = False,
    parallel_tools: bool = False,
    **kwargs,
):
    """Decorator for marking functions as benchmarks with specific optimizations.

    Usage:
        @benchmark(streaming=True, cache_control=True)
        async def my_benchmark(config: BenchmarkConfig) -> TimingResult:
            ...
    """
    def decorator(func: Callable) -> Callable:
        func._benchmark_config = {
            "streaming": streaming,
            "cache_control": cache_control,
            "parallel_tools": parallel_tools,
            **kwargs,
        }
        return func
    return decorator


class BenchmarkRunner:
    """Orchestrates benchmark execution."""

    def __init__(
        self,
        results_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.results_dir = results_dir or Path("results")
        self.verbose = verbose
        self._benchmarks: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        """Register a benchmark function."""
        self._benchmarks[name] = fn

    def list_benchmarks(self) -> list[str]:
        """List registered benchmark names."""
        return list(self._benchmarks.keys())

    async def run_single(
        self,
        fn: AsyncBenchmarkFn,
        config: BenchmarkConfig,
    ) -> TimingResult:
        """Run a single benchmark iteration."""
        try:
            result = await asyncio.wait_for(
                fn(config),
                timeout=config.timeout_seconds,
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Benchmark timed out after {config.timeout_seconds}s")

    async def run_benchmark(
        self,
        fn: AsyncBenchmarkFn,
        config: BenchmarkConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BenchmarkResult:
        """Run a complete benchmark with multiple iterations.

        Args:
            fn: Async benchmark function
            config: Benchmark configuration
            progress_callback: Optional callback(current, total) for progress updates
        """
        collector = LatencyCollector()
        errors: list[str] = []
        start_time = datetime.now()

        total_runs = config.warmup_runs + config.num_runs

        if self.verbose:
            print(f"\nRunning benchmark: {config.name}")
            print(f"  Model: {config.model}")
            print(f"  Optimizations: {[o.value for o in config.optimizations]}")
            print(f"  Warmup runs: {config.warmup_runs}")
            print(f"  Benchmark runs: {config.num_runs}")

        # Warmup runs
        for i in range(config.warmup_runs):
            if self.verbose:
                print(f"  Warmup {i + 1}/{config.warmup_runs}...", end="", flush=True)
            try:
                await self.run_single(fn, config)
                if self.verbose:
                    print(" done")
            except Exception as e:
                if self.verbose:
                    print(f" error: {e}")
            if progress_callback:
                progress_callback(i + 1, total_runs)

        # Actual benchmark runs
        for i in range(config.num_runs):
            if self.verbose:
                print(f"  Run {i + 1}/{config.num_runs}...", end="", flush=True)
            try:
                result = await self.run_single(fn, config)
                collector.add(result)
                if self.verbose:
                    print(f" {result.total_latency_ms:.1f}ms")
            except Exception as e:
                errors.append(str(e))
                if self.verbose:
                    print(f" error: {e}")
            if progress_callback:
                progress_callback(config.warmup_runs + i + 1, total_runs)

        end_time = datetime.now()
        stats = collector.stats()

        if self.verbose:
            print(f"\nResults for {config.name}:")
            print(f"  p50 latency: {stats['latency_p50_ms']:.1f}ms")
            print(f"  p95 latency: {stats['latency_p95_ms']:.1f}ms")
            print(f"  p99 latency: {stats['latency_p99_ms']:.1f}ms")
            if stats.get("ttft_p50_ms"):
                print(f"  p50 TTFT: {stats['ttft_p50_ms']:.1f}ms")

        return BenchmarkResult(
            config=config,
            timing_results=collector.results,
            stats=stats,
            start_time=start_time,
            end_time=end_time,
            errors=errors,
        )

    async def run_comparison(
        self,
        fn: AsyncBenchmarkFn,
        base_config: BenchmarkConfig,
        optimization_sets: list[set[OptimizationFlag]],
    ) -> list[BenchmarkResult]:
        """Run the same benchmark with different optimization configurations.

        Useful for A/B testing different optimization combinations.
        """
        results = []

        for opt_set in optimization_sets:
            config = BenchmarkConfig(
                name=f"{base_config.name}_{'-'.join(o.value for o in opt_set) or 'baseline'}",
                description=base_config.description,
                num_runs=base_config.num_runs,
                warmup_runs=base_config.warmup_runs,
                model=base_config.model,
                optimizations=opt_set,
                timeout_seconds=base_config.timeout_seconds,
                metadata=base_config.metadata,
            )

            result = await self.run_benchmark(fn, config)
            results.append(result)

            # Save individual result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = self.results_dir / f"{config.name}_{timestamp}.json"
            result.save(result_path)

        return results

    async def run_all(
        self,
        config: BenchmarkConfig,
    ) -> dict[str, BenchmarkResult]:
        """Run all registered benchmarks."""
        results = {}

        for name, fn in self._benchmarks.items():
            benchmark_config = BenchmarkConfig(
                name=name,
                description=config.description,
                num_runs=config.num_runs,
                warmup_runs=config.warmup_runs,
                model=config.model,
                optimizations=config.optimizations,
                timeout_seconds=config.timeout_seconds,
                metadata=config.metadata,
            )

            result = await self.run_benchmark(fn, benchmark_config)
            results[name] = result

        return results


class ScenarioRunner:
    """Runs predefined scenarios with different configurations."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self._scenarios: dict[str, dict] = {}

    def register_scenario(
        self,
        name: str,
        fn: AsyncBenchmarkFn,
        description: str = "",
        default_config: Optional[dict] = None,
    ) -> None:
        """Register a scenario for testing."""
        self._scenarios[name] = {
            "fn": fn,
            "description": description,
            "default_config": default_config or {},
        }

    async def run_scenario(
        self,
        name: str,
        config_overrides: Optional[dict] = None,
    ) -> BenchmarkResult:
        """Run a specific scenario."""
        if name not in self._scenarios:
            raise ValueError(f"Unknown scenario: {name}")

        scenario = self._scenarios[name]
        default_config = scenario["default_config"].copy()
        if config_overrides:
            default_config.update(config_overrides)

        config = BenchmarkConfig(
            name=name,
            description=scenario["description"],
            **default_config,
        )

        return await self.runner.run_benchmark(scenario["fn"], config)

    async def run_matrix(
        self,
        scenario_names: list[str],
        optimization_matrix: list[set[OptimizationFlag]],
        base_config: Optional[dict] = None,
    ) -> dict[str, list[BenchmarkResult]]:
        """Run multiple scenarios with multiple optimization configurations.

        Returns a matrix of results: {scenario_name: [results per optimization set]}
        """
        all_results = {}

        for scenario_name in scenario_names:
            if scenario_name not in self._scenarios:
                print(f"Warning: Unknown scenario {scenario_name}, skipping")
                continue

            scenario = self._scenarios[scenario_name]
            scenario_results = []

            for opt_set in optimization_matrix:
                config_dict = scenario["default_config"].copy()
                if base_config:
                    config_dict.update(base_config)

                config = BenchmarkConfig(
                    name=f"{scenario_name}_{'-'.join(o.value for o in opt_set) or 'baseline'}",
                    description=scenario["description"],
                    optimizations=opt_set,
                    **config_dict,
                )

                result = await self.runner.run_benchmark(scenario["fn"], config)
                scenario_results.append(result)

            all_results[scenario_name] = scenario_results

        return all_results
