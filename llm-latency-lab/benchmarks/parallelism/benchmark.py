"""
Parallelism benchmarks - Parallel tool calls and async patterns.

Tests the impact of parallel execution on wall-clock latency
for tool-using agents and batch processing.
"""

import asyncio
import random
import time
from typing import Any, Callable

import anthropic

from ...instrumentation.timing import Timer, TimingResult
from ...harness.runner import BenchmarkConfig, benchmark


# Simulated tools with variable latency
async def simulated_database_query(query: str, delay: float = 0.5) -> dict:
    """Simulate a database query with network latency."""
    await asyncio.sleep(delay + random.uniform(0, 0.1))
    return {"query": query, "results": [f"result_{i}" for i in range(3)]}


async def simulated_api_call(endpoint: str, delay: float = 0.3) -> dict:
    """Simulate an external API call."""
    await asyncio.sleep(delay + random.uniform(0, 0.1))
    return {"endpoint": endpoint, "status": "success", "data": {"value": 42}}


async def simulated_file_read(path: str, delay: float = 0.1) -> dict:
    """Simulate reading a file."""
    await asyncio.sleep(delay + random.uniform(0, 0.05))
    return {"path": path, "content": f"Content of {path}"}


async def simulated_computation(input_data: str, delay: float = 0.4) -> dict:
    """Simulate a CPU-intensive computation."""
    await asyncio.sleep(delay + random.uniform(0, 0.1))
    return {"input": input_data, "result": f"Processed: {input_data}"}


# Tool definitions for Anthropic API
TOOLS = [
    {
        "name": "database_query",
        "description": "Query the database for information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query to execute"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "api_call",
        "description": "Make an external API call",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API endpoint to call"}
            },
            "required": ["endpoint"],
        },
    },
    {
        "name": "read_file",
        "description": "Read contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "compute",
        "description": "Perform a computation on data",
        "input_schema": {
            "type": "object",
            "properties": {
                "input_data": {"type": "string", "description": "Input data to process"}
            },
            "required": ["input_data"],
        },
    },
]


TOOL_HANDLERS = {
    "database_query": lambda args: simulated_database_query(args["query"]),
    "api_call": lambda args: simulated_api_call(args["endpoint"]),
    "read_file": lambda args: simulated_file_read(args["path"]),
    "compute": lambda args: simulated_computation(args["input_data"]),
}


async def execute_tool(name: str, args: dict) -> dict:
    """Execute a tool by name."""
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return {"error": f"Unknown tool: {name}"}
    return await handler(args)


@benchmark(parallel_tools=False)
async def sequential_tool_execution(config: BenchmarkConfig) -> TimingResult:
    """Benchmark sequential tool execution.

    Executes tools one at a time.
    """
    client = anthropic.AsyncAnthropic()
    timer = Timer("sequential_tools")
    prompt = config.metadata.get(
        "prompt",
        "I need you to: 1) Query the database for user stats, "
        "2) Call the analytics API, 3) Read the config file, "
        "4) Compute the monthly summary. Use the available tools."
    )

    timer.start()

    response = await client.messages.create(
        model=config.model,
        max_tokens=1024,
        tools=TOOLS,
        messages=[{"role": "user", "content": prompt}],
    )

    tool_calls_count = 0
    tool_execution_time = 0

    # Handle tool calls sequentially
    while response.stop_reason == "tool_use":
        tool_results = []
        for content in response.content:
            if content.type == "tool_use":
                tool_calls_count += 1
                tool_start = time.perf_counter()

                # Execute sequentially (one at a time)
                result = await execute_tool(content.name, content.input)

                tool_execution_time += time.perf_counter() - tool_start
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content.id,
                    "content": str(result),
                })

        # Continue conversation
        response = await client.messages.create(
            model=config.model,
            max_tokens=1024,
            tools=TOOLS,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ],
        )

    timer.stop()

    return timer.to_result(
        metadata={
            "tool_calls": tool_calls_count,
            "tool_execution_time_ms": tool_execution_time * 1000,
        }
    )


@benchmark(parallel_tools=True)
async def parallel_tool_execution(config: BenchmarkConfig) -> TimingResult:
    """Benchmark parallel tool execution.

    Executes all tools concurrently.
    """
    client = anthropic.AsyncAnthropic()
    timer = Timer("parallel_tools")
    prompt = config.metadata.get(
        "prompt",
        "I need you to: 1) Query the database for user stats, "
        "2) Call the analytics API, 3) Read the config file, "
        "4) Compute the monthly summary. Use the available tools."
    )

    timer.start()

    response = await client.messages.create(
        model=config.model,
        max_tokens=1024,
        tools=TOOLS,
        messages=[{"role": "user", "content": prompt}],
    )

    tool_calls_count = 0
    tool_execution_time = 0

    # Handle tool calls in parallel
    while response.stop_reason == "tool_use":
        tool_uses = [c for c in response.content if c.type == "tool_use"]
        tool_calls_count += len(tool_uses)

        tool_start = time.perf_counter()

        # Execute all tools concurrently
        tasks = [execute_tool(tu.name, tu.input) for tu in tool_uses]
        results = await asyncio.gather(*tasks)

        tool_execution_time += time.perf_counter() - tool_start

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": str(result),
            }
            for tu, result in zip(tool_uses, results)
        ]

        # Continue conversation
        response = await client.messages.create(
            model=config.model,
            max_tokens=1024,
            tools=TOOLS,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ],
        )

    timer.stop()

    return timer.to_result(
        metadata={
            "tool_calls": tool_calls_count,
            "tool_execution_time_ms": tool_execution_time * 1000,
        }
    )


async def compare_parallel_vs_sequential(
    model: str = "claude-sonnet-4-20250514",
    num_runs: int = 5,
) -> dict:
    """Compare parallel vs sequential tool execution."""
    from ...harness.runner import BenchmarkRunner
    from ...harness.reporter import ConsoleReporter

    runner = BenchmarkRunner()
    reporter = ConsoleReporter()

    sequential_config = BenchmarkConfig(
        name="sequential_tools",
        description="Sequential tool execution",
        model=model,
        num_runs=num_runs,
        warmup_runs=1,
        parallel_tools=False,
    )

    parallel_config = BenchmarkConfig(
        name="parallel_tools",
        description="Parallel tool execution",
        model=model,
        num_runs=num_runs,
        warmup_runs=1,
        parallel_tools=True,
    )

    print("\n" + "=" * 60)
    print("Parallel vs Sequential Tool Execution")
    print("=" * 60)

    sequential_result = await runner.run_benchmark(
        sequential_tool_execution, sequential_config
    )
    parallel_result = await runner.run_benchmark(
        parallel_tool_execution, parallel_config
    )

    print(reporter.comparison_table([sequential_result, parallel_result]))

    # Calculate improvement
    seq_p50 = sequential_result.stats["latency_p50_ms"]
    par_p50 = parallel_result.stats["latency_p50_ms"]
    improvement = ((seq_p50 - par_p50) / seq_p50) * 100

    print(f"\nParallel execution improvement: {improvement:.1f}%")

    return {
        "sequential": sequential_result,
        "parallel": parallel_result,
    }


async def batch_processing_comparison(
    prompts: list[str],
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Compare sequential vs parallel batch processing of multiple prompts."""
    client = anthropic.AsyncAnthropic()

    if not prompts:
        prompts = [
            "What is Python?",
            "Explain JavaScript.",
            "What is Rust?",
            "Describe Go programming language.",
            "What is TypeScript?",
        ]

    print("\n" + "=" * 60)
    print("Batch Processing: Sequential vs Parallel")
    print("=" * 60)
    print(f"Processing {len(prompts)} prompts")

    # Sequential processing
    seq_timer = Timer("sequential_batch")
    seq_timer.start()

    seq_results = []
    for prompt in prompts:
        response = await client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        seq_results.append(response)

    seq_timer.stop()
    print(f"\nSequential: {seq_timer.elapsed_ms:.0f}ms total")

    # Parallel processing
    par_timer = Timer("parallel_batch")
    par_timer.start()

    tasks = [
        client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        for prompt in prompts
    ]
    par_results = await asyncio.gather(*tasks)

    par_timer.stop()
    print(f"Parallel: {par_timer.elapsed_ms:.0f}ms total")

    improvement = ((seq_timer.elapsed_ms - par_timer.elapsed_ms) / seq_timer.elapsed_ms) * 100
    print(f"\nImprovement: {improvement:.1f}%")
    print(f"Speedup: {seq_timer.elapsed_ms / par_timer.elapsed_ms:.2f}x")

    return {
        "sequential_ms": seq_timer.elapsed_ms,
        "parallel_ms": par_timer.elapsed_ms,
        "improvement_pct": improvement,
        "speedup": seq_timer.elapsed_ms / par_timer.elapsed_ms,
    }


class ParallelismBenchmarkSuite:
    """Suite of parallelism-related benchmarks."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    async def run_all(self, num_runs: int = 5) -> dict:
        """Run all parallelism benchmarks."""
        results = {}

        print("\n" + "=" * 70)
        print("PARALLELISM BENCHMARK SUITE")
        print("=" * 70)

        # Tool execution comparison
        results["tool_execution"] = await compare_parallel_vs_sequential(
            model=self.model,
            num_runs=num_runs,
        )

        # Batch processing comparison
        results["batch_processing"] = await batch_processing_comparison(
            prompts=[
                "What is machine learning?",
                "Explain neural networks.",
                "What is deep learning?",
                "Describe NLP.",
                "What is computer vision?",
            ],
            model=self.model,
        )

        return results

    async def run_concurrency_scaling(
        self,
        max_concurrency: int = 10,
    ) -> list[dict]:
        """Test how latency scales with different concurrency levels."""
        client = anthropic.AsyncAnthropic()
        results = []

        prompt = "What is AI?"

        print("\n" + "=" * 60)
        print("Concurrency Scaling Test")
        print("=" * 60)

        for concurrency in [1, 2, 4, 6, 8, 10]:
            if concurrency > max_concurrency:
                break

            timer = Timer(f"concurrency_{concurrency}")
            timer.start()

            tasks = [
                client.messages.create(
                    model=self.model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                for _ in range(concurrency)
            ]
            await asyncio.gather(*tasks)

            timer.stop()

            avg_per_request = timer.elapsed_ms / concurrency
            results.append({
                "concurrency": concurrency,
                "total_ms": timer.elapsed_ms,
                "avg_per_request_ms": avg_per_request,
            })

            print(f"Concurrency {concurrency}: {timer.elapsed_ms:.0f}ms total, "
                  f"{avg_per_request:.0f}ms avg per request")

        return results
