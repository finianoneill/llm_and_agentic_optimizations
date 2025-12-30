"""
Streaming benchmarks - TTFT vs full response comparison.

Tests the impact of streaming on perceived latency by measuring
Time to First Token (TTFT) compared to waiting for the full response.
"""

import asyncio
from typing import Optional

import anthropic

from ...instrumentation.timing import StreamingTimer, Timer, TimingResult
from ...harness.runner import BenchmarkConfig, benchmark


@benchmark(streaming=True)
async def streaming_response(config: BenchmarkConfig) -> TimingResult:
    """Benchmark streaming response with TTFT tracking.

    Measures:
    - Time to first token
    - Total latency
    - Inter-token latency
    - Token throughput
    """
    client = anthropic.AsyncAnthropic()
    timer = StreamingTimer("streaming_response")
    prompt = config.metadata.get("prompt", "Explain quantum computing in 3 paragraphs.")

    timer.start()

    total_tokens = 0
    async with client.messages.stream(
        model=config.model,
        max_tokens=config.metadata.get("max_tokens", 500),
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            timer.record_chunk()
            total_tokens += len(text.split())  # Approximate token count

    timer.stop()

    # Get actual token counts from the final message
    final_message = await stream.get_final_message()

    return timer.to_result(
        input_tokens=final_message.usage.input_tokens,
        output_tokens=final_message.usage.output_tokens,
        total_tokens=final_message.usage.input_tokens + final_message.usage.output_tokens,
    )


@benchmark(streaming=False)
async def non_streaming_response(config: BenchmarkConfig) -> TimingResult:
    """Benchmark non-streaming response (baseline).

    Measures total latency without streaming.
    """
    client = anthropic.AsyncAnthropic()
    timer = Timer("non_streaming_response")
    prompt = config.metadata.get("prompt", "Explain quantum computing in 3 paragraphs.")

    timer.start()

    response = await client.messages.create(
        model=config.model,
        max_tokens=config.metadata.get("max_tokens", 500),
        messages=[{"role": "user", "content": prompt}],
    )

    timer.stop()

    return timer.to_result(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )


async def compare_streaming_vs_non_streaming(
    prompt: str = "Explain quantum computing in 3 paragraphs.",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 500,
    num_runs: int = 10,
) -> dict:
    """Run a comparison between streaming and non-streaming responses.

    Returns a dictionary with both results for comparison.
    """
    from ...harness.runner import BenchmarkRunner
    from ...harness.reporter import ConsoleReporter

    runner = BenchmarkRunner()
    reporter = ConsoleReporter()

    metadata = {"prompt": prompt, "max_tokens": max_tokens}

    # Streaming config
    streaming_config = BenchmarkConfig(
        name="streaming",
        description="Streaming response with TTFT tracking",
        model=model,
        num_runs=num_runs,
        warmup_runs=2,
        streaming=True,
        metadata=metadata,
    )

    # Non-streaming config
    non_streaming_config = BenchmarkConfig(
        name="non_streaming",
        description="Non-streaming response baseline",
        model=model,
        num_runs=num_runs,
        warmup_runs=2,
        streaming=False,
        metadata=metadata,
    )

    # Run benchmarks
    streaming_result = await runner.run_benchmark(streaming_response, streaming_config)
    non_streaming_result = await runner.run_benchmark(non_streaming_response, non_streaming_config)

    # Print comparison
    print(reporter.comparison_table([streaming_result, non_streaming_result]))

    # Calculate TTFT advantage
    if streaming_result.stats.get("ttft_p50_ms"):
        ttft = streaming_result.stats["ttft_p50_ms"]
        total = non_streaming_result.stats["latency_p50_ms"]
        perceived_improvement = ((total - ttft) / total) * 100
        print(f"\nPerceived latency improvement from streaming: {perceived_improvement:.1f}%")
        print(f"  User sees first content in {ttft:.0f}ms vs waiting {total:.0f}ms")

    return {
        "streaming": streaming_result,
        "non_streaming": non_streaming_result,
    }


class StreamingBenchmarkSuite:
    """Suite of streaming-related benchmarks."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.runner = None
        self.reporter = None

    def setup(self):
        """Initialize runner and reporter."""
        from ...harness.runner import BenchmarkRunner
        from ...harness.reporter import ConsoleReporter

        self.runner = BenchmarkRunner()
        self.reporter = ConsoleReporter()

    async def run_response_length_comparison(
        self,
        lengths: list[int] = [100, 250, 500, 1000],
        num_runs: int = 5,
    ) -> list[dict]:
        """Compare streaming benefits across different response lengths.

        Hypothesis: Streaming benefits increase with response length.
        """
        if not self.runner:
            self.setup()

        results = []
        prompt = "Explain the history and importance of machine learning."

        for max_tokens in lengths:
            print(f"\n--- Testing max_tokens={max_tokens} ---")
            comparison = await compare_streaming_vs_non_streaming(
                prompt=prompt,
                model=self.model,
                max_tokens=max_tokens,
                num_runs=num_runs,
            )
            comparison["max_tokens"] = max_tokens
            results.append(comparison)

        # Summary
        print("\n" + "=" * 60)
        print("Summary: TTFT vs Total Latency by Response Length")
        print("=" * 60)
        for r in results:
            ttft = r["streaming"].stats.get("ttft_p50_ms", 0)
            total = r["non_streaming"].stats["latency_p50_ms"]
            print(f"  max_tokens={r['max_tokens']}: TTFT={ttft:.0f}ms, Total={total:.0f}ms")

        return results

    async def run_prompt_complexity_comparison(
        self,
        num_runs: int = 5,
    ) -> list[dict]:
        """Compare streaming benefits across different prompt complexities.

        Tests simple vs complex prompts to see TTFT differences.
        """
        if not self.runner:
            self.setup()

        prompts = [
            ("simple", "What is 2+2?"),
            ("medium", "Explain how photosynthesis works in plants."),
            ("complex", "Compare and contrast the economic theories of Keynesian economics "
                       "and Austrian economics, discussing their historical context, key proponents, "
                       "main principles, and modern applications."),
        ]

        results = []
        for complexity, prompt in prompts:
            print(f"\n--- Testing {complexity} prompt ---")
            comparison = await compare_streaming_vs_non_streaming(
                prompt=prompt,
                model=self.model,
                max_tokens=500,
                num_runs=num_runs,
            )
            comparison["complexity"] = complexity
            results.append(comparison)

        return results
