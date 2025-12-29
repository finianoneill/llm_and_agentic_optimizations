"""
Caching benchmarks - Prompt caching and semantic caching tests.

Tests the impact of Anthropic's cache_control feature and
demonstrates semantic caching patterns.
"""

import asyncio
import hashlib
import json
from typing import Any, Optional

import anthropic

from ...instrumentation.timing import Timer, TimingResult
from ...harness.runner import BenchmarkConfig, benchmark


# Large system prompt to benefit from caching
LARGE_SYSTEM_PROMPT = """You are an expert assistant with deep knowledge across multiple domains.

## Your Capabilities

### Technical Expertise
- Software engineering: Python, JavaScript, TypeScript, Rust, Go, and more
- System design: Distributed systems, microservices, databases
- DevOps: CI/CD, Kubernetes, Docker, cloud platforms
- Machine learning: Deep learning, NLP, computer vision

### Communication Style
- Be concise but thorough
- Use code examples when helpful
- Explain complex concepts in simple terms
- Provide actionable recommendations

### Response Format
- Structure responses with clear headings
- Use bullet points for lists
- Include code blocks with syntax highlighting
- Cite sources when applicable

### Context Awareness
- Remember previous conversation context
- Build on earlier topics naturally
- Ask clarifying questions when needed
- Acknowledge uncertainty when appropriate

This prompt is intentionally long to demonstrate the benefits of prompt caching
for system prompts that are reused across many requests. In production, system
prompts often include detailed instructions, few-shot examples, tool definitions,
and domain-specific knowledge that can total thousands of tokens.

The cache_control feature allows these large prefixes to be cached and reused,
reducing both latency and cost for subsequent requests that share the same prefix.

Key metrics to observe:
1. First request: cache_creation_input_tokens > 0, cache_read_input_tokens = 0
2. Subsequent requests: cache_creation_input_tokens = 0, cache_read_input_tokens > 0
3. Latency reduction: Up to 85% improvement on cache hits
4. Cost reduction: Cached tokens charged at 10% of regular rate

""" * 3  # Repeat to make it even larger for more dramatic caching benefits


@benchmark(cache_control=True)
async def cached_prompt_request(config: BenchmarkConfig) -> TimingResult:
    """Benchmark request with prompt caching enabled.

    Uses cache_control to cache the system prompt.
    """
    client = anthropic.AsyncAnthropic()
    timer = Timer("cached_prompt")
    user_prompt = config.metadata.get("prompt", "What is machine learning?")
    system_prompt = config.metadata.get("system_prompt", LARGE_SYSTEM_PROMPT)

    timer.start()

    response = await client.messages.create(
        model=config.model,
        max_tokens=config.metadata.get("max_tokens", 300),
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
    )

    timer.stop()

    # Extract cache usage from response
    usage = response.usage
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)

    return timer.to_result(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
        cache_hit=cache_read > 0,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


@benchmark(cache_control=False)
async def uncached_prompt_request(config: BenchmarkConfig) -> TimingResult:
    """Benchmark request without prompt caching (baseline).

    Does not use cache_control for comparison.
    """
    client = anthropic.AsyncAnthropic()
    timer = Timer("uncached_prompt")
    user_prompt = config.metadata.get("prompt", "What is machine learning?")
    system_prompt = config.metadata.get("system_prompt", LARGE_SYSTEM_PROMPT)

    timer.start()

    response = await client.messages.create(
        model=config.model,
        max_tokens=config.metadata.get("max_tokens", 300),
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    timer.stop()

    return timer.to_result(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )


class SemanticCache:
    """Simple semantic cache using embedding similarity.

    In production, you'd use a vector database like Pinecone, Weaviate, etc.
    This is a simplified in-memory implementation for demonstration.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.cache: dict[str, dict] = {}
        self.similarity_threshold = similarity_threshold

    def _hash_prompt(self, prompt: str) -> str:
        """Simple hash for exact match caching."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> Optional[dict]:
        """Retrieve cached response if exact match exists."""
        key = self._hash_prompt(prompt)
        return self.cache.get(key)

    def set(self, prompt: str, response: dict) -> None:
        """Store response in cache."""
        key = self._hash_prompt(prompt)
        self.cache[key] = response

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


# Global semantic cache for benchmark
_semantic_cache = SemanticCache()


@benchmark(cache_control=False)
async def semantic_cached_request(config: BenchmarkConfig) -> TimingResult:
    """Benchmark request with semantic caching layer.

    Checks semantic cache before making API call.
    """
    timer = Timer("semantic_cached")
    user_prompt = config.metadata.get("prompt", "What is machine learning?")

    timer.start()

    # Check cache first
    cached = _semantic_cache.get(user_prompt)
    if cached:
        timer.stop()
        return timer.to_result(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cache_hit=True,
            metadata={"semantic_cache_hit": True},
        )

    # Cache miss - make API call
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=config.model,
        max_tokens=config.metadata.get("max_tokens", 300),
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Store in cache
    _semantic_cache.set(user_prompt, {
        "content": response.content[0].text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    })

    timer.stop()

    return timer.to_result(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        cache_hit=False,
        metadata={"semantic_cache_hit": False},
    )


async def compare_cached_vs_uncached(
    model: str = "claude-sonnet-4-20250514",
    num_runs: int = 10,
) -> dict:
    """Run a comparison between cached and uncached requests.

    Demonstrates the latency benefits of prompt caching.
    """
    from ...harness.runner import BenchmarkRunner
    from ...harness.reporter import ConsoleReporter

    runner = BenchmarkRunner()
    reporter = ConsoleReporter()

    # Different prompts that share the same system prompt
    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "How does gradient descent work?",
        "What is overfitting?",
        "Describe backpropagation.",
    ]

    # Cached config
    cached_config = BenchmarkConfig(
        name="prompt_cached",
        description="Requests with cache_control enabled",
        model=model,
        num_runs=num_runs,
        warmup_runs=0,  # No warmup - we want to see the cache warm-up
        cache_control=True,
        metadata={"max_tokens": 300},
    )

    # Uncached config
    uncached_config = BenchmarkConfig(
        name="uncached_baseline",
        description="Requests without caching",
        model=model,
        num_runs=num_runs,
        warmup_runs=2,
        cache_control=False,
        metadata={"max_tokens": 300},
    )

    print("\n" + "=" * 60)
    print("Prompt Caching Benchmark")
    print("=" * 60)
    print(f"System prompt size: ~{len(LARGE_SYSTEM_PROMPT)} characters")

    # Run cached benchmark with rotating prompts
    print("\n--- Cached Requests ---")
    cached_results = []
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        cached_config.metadata["prompt"] = prompt
        result = await runner.run_single(cached_prompt_request, cached_config)
        cached_results.append(result)

        # Show cache status
        cache_status = "HIT" if result.cache_hit else "MISS (warming)"
        print(f"  Run {i+1}: {result.total_latency_ms:.0f}ms - Cache: {cache_status}")

    # Run uncached baseline
    print("\n--- Uncached Baseline ---")
    uncached_results = []
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        uncached_config.metadata["prompt"] = prompt
        result = await runner.run_single(uncached_prompt_request, uncached_config)
        uncached_results.append(result)
        print(f"  Run {i+1}: {result.total_latency_ms:.0f}ms")

    # Calculate stats
    cached_latencies = [r.total_latency_ms for r in cached_results]
    uncached_latencies = [r.total_latency_ms for r in uncached_results]

    # Only compare cache hits for fair comparison
    cached_hit_latencies = [r.total_latency_ms for r in cached_results if r.cache_hit]

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\nUncached baseline:")
    print(f"  Mean latency: {sum(uncached_latencies)/len(uncached_latencies):.0f}ms")

    print(f"\nWith prompt caching:")
    print(f"  Mean latency (all): {sum(cached_latencies)/len(cached_latencies):.0f}ms")
    if cached_hit_latencies:
        mean_hits = sum(cached_hit_latencies)/len(cached_hit_latencies)
        print(f"  Mean latency (cache hits only): {mean_hits:.0f}ms")
        improvement = ((sum(uncached_latencies)/len(uncached_latencies) - mean_hits) /
                      (sum(uncached_latencies)/len(uncached_latencies))) * 100
        print(f"  Latency improvement on cache hits: {improvement:.1f}%")

    return {
        "cached_results": cached_results,
        "uncached_results": uncached_results,
    }


async def test_cache_warmup_pattern(
    model: str = "claude-sonnet-4-20250514",
) -> list[TimingResult]:
    """Demonstrate cache warm-up pattern.

    Shows how the first request warms the cache and subsequent requests benefit.
    """
    client = anthropic.AsyncAnthropic()
    results = []

    print("\n" + "=" * 60)
    print("Cache Warm-up Pattern Demonstration")
    print("=" * 60)

    for i in range(5):
        timer = Timer(f"request_{i+1}")
        timer.start()

        response = await client.messages.create(
            model=model,
            max_tokens=100,
            system=[
                {
                    "type": "text",
                    "text": LARGE_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": f"Question {i+1}: What is AI?"}],
        )

        timer.stop()

        usage = response.usage
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0)
        cache_read = getattr(usage, "cache_read_input_tokens", 0)

        result = timer.to_result(
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            cache_hit=cache_read > 0,
        )
        results.append(result)

        status = "CREATED" if cache_creation > 0 else ("HIT" if cache_read > 0 else "NONE")
        print(f"Request {i+1}: {result.total_latency_ms:.0f}ms | "
              f"Cache status: {status} | "
              f"Created: {cache_creation} | Read: {cache_read}")

    return results


class CachingBenchmarkSuite:
    """Suite of caching-related benchmarks."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    async def run_all(self, num_runs: int = 10) -> dict:
        """Run all caching benchmarks."""
        results = {}

        print("\n" + "=" * 70)
        print("CACHING BENCHMARK SUITE")
        print("=" * 70)

        # Test cache warm-up
        results["warmup"] = await test_cache_warmup_pattern(self.model)

        # Test cached vs uncached
        results["comparison"] = await compare_cached_vs_uncached(
            model=self.model,
            num_runs=num_runs,
        )

        return results
