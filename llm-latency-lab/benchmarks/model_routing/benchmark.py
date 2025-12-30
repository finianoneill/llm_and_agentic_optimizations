"""
Model routing benchmarks - Small model classifier to large model generator.

Tests the latency and cost benefits of using a smaller, faster model
to classify/route requests before invoking a larger model when needed.
"""

import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import anthropic

from instrumentation.timing import Timer, TimingResult
from harness.runner import BenchmarkConfig, benchmark


class TaskComplexity(Enum):
    """Task complexity levels for routing decisions."""
    SIMPLE = "simple"      # Can be handled by small model
    MODERATE = "moderate"  # May need larger model
    COMPLEX = "complex"    # Requires larger model


@dataclass
class RoutingDecision:
    """Result of a routing classification."""
    complexity: TaskComplexity
    recommended_model: str
    confidence: float
    reasoning: str


# Model configurations
SMALL_MODEL = "claude-3-5-haiku-20241022"  # Fast, cheap classifier
LARGE_MODEL = "claude-sonnet-4-20250514"    # Powerful generator


CLASSIFIER_SYSTEM_PROMPT = """You are a task complexity classifier. Analyze the user's request and classify it into one of three categories:

1. SIMPLE: Quick factual questions, simple calculations, basic translations, formatting tasks
2. MODERATE: Multi-step reasoning, summarization, code explanation, moderate analysis
3. COMPLEX: Creative writing, complex code generation, deep analysis, multi-faceted problems

Respond with JSON only:
{
    "complexity": "simple|moderate|complex",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}
"""


async def classify_request(
    prompt: str,
    client: anthropic.AsyncAnthropic,
) -> RoutingDecision:
    """Use a small model to classify request complexity."""
    response = await client.messages.create(
        model=SMALL_MODEL,
        max_tokens=200,
        system=CLASSIFIER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        result = json.loads(response.content[0].text)
        complexity = TaskComplexity(result["complexity"])
        confidence = float(result.get("confidence", 0.8))
        reasoning = result.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        # Default to complex if classification fails
        complexity = TaskComplexity.COMPLEX
        confidence = 0.5
        reasoning = "Classification failed, defaulting to complex"

    # Determine recommended model based on complexity
    if complexity == TaskComplexity.SIMPLE:
        recommended_model = SMALL_MODEL
    elif complexity == TaskComplexity.MODERATE:
        # Could go either way - use confidence threshold
        recommended_model = SMALL_MODEL if confidence > 0.8 else LARGE_MODEL
    else:
        recommended_model = LARGE_MODEL

    return RoutingDecision(
        complexity=complexity,
        recommended_model=recommended_model,
        confidence=confidence,
        reasoning=reasoning,
    )


@benchmark()
async def routed_request(config: BenchmarkConfig) -> TimingResult:
    """Benchmark a request with model routing.

    First classifies the request, then routes to appropriate model.
    """
    client = anthropic.AsyncAnthropic()
    timer = Timer("routed_request")
    prompt = config.metadata.get("prompt", "What is 2+2?")

    timer.start()

    # Step 1: Classify
    classification_start = asyncio.get_event_loop().time()
    routing = await classify_request(prompt, client)
    classification_time = (asyncio.get_event_loop().time() - classification_start) * 1000

    # Step 2: Generate with routed model
    generation_start = asyncio.get_event_loop().time()
    response = await client.messages.create(
        model=routing.recommended_model,
        max_tokens=config.metadata.get("max_tokens", 500),
        messages=[{"role": "user", "content": prompt}],
    )
    generation_time = (asyncio.get_event_loop().time() - generation_start) * 1000

    timer.stop()

    return timer.to_result(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        metadata={
            "complexity": routing.complexity.value,
            "routed_model": routing.recommended_model,
            "confidence": routing.confidence,
            "classification_time_ms": classification_time,
            "generation_time_ms": generation_time,
        }
    )


@benchmark()
async def direct_large_model_request(config: BenchmarkConfig) -> TimingResult:
    """Benchmark direct request to large model (baseline).

    No routing - always uses the large model.
    """
    client = anthropic.AsyncAnthropic()
    timer = Timer("direct_large_model")
    prompt = config.metadata.get("prompt", "What is 2+2?")

    timer.start()

    response = await client.messages.create(
        model=LARGE_MODEL,
        max_tokens=config.metadata.get("max_tokens", 500),
        messages=[{"role": "user", "content": prompt}],
    )

    timer.stop()

    return timer.to_result(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        metadata={"model": LARGE_MODEL}
    )


@benchmark()
async def direct_small_model_request(config: BenchmarkConfig) -> TimingResult:
    """Benchmark direct request to small model.

    For comparison - shows small model baseline latency.
    """
    client = anthropic.AsyncAnthropic()
    timer = Timer("direct_small_model")
    prompt = config.metadata.get("prompt", "What is 2+2?")

    timer.start()

    response = await client.messages.create(
        model=SMALL_MODEL,
        max_tokens=config.metadata.get("max_tokens", 500),
        messages=[{"role": "user", "content": prompt}],
    )

    timer.stop()

    return timer.to_result(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        metadata={"model": SMALL_MODEL}
    )


async def compare_routing_strategies(
    prompts: Optional[list[tuple[str, str]]] = None,
    num_runs: int = 3,
) -> dict:
    """Compare routing strategies across different task types.

    Args:
        prompts: List of (complexity_label, prompt) tuples
        num_runs: Number of runs per prompt
    """
    from harness.runner import BenchmarkRunner
    from harness.reporter import ConsoleReporter

    runner = BenchmarkRunner(verbose=False)
    reporter = ConsoleReporter()

    if prompts is None:
        prompts = [
            ("simple", "What is 2+2?"),
            ("simple", "Translate 'hello' to Spanish."),
            ("moderate", "Explain how photosynthesis works in plants."),
            ("moderate", "Summarize the key differences between Python and JavaScript."),
            ("complex", "Write a Python function that implements a binary search tree with insert, delete, and search operations."),
            ("complex", "Analyze the economic impacts of artificial intelligence on the job market over the next decade."),
        ]

    print("\n" + "=" * 70)
    print("Model Routing Strategy Comparison")
    print("=" * 70)
    print(f"Small model: {SMALL_MODEL}")
    print(f"Large model: {LARGE_MODEL}")
    print(f"Testing {len(prompts)} prompts, {num_runs} runs each")

    results = {
        "routed": [],
        "direct_large": [],
        "direct_small": [],
    }

    for complexity_label, prompt in prompts:
        print(f"\n--- {complexity_label.upper()}: {prompt[:50]}... ---")

        metadata = {"prompt": prompt, "max_tokens": 500}

        # Test routed approach
        routed_config = BenchmarkConfig(
            name="routed",
            model=LARGE_MODEL,
            num_runs=num_runs,
            warmup_runs=0,
            metadata=metadata,
        )

        for _ in range(num_runs):
            result = await runner.run_single(routed_request, routed_config)
            result.metadata["expected_complexity"] = complexity_label
            results["routed"].append(result)

        # Test direct large model
        large_config = BenchmarkConfig(
            name="direct_large",
            model=LARGE_MODEL,
            num_runs=num_runs,
            warmup_runs=0,
            metadata=metadata,
        )

        for _ in range(num_runs):
            result = await runner.run_single(direct_large_model_request, large_config)
            results["direct_large"].append(result)

        # Test direct small model
        small_config = BenchmarkConfig(
            name="direct_small",
            model=SMALL_MODEL,
            num_runs=num_runs,
            warmup_runs=0,
            metadata=metadata,
        )

        for _ in range(num_runs):
            result = await runner.run_single(direct_small_model_request, small_config)
            results["direct_small"].append(result)

    # Analyze results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for strategy, strategy_results in results.items():
        latencies = [r.total_latency_ms for r in strategy_results]
        avg = sum(latencies) / len(latencies) if latencies else 0
        print(f"\n{strategy}:")
        print(f"  Average latency: {avg:.0f}ms")

        if strategy == "routed":
            # Analyze routing decisions
            routed_to_small = sum(1 for r in strategy_results
                                 if r.metadata.get("routed_model") == SMALL_MODEL)
            routed_to_large = len(strategy_results) - routed_to_small
            print(f"  Routed to small: {routed_to_small}")
            print(f"  Routed to large: {routed_to_large}")

            # Average classification overhead
            class_times = [r.metadata.get("classification_time_ms", 0)
                          for r in strategy_results]
            avg_class = sum(class_times) / len(class_times) if class_times else 0
            print(f"  Avg classification overhead: {avg_class:.0f}ms")

    return results


class RoutingBenchmarkSuite:
    """Suite of model routing benchmarks."""

    def __init__(
        self,
        small_model: str = SMALL_MODEL,
        large_model: str = LARGE_MODEL,
    ):
        self.small_model = small_model
        self.large_model = large_model

    async def run_all(self, num_runs: int = 3) -> dict:
        """Run all routing benchmarks."""
        print("\n" + "=" * 70)
        print("MODEL ROUTING BENCHMARK SUITE")
        print("=" * 70)

        results = await compare_routing_strategies(num_runs=num_runs)
        return results

    async def test_classification_accuracy(
        self,
        test_cases: Optional[list[tuple[str, str]]] = None,
    ) -> dict:
        """Test the accuracy of the classification model."""
        client = anthropic.AsyncAnthropic()

        if test_cases is None:
            test_cases = [
                ("simple", "What is 2+2?"),
                ("simple", "What is the capital of France?"),
                ("simple", "How many days in a week?"),
                ("moderate", "Explain the difference between TCP and UDP."),
                ("moderate", "Summarize the causes of World War I."),
                ("moderate", "What are the pros and cons of remote work?"),
                ("complex", "Design a scalable microservices architecture for an e-commerce platform."),
                ("complex", "Write a comprehensive guide to implementing OAuth 2.0 from scratch."),
                ("complex", "Analyze the philosophical implications of artificial general intelligence."),
            ]

        print("\n" + "=" * 60)
        print("Classification Accuracy Test")
        print("=" * 60)

        correct = 0
        total = len(test_cases)

        for expected, prompt in test_cases:
            routing = await classify_request(prompt, client)
            predicted = routing.complexity.value
            is_correct = predicted == expected

            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"

            print(f"{status} Expected: {expected}, Got: {predicted} "
                  f"(conf: {routing.confidence:.2f}) - {prompt[:40]}...")

        accuracy = correct / total * 100
        print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
