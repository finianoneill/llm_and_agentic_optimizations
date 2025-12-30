#!/usr/bin/env python3
"""
LLM Latency Lab - Main entry point for running benchmarks.

Usage:
    python main.py [command] [options]

Commands:
    streaming   - Run streaming vs non-streaming benchmarks
    caching     - Run prompt caching benchmarks
    parallel    - Run parallelism benchmarks
    routing     - Run model routing benchmarks
    topology    - Run agent topology benchmarks
    all         - Run all benchmark suites
    baseline    - Run baseline scenarios
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add llm-latency-lab to path for package imports
sys.path.insert(0, str(Path(__file__).parent / "llm-latency-lab"))


async def run_streaming_benchmarks(args):
    """Run streaming benchmarks."""
    from benchmarks.streaming import StreamingBenchmarkSuite, compare_streaming_vs_non_streaming

    if args.quick:
        await compare_streaming_vs_non_streaming(
            model=args.model,
            num_runs=3,
        )
    else:
        suite = StreamingBenchmarkSuite(model=args.model)
        await suite.run_response_length_comparison(num_runs=args.runs)


async def run_caching_benchmarks(args):
    """Run caching benchmarks."""
    from benchmarks.caching import CachingBenchmarkSuite

    suite = CachingBenchmarkSuite(model=args.model)
    await suite.run_all(num_runs=args.runs)


async def run_parallel_benchmarks(args):
    """Run parallelism benchmarks."""
    from benchmarks.parallelism import ParallelismBenchmarkSuite

    suite = ParallelismBenchmarkSuite(model=args.model)
    await suite.run_all(num_runs=args.runs)


async def run_routing_benchmarks(args):
    """Run model routing benchmarks."""
    from benchmarks.model_routing import RoutingBenchmarkSuite

    suite = RoutingBenchmarkSuite()
    await suite.run_all(num_runs=args.runs)


async def run_topology_benchmarks(args):
    """Run agent topology benchmarks."""
    from benchmarks.agent_topology import AgentTopologyBenchmarkSuite

    suite = AgentTopologyBenchmarkSuite(model=args.model)
    await suite.run_all(num_runs=args.runs)


async def run_all_benchmarks(args):
    """Run all benchmark suites."""
    print("=" * 70)
    print("LLM LATENCY LAB - FULL BENCHMARK SUITE")
    print("=" * 70)

    print("\n[1/5] STREAMING BENCHMARKS")
    await run_streaming_benchmarks(args)

    print("\n[2/5] CACHING BENCHMARKS")
    await run_caching_benchmarks(args)

    print("\n[3/5] PARALLELISM BENCHMARKS")
    await run_parallel_benchmarks(args)

    print("\n[4/5] MODEL ROUTING BENCHMARKS")
    await run_routing_benchmarks(args)

    print("\n[5/5] AGENT TOPOLOGY BENCHMARKS")
    await run_topology_benchmarks(args)

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)


async def run_baseline_scenarios(args):
    """Run baseline scenarios to establish performance baselines."""
    from scenarios import get_baseline_scenarios
    from instrumentation.timing import Timer, LatencyCollector
    from harness.reporter import ConsoleReporter

    reporter = ConsoleReporter()
    collector = LatencyCollector()

    scenarios = get_baseline_scenarios()

    print("=" * 70)
    print("BASELINE SCENARIO BENCHMARKS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Runs per scenario: {args.runs}")

    if args.max_account:
        print("Authentication: Claude Max Account (via Claude Agent SDK)")
        from instrumentation.claude_sdk_client import ClaudeMaxClient

        # Map full model name to SDK model identifier
        model_map = {
            "claude-sonnet-4-20250514": "sonnet",
            "claude-opus-4-5-20251101": "opus",
            "claude-3-5-haiku-20241022": "haiku",
        }
        sdk_model = model_map.get(args.model, "sonnet")
        client = ClaudeMaxClient(model=sdk_model)

        for scenario in scenarios:
            print(f"\n--- {scenario.name} ({scenario.category}) ---")
            print(f"Prompt: {scenario.prompt[:60]}...")

            for i in range(args.runs):
                timer = Timer(scenario.name)
                timer.start()

                response = await client.create_message(
                    prompt=scenario.prompt,
                    max_tokens=scenario.max_tokens,
                    system_prompt=scenario.system_prompt,
                )

                timer.stop()
                result = timer.to_result(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                collector.add(result)
                print(f"  Run {i+1}: {result.total_latency_ms:.0f}ms")
    else:
        print("Authentication: API Key (via Anthropic SDK)")
        import anthropic

        client = anthropic.AsyncAnthropic()

        for scenario in scenarios:
            print(f"\n--- {scenario.name} ({scenario.category}) ---")
            print(f"Prompt: {scenario.prompt[:60]}...")

            for i in range(args.runs):
                timer = Timer(scenario.name)
                timer.start()

                kwargs = {
                    "model": args.model,
                    "max_tokens": scenario.max_tokens,
                    "messages": [{"role": "user", "content": scenario.prompt}],
                }
                if scenario.system_prompt:
                    kwargs["system"] = scenario.system_prompt

                response = await client.messages.create(**kwargs)

                timer.stop()
                result = timer.to_result(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                collector.add(result)
                print(f"  Run {i+1}: {result.total_latency_ms:.0f}ms")

    stats = collector.stats()
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    print(f"  Total runs: {stats['count']}")
    print(f"  p50 latency: {stats['latency_p50_ms']:.0f}ms")
    print(f"  p95 latency: {stats['latency_p95_ms']:.0f}ms")
    print(f"  p99 latency: {stats['latency_p99_ms']:.0f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Latency Lab - Benchmark LLM optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py streaming --quick
    python main.py caching --runs 10
    python main.py all --model claude-sonnet-4-20250514
    python main.py baseline
        """,
    )

    parser.add_argument(
        "command",
        choices=["streaming", "caching", "parallel", "routing", "topology", "all", "baseline"],
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use for benchmarks (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per benchmark (default: 5)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick version of benchmarks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--max-account",
        action="store_true",
        help="Use Claude Max account authentication via Claude Agent SDK instead of API key",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Map commands to functions
    commands = {
        "streaming": run_streaming_benchmarks,
        "caching": run_caching_benchmarks,
        "parallel": run_parallel_benchmarks,
        "routing": run_routing_benchmarks,
        "topology": run_topology_benchmarks,
        "all": run_all_benchmarks,
        "baseline": run_baseline_scenarios,
    }

    # Run the selected command
    try:
        asyncio.run(commands[args.command](args))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
