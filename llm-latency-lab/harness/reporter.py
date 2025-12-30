"""
Results aggregation and visualization for benchmark results.

Provides CLI tables, charts, and comparison reports.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .runner import BenchmarkResult


@dataclass
class ComparisonReport:
    """Report comparing multiple benchmark results."""

    baseline: BenchmarkResult
    variants: list[BenchmarkResult]
    name: str = "comparison"

    def latency_improvement(self, variant: BenchmarkResult) -> float:
        """Calculate latency improvement as a percentage."""
        baseline_p50 = self.baseline.stats.get("latency_p50_ms", 0)
        variant_p50 = variant.stats.get("latency_p50_ms", 0)
        if baseline_p50 <= 0:
            return 0.0
        return ((baseline_p50 - variant_p50) / baseline_p50) * 100

    def ttft_improvement(self, variant: BenchmarkResult) -> Optional[float]:
        """Calculate TTFT improvement as a percentage."""
        baseline_ttft = self.baseline.stats.get("ttft_p50_ms")
        variant_ttft = variant.stats.get("ttft_p50_ms")
        if baseline_ttft is None or variant_ttft is None or baseline_ttft <= 0:
            return None
        return ((baseline_ttft - variant_ttft) / baseline_ttft) * 100


class ConsoleReporter:
    """Generates console/CLI reports."""

    def __init__(self, use_color: bool = True):
        self.use_color = use_color

    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color if enabled."""
        if not self.use_color:
            return text

        colors = {
            "green": "\033[92m",
            "red": "\033[91m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "bold": "\033[1m",
            "reset": "\033[0m",
        }

        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def format_duration(self, ms: float) -> str:
        """Format duration for display."""
        if ms < 1000:
            return f"{ms:.1f}ms"
        return f"{ms / 1000:.2f}s"

    def format_improvement(self, pct: float) -> str:
        """Format improvement percentage with color."""
        if pct > 0:
            return self._color(f"+{pct:.1f}%", "green")
        elif pct < 0:
            return self._color(f"{pct:.1f}%", "red")
        return f"{pct:.1f}%"

    def single_result(self, result: BenchmarkResult) -> str:
        """Generate report for a single benchmark result."""
        lines = []
        lines.append(self._color(f"\n{'=' * 60}", "blue"))
        lines.append(self._color(f"Benchmark: {result.config.name}", "bold"))
        lines.append(self._color(f"{'=' * 60}", "blue"))

        lines.append(f"\nConfiguration:")
        lines.append(f"  Model: {result.config.model}")
        lines.append(f"  Runs: {result.config.num_runs}")
        lines.append(f"  Optimizations: {[o.value for o in result.config.optimizations]}")

        lines.append(f"\nLatency Statistics:")
        stats = result.stats
        lines.append(f"  {'p50:':<8} {self.format_duration(stats['latency_p50_ms'])}")
        lines.append(f"  {'p95:':<8} {self.format_duration(stats['latency_p95_ms'])}")
        lines.append(f"  {'p99:':<8} {self.format_duration(stats['latency_p99_ms'])}")
        lines.append(f"  {'Mean:':<8} {self.format_duration(stats['latency_mean_ms'])}")
        lines.append(f"  {'Min:':<8} {self.format_duration(stats['latency_min_ms'])}")
        lines.append(f"  {'Max:':<8} {self.format_duration(stats['latency_max_ms'])}")

        if stats.get("ttft_p50_ms"):
            lines.append(f"\nTime to First Token:")
            lines.append(f"  {'p50:':<8} {self.format_duration(stats['ttft_p50_ms'])}")
            lines.append(f"  {'p95:':<8} {self.format_duration(stats['ttft_p95_ms'])}")
            lines.append(f"  {'p99:':<8} {self.format_duration(stats['ttft_p99_ms'])}")

        if stats.get("avg_tokens_per_second", 0) > 0:
            lines.append(f"\nThroughput:")
            lines.append(f"  Tokens/sec: {stats['avg_tokens_per_second']:.1f}")

        if stats.get("avg_cache_hit_rate", 0) > 0:
            lines.append(f"\nCache Performance:")
            lines.append(f"  Hit rate: {stats['avg_cache_hit_rate'] * 100:.1f}%")

        lines.append(f"\nExecution:")
        lines.append(f"  Success rate: {result.success_rate * 100:.1f}%")
        duration = (result.end_time - result.start_time).total_seconds()
        lines.append(f"  Total duration: {duration:.1f}s")

        if result.errors:
            lines.append(f"\n{self._color('Errors:', 'red')}")
            for error in result.errors[:5]:
                lines.append(f"  - {error}")
            if len(result.errors) > 5:
                lines.append(f"  ... and {len(result.errors) - 5} more")

        return "\n".join(lines)

    def comparison_table(self, results: list[BenchmarkResult]) -> str:
        """Generate a comparison table for multiple results."""
        if not results:
            return "No results to display"

        # Table headers
        headers = ["Config", "p50", "p95", "p99", "TTFT p50", "Tokens/s", "Success"]
        col_widths = [25, 12, 12, 12, 12, 10, 8]

        lines = []
        lines.append(self._color(f"\n{'=' * sum(col_widths)}", "blue"))
        lines.append(self._color("Benchmark Comparison", "bold"))
        lines.append(self._color(f"{'=' * sum(col_widths)}", "blue"))

        # Header row
        header_row = ""
        for i, header in enumerate(headers):
            header_row += f"{header:<{col_widths[i]}}"
        lines.append(self._color(header_row, "bold"))
        lines.append("-" * sum(col_widths))

        # Data rows
        for result in results:
            stats = result.stats
            row = []

            # Config name (truncate if needed)
            name = result.config.name[:22] + "..." if len(result.config.name) > 25 else result.config.name
            row.append(f"{name:<{col_widths[0]}}")

            # Latency metrics
            row.append(f"{self.format_duration(stats['latency_p50_ms']):<{col_widths[1]}}")
            row.append(f"{self.format_duration(stats['latency_p95_ms']):<{col_widths[2]}}")
            row.append(f"{self.format_duration(stats['latency_p99_ms']):<{col_widths[3]}}")

            # TTFT
            ttft = stats.get("ttft_p50_ms")
            ttft_str = self.format_duration(ttft) if ttft else "N/A"
            row.append(f"{ttft_str:<{col_widths[4]}}")

            # Throughput
            tps = stats.get("avg_tokens_per_second", 0)
            tps_str = f"{tps:.1f}" if tps > 0 else "N/A"
            row.append(f"{tps_str:<{col_widths[5]}}")

            # Success rate
            success = f"{result.success_rate * 100:.0f}%"
            row.append(f"{success:<{col_widths[6]}}")

            lines.append("".join(row))

        return "\n".join(lines)

    def comparison_report(self, report: ComparisonReport) -> str:
        """Generate a detailed comparison report."""
        lines = []
        lines.append(self._color(f"\n{'=' * 70}", "blue"))
        lines.append(self._color(f"Comparison Report: {report.name}", "bold"))
        lines.append(self._color(f"{'=' * 70}", "blue"))

        # Baseline
        lines.append(f"\n{self._color('Baseline:', 'bold')} {report.baseline.config.name}")
        baseline_stats = report.baseline.stats
        lines.append(f"  p50 latency: {self.format_duration(baseline_stats['latency_p50_ms'])}")

        # Variants
        lines.append(f"\n{self._color('Variants:', 'bold')}")

        for variant in report.variants:
            variant_stats = variant.stats
            improvement = report.latency_improvement(variant)
            ttft_improvement = report.ttft_improvement(variant)

            lines.append(f"\n  {variant.config.name}")
            lines.append(f"    p50 latency: {self.format_duration(variant_stats['latency_p50_ms'])} "
                        f"({self.format_improvement(improvement)} vs baseline)")

            if ttft_improvement is not None:
                lines.append(f"    p50 TTFT: {self.format_duration(variant_stats['ttft_p50_ms'])} "
                           f"({self.format_improvement(ttft_improvement)} vs baseline)")

            opts = [o.value for o in variant.config.optimizations]
            lines.append(f"    Optimizations: {opts or 'none'}")

        return "\n".join(lines)


class ChartReporter:
    """Generates visual charts using matplotlib."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("results/charts")
        self._matplotlib_available = False
        self._check_matplotlib()

    def _check_matplotlib(self):
        """Check if matplotlib is available."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            self._matplotlib_available = True
        except ImportError:
            self._matplotlib_available = False

    def latency_distribution(
        self,
        result: BenchmarkResult,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """Generate a latency distribution histogram."""
        if not self._matplotlib_available:
            print("Warning: matplotlib not available for charts")
            return None

        import matplotlib.pyplot as plt

        latencies = [r.total_latency_ms for r in result.timing_results]
        if not latencies:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(latencies, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(
            result.stats["latency_p50_ms"],
            color="r",
            linestyle="--",
            label=f'p50: {result.stats["latency_p50_ms"]:.1f}ms',
        )
        ax.axvline(
            result.stats["latency_p95_ms"],
            color="orange",
            linestyle="--",
            label=f'p95: {result.stats["latency_p95_ms"]:.1f}ms',
        )

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"Latency Distribution: {result.config.name}")
        ax.legend()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = filename or f"{result.config.name}_latency_dist.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return filepath

    def comparison_bar_chart(
        self,
        results: list[BenchmarkResult],
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """Generate a bar chart comparing multiple benchmark results."""
        if not self._matplotlib_available:
            print("Warning: matplotlib not available for charts")
            return None

        import matplotlib.pyplot as plt
        import numpy as np

        if not results:
            return None

        names = [r.config.name for r in results]
        p50s = [r.stats["latency_p50_ms"] for r in results]
        p95s = [r.stats["latency_p95_ms"] for r in results]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, p50s, width, label="p50", color="steelblue")
        ax.bar(x + width / 2, p95s, width, label="p95", color="coral")

        ax.set_xlabel("Configuration")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()

        fig.tight_layout()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = filename or "comparison_bar_chart.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return filepath

    def ttft_vs_total_latency(
        self,
        results: list[BenchmarkResult],
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """Generate a scatter plot of TTFT vs total latency."""
        if not self._matplotlib_available:
            print("Warning: matplotlib not available for charts")
            return None

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        for result in results:
            ttfts = []
            totals = []
            for tr in result.timing_results:
                if tr.ttft_ms is not None:
                    ttfts.append(tr.ttft_ms)
                    totals.append(tr.total_latency_ms)

            if ttfts and totals:
                ax.scatter(ttfts, totals, label=result.config.name, alpha=0.6)

        ax.set_xlabel("Time to First Token (ms)")
        ax.set_ylabel("Total Latency (ms)")
        ax.set_title("TTFT vs Total Latency")
        ax.legend()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = filename or "ttft_vs_total.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return filepath


class JSONReporter:
    """Exports results as JSON for further analysis."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("results")

    def save_result(self, result: BenchmarkResult) -> Path:
        """Save a single result to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.name}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return filepath

    def save_comparison(
        self,
        results: list[BenchmarkResult],
        name: str = "comparison",
    ) -> Path:
        """Save multiple results as a comparison JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = self.output_dir / filename

        data = {
            "name": name,
            "timestamp": timestamp,
            "results": [r.to_dict() for r in results],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def load_result(self, filepath: Path) -> dict:
        """Load a result from JSON."""
        with open(filepath) as f:
            return json.load(f)

    def load_all_results(self, pattern: str = "*.json") -> list[dict]:
        """Load all results matching a pattern."""
        results = []
        for filepath in self.output_dir.glob(pattern):
            results.append(self.load_result(filepath))
        return results
