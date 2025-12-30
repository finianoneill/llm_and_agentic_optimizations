"""
Agent topology benchmarks - Flat vs hierarchical supervisor patterns.

Tests the latency implications of different agent architectures
for multi-agent systems.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import anthropic

from instrumentation.timing import Timer, TimingResult
from harness.runner import BenchmarkConfig, benchmark


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_name: str
    output: str
    latency_ms: float
    tokens_used: int


class BaseAgent:
    """Base class for agents."""

    def __init__(
        self,
        name: str,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str = "",
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.client = anthropic.AsyncAnthropic()

    async def execute(self, task: str, context: Optional[dict] = None) -> AgentResult:
        """Execute a task and return the result."""
        timer = Timer(self.name)
        timer.start()

        messages = [{"role": "user", "content": task}]
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            messages[0]["content"] = f"Context:\n{context_str}\n\nTask: {task}"

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=self.system_prompt,
            messages=messages,
        )

        timer.stop()

        return AgentResult(
            agent_name=self.name,
            output=response.content[0].text,
            latency_ms=timer.elapsed_ms,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        )


class ResearchAgent(BaseAgent):
    """Agent specialized for research tasks."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        super().__init__(
            name="researcher",
            model=model,
            system_prompt="You are a research specialist. Provide concise, factual information."
        )


class AnalysisAgent(BaseAgent):
    """Agent specialized for analysis tasks."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        super().__init__(
            name="analyst",
            model=model,
            system_prompt="You are an analysis specialist. Provide clear analytical insights."
        )


class WriterAgent(BaseAgent):
    """Agent specialized for writing tasks."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        super().__init__(
            name="writer",
            model=model,
            system_prompt="You are a writing specialist. Create clear, well-structured content."
        )


class SupervisorAgent(BaseAgent):
    """Supervisor agent that coordinates other agents."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        super().__init__(
            name="supervisor",
            model=model,
            system_prompt="""You are a supervisor agent coordinating a team.
Your job is to:
1. Analyze the task
2. Delegate to specialists
3. Synthesize their outputs

Available specialists: researcher, analyst, writer

Respond with your delegation plan and synthesis."""
        )


# Flat topology: Single agent handles everything
@benchmark()
async def flat_topology_single_agent(config: BenchmarkConfig) -> TimingResult:
    """Benchmark flat topology with a single powerful agent.

    One agent handles the entire task without delegation.
    """
    timer = Timer("flat_single_agent")
    task = config.metadata.get(
        "task",
        "Research the impact of climate change on agriculture, "
        "analyze the economic implications, and write a brief summary."
    )

    timer.start()

    agent = BaseAgent(
        name="generalist",
        model=config.model,
        system_prompt="You are a capable generalist. Handle all aspects of the task."
    )
    result = await agent.execute(task)

    timer.stop()

    return timer.to_result(
        metadata={
            "topology": "flat_single",
            "agent_calls": 1,
            "total_tokens": result.tokens_used,
        }
    )


# Flat topology: Parallel specialists without coordinator
@benchmark()
async def flat_topology_parallel_specialists(config: BenchmarkConfig) -> TimingResult:
    """Benchmark flat topology with parallel specialists.

    Multiple specialists work in parallel, results combined at the end.
    """
    timer = Timer("flat_parallel_specialists")
    task = config.metadata.get(
        "task",
        "Research the impact of climate change on agriculture, "
        "analyze the economic implications, and write a brief summary."
    )

    timer.start()

    # Create specialists
    researcher = ResearchAgent(model=config.model)
    analyst = AnalysisAgent(model=config.model)
    writer = WriterAgent(model=config.model)

    # Execute in parallel
    research_task = researcher.execute(f"Research: {task}")
    analysis_task = analyst.execute(f"Analyze: {task}")
    writing_task = writer.execute(f"Write about: {task}")

    results = await asyncio.gather(research_task, analysis_task, writing_task)

    timer.stop()

    total_tokens = sum(r.tokens_used for r in results)
    agent_latencies = {r.agent_name: r.latency_ms for r in results}

    return timer.to_result(
        metadata={
            "topology": "flat_parallel",
            "agent_calls": 3,
            "total_tokens": total_tokens,
            "agent_latencies": agent_latencies,
            "critical_path_ms": max(r.latency_ms for r in results),
        }
    )


# Hierarchical topology: Supervisor coordinates specialists
@benchmark()
async def hierarchical_supervisor_topology(config: BenchmarkConfig) -> TimingResult:
    """Benchmark hierarchical topology with supervisor.

    Supervisor plans, delegates to specialists, then synthesizes.
    """
    timer = Timer("hierarchical_supervisor")
    task = config.metadata.get(
        "task",
        "Research the impact of climate change on agriculture, "
        "analyze the economic implications, and write a brief summary."
    )

    timer.start()

    supervisor = SupervisorAgent(model=config.model)

    # Step 1: Supervisor plans
    plan_result = await supervisor.execute(f"Plan how to handle this task: {task}")

    # Step 2: Specialists execute (could be parallel or sequential based on plan)
    researcher = ResearchAgent(model=config.model)
    analyst = AnalysisAgent(model=config.model)
    writer = WriterAgent(model=config.model)

    # For this benchmark, we execute sequentially as supervisor would coordinate
    research_result = await researcher.execute(
        f"Research: {task}",
        context={"supervisor_plan": plan_result.output}
    )

    analysis_result = await analyst.execute(
        f"Analyze: {task}",
        context={
            "supervisor_plan": plan_result.output,
            "research": research_result.output
        }
    )

    writing_result = await writer.execute(
        f"Write: {task}",
        context={
            "supervisor_plan": plan_result.output,
            "research": research_result.output,
            "analysis": analysis_result.output
        }
    )

    # Step 3: Supervisor synthesizes
    synthesis_result = await supervisor.execute(
        "Synthesize the outputs into a final response",
        context={
            "research": research_result.output,
            "analysis": analysis_result.output,
            "writing": writing_result.output,
        }
    )

    timer.stop()

    all_results = [plan_result, research_result, analysis_result, writing_result, synthesis_result]
    total_tokens = sum(r.tokens_used for r in all_results)

    return timer.to_result(
        metadata={
            "topology": "hierarchical",
            "agent_calls": 5,  # supervisor (2) + specialists (3)
            "total_tokens": total_tokens,
            "supervisor_overhead_ms": plan_result.latency_ms + synthesis_result.latency_ms,
        }
    )


# Hierarchical with parallel sub-agents
@benchmark()
async def hierarchical_parallel_topology(config: BenchmarkConfig) -> TimingResult:
    """Benchmark hierarchical topology with parallel specialist execution.

    Supervisor plans, specialists execute in parallel, supervisor synthesizes.
    """
    timer = Timer("hierarchical_parallel")
    task = config.metadata.get(
        "task",
        "Research the impact of climate change on agriculture, "
        "analyze the economic implications, and write a brief summary."
    )

    timer.start()

    supervisor = SupervisorAgent(model=config.model)

    # Step 1: Supervisor plans
    plan_result = await supervisor.execute(f"Plan how to handle this task: {task}")

    # Step 2: Specialists execute in parallel
    researcher = ResearchAgent(model=config.model)
    analyst = AnalysisAgent(model=config.model)
    writer = WriterAgent(model=config.model)

    specialist_results = await asyncio.gather(
        researcher.execute(f"Research: {task}", context={"plan": plan_result.output}),
        analyst.execute(f"Analyze: {task}", context={"plan": plan_result.output}),
        writer.execute(f"Draft: {task}", context={"plan": plan_result.output}),
    )

    research_result, analysis_result, writing_result = specialist_results

    # Step 3: Supervisor synthesizes
    synthesis_result = await supervisor.execute(
        "Synthesize the outputs into a final response",
        context={
            "research": research_result.output,
            "analysis": analysis_result.output,
            "writing": writing_result.output,
        }
    )

    timer.stop()

    all_results = [plan_result, *specialist_results, synthesis_result]
    total_tokens = sum(r.tokens_used for r in all_results)

    return timer.to_result(
        metadata={
            "topology": "hierarchical_parallel",
            "agent_calls": 5,
            "total_tokens": total_tokens,
            "supervisor_overhead_ms": plan_result.latency_ms + synthesis_result.latency_ms,
            "specialist_critical_path_ms": max(r.latency_ms for r in specialist_results),
        }
    )


async def compare_topologies(
    task: str = None,
    model: str = "claude-sonnet-4-20250514",
    num_runs: int = 3,
) -> dict:
    """Compare all agent topologies."""
    from harness.runner import BenchmarkRunner
    from harness.reporter import ConsoleReporter

    runner = BenchmarkRunner()
    reporter = ConsoleReporter()

    if task is None:
        task = (
            "Research the impact of climate change on agriculture, "
            "analyze the economic implications, and write a brief summary."
        )

    print("\n" + "=" * 70)
    print("Agent Topology Comparison")
    print("=" * 70)
    print(f"Task: {task[:60]}...")
    print(f"Model: {model}")

    metadata = {"task": task}

    results = []

    # Test each topology
    topologies = [
        ("Flat Single Agent", flat_topology_single_agent),
        ("Flat Parallel Specialists", flat_topology_parallel_specialists),
        ("Hierarchical Sequential", hierarchical_supervisor_topology),
        ("Hierarchical Parallel", hierarchical_parallel_topology),
    ]

    for name, fn in topologies:
        config = BenchmarkConfig(
            name=name,
            model=model,
            num_runs=num_runs,
            warmup_runs=1,
            metadata=metadata,
        )
        result = await runner.run_benchmark(fn, config)
        results.append(result)

    print(reporter.comparison_table(results))

    # Detailed analysis
    print("\n" + "=" * 70)
    print("Detailed Analysis")
    print("=" * 70)

    for result in results:
        stats = result.stats
        print(f"\n{result.config.name}:")
        print(f"  p50 latency: {stats['latency_p50_ms']:.0f}ms")

        # Get metadata from first timing result
        if result.timing_results:
            meta = result.timing_results[0].metadata
            print(f"  Agent calls: {meta.get('agent_calls', 'N/A')}")
            print(f"  Total tokens: {meta.get('total_tokens', 'N/A')}")
            if "supervisor_overhead_ms" in meta:
                print(f"  Supervisor overhead: {meta['supervisor_overhead_ms']:.0f}ms")
            if "critical_path_ms" in meta:
                print(f"  Critical path: {meta['critical_path_ms']:.0f}ms")

    return {name: result for name, result in zip([t[0] for t in topologies], results)}


class AgentTopologyBenchmarkSuite:
    """Suite of agent topology benchmarks."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    async def run_all(self, num_runs: int = 3) -> dict:
        """Run all topology benchmarks."""
        print("\n" + "=" * 70)
        print("AGENT TOPOLOGY BENCHMARK SUITE")
        print("=" * 70)

        return await compare_topologies(
            model=self.model,
            num_runs=num_runs,
        )

    async def test_task_complexity_impact(
        self,
        num_runs: int = 2,
    ) -> dict:
        """Test how task complexity affects topology performance."""
        tasks = {
            "simple": "What is the capital of France?",
            "moderate": "Explain the key differences between REST and GraphQL APIs.",
            "complex": (
                "Design a complete microservices architecture for a real-time "
                "collaborative document editing platform, considering scalability, "
                "consistency, and fault tolerance."
            ),
        }

        results = {}
        for complexity, task in tasks.items():
            print(f"\n--- Testing {complexity} task ---")
            results[complexity] = await compare_topologies(
                task=task,
                model=self.model,
                num_runs=num_runs,
            )

        return results
