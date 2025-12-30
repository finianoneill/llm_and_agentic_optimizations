"""
Agent topology benchmarks - Flat vs hierarchical supervisor patterns.
"""

from .benchmark import (
    AgentResult,
    BaseAgent,
    ResearchAgent,
    AnalysisAgent,
    WriterAgent,
    SupervisorAgent,
    flat_topology_single_agent,
    flat_topology_parallel_specialists,
    hierarchical_supervisor_topology,
    hierarchical_parallel_topology,
    compare_topologies,
    AgentTopologyBenchmarkSuite,
)

__all__ = [
    "AgentResult",
    "BaseAgent",
    "ResearchAgent",
    "AnalysisAgent",
    "WriterAgent",
    "SupervisorAgent",
    "flat_topology_single_agent",
    "flat_topology_parallel_specialists",
    "hierarchical_supervisor_topology",
    "hierarchical_parallel_topology",
    "compare_topologies",
    "AgentTopologyBenchmarkSuite",
]
