"""
Scenario definitions for LLM latency benchmarking.
"""

from .definitions import (
    Scenario,
    ALL_SCENARIOS,
    SIMPLE_QA_SCENARIOS,
    REASONING_SCENARIOS,
    TOOL_AGENT_SCENARIOS,
    RAG_SCENARIOS,
    RAG_DOCUMENTS,
    get_scenario,
    get_scenarios_by_category,
    list_scenarios,
    get_baseline_scenarios,
)

__all__ = [
    "Scenario",
    "ALL_SCENARIOS",
    "SIMPLE_QA_SCENARIOS",
    "REASONING_SCENARIOS",
    "TOOL_AGENT_SCENARIOS",
    "RAG_SCENARIOS",
    "RAG_DOCUMENTS",
    "get_scenario",
    "get_scenarios_by_category",
    "list_scenarios",
    "get_baseline_scenarios",
]
