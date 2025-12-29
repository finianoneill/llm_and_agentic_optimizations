"""
Baseline scenario definitions for LLM latency benchmarking.

Defines realistic task types that represent common LLM use cases:
1. Simple Q&A
2. Multi-step reasoning
3. Tool-heavy agent loop
4. RAG retrieval + synthesis
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Scenario:
    """Definition of a benchmark scenario."""

    name: str
    description: str
    category: str
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 500
    expected_tools: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "max_tokens": self.max_tokens,
            "expected_tools": self.expected_tools,
            "metadata": self.metadata,
        }


# ============================================================================
# Category 1: Simple Q&A
# ============================================================================

SIMPLE_QA_SCENARIOS = [
    Scenario(
        name="simple_factual",
        description="Simple factual question",
        category="simple_qa",
        prompt="What is the capital of Japan?",
        max_tokens=100,
    ),
    Scenario(
        name="simple_definition",
        description="Definition question",
        category="simple_qa",
        prompt="What is machine learning?",
        max_tokens=200,
    ),
    Scenario(
        name="simple_calculation",
        description="Simple math question",
        category="simple_qa",
        prompt="What is 15% of 240?",
        max_tokens=100,
    ),
    Scenario(
        name="simple_translation",
        description="Translation request",
        category="simple_qa",
        prompt="Translate 'Good morning, how are you?' to French.",
        max_tokens=100,
    ),
    Scenario(
        name="simple_comparison",
        description="Simple comparison",
        category="simple_qa",
        prompt="What's the difference between RAM and ROM?",
        max_tokens=300,
    ),
]


# ============================================================================
# Category 2: Multi-step Reasoning
# ============================================================================

REASONING_SCENARIOS = [
    Scenario(
        name="reasoning_math_word_problem",
        description="Math word problem requiring multiple steps",
        category="reasoning",
        prompt="""A train leaves Station A at 9:00 AM traveling at 60 mph toward Station B.
Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A.
The stations are 280 miles apart. At what time will the trains meet?""",
        max_tokens=500,
    ),
    Scenario(
        name="reasoning_logical_puzzle",
        description="Logical deduction puzzle",
        category="reasoning",
        prompt="""Three friends - Alice, Bob, and Charlie - have different jobs: teacher,
doctor, and engineer. Given these clues:
1. Alice is not the engineer
2. Bob is not the teacher
3. The doctor is younger than Bob
4. Charlie is the oldest

Who has which job?""",
        max_tokens=500,
    ),
    Scenario(
        name="reasoning_code_analysis",
        description="Code bug analysis",
        category="reasoning",
        prompt="""What's wrong with this Python code and how would you fix it?

def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# Usage
result = calculate_average([])
print(result)""",
        max_tokens=500,
    ),
    Scenario(
        name="reasoning_pros_cons",
        description="Analysis requiring weighing multiple factors",
        category="reasoning",
        prompt="""Should a startup use microservices or a monolithic architecture?
Consider factors like team size, development speed, scalability, and operational complexity.
Provide a recommendation with reasoning.""",
        max_tokens=800,
    ),
]


# ============================================================================
# Category 3: Tool-Heavy Agent Loop
# ============================================================================

TOOL_AGENT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use the tools when needed to answer the user's questions accurately.
Available tools: database_query, api_call, read_file, compute"""

TOOL_AGENT_SCENARIOS = [
    Scenario(
        name="tool_data_retrieval",
        description="Task requiring database and API calls",
        category="tool_agent",
        prompt="Get the user statistics from the database and cross-reference with the analytics API.",
        system_prompt=TOOL_AGENT_SYSTEM_PROMPT,
        max_tokens=500,
        expected_tools=["database_query", "api_call"],
    ),
    Scenario(
        name="tool_file_processing",
        description="Task requiring file reading and computation",
        category="tool_agent",
        prompt="Read the config file, extract the settings, and compute the derived values.",
        system_prompt=TOOL_AGENT_SYSTEM_PROMPT,
        max_tokens=500,
        expected_tools=["read_file", "compute"],
    ),
    Scenario(
        name="tool_multi_step",
        description="Complex task requiring multiple tool calls",
        category="tool_agent",
        prompt="""Complete this workflow:
1. Query the database for all active users
2. For each user, call the analytics API to get their metrics
3. Read the threshold config file
4. Compute which users exceed the thresholds""",
        system_prompt=TOOL_AGENT_SYSTEM_PROMPT,
        max_tokens=800,
        expected_tools=["database_query", "api_call", "read_file", "compute"],
    ),
]


# ============================================================================
# Category 4: RAG Retrieval + Synthesis
# ============================================================================

# Simulated document chunks for RAG scenarios
RAG_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Company Benefits Overview",
        "content": """Our company offers comprehensive benefits including:
- Health insurance: PPO and HMO options, covering 80% of premiums
- 401(k): 4% company match, vested immediately
- PTO: 20 days for first 2 years, 25 days after
- Remote work: Hybrid model, 3 days in office minimum
- Professional development: $2000 annual budget"""
    },
    {
        "id": "doc2",
        "title": "Engineering Team Standards",
        "content": """Our engineering standards require:
- Code review: All PRs require 2 approvals
- Testing: 80% minimum code coverage
- Documentation: All public APIs must be documented
- Deployment: CI/CD pipeline with staging environment
- Security: OWASP top 10 compliance mandatory"""
    },
    {
        "id": "doc3",
        "title": "Product Roadmap Q1 2024",
        "content": """Q1 2024 priorities:
1. Mobile app launch - target March release
2. API v2 migration - deprecate v1 by February
3. New dashboard features - real-time analytics
4. Performance improvements - 50% latency reduction
5. Security audit - SOC 2 compliance preparation"""
    },
]

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use the retrieved documents to answer accurately. If the information isn't in the context, say so.

Retrieved Documents:
{documents}
"""


RAG_SCENARIOS = [
    Scenario(
        name="rag_simple_lookup",
        description="Simple fact lookup from documents",
        category="rag",
        prompt="What is the company's 401(k) match percentage?",
        system_prompt=RAG_SYSTEM_PROMPT.format(
            documents="\n\n".join(f"[{d['title']}]\n{d['content']}" for d in RAG_DOCUMENTS[:1])
        ),
        max_tokens=200,
        metadata={"num_documents": 1},
    ),
    Scenario(
        name="rag_multi_doc_synthesis",
        description="Synthesize information from multiple documents",
        category="rag",
        prompt="What are the key engineering practices and upcoming priorities for Q1?",
        system_prompt=RAG_SYSTEM_PROMPT.format(
            documents="\n\n".join(f"[{d['title']}]\n{d['content']}" for d in RAG_DOCUMENTS)
        ),
        max_tokens=500,
        metadata={"num_documents": 3},
    ),
    Scenario(
        name="rag_complex_question",
        description="Complex question requiring reasoning over documents",
        category="rag",
        prompt="""Based on the company information, what would you recommend for a new
engineer joining the team in terms of:
1. Understanding the codebase standards
2. Planning their first 90 days
3. Taking advantage of company benefits""",
        system_prompt=RAG_SYSTEM_PROMPT.format(
            documents="\n\n".join(f"[{d['title']}]\n{d['content']}" for d in RAG_DOCUMENTS)
        ),
        max_tokens=800,
        metadata={"num_documents": 3},
    ),
]


# ============================================================================
# Scenario Registry
# ============================================================================

ALL_SCENARIOS = {
    "simple_qa": SIMPLE_QA_SCENARIOS,
    "reasoning": REASONING_SCENARIOS,
    "tool_agent": TOOL_AGENT_SCENARIOS,
    "rag": RAG_SCENARIOS,
}


def get_scenario(name: str) -> Optional[Scenario]:
    """Get a scenario by name."""
    for category_scenarios in ALL_SCENARIOS.values():
        for scenario in category_scenarios:
            if scenario.name == name:
                return scenario
    return None


def get_scenarios_by_category(category: str) -> list[Scenario]:
    """Get all scenarios in a category."""
    return ALL_SCENARIOS.get(category, [])


def list_scenarios() -> dict[str, list[str]]:
    """List all available scenarios by category."""
    return {
        category: [s.name for s in scenarios]
        for category, scenarios in ALL_SCENARIOS.items()
    }


def get_baseline_scenarios() -> list[Scenario]:
    """Get one representative scenario from each category for baseline testing."""
    return [
        SIMPLE_QA_SCENARIOS[0],     # simple_factual
        REASONING_SCENARIOS[0],      # reasoning_math_word_problem
        TOOL_AGENT_SCENARIOS[2],     # tool_multi_step
        RAG_SCENARIOS[1],            # rag_multi_doc_synthesis
    ]
