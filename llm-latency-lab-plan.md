# LLM Latency Lab — Project Plan

**Reference ID:** `llm-latency-lab-2024-12-29`  
**Context:** Planning document for a demo repo that demonstrates and tests various optimizations to improve LLM and agentic system latency.

---

## Repository Structure

```
llm-latency-lab/
├── benchmarks/
│   ├── streaming/          # TTFT vs full response comparison
│   ├── caching/            # Prompt caching, semantic caching
│   ├── parallelism/        # Parallel tool calls, async patterns
│   ├── model_routing/      # Small model → large model routing
│   └── agent_topology/     # Flat vs hierarchical supervisor
├── instrumentation/
│   ├── timing.py           # Decorators, context managers
│   └── traces.py           # OpenTelemetry / Langfuse integration
├── harness/
│   ├── runner.py           # Benchmark orchestrator
│   └── reporter.py         # Results aggregation, visualization
├── scenarios/
│   └── ...                 # Realistic task definitions
└── results/
    └── ...                 # Stored benchmark outputs
```

---

## Key Optimization Categories to Test

| Category | Techniques | Expected Impact |
|----------|------------|-----------------|
| **Streaming** | TTFT measurement, chunked processing | Perceived latency ↓ 60-80% |
| **Prompt Caching** | Anthropic cache_control, prefix caching | Latency ↓ up to 85% on cache hits |
| **Parallelism** | Async tool execution, concurrent agent calls | Wall-clock ↓ proportional to parallelizable work |
| **Model Routing** | Haiku for classification → Sonnet for generation | Cost ↓, latency ↓ for simple tasks |
| **Speculative Execution** | Pre-fetch likely tool results | Latency ↓ when predictions accurate |
| **Connection Reuse** | HTTP keep-alive, connection pooling | ~50-150ms saved per request |

---

## Implementation Approach

### 1. Instrumentation First
Create a clean timing harness that captures:
- **TTFT** (Time to First Token)
- **Total latency**
- **Token throughput** (tokens/sec)
- **Cache hit rates**

OpenTelemetry spans work well here, optionally push to Langfuse for tracing.

### 2. Baseline Scenarios
Define 3-5 representative tasks:
- Simple Q&A
- Multi-step reasoning
- Tool-heavy agent loop
- RAG retrieval + synthesis

Run these unoptimized to establish baselines.

### 3. Optimization Toggles
Implement each optimization as a configurable flag for clean A/B testing:

```python
@benchmark(streaming=True, cache_control=True, parallel_tools=True)
async def run_scenario(config: BenchmarkConfig): ...
```

### 4. Statistical Rigor
- Run each configuration N times (20-50 runs recommended)
- Report p50/p95/p99 latencies
- Variance analysis matters

### 5. Visualization
- CLI table output for quick comparison
- Matplotlib PNGs or lightweight dashboard for visual reports

---

## Next Steps / Open Questions

1. **Breadth vs Depth:** Start with full skeleton or deep-dive on one optimization?
2. **Tech Stack Options:**
   - FastAPI + async patterns
   - LangGraph-specific benchmarks (supervisor overhead, checkpoint latency)
3. **Priority Optimization:** Prompt caching often yields the biggest win

---

## Claude Code Reference

When you're ready to build, you can reference this plan:

```bash
# In Claude Code, you can say:
# "Let's build the llm-latency-lab repo based on our earlier plan"
# or reference this file directly if you've downloaded it
```

---

*Generated: 2024-12-29*
