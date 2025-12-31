# LLM Latency Lab

A comprehensive benchmark suite for testing and demonstrating various optimizations to improve LLM and agentic system latency.

## Overview

This project provides tools and benchmarks for measuring the impact of common latency optimization techniques when working with Large Language Models, specifically the Anthropic Claude API.

## Installation

```bash
pip install -r requirements.txt
```

### Authentication

This project uses the Claude Agent SDK with Claude Max account authentication. To set up:

1. Install Claude Code CLI:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. Authenticate with your Claude Max account:
   ```bash
   claude login
   ```

Once authenticated, the benchmarks will use your Claude Max subscription automatically.

## Quick Start

```bash
# Run a quick streaming benchmark
python main.py streaming --quick

# Run all benchmarks
python main.py all

# Run specific benchmark with custom settings
python main.py caching --runs 10 --model claude-sonnet-4-20250514
```

## Benchmark Categories

### 1. Streaming (`benchmarks/streaming/`)

Compares streaming vs non-streaming responses:
- **TTFT (Time to First Token)**: How quickly users see initial content
- **Perceived latency**: Up to 60-80% improvement with streaming

**CLI Command:**
```bash
python main.py streaming
```

```python
from benchmarks.streaming import compare_streaming_vs_non_streaming

results = await compare_streaming_vs_non_streaming(
    prompt="Explain quantum computing",
    num_runs=10
)
```

### 2. Prompt Caching (`benchmarks/caching/`)

Tests Anthropic's `cache_control` feature:
- **Cache warm-up patterns**: First request creates cache, subsequent requests hit it
- **Latency reduction**: Up to 85% on cache hits
- **Cost reduction**: Cached tokens at 10% of regular rate

**CLI Command:**
```bash
python main.py caching
```

```python
from benchmarks.caching import CachingBenchmarkSuite

suite = CachingBenchmarkSuite()
results = await suite.run_all()
```

### 3. Parallelism (`benchmarks/parallelism/`)

Tests parallel execution patterns:
- **Parallel tool calls**: Execute independent tools concurrently
- **Batch processing**: Process multiple prompts in parallel
- **Concurrency scaling**: How performance scales with concurrency

**CLI Command:**
```bash
python main.py parallel
```

```python
from benchmarks.parallelism import compare_parallel_vs_sequential

results = await compare_parallel_vs_sequential(num_runs=5)
```

### 4. Model Routing (`benchmarks/model_routing/`)

Tests small model → large model routing:
- **Classification accuracy**: How well small models identify task complexity
- **Latency/cost tradeoffs**: When routing saves time vs adds overhead

**CLI Command:**
```bash
python main.py routing
```

```python
from benchmarks.model_routing import compare_routing_strategies

results = await compare_routing_strategies(num_runs=3)
```

### 5. Agent Topology (`benchmarks/agent_topology/`)

Compares different multi-agent architectures:
- **Flat single agent**: One agent handles everything
- **Flat parallel**: Multiple specialists work in parallel
- **Hierarchical**: Supervisor coordinates specialists

**CLI Command:**
```bash
python main.py topology
```

```python
from benchmarks.agent_topology import compare_topologies

results = await compare_topologies(num_runs=3)
```

## Docker Compose Setup

A full-stack Docker application is available with a web UI, API, and LLM observability via self-hosted Langfuse v3.

### Architecture

```
Traefik v3.6 (Reverse Proxy :80)
├── app.localhost      → Streamlit (Dashboard UI)
│   └── /api/*         → FastAPI (Benchmark API)
├── langfuse.localhost → Langfuse Web (LLM Observability)
└── traefik.localhost  → Traefik Dashboard

Langfuse v3 Stack:
├── langfuse-web       → Web UI & API
├── langfuse-worker    → Background job processing
├── PostgreSQL         → Primary database
├── ClickHouse         → Analytics/traces storage
├── Redis              → Cache & queue
└── MinIO              → S3-compatible object storage
```

### Prerequisites

1. Ensure Claude Code CLI is installed and authenticated on your host:
   ```bash
   npm install -g @anthropic-ai/claude-code
   claude login
   ```

2. Docker and Docker Compose v2+ installed

### Quick Start

```bash
# From the repository root, start all services
docker compose -f docker/docker-compose.yml up -d --build

# View logs
docker compose -f docker/docker-compose.yml logs -f

# View logs for a specific service
docker compose -f docker/docker-compose.yml logs -f langfuse-web
```

### Access Points

Once running, access the services at:

| URL | Service |
|-----|---------|
| http://app.localhost | Streamlit Dashboard UI |
| http://app.localhost/api/docs | FastAPI Swagger Docs |
| http://langfuse.localhost | Langfuse Observability Console |
| http://traefik.localhost | Traefik Dashboard |

> **Note:** The `.localhost` TLD resolves to `127.0.0.1` automatically in modern browsers without `/etc/hosts` configuration.

### Environment Variables

Create a `docker/.env` file to customize secrets (recommended for production):

```bash
# PostgreSQL
POSTGRES_PASSWORD=your-secure-password

# ClickHouse
CLICKHOUSE_PASSWORD=your-clickhouse-password

# Redis
REDIS_AUTH=your-redis-password

# MinIO
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=your-minio-password

# Langfuse Auth (use long random strings)
NEXTAUTH_SECRET=your-nextauth-secret-min-32-chars
SALT=your-salt-for-api-keys
ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000  # 64 hex chars

# Langfuse API Keys (create in Langfuse UI after first login)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/benchmarks` | GET | List available benchmarks |
| `/api/benchmarks/{type}/run` | POST | Start a benchmark (returns job ID) |
| `/api/jobs` | GET | List all jobs |
| `/api/jobs/{job_id}` | GET | Get job status |
| `/api/ws/{job_id}` | WS | WebSocket for real-time progress |
| `/api/results` | GET | List saved results |
| `/api/results/{filename}` | GET | Get specific result |

### Services

| Container | Image | Purpose |
|-----------|-------|---------|
| llm-lab-traefik | traefik:v3.6 | Reverse proxy & routing |
| llm-lab-ui | docker-streamlit | Dashboard UI |
| llm-lab-api | docker-fastapi | Benchmark API |
| llm-lab-langfuse-web | langfuse/langfuse:3 | Langfuse web interface |
| llm-lab-langfuse-worker | langfuse/langfuse-worker:3 | Background processing |
| llm-lab-postgres | postgres:15 | Primary database |
| llm-lab-clickhouse | clickhouse/clickhouse-server:24.3 | Analytics database |
| llm-lab-redis | redis:7 | Cache & queue |
| llm-lab-minio | minio/minio:latest | Object storage |

### Stopping Services

```bash
# Stop all services
docker compose -f docker/docker-compose.yml down

# Stop and remove volumes (deletes all data)
docker compose -f docker/docker-compose.yml down -v
```

### Troubleshooting

**Traefik returning 404:**
- Ensure you're using Traefik v3.6+ (older versions have Docker socket issues on macOS)
- Check that Docker Desktop is running and the socket is accessible

**Langfuse not starting:**
- Wait for PostgreSQL, ClickHouse, Redis, and MinIO to become healthy first
- Check logs: `docker compose -f docker/docker-compose.yml logs langfuse-web`

## Project Structure

```
.
├── main.py                 # CLI entry point
├── requirements.txt
├── llm-latency-lab/
│   ├── benchmarks/
│   │   ├── streaming/      # TTFT vs full response comparison
│   │   ├── caching/        # Prompt caching, semantic caching
│   │   ├── parallelism/    # Parallel tool calls, async patterns
│   │   ├── model_routing/  # Small model → large model routing
│   │   └── agent_topology/ # Flat vs hierarchical supervisor
│   ├── instrumentation/
│   │   ├── timing.py       # Decorators, context managers
│   │   ├── traces.py       # OpenTelemetry / Langfuse integration
│   │   └── claude_sdk_client.py
│   ├── harness/
│   │   ├── runner.py       # Benchmark orchestrator
│   │   └── reporter.py     # Results aggregation, visualization
│   ├── scenarios/
│   │   └── definitions.py  # Realistic task definitions
│   └── results/            # Stored benchmark outputs
└── docker/
    ├── docker-compose.yml  # Full stack orchestration
    ├── .env.example        # Environment template
    ├── traefik/
    │   └── traefik.yml     # Traefik static config
    ├── fastapi/
    │   ├── Dockerfile
    │   └── app/            # Benchmark API source
    └── streamlit/
        ├── Dockerfile
        └── app.py          # Dashboard UI source
```

## Instrumentation

### Timing Utilities

```python
from instrumentation.timing import timed, async_timed, Timer, StreamingTimer

# Context manager
async with async_timed("my_operation") as timer:
    await do_something()
print(f"Elapsed: {timer.elapsed_ms}ms")

# For streaming responses
timer = StreamingTimer("streaming")
timer.start()
async for chunk in stream:
    timer.record_chunk()
timer.stop()
print(f"TTFT: {timer.ttft_ms}ms")
```

### Tracing Integration

```python
from instrumentation.traces import init_tracing, TracingConfig

# Enable OpenTelemetry tracing
config = TracingConfig(
    service_name="my-benchmark",
    enable_console_export=True,
    enable_langfuse=False,
)
init_tracing(config)
```

## Statistical Analysis

Each benchmark runs multiple iterations and reports:
- **p50/p95/p99 latencies**: Percentile distributions
- **Mean/min/max**: Basic statistics
- **Token throughput**: Tokens per second
- **Cache hit rates**: For caching benchmarks

Results are saved as JSON for further analysis.

## Example Results

Running streaming benchmark:
```
======================================================
Benchmark Comparison
======================================================
Config                   p50          p95          p99          TTFT p50
streaming               1234ms       1456ms       1578ms       89ms
non_streaming           1345ms       1567ms       1689ms       N/A

Perceived latency improvement from streaming: 93.4%
User sees first content in 89ms vs waiting 1345ms
```

## License

MIT License

Copyright (c) 2025 Finian O'Neill

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
