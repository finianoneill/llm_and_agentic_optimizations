# LLM Latency Lab

A comprehensive benchmark suite for testing and demonstrating various optimizations to improve LLM and agentic system latency, with built-in observability via **Langfuse**.

## Overview

This project provides tools and benchmarks for measuring the impact of common latency optimization techniques when working with Large Language Models, specifically the Anthropic Claude API.

### Key Features

- **Latency Benchmarks**: Streaming, caching, parallelism, model routing, and agent topology
- **Langfuse Integration**: Automatic tracing of all LLM calls with token usage, latency metrics, and cost tracking
- **Claude Max Support**: Uses Claude Agent SDK with consumer subscription authentication
- **Docker Stack**: Full-stack deployment with self-hosted Langfuse v3 for complete observability
- **PostgreSQL App Database**: Persistent storage for benchmark jobs, results, and timing data with Alembic migrations

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

All benchmarks automatically trace LLM calls to Langfuse when configured, capturing token usage, latency metrics, and cost data.

### 1. Streaming (`benchmarks/streaming/`)

Compares streaming vs non-streaming responses:
- **TTFT (Time to First Token)**: How quickly users see initial content
- **Perceived latency**: Up to 60-80% improvement with streaming
- **Langfuse Traces**: Captures TTFT, total latency, and chunk timing

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
- **Langfuse Traces**: Tracks cache creation and read token counts

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

A full-stack Docker application is available with a web UI, API, and **complete LLM observability via self-hosted Langfuse v3**.

### Langfuse Observability

The Docker stack includes a fully self-hosted Langfuse v3 installation for LLM observability:

- **Trace Dashboard**: View all LLM calls with latency, tokens, and costs
- **Session Tracking**: Group related benchmark runs into sessions
- **Prompt Management**: Compare prompt variations and their performance
- **Analytics**: Aggregate metrics across benchmark runs

Access the Langfuse console at `http://langfuse.localhost` after starting the stack.

### Architecture

```
Traefik v3.6 (Reverse Proxy :80)
├── app.localhost      → Streamlit (Dashboard UI)
│   └── /api/*         → FastAPI (Benchmark API)
├── langfuse.localhost → Langfuse Web (LLM Observability)
└── traefik.localhost  → Traefik Dashboard

Application Stack:
├── app-postgres       → Benchmark jobs, results & timing data (PostgreSQL 15)
└── FastAPI            → Benchmark API with SQLAlchemy ORM

Langfuse v3 Stack:
├── langfuse-web       → Web UI & API
├── langfuse-worker    → Background job processing
├── PostgreSQL         → Langfuse database (separate from app DB)
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
# 1. Start the Claude proxy on your host (required for Claude Max auth)
cd docker/claude-proxy
pip install -r requirements.txt
python proxy.py &

# 2. From the repository root, start all services
cd ../..
docker compose -f docker/docker-compose.yml up -d --build

# View logs
docker compose -f docker/docker-compose.yml logs -f

# View logs for a specific service
docker compose -f docker/docker-compose.yml logs -f langfuse-web
```

> **Note:** The Claude proxy runs on your host machine and provides access to Claude Max authentication via macOS Keychain. The Docker containers connect to it via `host.docker.internal:8765`.

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
# Application Database (benchmark data)
APP_DB_PASSWORD=your-app-db-password

# Langfuse PostgreSQL
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

#### Setting Up Langfuse API Keys

1. Start the Docker stack: `docker compose -f docker/docker-compose.yml up -d`
2. Navigate to `http://langfuse.localhost` and create an account
3. Create a new project and generate API keys (Settings → API Keys)
4. Add the keys to your `docker/.env` file
5. Restart the API container: `docker compose -f docker/docker-compose.yml restart llm-lab-api`

Once configured, all benchmark LLM calls will appear in your Langfuse dashboard with full tracing.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check (includes Langfuse status) |
| `/api/benchmarks` | GET | List available benchmarks |
| `/api/benchmarks/{type}/run` | POST | Start a benchmark (returns job ID) |
| `/api/jobs` | GET | List all jobs |
| `/api/jobs/{job_id}` | GET | Get job status |
| `/api/ws/{job_id}` | WS | WebSocket for real-time progress |
| `/api/results` | GET | List saved results (from database) |
| `/api/results/{job_id}` | GET | Get results for a specific job |
| `/api/results/{job_id}/csv` | GET | Export job timing data as CSV |
| `/api/results/export/csv` | GET | Export all timing data as CSV |

### Services

| Container | Image | Purpose |
|-----------|-------|---------|
| llm-lab-traefik | traefik:v3.6 | Reverse proxy & routing |
| llm-lab-ui | docker-streamlit | Dashboard UI |
| llm-lab-api | docker-fastapi | Benchmark API |
| llm-lab-app-db | postgres:15 | Application database (jobs, results, timing) |
| llm-lab-langfuse-web | langfuse/langfuse:3 | Langfuse web interface |
| llm-lab-langfuse-worker | langfuse/langfuse-worker:3 | Background processing |
| llm-lab-postgres | postgres:15 | Langfuse database |
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

### Application Database

The application uses a dedicated PostgreSQL database (`llm-lab-app-db`) separate from Langfuse's database. This stores all benchmark jobs, results, and individual timing measurements.

#### Database Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ jobs                                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ id (PK)           │ benchmark_type │ state    │ model      │ runs          │
│ max_tokens        │ prompt         │ quick    │ created_at │ started_at    │
│ completed_at      │ error          │          │            │               │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        │ 1:N
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ benchmark_results                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ id (PK)           │ job_id (FK)    │ name     │ description │ success_rate │
│ stats (JSON)      │ start_time     │ end_time │ errors      │ created_at   │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        │ 1:N
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ timing_results                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ id (PK)           │ benchmark_result_id (FK) │ name           │             │
│ total_latency_ms  │ ttft_ms                  │ input_tokens   │             │
│ output_tokens     │ tokens_per_second        │ cache_hit      │             │
│ cache_hit_rate    │ created_at               │                │             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Alembic Migrations

The database schema is managed with Alembic. Migrations run automatically when the FastAPI container starts.

```bash
# View migration history
docker exec llm-lab-api alembic history

# Create a new migration (after modifying models)
docker exec llm-lab-api alembic revision --autogenerate -m "description"

# Apply pending migrations manually
docker exec llm-lab-api alembic upgrade head

# Rollback one migration
docker exec llm-lab-api alembic downgrade -1
```

#### Accessing the Database

```bash
# Connect to the app database
docker exec -it llm-lab-app-db psql -U llm_lab -d llm_lab

# Example queries
SELECT * FROM jobs ORDER BY created_at DESC LIMIT 10;
SELECT COUNT(*) FROM timing_results;
```

### Claude Proxy (Required for Claude Max)

The Docker containers cannot access the macOS Keychain where Claude Max credentials are stored. A lightweight proxy server bridges this gap by running on your host machine.

**How it works:**
```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────┐
│  Docker Container   │────▶│   Claude Proxy      │────▶│  Claude CLI │
│  (llm-lab-api)      │     │   (host:8765)       │     │  (Keychain) │
└─────────────────────┘     └─────────────────────┘     └─────────────┘
        Calls                    Forwards to               Authenticated
   host.docker.internal          claude CLI                via macOS
```

**Starting the proxy:**

```bash
# Terminal 1: Start the proxy (keep running)
cd docker/claude-proxy
pip install -r requirements.txt
python proxy.py
```

You should see:
```
Starting Claude CLI Proxy on http://localhost:8765
Container can reach this at http://host.docker.internal:8765
```

**Testing the proxy:**

```bash
# Test health endpoint
curl http://localhost:8765/health

# Test a query
curl -X POST http://localhost:8765/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Say hello", "model": "haiku"}'
```

**Proxy endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Send a prompt to Claude |
| `/query/stream` | POST | Streaming response (SSE) |

**Running as a background service (optional):**

```bash
# Using nohup
nohup python docker/claude-proxy/proxy.py > /tmp/claude-proxy.log 2>&1 &

# Check if running
curl http://localhost:8765/health

# Stop the proxy
pkill -f "python.*proxy.py"
```

### Troubleshooting

**Traefik returning 404:**
- Ensure you're using Traefik v3.6+ (older versions have Docker socket issues on macOS)
- Check that Docker Desktop is running and the socket is accessible

**Langfuse not starting:**
- Wait for PostgreSQL, ClickHouse, Redis, and MinIO to become healthy first
- Check logs: `docker compose -f docker/docker-compose.yml logs langfuse-web`

**Benchmarks failing with "Connection refused":**
- Ensure the Claude proxy is running on the host: `curl http://localhost:8765/health`
- Check that the container can reach the host: `docker exec llm-lab-api curl http://host.docker.internal:8765/health`

**Claude proxy returning "Invalid API key":**
- Run `claude login` on your host to authenticate
- Verify authentication: `claude -p "test" --output-format json`

**Langfuse traces not appearing:**
- Check API keys are set: `docker exec llm-lab-api env | grep LANGFUSE`
- Verify Langfuse is healthy: `curl http://langfuse.localhost/api/public/health`
- Check API logs for tracing errors: `docker compose -f docker/docker-compose.yml logs llm-lab-api | grep Langfuse`

**App database connection issues:**
- Check database is healthy: `docker exec llm-lab-app-db pg_isready -U llm_lab`
- Verify DATABASE_URL: `docker exec llm-lab-api env | grep DATABASE_URL`
- Check migration status: `docker exec llm-lab-api alembic current`
- View database logs: `docker compose -f docker/docker-compose.yml logs llm-lab-app-db`

**Missing benchmark results:**
- Ensure the app-postgres container is running: `docker ps | grep llm-lab-app-db`
- Check for migration errors in API logs: `docker compose -f docker/docker-compose.yml logs llm-lab-api | grep -i alembic`

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
│   │   ├── timing.py       # Timer utilities for latency measurement
│   │   ├── traces.py       # Langfuse tracing and @observe decorator
│   │   └── claude_sdk_client.py  # Claude client with auto-tracing
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
    │   ├── alembic.ini     # Alembic configuration
    │   ├── alembic/        # Database migrations
    │   │   ├── env.py
    │   │   └── versions/   # Migration scripts
    │   └── app/
    │       ├── main.py     # FastAPI application
    │       ├── db/         # Database layer
    │       │   ├── models.py   # SQLAlchemy ORM models
    │       │   ├── crud.py     # CRUD operations
    │       │   └── session.py  # Database session management
    │       └── routers/    # API route handlers
    ├── streamlit/
    │   ├── Dockerfile
    │   └── app.py          # Dashboard UI source
    └── claude-proxy/
        ├── proxy.py        # Host proxy for Claude Max auth
        └── requirements.txt
```

## Instrumentation

### Langfuse LLM Observability

All LLM calls in the benchmark suite are **automatically traced to Langfuse** when configured. This provides:

- **Token Usage Tracking**: Input, output, and cached token counts per call
- **Latency Metrics**: Total latency, time-to-first-token (TTFT) for streaming
- **Cost Tracking**: Automatic cost calculation based on token usage
- **Trace Hierarchy**: Nested spans for complex benchmark operations

#### Configuration

Set these environment variables to enable Langfuse tracing:

```bash
export LANGFUSE_HOST=http://langfuse.localhost  # Or https://cloud.langfuse.com
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
```

#### Automatic Tracing

The `ClaudeMaxClient` automatically traces all LLM calls:

```python
from instrumentation.claude_sdk_client import ClaudeMaxClient

client = ClaudeMaxClient(model="sonnet")

# This call is automatically traced to Langfuse
response = await client.create_message(
    prompt="Explain quantum computing",
    trace_name="my_operation",  # Optional: custom trace name
    trace_metadata={"benchmark": "streaming"},  # Optional: custom metadata
)
```

#### Using the @observe Decorator

For custom functions, use the Langfuse `@observe` decorator:

```python
from instrumentation.traces import get_observe_decorator

observe = get_observe_decorator()

@observe(name="my_benchmark_function")
async def run_benchmark():
    # All LLM calls within are nested under this trace
    ...
```

### Timing Utilities

```python
from instrumentation.timing import Timer, StreamingTimer

# For non-streaming operations
timer = Timer("my_operation")
timer.start()
result = await do_something()
timer.stop()
print(f"Elapsed: {timer.elapsed_ms}ms")

# For streaming responses with TTFT tracking
timer = StreamingTimer("streaming")
timer.start()
async for chunk in stream:
    timer.record_chunk()
timer.stop()
print(f"TTFT: {timer.ttft_ms}ms, Total: {timer.elapsed_ms}ms")
```

### OpenTelemetry Integration

For additional distributed tracing beyond Langfuse:

```python
from instrumentation.traces import init_tracing, TracingConfig

config = TracingConfig(
    service_name="my-benchmark",
    enable_console_export=True,
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
