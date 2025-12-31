"""Benchmark API endpoints with job management.

All benchmark runs are traced to Langfuse when configured.
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.models import (
    BenchmarkType,
    BenchmarkRequest,
    JobResponse,
    JobStatus,
    JobState,
    JobListResponse,
    BenchmarkInfo,
    ProgressUpdate,
)

# Import Langfuse tracing utilities
try:
    from instrumentation.traces import (
        get_langfuse_tracer,
        flush_langfuse,
        LANGFUSE_AVAILABLE,
    )
except ImportError:
    LANGFUSE_AVAILABLE = False
    get_langfuse_tracer = lambda: None
    flush_langfuse = lambda: None

RESULTS_DIR = Path("/app/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api", tags=["benchmarks"])


def save_result_to_file(job: JobStatus) -> str:
    """Save job results to a JSON file. Returns the filename."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{job.benchmark_type.value}_{timestamp}_{job.job_id}.json"
    filepath = RESULTS_DIR / filename

    result_data = {
        "job_id": job.job_id,
        "benchmark_type": job.benchmark_type.value,
        "timestamp": timestamp,
        "config": {
            "model": job.request.model,
            "runs": job.request.runs,
            "max_tokens": job.request.max_tokens,
            "prompt": job.request.prompt,
        },
        "results": job.results,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }

    # Extract stats from results if available
    if job.results and isinstance(job.results, list) and len(job.results) > 0:
        first_result = job.results[0]
        if isinstance(first_result, dict) and "stats" in first_result:
            result_data["stats"] = first_result["stats"]

    with open(filepath, "w") as f:
        json.dump(result_data, f, indent=2, default=str)

    return filename

# In-memory job store
jobs: dict[str, JobStatus] = {}

# WebSocket connections per job
websocket_connections: dict[str, list[WebSocket]] = {}


class WebSocketManager:
    """Manages WebSocket connections for job progress updates."""

    def __init__(self):
        self.connections: dict[str, list[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        """Accept a new WebSocket connection for a job."""
        await websocket.accept()
        if job_id not in self.connections:
            self.connections[job_id] = []
        self.connections[job_id].append(websocket)

    def disconnect(self, job_id: str, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if job_id in self.connections:
            if websocket in self.connections[job_id]:
                self.connections[job_id].remove(websocket)
            if not self.connections[job_id]:
                del self.connections[job_id]

    async def broadcast(self, job_id: str, message: dict):
        """Send a message to all connections for a job."""
        if job_id in self.connections:
            disconnected = []
            for ws in self.connections[job_id]:
                try:
                    await ws.send_json(message)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                self.disconnect(job_id, ws)


ws_manager = WebSocketManager()


# Available benchmarks info
BENCHMARK_INFO = {
    BenchmarkType.STREAMING: BenchmarkInfo(
        type=BenchmarkType.STREAMING,
        name="Streaming Benchmark",
        description="Compare streaming vs non-streaming responses, measuring TTFT",
        expected_duration="2-5 minutes",
        metrics=["TTFT", "Total Latency", "Tokens/sec"],
    ),
    BenchmarkType.CACHING: BenchmarkInfo(
        type=BenchmarkType.CACHING,
        name="Prompt Caching Benchmark",
        description="Test Anthropic's cache_control feature for latency reduction",
        expected_duration="3-6 minutes",
        metrics=["Cache Hit Rate", "Latency Reduction", "Cost Savings"],
    ),
    BenchmarkType.PARALLEL: BenchmarkInfo(
        type=BenchmarkType.PARALLEL,
        name="Parallelism Benchmark",
        description="Compare parallel vs sequential tool execution",
        expected_duration="2-4 minutes",
        metrics=["Execution Time", "Concurrency Scaling"],
    ),
    BenchmarkType.ROUTING: BenchmarkInfo(
        type=BenchmarkType.ROUTING,
        name="Model Routing Benchmark",
        description="Test small model routing to large model patterns",
        expected_duration="3-5 minutes",
        metrics=["Classification Accuracy", "Latency/Cost Tradeoff"],
    ),
    BenchmarkType.TOPOLOGY: BenchmarkInfo(
        type=BenchmarkType.TOPOLOGY,
        name="Agent Topology Benchmark",
        description="Compare flat vs hierarchical multi-agent architectures",
        expected_duration="4-8 minutes",
        metrics=["Total Latency", "Coordination Overhead"],
    ),
    BenchmarkType.ALL: BenchmarkInfo(
        type=BenchmarkType.ALL,
        name="All Benchmarks",
        description="Run all benchmark suites sequentially",
        expected_duration="15-30 minutes",
        metrics=["All metrics from individual benchmarks"],
    ),
}


def create_progress_callback(job_id: str) -> Callable:
    """Create a progress callback that updates job status and broadcasts to WebSockets."""

    async def callback(
        current_run: int,
        total_runs: int,
        benchmark_name: str,
        message: str = "",
        metrics: dict = None,
    ):
        if job_id in jobs:
            job = jobs[job_id]
            job.progress = ProgressUpdate(
                job_id=job_id,
                state=JobState.RUNNING,
                current_run=current_run,
                total_runs=total_runs,
                current_benchmark=benchmark_name,
                message=message,
                metrics=metrics or {},
            )
            await ws_manager.broadcast(job_id, job.progress.model_dump(mode="json"))

    return callback


async def run_benchmark_task(job_id: str, benchmark_type: BenchmarkType, request: BenchmarkRequest):
    """Background task to run a benchmark.

    Individual LLM calls are traced to Langfuse automatically.
    """
    job = jobs[job_id]
    job.state = JobState.RUNNING
    job.started_at = datetime.utcnow()
    job.progress.state = JobState.RUNNING

    # Initialize Langfuse tracer (ensures env vars are set for individual traces)
    if LANGFUSE_AVAILABLE:
        get_langfuse_tracer()

    try:
        # Import benchmark modules
        if benchmark_type == BenchmarkType.STREAMING:
            from benchmarks.streaming import compare_streaming_vs_non_streaming

            results = await compare_streaming_vs_non_streaming(
                prompt=request.prompt or "Explain quantum computing in 3 paragraphs.",
                model=request.model,
                max_tokens=request.max_tokens,
                num_runs=request.runs,
            )
            job.results = [
                {"name": "streaming", "stats": results["streaming"].stats},
                {"name": "non_streaming", "stats": results["non_streaming"].stats},
            ]

        elif benchmark_type == BenchmarkType.CACHING:
            from benchmarks.caching import CachingBenchmarkSuite

            suite = CachingBenchmarkSuite(model=request.model)
            results = await suite.run_all(num_runs=request.runs)
            job.results = results

        elif benchmark_type == BenchmarkType.PARALLEL:
            from benchmarks.parallelism import ParallelismBenchmarkSuite

            suite = ParallelismBenchmarkSuite(model=request.model)
            results = await suite.run_all(num_runs=request.runs)
            job.results = results

        elif benchmark_type == BenchmarkType.ROUTING:
            from benchmarks.model_routing import RoutingBenchmarkSuite

            suite = RoutingBenchmarkSuite()
            results = await suite.run_all(num_runs=request.runs)
            job.results = results

        elif benchmark_type == BenchmarkType.TOPOLOGY:
            from benchmarks.agent_topology import AgentTopologyBenchmarkSuite

            suite = AgentTopologyBenchmarkSuite(model=request.model)
            results = await suite.run_all(num_runs=request.runs)
            job.results = results

        elif benchmark_type == BenchmarkType.ALL:
            # Run all benchmarks sequentially
            all_results = {}

            for btype in [
                BenchmarkType.STREAMING,
                BenchmarkType.CACHING,
                BenchmarkType.PARALLEL,
                BenchmarkType.ROUTING,
                BenchmarkType.TOPOLOGY,
            ]:
                job.progress.current_benchmark = btype.value
                job.progress.message = f"Running {btype.value} benchmark..."
                await ws_manager.broadcast(job_id, job.progress.model_dump(mode="json"))

                # Recursively run each benchmark type
                sub_request = request.model_copy()
                await run_benchmark_task(f"{job_id}_{btype.value}", btype, sub_request)
                all_results[btype.value] = jobs.get(f"{job_id}_{btype.value}", {})

            job.results = all_results

        job.state = JobState.COMPLETED
        job.completed_at = datetime.utcnow()
        job.progress.state = JobState.COMPLETED
        job.progress.message = "Benchmark completed successfully"

        # Save results to file for persistence
        try:
            filename = save_result_to_file(job)
            job.result_file = filename
            job.progress.message = f"Benchmark completed. Results saved to {filename}"
        except Exception as save_error:
            job.progress.message = f"Benchmark completed but failed to save: {save_error}"

    except Exception as e:
        job.state = JobState.FAILED
        job.completed_at = datetime.utcnow()
        job.error = str(e)
        job.progress.state = JobState.FAILED
        job.progress.message = f"Benchmark failed: {e}"

    finally:
        # Flush Langfuse traces
        if LANGFUSE_AVAILABLE:
            flush_langfuse()

    # Broadcast final status
    await ws_manager.broadcast(job_id, job.progress.model_dump(mode="json"))


@router.get("/benchmarks", response_model=list[BenchmarkInfo])
async def list_benchmarks():
    """List all available benchmark types."""
    return list(BENCHMARK_INFO.values())


@router.get("/benchmarks/{benchmark_type}", response_model=BenchmarkInfo)
async def get_benchmark_info(benchmark_type: BenchmarkType):
    """Get information about a specific benchmark type."""
    if benchmark_type not in BENCHMARK_INFO:
        raise HTTPException(status_code=404, detail="Benchmark type not found")
    return BENCHMARK_INFO[benchmark_type]


@router.post("/benchmarks/{benchmark_type}/run", response_model=JobResponse)
async def run_benchmark(benchmark_type: BenchmarkType, request: BenchmarkRequest = None):
    """Start a benchmark run. Returns immediately with a job ID."""
    if request is None:
        request = BenchmarkRequest()

    if request.quick:
        request.runs = 3

    job_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow()

    job = JobStatus(
        job_id=job_id,
        benchmark_type=benchmark_type,
        state=JobState.PENDING,
        request=request,
        created_at=now,
        progress=ProgressUpdate(
            job_id=job_id,
            state=JobState.PENDING,
            message="Job created, waiting to start...",
        ),
    )
    jobs[job_id] = job

    # Start background task
    asyncio.create_task(run_benchmark_task(job_id, benchmark_type, request))

    return JobResponse(
        job_id=job_id,
        message=f"Benchmark '{benchmark_type.value}' started",
        websocket_url=f"/api/ws/{job_id}",
    )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all jobs (running and completed)."""
    return JobListResponse(
        jobs=list(jobs.values()),
        total=len(jobs),
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a specific job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.state == JobState.RUNNING:
        job.state = JobState.CANCELLED
        job.completed_at = datetime.utcnow()
        job.progress.state = JobState.CANCELLED
        job.progress.message = "Job cancelled by user"
        await ws_manager.broadcast(job_id, job.progress.model_dump(mode="json"))

    return {"message": "Job cancelled"}


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress updates."""
    if job_id not in jobs:
        await websocket.close(code=4004, reason="Job not found")
        return

    await ws_manager.connect(job_id, websocket)

    try:
        # Send current status immediately
        job = jobs[job_id]
        await websocket.send_json(job.progress.model_dump(mode="json"))

        # Keep connection alive and wait for messages
        while True:
            try:
                # Wait for any message (ping/pong handled automatically)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back for ping
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, websocket)
    except Exception:
        ws_manager.disconnect(job_id, websocket)
