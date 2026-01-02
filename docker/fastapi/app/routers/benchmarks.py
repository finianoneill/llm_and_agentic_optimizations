"""Benchmark API endpoints with job management.

All benchmark runs are traced to Langfuse when configured.
Jobs are persisted to PostgreSQL database.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

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
from app.db.session import get_db
from app.db import crud

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


router = APIRouter(prefix="/api", tags=["benchmarks"])


# In-memory store for active jobs (needed for WebSocket real-time updates)
# Jobs are persisted to database on completion
active_jobs: dict[str, JobStatus] = {}


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


def job_to_status(job, results=None) -> JobStatus:
    """Convert database job to JobStatus Pydantic model."""
    return JobStatus(
        job_id=job.id,
        benchmark_type=BenchmarkType(job.benchmark_type),
        state=JobState(job.state),
        request=BenchmarkRequest(
            model=job.model,
            runs=job.runs,
            max_tokens=job.max_tokens,
            prompt=job.prompt,
            quick=job.quick,
        ),
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=ProgressUpdate(
            job_id=job.id,
            state=JobState(job.state),
            message="Job completed" if job.state == "completed" else job.error or "",
        ),
        results=[r.to_dict(include_model=job.model) for r in results] if results else None,
        error=job.error,
    )


def save_results_to_db(job_id: str, results: list, db: Session):
    """Save benchmark results to database."""
    for result in results:
        if isinstance(result, dict):
            name = result.get("name", "unknown")
            stats = result.get("stats", {})

            # Convert stats object to dict if needed
            if hasattr(stats, "__dict__"):
                stats = vars(stats)
            elif hasattr(stats, "model_dump"):
                stats = stats.model_dump()

            crud.save_benchmark_result(
                db=db,
                job_id=job_id,
                name=name,
                stats=stats,
                description=result.get("description", ""),
                success_rate=result.get("success_rate", 1.0),
                errors=result.get("errors", []),
            )


async def run_benchmark_task(job_id: str, benchmark_type: BenchmarkType, request: BenchmarkRequest):
    """Background task to run a benchmark.

    Individual LLM calls are traced to Langfuse automatically.
    Results are saved to PostgreSQL on completion.
    """
    from app.db.session import SessionLocal

    job = active_jobs[job_id]
    job.state = JobState.RUNNING
    job.started_at = datetime.utcnow()
    job.progress.state = JobState.RUNNING

    # Update database
    db = SessionLocal()
    try:
        crud.update_job_state(db, job_id, "running", started_at=job.started_at)
    finally:
        db.close()

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
            all_results = []

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

                # Run each benchmark
                if btype == BenchmarkType.STREAMING:
                    from benchmarks.streaming import compare_streaming_vs_non_streaming
                    results = await compare_streaming_vs_non_streaming(
                        prompt=request.prompt or "Explain quantum computing in 3 paragraphs.",
                        model=request.model,
                        max_tokens=request.max_tokens,
                        num_runs=request.runs,
                    )
                    all_results.extend([
                        {"name": "streaming", "stats": results["streaming"].stats},
                        {"name": "non_streaming", "stats": results["non_streaming"].stats},
                    ])
                elif btype == BenchmarkType.CACHING:
                    from benchmarks.caching import CachingBenchmarkSuite
                    suite = CachingBenchmarkSuite(model=request.model)
                    all_results.extend(await suite.run_all(num_runs=request.runs))
                elif btype == BenchmarkType.PARALLEL:
                    from benchmarks.parallelism import ParallelismBenchmarkSuite
                    suite = ParallelismBenchmarkSuite(model=request.model)
                    all_results.extend(await suite.run_all(num_runs=request.runs))
                elif btype == BenchmarkType.ROUTING:
                    from benchmarks.model_routing import RoutingBenchmarkSuite
                    suite = RoutingBenchmarkSuite()
                    all_results.extend(await suite.run_all(num_runs=request.runs))
                elif btype == BenchmarkType.TOPOLOGY:
                    from benchmarks.agent_topology import AgentTopologyBenchmarkSuite
                    suite = AgentTopologyBenchmarkSuite(model=request.model)
                    all_results.extend(await suite.run_all(num_runs=request.runs))

            job.results = all_results

        job.state = JobState.COMPLETED
        job.completed_at = datetime.utcnow()
        job.progress.state = JobState.COMPLETED
        job.progress.message = "Benchmark completed successfully"

        # Save results to database
        db = SessionLocal()
        try:
            crud.update_job_state(
                db, job_id, "completed",
                completed_at=job.completed_at,
            )
            if job.results:
                save_results_to_db(job_id, job.results, db)
            job.progress.message = "Benchmark completed. Results saved to database."
        except Exception as save_error:
            job.progress.message = f"Benchmark completed but failed to save: {save_error}"
        finally:
            db.close()

    except Exception as e:
        job.state = JobState.FAILED
        job.completed_at = datetime.utcnow()
        job.error = str(e)
        job.progress.state = JobState.FAILED
        job.progress.message = f"Benchmark failed: {e}"

        # Update database with error
        db = SessionLocal()
        try:
            crud.update_job_state(
                db, job_id, "failed",
                completed_at=job.completed_at,
                error=str(e),
            )
        finally:
            db.close()

    finally:
        # Flush Langfuse traces
        if LANGFUSE_AVAILABLE:
            flush_langfuse()

    # Broadcast final status
    await ws_manager.broadcast(job_id, job.progress.model_dump(mode="json"))

    # Remove from active jobs after a delay (keep for WebSocket reconnection)
    await asyncio.sleep(60)
    if job_id in active_jobs:
        del active_jobs[job_id]


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
async def run_benchmark(
    benchmark_type: BenchmarkType,
    request: BenchmarkRequest = None,
    db: Session = Depends(get_db),
):
    """Start a benchmark run. Returns immediately with a job ID."""
    if request is None:
        request = BenchmarkRequest()

    if request.quick:
        request.runs = 3

    job_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow()

    # Create job in database
    crud.create_job(
        db=db,
        job_id=job_id,
        benchmark_type=benchmark_type.value,
        model=request.model,
        runs=request.runs,
        max_tokens=request.max_tokens,
        prompt=request.prompt,
        quick=request.quick,
    )

    # Create in-memory job for WebSocket updates
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
    active_jobs[job_id] = job

    # Start background task
    asyncio.create_task(run_benchmark_task(job_id, benchmark_type, request))

    return JobResponse(
        job_id=job_id,
        message=f"Benchmark '{benchmark_type.value}' started",
        websocket_url=f"/api/ws/{job_id}",
    )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(db: Session = Depends(get_db)):
    """List all jobs (running and completed)."""
    # Get jobs from database
    db_jobs = crud.get_jobs(db)
    total = crud.get_job_count(db)

    # Merge with active jobs (which have real-time status)
    jobs_list = []
    for db_job in db_jobs:
        if db_job.id in active_jobs:
            # Use active job for real-time status
            jobs_list.append(active_jobs[db_job.id])
        else:
            # Use database job
            results = crud.get_results_for_job(db, db_job.id)
            jobs_list.append(job_to_status(db_job, results))

    return JobListResponse(
        jobs=jobs_list,
        total=total,
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get the status of a specific job."""
    # Check active jobs first (for real-time status)
    if job_id in active_jobs:
        return active_jobs[job_id]

    # Check database
    db_job = crud.get_job(db, job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")

    results = crud.get_results_for_job(db, job_id)
    return job_to_status(db_job, results)


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel a running job."""
    # Check active jobs
    if job_id in active_jobs:
        job = active_jobs[job_id]
        if job.state == JobState.RUNNING:
            job.state = JobState.CANCELLED
            job.completed_at = datetime.utcnow()
            job.progress.state = JobState.CANCELLED
            job.progress.message = "Job cancelled by user"

            # Update database
            crud.update_job_state(
                db, job_id, "cancelled",
                completed_at=job.completed_at,
            )

            await ws_manager.broadcast(job_id, job.progress.model_dump(mode="json"))
        return {"message": "Job cancelled"}

    # Check database
    db_job = crud.get_job(db, job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")

    if db_job.state == "running":
        crud.update_job_state(db, job_id, "cancelled", completed_at=datetime.utcnow())

    return {"message": "Job cancelled"}


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress updates."""
    # Check if job exists (active or in database)
    from app.db.session import SessionLocal

    if job_id not in active_jobs:
        db = SessionLocal()
        try:
            db_job = crud.get_job(db, job_id)
            if not db_job:
                await websocket.close(code=4004, reason="Job not found")
                return
        finally:
            db.close()

    await ws_manager.connect(job_id, websocket)

    try:
        # Send current status immediately
        if job_id in active_jobs:
            job = active_jobs[job_id]
            await websocket.send_json(job.progress.model_dump(mode="json"))
        else:
            # Job completed, send final status
            db = SessionLocal()
            try:
                db_job = crud.get_job(db, job_id)
                if db_job:
                    await websocket.send_json({
                        "job_id": job_id,
                        "state": db_job.state,
                        "message": "Job completed" if db_job.state == "completed" else db_job.error or "",
                    })
            finally:
                db.close()

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
