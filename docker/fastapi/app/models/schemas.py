"""Pydantic schemas for API request/response models."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class BenchmarkType(str, Enum):
    """Available benchmark types."""

    STREAMING = "streaming"
    CACHING = "caching"
    PARALLEL = "parallel"
    ROUTING = "routing"
    TOPOLOGY = "topology"
    ALL = "all"


class JobState(str, Enum):
    """Job execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BenchmarkRequest(BaseModel):
    """Request to start a benchmark run."""

    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model to use for benchmarks",
    )
    runs: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of benchmark iterations",
    )
    quick: bool = Field(
        default=False,
        description="Use quick mode (fewer runs)",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for the benchmark",
    )
    max_tokens: int = Field(
        default=500,
        ge=50,
        le=4096,
        description="Maximum tokens in response",
    )


class ProgressUpdate(BaseModel):
    """Real-time progress update for a running job."""

    job_id: str
    state: JobState
    current_run: int = 0
    total_runs: int = 0
    current_benchmark: Optional[str] = None
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: dict[str, Any] = Field(default_factory=dict)


class TimingResultSchema(BaseModel):
    """Timing result for a single benchmark run."""

    name: str
    total_latency_ms: float
    ttft_ms: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    cache_hit: bool = False
    cache_hit_rate: float = 0.0


class StatsSchema(BaseModel):
    """Aggregated statistics from benchmark runs."""

    count: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    avg_tokens_per_second: float = 0.0
    avg_cache_hit_rate: float = 0.0


class BenchmarkResultSchema(BaseModel):
    """Result of a completed benchmark."""

    name: str
    description: str = ""
    model: str
    timing_results: list[TimingResultSchema] = Field(default_factory=list)
    stats: StatsSchema = Field(default_factory=StatsSchema)
    success_rate: float = 1.0
    start_time: datetime
    end_time: datetime
    errors: list[str] = Field(default_factory=list)


class JobStatus(BaseModel):
    """Full status of a benchmark job."""

    job_id: str
    benchmark_type: BenchmarkType
    state: JobState
    request: BenchmarkRequest
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: ProgressUpdate
    results: Optional[list[BenchmarkResultSchema]] = None
    result_file: Optional[str] = None
    error: Optional[str] = None


class JobResponse(BaseModel):
    """Response when a job is created."""

    job_id: str
    message: str = "Job created successfully"
    websocket_url: str


class JobListResponse(BaseModel):
    """Response listing all jobs."""

    jobs: list[JobStatus]
    total: int


class BenchmarkInfo(BaseModel):
    """Information about an available benchmark."""

    type: BenchmarkType
    name: str
    description: str
    expected_duration: str
    metrics: list[str]


class ResultSummary(BaseModel):
    """Summary of a saved result file."""

    filename: str
    benchmark_type: str
    model: str
    created_at: datetime
    stats: StatsSchema


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    services: dict[str, str] = Field(default_factory=dict)
