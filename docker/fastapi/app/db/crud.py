"""CRUD operations for benchmark data."""

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import Job, BenchmarkResult, TimingResult


# ============== Job Operations ==============

def create_job(
    db: Session,
    job_id: str,
    benchmark_type: str,
    model: str,
    runs: int = 5,
    max_tokens: int = 500,
    prompt: Optional[str] = None,
    quick: bool = False,
) -> Job:
    """Create a new benchmark job."""
    job = Job(
        id=job_id,
        benchmark_type=benchmark_type,
        state="pending",
        model=model,
        runs=runs,
        max_tokens=max_tokens,
        prompt=prompt,
        quick=quick,
        created_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> Optional[Job]:
    """Get a job by ID."""
    return db.query(Job).filter(Job.id == job_id).first()


def get_jobs(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    benchmark_type: Optional[str] = None,
    state: Optional[str] = None,
) -> list[Job]:
    """Get all jobs with optional filtering."""
    query = db.query(Job)

    if benchmark_type:
        query = query.filter(Job.benchmark_type == benchmark_type)
    if state:
        query = query.filter(Job.state == state)

    return query.order_by(Job.created_at.desc()).offset(skip).limit(limit).all()


def get_job_count(
    db: Session,
    benchmark_type: Optional[str] = None,
    state: Optional[str] = None,
) -> int:
    """Get total count of jobs."""
    query = db.query(Job)

    if benchmark_type:
        query = query.filter(Job.benchmark_type == benchmark_type)
    if state:
        query = query.filter(Job.state == state)

    return query.count()


def update_job_state(
    db: Session,
    job_id: str,
    state: str,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    error: Optional[str] = None,
) -> Optional[Job]:
    """Update job state."""
    job = get_job(db, job_id)
    if not job:
        return None

    job.state = state
    if started_at:
        job.started_at = started_at
    if completed_at:
        job.completed_at = completed_at
    if error is not None:
        job.error = error

    db.commit()
    db.refresh(job)
    return job


def delete_job(db: Session, job_id: str) -> bool:
    """Delete a job and all related results."""
    job = get_job(db, job_id)
    if not job:
        return False

    db.delete(job)
    db.commit()
    return True


# ============== Benchmark Result Operations ==============

def save_benchmark_result(
    db: Session,
    job_id: str,
    name: str,
    stats: dict,
    description: str = "",
    success_rate: float = 1.0,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    errors: Optional[list[str]] = None,
    timing_results: Optional[list[dict]] = None,
) -> BenchmarkResult:
    """Save a benchmark result with optional timing data."""
    result = BenchmarkResult(
        job_id=job_id,
        name=name,
        description=description,
        success_rate=success_rate,
        stats=stats,
        start_time=start_time,
        end_time=end_time,
        errors=errors or [],
        created_at=datetime.utcnow(),
    )
    db.add(result)
    db.flush()  # Get the result ID

    # Add timing results if provided
    if timing_results:
        for tr in timing_results:
            timing = TimingResult(
                benchmark_result_id=result.id,
                name=tr.get("name", name),
                total_latency_ms=tr.get("total_latency_ms", 0.0),
                ttft_ms=tr.get("ttft_ms"),
                input_tokens=tr.get("input_tokens", 0),
                output_tokens=tr.get("output_tokens", 0),
                tokens_per_second=tr.get("tokens_per_second", 0.0),
                cache_hit=tr.get("cache_hit", False),
                cache_hit_rate=tr.get("cache_hit_rate", 0.0),
            )
            db.add(timing)

    db.commit()
    db.refresh(result)
    return result


def get_benchmark_results(
    db: Session,
    job_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> list[BenchmarkResult]:
    """Get benchmark results with optional job filter."""
    query = db.query(BenchmarkResult)

    if job_id:
        query = query.filter(BenchmarkResult.job_id == job_id)

    return query.order_by(BenchmarkResult.created_at.desc()).offset(skip).limit(limit).all()


def get_results_for_job(db: Session, job_id: str) -> list[BenchmarkResult]:
    """Get all results for a specific job."""
    return db.query(BenchmarkResult).filter(BenchmarkResult.job_id == job_id).all()


def get_timing_results_for_job(db: Session, job_id: str) -> list[TimingResult]:
    """Get all timing results for a job (for CSV export)."""
    return (
        db.query(TimingResult)
        .join(BenchmarkResult)
        .filter(BenchmarkResult.job_id == job_id)
        .order_by(TimingResult.created_at)
        .all()
    )


def get_all_timing_results(
    db: Session,
    benchmark_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 1000,
) -> list[tuple]:
    """Get timing results with job info for CSV export."""
    query = (
        db.query(
            Job.id,
            Job.benchmark_type,
            Job.model,
            TimingResult.total_latency_ms,
            TimingResult.ttft_ms,
            TimingResult.input_tokens,
            TimingResult.output_tokens,
            TimingResult.tokens_per_second,
            TimingResult.cache_hit,
            TimingResult.created_at,
        )
        .join(BenchmarkResult, BenchmarkResult.job_id == Job.id)
        .join(TimingResult, TimingResult.benchmark_result_id == BenchmarkResult.id)
    )

    if benchmark_type:
        query = query.filter(Job.benchmark_type == benchmark_type)

    return query.order_by(TimingResult.created_at.desc()).offset(skip).limit(limit).all()
