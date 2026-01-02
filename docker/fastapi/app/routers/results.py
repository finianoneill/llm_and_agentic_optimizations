"""Results API endpoints for retrieving benchmark results from database."""

import csv
import io
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.models import ResultSummary, StatsSchema
from app.db.session import get_db
from app.db import crud

router = APIRouter(prefix="/api/results", tags=["results"])


def result_to_summary(result, job) -> ResultSummary:
    """Convert database result to ResultSummary."""
    stats_data = result.stats or {}
    return ResultSummary(
        job_id=job.id,
        result_name=result.name,
        benchmark_type=job.benchmark_type,
        model=job.model,
        created_at=result.created_at or job.created_at,
        stats=StatsSchema(**stats_data) if stats_data else StatsSchema(),
    )


@router.get("", response_model=list[ResultSummary])
async def list_results(
    benchmark_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List all benchmark results."""
    # Get completed jobs
    jobs = crud.get_jobs(db, skip=skip, limit=limit, state="completed")

    if benchmark_type:
        jobs = [j for j in jobs if j.benchmark_type == benchmark_type]

    results = []
    for job in jobs:
        job_results = crud.get_results_for_job(db, job.id)
        for result in job_results:
            results.append(result_to_summary(result, job))

    # Sort by created_at descending (newest first)
    results.sort(key=lambda x: x.created_at, reverse=True)
    return results


@router.get("/{job_id}")
async def get_result(job_id: str, db: Session = Depends(get_db)):
    """Get results for a specific job."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    results = crud.get_results_for_job(db, job_id)
    if not results:
        raise HTTPException(status_code=404, detail="No results found for this job")

    return {
        "job_id": job.id,
        "benchmark_type": job.benchmark_type,
        "model": job.model,
        "config": {
            "runs": job.runs,
            "max_tokens": job.max_tokens,
            "prompt": job.prompt,
        },
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "results": [r.to_dict(include_model=job.model) for r in results],
    }


@router.get("/{job_id}/export")
async def export_result_csv(job_id: str, db: Session = Depends(get_db)):
    """Export job results as CSV file."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get timing results for this job
    timing_results = crud.get_timing_results_for_job(db, job_id)

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        "job_id",
        "benchmark_type",
        "model",
        "run_name",
        "total_latency_ms",
        "ttft_ms",
        "input_tokens",
        "output_tokens",
        "tokens_per_second",
        "cache_hit",
        "cache_hit_rate",
        "timestamp",
    ])

    # Write timing results
    for tr in timing_results:
        writer.writerow([
            job.id,
            job.benchmark_type,
            job.model,
            tr.name,
            tr.total_latency_ms,
            tr.ttft_ms if tr.ttft_ms else "",
            tr.input_tokens,
            tr.output_tokens,
            tr.tokens_per_second,
            tr.cache_hit,
            tr.cache_hit_rate,
            tr.created_at.isoformat() if tr.created_at else "",
        ])

    # If no timing results, write aggregated stats from benchmark results
    if not timing_results:
        results = crud.get_results_for_job(db, job_id)
        for result in results:
            stats = result.stats or {}
            writer.writerow([
                job.id,
                job.benchmark_type,
                job.model,
                result.name,
                stats.get("latency_mean_ms", ""),
                stats.get("ttft_p50_ms", ""),
                "",  # input_tokens
                "",  # output_tokens
                stats.get("avg_tokens_per_second", ""),
                "",  # cache_hit
                stats.get("avg_cache_hit_rate", ""),
                result.created_at.isoformat() if result.created_at else "",
            ])

    output.seek(0)

    # Generate filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{job.benchmark_type}_{job_id}_{timestamp}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.delete("/{job_id}")
async def delete_result(job_id: str, db: Session = Depends(get_db)):
    """Delete a job and all its results."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Don't allow deleting running jobs
    if job.state == "running":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running job. Cancel it first.",
        )

    deleted = crud.delete_job(db, job_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete job")

    return {"message": f"Job '{job_id}' and all results deleted"}
