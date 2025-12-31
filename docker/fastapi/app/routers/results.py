"""Results API endpoints for retrieving saved benchmark results."""

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.models import ResultSummary, StatsSchema

router = APIRouter(prefix="/api/results", tags=["results"])

RESULTS_DIR = Path("/app/results")


def parse_result_file(filepath: Path) -> ResultSummary:
    """Parse a result JSON file into a summary."""
    try:
        with open(filepath) as f:
            data = json.load(f)

        # Handle both single result and comparison formats
        if "results" in data:
            # Comparison format
            first_result = data["results"][0] if data["results"] else {}
            stats_data = first_result.get("stats", {})
            benchmark_type = data.get("name", "comparison")
            model = first_result.get("config", {}).get("model", "unknown")
        else:
            # Single result format
            stats_data = data.get("stats", {})
            benchmark_type = data.get("config", {}).get("name", "unknown")
            model = data.get("config", {}).get("model", "unknown")

        # Parse timestamp from filename or data
        timestamp_str = data.get("timestamp") or filepath.stem.split("_")[-2:]
        if isinstance(timestamp_str, list):
            timestamp_str = "_".join(timestamp_str)
        try:
            created_at = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except (ValueError, TypeError):
            created_at = datetime.fromtimestamp(filepath.stat().st_mtime)

        return ResultSummary(
            filename=filepath.name,
            benchmark_type=benchmark_type,
            model=model,
            created_at=created_at,
            stats=StatsSchema(**stats_data) if stats_data else StatsSchema(),
        )
    except Exception as e:
        # Return minimal summary on parse error
        return ResultSummary(
            filename=filepath.name,
            benchmark_type="unknown",
            model="unknown",
            created_at=datetime.fromtimestamp(filepath.stat().st_mtime),
            stats=StatsSchema(),
        )


@router.get("", response_model=list[ResultSummary])
async def list_results():
    """List all saved result files."""
    if not RESULTS_DIR.exists():
        return []

    results = []
    for filepath in RESULTS_DIR.glob("*.json"):
        summary = parse_result_file(filepath)
        results.append(summary)

    # Sort by created_at descending (newest first)
    results.sort(key=lambda x: x.created_at, reverse=True)
    return results


@router.get("/{filename}")
async def get_result(filename: str):
    """Get a specific result file."""
    filepath = RESULTS_DIR / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    if not filepath.suffix == ".json":
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Ensure we're not escaping the results directory
    try:
        filepath.resolve().relative_to(RESULTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    with open(filepath) as f:
        return json.load(f)


@router.get("/{filename}/download")
async def download_result(filename: str):
    """Download a result file."""
    filepath = RESULTS_DIR / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    # Ensure we're not escaping the results directory
    try:
        filepath.resolve().relative_to(RESULTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/json",
    )


@router.delete("/{filename}")
async def delete_result(filename: str):
    """Delete a result file."""
    filepath = RESULTS_DIR / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    # Ensure we're not escaping the results directory
    try:
        filepath.resolve().relative_to(RESULTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    filepath.unlink()
    return {"message": f"Result file '{filename}' deleted"}
