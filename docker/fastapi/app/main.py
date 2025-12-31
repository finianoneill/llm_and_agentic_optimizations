"""LLM Latency Lab API - FastAPI application."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import HealthResponse
from app.routers import benchmarks_router, results_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Starting LLM Latency Lab API...")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
    print(f"Langfuse Host: {os.environ.get('LANGFUSE_HOST', 'not set')}")

    # Verify benchmark modules are importable
    try:
        import benchmarks  # noqa
        print("Benchmark modules loaded successfully")
    except ImportError as e:
        print(f"Warning: Could not import benchmark modules: {e}")

    yield

    # Shutdown
    print("Shutting down LLM Latency Lab API...")


app = FastAPI(
    title="LLM Latency Lab API",
    description="API for running and managing LLM latency benchmarks",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit and local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(benchmarks_router)
app.include_router(results_router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    services = {
        "api": "healthy",
    }

    # Check if Langfuse is configured
    if os.environ.get("LANGFUSE_HOST"):
        services["langfuse"] = "configured"

    # Check if Claude credentials are available
    claude_config = os.path.expanduser("~/.claude")
    if os.path.exists(claude_config):
        services["claude_auth"] = "available"
    else:
        services["claude_auth"] = "missing"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services=services,
    )


@app.get("/api", tags=["root"])
async def api_root():
    """API root - list available endpoints."""
    return {
        "message": "LLM Latency Lab API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "benchmarks": "/api/benchmarks",
            "jobs": "/api/jobs",
            "results": "/api/results",
            "websocket": "/api/ws/{job_id}",
        },
        "documentation": "/docs",
    }
