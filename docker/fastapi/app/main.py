"""LLM Latency Lab API - FastAPI application.

This application provides an API for running LLM latency benchmarks.
All LLM calls are traced to Langfuse when configured.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import HealthResponse
from app.routers import benchmarks_router, results_router
from app.db.session import init_db, check_db_connection

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Starting LLM Latency Lab API...")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
    print(f"Langfuse Host: {os.environ.get('LANGFUSE_HOST', 'not set')}")
    print(f"Database URL: {os.environ.get('DATABASE_URL', 'not set')}")

    # Initialize database
    try:
        print("Initializing database...")
        init_db()
        if check_db_connection():
            print("Database connection verified")
        else:
            print("WARNING: Database connection check failed")
    except Exception as e:
        print(f"ERROR: Failed to initialize database: {e}")

    # Initialize Langfuse tracing
    if LANGFUSE_AVAILABLE:
        langfuse_tracer = get_langfuse_tracer()
        if langfuse_tracer:
            print("Langfuse tracing initialized successfully")
        else:
            print("Langfuse available but not configured (missing keys)")
    else:
        print("Langfuse not available (package not installed)")

    # Verify benchmark modules are importable
    try:
        import benchmarks  # noqa
        print("Benchmark modules loaded successfully")
    except ImportError as e:
        print(f"Warning: Could not import benchmark modules: {e}")

    yield

    # Shutdown
    print("Shutting down LLM Latency Lab API...")

    # Flush Langfuse traces before shutdown
    if LANGFUSE_AVAILABLE:
        print("Flushing Langfuse traces...")
        flush_langfuse()
        print("Langfuse traces flushed")


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

    # Check database status
    if check_db_connection():
        services["database"] = "healthy"
    else:
        services["database"] = "unhealthy"

    # Check Langfuse status
    if LANGFUSE_AVAILABLE:
        langfuse_host = os.environ.get("LANGFUSE_HOST")
        langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

        if langfuse_host and langfuse_public_key and langfuse_secret_key:
            # Try to get the tracer to verify it's working
            tracer = get_langfuse_tracer()
            if tracer:
                services["langfuse"] = "active"
                services["langfuse_host"] = langfuse_host
            else:
                services["langfuse"] = "configured_but_failed"
        elif langfuse_host:
            services["langfuse"] = "host_configured_missing_keys"
        else:
            services["langfuse"] = "not_configured"
    else:
        services["langfuse"] = "not_installed"

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
