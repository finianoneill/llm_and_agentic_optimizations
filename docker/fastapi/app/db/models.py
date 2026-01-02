"""SQLAlchemy ORM models for benchmark data."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Job(Base):
    """Benchmark job record."""

    __tablename__ = "jobs"

    id = Column(String(8), primary_key=True)
    benchmark_type = Column(String(50), nullable=False)
    state = Column(String(20), nullable=False, default="pending")
    model = Column(String(100), nullable=False)
    runs = Column(Integer, nullable=False, default=5)
    max_tokens = Column(Integer, nullable=False, default=500)
    prompt = Column(Text, nullable=True)
    quick = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

    # Relationships
    results = relationship(
        "BenchmarkResult",
        back_populates="job",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "job_id": self.id,
            "benchmark_type": self.benchmark_type,
            "state": self.state,
            "request": {
                "model": self.model,
                "runs": self.runs,
                "max_tokens": self.max_tokens,
                "prompt": self.prompt,
                "quick": self.quick,
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "results": [r.to_dict(include_model=self.model) for r in self.results] if self.results else None,
        }


class BenchmarkResult(Base):
    """Benchmark result containing stats and timing data."""

    __tablename__ = "benchmark_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(8), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    success_rate = Column(Float, nullable=False, default=1.0)
    stats = Column(JSON, nullable=True)  # Store aggregated stats as JSONB
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    errors = Column(JSON, nullable=True)  # List of error messages
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    job = relationship("Job", back_populates="results")
    timing_results = relationship(
        "TimingResult",
        back_populates="benchmark_result",
        cascade="all, delete-orphan",
    )

    def to_dict(self, include_model: str = None) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "description": self.description,
            "model": include_model,
            "stats": self.stats,
            "success_rate": self.success_rate,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "errors": self.errors or [],
            "timing_results": [t.to_dict() for t in self.timing_results] if self.timing_results else [],
        }


class TimingResult(Base):
    """Individual timing result from a single benchmark run."""

    __tablename__ = "timing_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_result_id = Column(
        Integer,
        ForeignKey("benchmark_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    name = Column(String(100), nullable=False)
    total_latency_ms = Column(Float, nullable=False)
    ttft_ms = Column(Float, nullable=True)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    tokens_per_second = Column(Float, nullable=False, default=0.0)
    cache_hit = Column(Boolean, nullable=False, default=False)
    cache_hit_rate = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    benchmark_result = relationship("BenchmarkResult", back_populates="timing_results")

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "total_latency_ms": self.total_latency_ms,
            "ttft_ms": self.ttft_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_per_second": self.tokens_per_second,
            "cache_hit": self.cache_hit,
            "cache_hit_rate": self.cache_hit_rate,
        }
