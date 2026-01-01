"""Initial schema for benchmark data.

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create jobs table
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(8), primary_key=True),
        sa.Column("benchmark_type", sa.String(50), nullable=False),
        sa.Column("state", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("runs", sa.Integer, nullable=False, server_default="5"),
        sa.Column("max_tokens", sa.Integer, nullable=False, server_default="500"),
        sa.Column("prompt", sa.Text, nullable=True),
        sa.Column("quick", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime, nullable=True),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
    )

    # Create index on jobs for common queries
    op.create_index("ix_jobs_benchmark_type", "jobs", ["benchmark_type"])
    op.create_index("ix_jobs_state", "jobs", ["state"])
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"])

    # Create benchmark_results table
    op.create_table(
        "benchmark_results",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            sa.String(8),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("success_rate", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("stats", sa.JSON, nullable=True),
        sa.Column("start_time", sa.DateTime, nullable=True),
        sa.Column("end_time", sa.DateTime, nullable=True),
        sa.Column("errors", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # Create index on benchmark_results
    op.create_index("ix_benchmark_results_job_id", "benchmark_results", ["job_id"])

    # Create timing_results table
    op.create_table(
        "timing_results",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "benchmark_result_id",
            sa.Integer,
            sa.ForeignKey("benchmark_results.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("total_latency_ms", sa.Float, nullable=False),
        sa.Column("ttft_ms", sa.Float, nullable=True),
        sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tokens_per_second", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("cache_hit", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("cache_hit_rate", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # Create index on timing_results
    op.create_index(
        "ix_timing_results_benchmark_result_id",
        "timing_results",
        ["benchmark_result_id"],
    )


def downgrade() -> None:
    op.drop_table("timing_results")
    op.drop_table("benchmark_results")
    op.drop_table("jobs")
