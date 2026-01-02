"""LLM Latency Lab - Streamlit Dashboard."""

import json
import os
import threading
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import websocket

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://fastapi:8000")

st.set_page_config(
    page_title="LLM Latency Lab",
    page_icon="🧪",
    layout="wide",
)


def get_api(endpoint: str):
    """Make a GET request to the API."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def post_api(endpoint: str, data: dict = None):
    """Make a POST request to the API."""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data or {}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def format_duration(ms: float) -> str:
    """Format duration in milliseconds to human-readable string."""
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


# Sidebar navigation
st.sidebar.title("🧪 LLM Latency Lab")

page = st.sidebar.radio(
    "Navigation",
    ["🚀 Run Benchmarks", "📊 View Results", "🔄 Active Jobs", "❤️ Health"],
)

# Check API health
health = get_api("/api/health")
if health:
    status_color = "🟢" if health.get("status") == "healthy" else "🔴"
    st.sidebar.markdown(f"**API Status:** {status_color} {health.get('status', 'unknown')}")

    # Show service statuses
    services = health.get("services", {})
    if services.get("claude_auth") == "available":
        st.sidebar.markdown("**Claude Auth:** 🟢 Available")
    else:
        st.sidebar.markdown("**Claude Auth:** 🔴 Missing")
else:
    st.sidebar.markdown("**API Status:** 🔴 Unreachable")


# Page: Run Benchmarks
if page == "🚀 Run Benchmarks":
    st.title("🚀 Run Benchmarks")
    st.markdown("Start a new benchmark run to measure LLM optimization performance.")

    # Get available benchmarks
    benchmarks = get_api("/api/benchmarks")

    if benchmarks:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Benchmark selection
            benchmark_options = {b["name"]: b for b in benchmarks}
            selected_name = st.selectbox(
                "Select Benchmark",
                options=list(benchmark_options.keys()),
            )
            selected_benchmark = benchmark_options[selected_name]

            st.markdown(f"**Description:** {selected_benchmark['description']}")
            st.markdown(f"**Expected Duration:** {selected_benchmark['expected_duration']}")
            st.markdown(f"**Metrics:** {', '.join(selected_benchmark['metrics'])}")

        with col2:
            st.subheader("Configuration")

            model = st.selectbox(
                "Model",
                ["claude-sonnet-4-20250514", "claude-opus-4-5-20251101", "claude-3-5-haiku-20241022"],
            )

            runs = st.slider("Number of Runs", min_value=1, max_value=20, value=5)

            quick_mode = st.checkbox("Quick Mode (3 runs)", value=False)

            max_tokens = st.number_input("Max Tokens", min_value=50, max_value=4096, value=500)

        # Custom prompt (optional)
        custom_prompt = st.text_area(
            "Custom Prompt (optional)",
            placeholder="Leave empty to use default prompts",
            height=100,
        )

        if st.button("🚀 Start Benchmark", type="primary"):
            request_data = {
                "model": model,
                "runs": 3 if quick_mode else runs,
                "quick": quick_mode,
                "max_tokens": max_tokens,
            }
            if custom_prompt:
                request_data["prompt"] = custom_prompt

            result = post_api(f"/api/benchmarks/{selected_benchmark['type']}/run", request_data)

            if result:
                st.success(f"Benchmark started! Job ID: `{result['job_id']}`")
                st.info(f"WebSocket URL: `{result['websocket_url']}`")
                st.markdown("Go to **🔄 Active Jobs** to monitor progress.")


# Page: View Results
elif page == "📊 View Results":
    st.title("📊 Benchmark Results")
    st.markdown("Browse and compare saved benchmark results.")

    results = get_api("/api/results")

    if results is not None:
        if not results:
            st.info("No results found. Run a benchmark to generate results.")
        else:
            # Results table
            df = pd.DataFrame([
                {
                    "Job ID": r["job_id"],
                    "Result": r["result_name"],
                    "Benchmark": r["benchmark_type"],
                    "Model": r["model"],
                    "Created": r["created_at"],
                    "p50 Latency": format_duration(r["stats"]["latency_p50_ms"]),
                    "p95 Latency": format_duration(r["stats"]["latency_p95_ms"]),
                    "TTFT p50": format_duration(r["stats"]["ttft_p50_ms"]) if r["stats"].get("ttft_p50_ms") else "N/A",
                }
                for r in results
            ])

            st.dataframe(df, width="stretch")

            # Result details - get unique job IDs
            st.subheader("Result Details")
            job_ids = list(dict.fromkeys([r["job_id"] for r in results]))  # Unique, preserve order
            selected_job_id = st.selectbox(
                "Select a job to view details",
                options=job_ids,
                format_func=lambda x: f"{x} - {next((r['benchmark_type'] for r in results if r['job_id'] == x), 'unknown')}",
            )

            if selected_job_id:
                result_data = get_api(f"/api/results/{selected_job_id}")

                if result_data:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.json(result_data.get("config", {}))

                    with col2:
                        # Get stats from first result if available
                        result_list = result_data.get("results", [])
                        if result_list:
                            stats = result_list[0].get("stats", {})
                            if stats:
                                # Create latency chart
                                latency_data = {
                                    "Percentile": ["p50", "p95", "p99"],
                                    "Latency (ms)": [
                                        stats.get("latency_p50_ms", 0),
                                        stats.get("latency_p95_ms", 0),
                                        stats.get("latency_p99_ms", 0),
                                    ],
                                }
                                fig = px.bar(
                                    latency_data,
                                    x="Percentile",
                                    y="Latency (ms)",
                                    title="Latency Distribution",
                                )
                                st.plotly_chart(fig, width="stretch")

                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download JSON",
                            data=json.dumps(result_data, indent=2),
                            file_name=f"{result_data.get('benchmark_type', 'result')}_{selected_job_id}.json",
                            mime="application/json",
                        )
                    with col2:
                        # Fetch CSV from API and provide download
                        try:
                            csv_response = requests.get(
                                f"{API_BASE_URL}/api/results/{selected_job_id}/export",
                                timeout=10
                            )
                            if csv_response.status_code == 200:
                                st.download_button(
                                    "📥 Download CSV",
                                    data=csv_response.text,
                                    file_name=f"{result_data.get('benchmark_type', 'result')}_{selected_job_id}.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.warning("CSV export not available")
                        except requests.RequestException:
                            st.warning("Could not fetch CSV")
    else:
        st.error("Could not fetch results from API.")


# Page: Active Jobs
elif page == "🔄 Active Jobs":
    st.title("🔄 Active Jobs")
    st.markdown("Monitor running and completed benchmark jobs.")

    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (every 5s)", value=True)

    if auto_refresh:
        time.sleep(0.1)  # Small delay to prevent rapid refreshes
        st.empty()

    jobs_response = get_api("/api/jobs")

    if jobs_response:
        jobs = jobs_response.get("jobs", [])

        if not jobs:
            st.info("No jobs found. Start a benchmark to create a job.")
        else:
            # Group by state
            running_jobs = [j for j in jobs if j["state"] == "running"]
            pending_jobs = [j for j in jobs if j["state"] == "pending"]
            completed_jobs = [j for j in jobs if j["state"] == "completed"]
            failed_jobs = [j for j in jobs if j["state"] == "failed"]

            # Running jobs
            if running_jobs or pending_jobs:
                st.subheader("🔄 In Progress")
                for job in running_jobs + pending_jobs:
                    with st.expander(f"Job {job['job_id']} - {job['benchmark_type']} ({job['state']})", expanded=True):
                        progress = job.get("progress", {})

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**State:** {job['state']}")
                            st.markdown(f"**Benchmark:** {job['benchmark_type']}")
                            st.markdown(f"**Created:** {job['created_at']}")

                        with col2:
                            if progress.get("total_runs", 0) > 0:
                                pct = progress.get("current_run", 0) / progress["total_runs"]
                                st.progress(pct)
                                st.markdown(f"Run {progress['current_run']}/{progress['total_runs']}")

                            st.markdown(f"**Message:** {progress.get('message', 'Waiting...')}")

            # Completed jobs
            if completed_jobs:
                st.subheader("✅ Completed")
                for job in completed_jobs[:10]:  # Show last 10
                    with st.expander(f"Job {job['job_id']} - {job['benchmark_type']}"):
                        st.markdown(f"**Completed:** {job.get('completed_at', 'N/A')}")
                        st.markdown(f"**Model:** {job['request'].get('model', 'N/A')}")
                        st.markdown(f"**Runs:** {job['request'].get('runs', 'N/A')}")

                        if job.get("results"):
                            st.json(job["results"])

            # Failed jobs
            if failed_jobs:
                st.subheader("❌ Failed")
                for job in failed_jobs[:5]:
                    with st.expander(f"Job {job['job_id']} - {job['benchmark_type']}", expanded=False):
                        st.error(f"Error: {job.get('error', 'Unknown error')}")

    if auto_refresh:
        time.sleep(5)
        st.rerun()


# Page: Health
elif page == "❤️ Health":
    st.title("❤️ System Health")

    if health:
        st.success(f"API Status: {health.get('status', 'unknown')}")
        st.markdown(f"**Version:** {health.get('version', 'unknown')}")

        st.subheader("Services")
        services = health.get("services", {})
        for service, status in services.items():
            icon = "🟢" if status in ["healthy", "available", "configured", "active"] else "🔴"
            st.markdown(f"- **{service}:** {icon} {status}")

        st.subheader("Configuration")
        st.markdown(f"**API Base URL:** `{API_BASE_URL}`")

        # Test connectivity
        st.subheader("Connectivity Tests")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Test API Connection"):
                try:
                    response = requests.get(f"{API_BASE_URL}/api", timeout=5)
                    st.success(f"API reachable: {response.status_code}")
                except Exception as e:
                    st.error(f"API unreachable: {e}")

        with col2:
            if st.button("Test Benchmarks Endpoint"):
                benchmarks = get_api("/api/benchmarks")
                if benchmarks:
                    st.success(f"Found {len(benchmarks)} benchmark types")
                else:
                    st.error("Could not fetch benchmarks")
    else:
        st.error("Could not connect to API")
        st.markdown(f"**Attempted URL:** `{API_BASE_URL}`")
        st.markdown("Make sure the FastAPI service is running.")
