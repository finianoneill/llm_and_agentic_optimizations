#!/usr/bin/env python3
"""
Claude CLI Proxy Server

Runs on the host machine and proxies requests to Claude CLI,
which has access to the macOS Keychain for authentication.

Usage:
    python proxy.py

The container can then call http://host.docker.internal:8765/query
"""

import asyncio
import json
import subprocess
from aiohttp import web


async def health(request):
    """Health check endpoint."""
    return web.json_response({"status": "healthy", "service": "claude-proxy"})


async def query(request):
    """
    Proxy a query to Claude CLI.

    Expects JSON body:
    {
        "prompt": "Your prompt here",
        "model": "sonnet",  # optional: opus, sonnet, haiku
        "max_tokens": 1024  # optional
    }
    """
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        model = data.get("model", "sonnet")

        if not prompt:
            return web.json_response({"error": "No prompt provided"}, status=400)

        # Build Claude CLI command
        cmd = [
            "claude",
            "-p", prompt,
            "--model", model,
            "--output-format", "json",
        ]

        # Run Claude CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return web.json_response({
                "error": f"Claude CLI failed: {result.stderr}",
                "returncode": result.returncode,
            }, status=500)

        # Parse Claude's JSON output
        try:
            response = json.loads(result.stdout)
            return web.json_response(response)
        except json.JSONDecodeError:
            return web.json_response({
                "result": result.stdout,
                "raw": True,
            })

    except asyncio.TimeoutError:
        return web.json_response({"error": "Request timed out"}, status=504)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def query_streaming(request):
    """
    Streaming query to Claude CLI.
    Returns Server-Sent Events.

    Note: Claude CLI doesn't have a true streaming mode via subprocess,
    so we fall back to returning the full response as a single chunk.
    """
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        model = data.get("model", "sonnet")

        if not prompt:
            return web.json_response({"error": "No prompt provided"}, status=400)

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        # Run Claude CLI (non-streaming, but return as SSE)
        cmd = [
            "claude",
            "-p", prompt,
            "--model", model,
            "--output-format", "json",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0 and stdout:
            try:
                result = json.loads(stdout.decode())
                # Send the result text as a single chunk
                text = result.get("result", "")
                if text:
                    await response.write(f"data: {json.dumps({'content': text})}\n\n".encode())
            except json.JSONDecodeError:
                await response.write(f"data: {json.dumps({'content': stdout.decode()})}\n\n".encode())
        elif stderr:
            await response.write(f"data: {json.dumps({'error': stderr.decode()})}\n\n".encode())

        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()

        return response

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


def create_app():
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/query", query)
    app.router.add_post("/query/stream", query_streaming)
    return app


if __name__ == "__main__":
    print("Starting Claude CLI Proxy on http://localhost:8765")
    print("Container can reach this at http://host.docker.internal:8765")
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=8765)
