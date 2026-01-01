"""
Claude Agent SDK client wrapper for Max account authentication.

This module provides a wrapper around the Claude Agent SDK that allows
authentication with Claude Max accounts (consumer subscription) instead
of requiring an API key.

The SDK uses Claude Code CLI as its runtime, which handles authentication.
To use with Max account:
1. Install Claude Code CLI: npm install -g @anthropic-ai/claude-code
2. Authenticate: claude login
3. Use this wrapper instead of anthropic.AsyncAnthropic()

For Docker deployments, set CLAUDE_PROXY_URL to point to the host proxy:
    CLAUDE_PROXY_URL=http://host.docker.internal:8765

Note: The Claude Agent SDK is designed for agentic workflows. For simple
message completion tasks, it may have slightly different behavior than
the direct Anthropic API.

Langfuse Integration:
    All LLM calls are automatically traced to Langfuse when configured.
    Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST
    environment variables to enable tracing.
"""

import os
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Optional

# Import Langfuse tracing utilities
try:
    from instrumentation.traces import (
        get_langfuse_tracer,
        flush_langfuse,
        LANGFUSE_AVAILABLE,
        observe,
        get_client as get_langfuse_client,
    )
except ImportError:
    # Fallback for when running outside the package
    LANGFUSE_AVAILABLE = False
    get_langfuse_tracer = lambda: None
    flush_langfuse = lambda: None
    observe = None
    get_langfuse_client = None


@dataclass
class Usage:
    """Usage information from Claude response."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class Message:
    """Simplified message response compatible with existing code."""
    content: str
    usage: Usage
    stop_reason: str = "end_turn"
    model: str = ""


class ClaudeMaxClient:
    """
    Client wrapper for Claude Agent SDK with Max account authentication.

    This provides a similar interface to anthropic.AsyncAnthropic() but
    uses the Claude Agent SDK under the hood, which supports Max account
    authentication through Claude Code CLI.

    For Docker deployments, set CLAUDE_PROXY_URL environment variable
    to use the host proxy instead of direct SDK calls.

    Usage:
        client = ClaudeMaxClient()
        response = await client.create_message(
            prompt="What is the capital of France?",
            max_tokens=500
        )
        print(response.content)
        print(f"Tokens used: {response.usage.output_tokens}")
    """

    def __init__(self, model: str = "sonnet", enable_tracing: bool = True):
        """
        Initialize the Claude Max client.

        Args:
            model: Model to use - "opus", "sonnet", or "haiku"
            enable_tracing: Whether to enable Langfuse tracing (default: True)
        """
        self.model = model
        self._sdk_available = None
        self._proxy_url = os.environ.get("CLAUDE_PROXY_URL")
        self._enable_tracing = enable_tracing and LANGFUSE_AVAILABLE

        if self._enable_tracing:
            # Initialize Langfuse by getting the tracer (sets env vars)
            tracer = get_langfuse_tracer()
            if tracer:
                print(f"[ClaudeMaxClient] Langfuse tracing enabled for model {model}")
            else:
                print(f"[ClaudeMaxClient] Langfuse tracing NOT available")

    def _get_full_model_name(self, model: str) -> str:
        """Convert short model name to full model identifier for tracing."""
        model_map = {
            "sonnet": "claude-sonnet-4-20250514",
            "opus": "claude-opus-4-5-20251101",
            "haiku": "claude-3-5-haiku-20241022",
        }
        return model_map.get(model, model)

    async def _check_sdk(self) -> bool:
        """Check if the Claude Agent SDK is available."""
        if self._sdk_available is not None:
            return self._sdk_available

        try:
            from claude_agent_sdk import query
            self._sdk_available = True
        except ImportError:
            self._sdk_available = False

        return self._sdk_available

    async def _create_message_via_proxy(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Message:
        """Create a message using the host proxy."""
        import aiohttp

        model_to_use = model or self.model
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._proxy_url}/query",
                json={
                    "prompt": full_prompt,
                    "model": model_to_use,
                    "max_tokens": max_tokens,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()

                if "error" in data:
                    raise Exception(data["error"])

                # Parse the Claude CLI JSON response
                result_text = data.get("result", "")
                usage = data.get("usage", {})

                return Message(
                    content=result_text,
                    usage=Usage(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                    ),
                    model=model_to_use,
                )

    async def create_message(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        model: str | None = None,
        trace_name: str | None = None,
        trace_metadata: dict | None = None,
    ) -> Message:
        """
        Create a message using Claude Agent SDK or proxy.

        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens in response (note: SDK manages this internally)
            system_prompt: Optional system prompt
            model: Override the default model ("opus", "sonnet", or "haiku")
            trace_name: Optional name for the Langfuse trace
            trace_metadata: Optional metadata to attach to the trace

        Returns:
            Message object with content and usage info
        """
        model_to_use = model or self.model
        full_model_name = self._get_full_model_name(model_to_use)
        start_time = time.time()

        # Use proxy if available (for Docker deployments)
        if self._proxy_url:
            response = await self._create_message_via_proxy(
                prompt=prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                model=model,
            )
        else:
            response = await self._create_message_direct(
                prompt=prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                model=model_to_use,
            )

        # Record to Langfuse if enabled
        if self._enable_tracing and LANGFUSE_AVAILABLE and get_langfuse_client:
            try:
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                client = get_langfuse_client()

                with client.start_as_current_observation(
                    name=trace_name or "llm_call",
                    as_type="generation",
                    model=full_model_name,
                ) as obs:
                    # Update with input, output, usage, and metadata
                    obs.update(
                        input={"prompt": prompt[:500], "system_prompt": system_prompt[:200] if system_prompt else None},
                        output=response.content[:1000] if response.content else None,
                        usage={
                            "input": response.usage.input_tokens,
                            "output": response.usage.output_tokens,
                            "total": response.usage.input_tokens + response.usage.output_tokens,
                        },
                        metadata={
                            "max_tokens": max_tokens,
                            "latency_ms": latency_ms,
                            "cache_creation_tokens": response.usage.cache_creation_input_tokens,
                            "cache_read_tokens": response.usage.cache_read_input_tokens,
                            **(trace_metadata or {}),
                        },
                    )

                print(f"[Langfuse] Generation recorded: {trace_name or 'llm_call'}")
            except Exception as e:
                print(f"[Langfuse] Error recording generation: {e}")

        return response

    async def _create_message_direct(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Message:
        """Create a message directly using Claude Agent SDK."""
        from claude_agent_sdk import (
            query,
            ClaudeAgentOptions,
            ResultMessage,
            AssistantMessage,
            TextBlock,
        )

        model_to_use = model or self.model

        # Build the options
        options = ClaudeAgentOptions(
            model=model_to_use,
            allowed_tools=[],  # No tools for simple message completion
        )

        # If system prompt provided, prepend to the prompt
        full_prompt = prompt
        if system_prompt:
            options.system_prompt = system_prompt

        content_parts = []
        usage_data = Usage()
        result_model = model_to_use

        async for message in query(prompt=full_prompt, options=options):
            # Handle ResultMessage (final message with usage info)
            if isinstance(message, ResultMessage):
                if message.result:
                    content_parts.append(message.result)
                if message.usage:
                    usage_data = Usage(
                        input_tokens=message.usage.get("input_tokens", 0),
                        output_tokens=message.usage.get("output_tokens", 0),
                        cache_creation_input_tokens=message.usage.get("cache_creation_input_tokens", 0),
                        cache_read_input_tokens=message.usage.get("cache_read_input_tokens", 0),
                    )
            # Handle AssistantMessage (streaming content)
            elif isinstance(message, AssistantMessage):
                result_model = message.model
                for block in message.content:
                    if isinstance(block, TextBlock):
                        content_parts.append(block.text)

        response_text = "".join(content_parts)

        return Message(
            content=response_text,
            usage=usage_data,
            model=result_model,
        )

    async def _create_message_streaming_via_proxy(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Create a streaming message using the host proxy."""
        import aiohttp

        model_to_use = model or self.model
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._proxy_url}/query/stream",
                json={
                    "prompt": full_prompt,
                    "model": model_to_use,
                    "max_tokens": max_tokens,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                async for line in resp.content:
                    line = line.decode().strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            parsed = json.loads(data)
                            if "result" in parsed:
                                yield parsed["result"]
                            elif "content" in parsed:
                                yield parsed["content"]
                            else:
                                # Raw text chunk
                                yield data
                        except:
                            # Not JSON, yield as-is
                            if data:
                                yield data

    async def create_message_streaming(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        model: str | None = None,
        trace_name: str | None = None,
        trace_metadata: dict | None = None,
    ) -> AsyncIterator[str]:
        """
        Create a streaming message using Claude Agent SDK or proxy.

        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens in response (note: SDK manages this internally)
            system_prompt: Optional system prompt
            model: Override the default model
            trace_name: Optional name for the Langfuse trace
            trace_metadata: Optional metadata to attach to the trace

        Yields:
            Text chunks as they arrive
        """
        model_to_use = model or self.model
        full_model_name = self._get_full_model_name(model_to_use)
        start_time = time.time()
        first_token_time = None
        total_chunks = 0
        accumulated_text = []

        # Use proxy if available (for Docker deployments)
        if self._proxy_url:
            async for chunk in self._create_message_streaming_via_proxy(
                prompt=prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                model=model,
            ):
                if first_token_time is None:
                    first_token_time = time.time()
                total_chunks += 1
                accumulated_text.append(chunk)
                yield chunk
        else:
            from claude_agent_sdk import (
                query,
                ClaudeAgentOptions,
                ResultMessage,
                AssistantMessage,
                TextBlock,
            )

            options = ClaudeAgentOptions(
                model=model_to_use,
                allowed_tools=[],
            )

            if system_prompt:
                options.system_prompt = system_prompt

            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    if message.result:
                        if first_token_time is None:
                            first_token_time = time.time()
                        total_chunks += 1
                        accumulated_text.append(message.result)
                        yield message.result
                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            if first_token_time is None:
                                first_token_time = time.time()
                            total_chunks += 1
                            accumulated_text.append(block.text)
                            yield block.text

        # Record to Langfuse after streaming completes
        if self._enable_tracing and LANGFUSE_AVAILABLE and get_langfuse_client:
            try:
                end_time = time.time()
                total_latency_ms = (end_time - start_time) * 1000
                ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else None
                full_response = "".join(accumulated_text)
                client = get_langfuse_client()

                with client.start_as_current_observation(
                    name=trace_name or "llm_streaming_call",
                    as_type="generation",
                    model=full_model_name,
                ) as obs:
                    # Update with input, output, and metadata
                    obs.update(
                        input={"prompt": prompt[:500], "system_prompt": system_prompt[:200] if system_prompt else None},
                        output=full_response[:1000] if full_response else None,
                        metadata={
                            "streaming": True,
                            "max_tokens": max_tokens,
                            "total_latency_ms": total_latency_ms,
                            "ttft_ms": ttft_ms,
                            "total_chunks": total_chunks,
                            "response_length": len(full_response),
                            **(trace_metadata or {}),
                        },
                    )

                print(f"[Langfuse] Streaming generation recorded: {trace_name or 'llm_streaming_call'}")
            except Exception as e:
                print(f"[Langfuse] Error recording streaming generation: {e}")


def get_client(model: str = "sonnet") -> ClaudeMaxClient:
    """
    Factory function to get a Claude client.

    Args:
        model: Model to use - "opus", "sonnet", or "haiku"

    Returns:
        ClaudeMaxClient instance
    """
    return ClaudeMaxClient(model=model)
