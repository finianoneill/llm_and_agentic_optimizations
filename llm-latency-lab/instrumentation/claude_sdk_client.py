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
"""

import os
from dataclasses import dataclass
from typing import AsyncIterator


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

    def __init__(self, model: str = "sonnet"):
        """
        Initialize the Claude Max client.

        Args:
            model: Model to use - "opus", "sonnet", or "haiku"
        """
        self.model = model
        self._sdk_available = None
        self._proxy_url = os.environ.get("CLAUDE_PROXY_URL")

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
    ) -> Message:
        """
        Create a message using Claude Agent SDK or proxy.

        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens in response (note: SDK manages this internally)
            system_prompt: Optional system prompt
            model: Override the default model ("opus", "sonnet", or "haiku")

        Returns:
            Message object with content and usage info
        """
        # Use proxy if available (for Docker deployments)
        if self._proxy_url:
            return await self._create_message_via_proxy(
                prompt=prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                model=model,
            )

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

    async def create_message_streaming(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Create a streaming message using Claude Agent SDK.

        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens in response (note: SDK manages this internally)
            system_prompt: Optional system prompt
            model: Override the default model

        Yields:
            Text chunks as they arrive
        """
        from claude_agent_sdk import (
            query,
            ClaudeAgentOptions,
            ResultMessage,
            AssistantMessage,
            TextBlock,
        )

        model_to_use = model or self.model

        options = ClaudeAgentOptions(
            model=model_to_use,
            allowed_tools=[],
        )

        if system_prompt:
            options.system_prompt = system_prompt

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                if message.result:
                    yield message.result
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        yield block.text


def get_client(model: str = "sonnet") -> ClaudeMaxClient:
    """
    Factory function to get a Claude client.

    Args:
        model: Model to use - "opus", "sonnet", or "haiku"

    Returns:
        ClaudeMaxClient instance
    """
    return ClaudeMaxClient(model=model)
