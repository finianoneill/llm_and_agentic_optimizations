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

Note: The Claude Agent SDK is designed for agentic workflows. For simple
message completion tasks, it may have slightly different behavior than
the direct Anthropic API.
"""

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

    async def create_message(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Message:
        """
        Create a message using Claude Agent SDK.

        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens in response (note: SDK manages this internally)
            system_prompt: Optional system prompt
            model: Override the default model ("opus", "sonnet", or "haiku")

        Returns:
            Message object with content and usage info
        """
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


async def get_client(use_max_account: bool = False, model: str = "sonnet"):
    """
    Factory function to get the appropriate Claude client.

    Args:
        use_max_account: If True, use Claude Agent SDK with Max auth.
                        If False, use standard Anthropic SDK with API key.
        model: Model to use (for Max client)

    Returns:
        Either ClaudeMaxClient or anthropic.AsyncAnthropic client
    """
    if use_max_account:
        return ClaudeMaxClient(model=model)
    else:
        import anthropic
        return anthropic.AsyncAnthropic()
