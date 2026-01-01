"""
Tracing utilities for LLM latency benchmarking.

Provides OpenTelemetry integration and optional Langfuse tracing
for distributed tracing and observability.
"""

import functools
import os
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Callable, Iterator, Optional

# OpenTelemetry imports - optional dependency
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

# Langfuse imports - optional dependency
try:
    from langfuse import Langfuse, observe, get_client
    LANGFUSE_AVAILABLE = True
    print("[Langfuse] SDK imported successfully")
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    observe = None
    get_client = None
    print(f"[Langfuse] SDK import failed: {e}")


class TracingConfig:
    """Configuration for tracing setup."""

    def __init__(
        self,
        service_name: str = "llm-latency-lab",
        enable_console_export: bool = True,
        enable_langfuse: bool = False,
        langfuse_public_key: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        langfuse_host: Optional[str] = None,
    ):
        self.service_name = service_name
        self.enable_console_export = enable_console_export
        self.enable_langfuse = enable_langfuse
        self.langfuse_public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.langfuse_host = langfuse_host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


class Tracer:
    """Unified tracer supporting OpenTelemetry and Langfuse."""

    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig()
        self._otel_tracer = None
        self._langfuse_client = None
        self._initialized = False

    def initialize(self) -> "Tracer":
        """Initialize tracing backends."""
        if self._initialized:
            return self

        # Initialize OpenTelemetry
        if OTEL_AVAILABLE:
            resource = Resource.create({"service.name": self.config.service_name})
            provider = TracerProvider(resource=resource)

            if self.config.enable_console_export:
                processor = SimpleSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(processor)

            trace.set_tracer_provider(provider)
            self._otel_tracer = trace.get_tracer(self.config.service_name)

        # Initialize Langfuse
        if LANGFUSE_AVAILABLE and self.config.enable_langfuse:
            if self.config.langfuse_public_key and self.config.langfuse_secret_key:
                self._langfuse_client = Langfuse(
                    public_key=self.config.langfuse_public_key,
                    secret_key=self.config.langfuse_secret_key,
                    host=self.config.langfuse_host,
                )

        self._initialized = True
        return self

    def shutdown(self) -> None:
        """Shutdown tracing backends."""
        if OTEL_AVAILABLE and trace.get_tracer_provider():
            trace.get_tracer_provider().shutdown()

        if self._langfuse_client:
            self._langfuse_client.flush()

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[dict] = None,
    ) -> Iterator[Any]:
        """Create a traced span for synchronous operations.

        Usage:
            with tracer.span("my_operation", {"key": "value"}) as span:
                # do work
                span.set_attribute("result", "success")
        """
        if not self._initialized:
            self.initialize()

        span_obj = None
        if self._otel_tracer:
            span_obj = self._otel_tracer.start_span(name)
            if attributes:
                for key, value in attributes.items():
                    span_obj.set_attribute(key, value)

        try:
            yield span_obj
        except Exception as e:
            if span_obj and OTEL_AVAILABLE:
                span_obj.set_status(Status(StatusCode.ERROR, str(e)))
                span_obj.record_exception(e)
            raise
        finally:
            if span_obj:
                span_obj.end()

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        attributes: Optional[dict] = None,
    ) -> AsyncIterator[Any]:
        """Create a traced span for async operations.

        Usage:
            async with tracer.async_span("my_operation") as span:
                # do async work
        """
        if not self._initialized:
            self.initialize()

        span_obj = None
        if self._otel_tracer:
            span_obj = self._otel_tracer.start_span(name)
            if attributes:
                for key, value in attributes.items():
                    span_obj.set_attribute(key, value)

        try:
            yield span_obj
        except Exception as e:
            if span_obj and OTEL_AVAILABLE:
                span_obj.set_status(Status(StatusCode.ERROR, str(e)))
                span_obj.record_exception(e)
            raise
        finally:
            if span_obj:
                span_obj.end()

    def trace_llm_call(
        self,
        name: str = "llm_call",
        model: Optional[str] = None,
        **kwargs,
    ) -> Callable:
        """Decorator for tracing LLM calls.

        Usage:
            @tracer.trace_llm_call("generate_response", model="claude-3-sonnet")
            async def generate_response(prompt: str):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **func_kwargs):
                attributes = {
                    "llm.model": model or "unknown",
                    "llm.operation": name,
                    **kwargs,
                }
                async with self.async_span(name, attributes) as span:
                    result = await func(*args, **func_kwargs)
                    if span and hasattr(result, "usage"):
                        span.set_attribute("llm.input_tokens", getattr(result.usage, "input_tokens", 0))
                        span.set_attribute("llm.output_tokens", getattr(result.usage, "output_tokens", 0))
                    return result

            @functools.wraps(func)
            def sync_wrapper(*args, **func_kwargs):
                attributes = {
                    "llm.model": model or "unknown",
                    "llm.operation": name,
                    **kwargs,
                }
                with self.span(name, attributes) as span:
                    result = func(*args, **func_kwargs)
                    if span and hasattr(result, "usage"):
                        span.set_attribute("llm.input_tokens", getattr(result.usage, "input_tokens", 0))
                        span.set_attribute("llm.output_tokens", getattr(result.usage, "output_tokens", 0))
                    return result

            if asyncio_iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator


def asyncio_iscoroutinefunction(func: Callable) -> bool:
    """Check if a function is a coroutine function."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


class LangfuseTracer:
    """Langfuse-specific tracer for LLM observability.

    Uses the new Langfuse SDK API with get_client() and context managers.
    """

    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig(enable_langfuse=True)
        self._initialized = False

    def initialize(self) -> "LangfuseTracer":
        """Initialize Langfuse client via environment variables.

        The new Langfuse SDK uses environment variables for configuration:
        - LANGFUSE_PUBLIC_KEY
        - LANGFUSE_SECRET_KEY
        - LANGFUSE_HOST
        """
        if self._initialized:
            return self

        if not LANGFUSE_AVAILABLE:
            raise ImportError("Langfuse is not installed. Run: pip install langfuse")

        # Set environment variables for the Langfuse SDK
        import os
        if self.config.langfuse_public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.config.langfuse_public_key
        if self.config.langfuse_secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = self.config.langfuse_secret_key
        if self.config.langfuse_host:
            os.environ["LANGFUSE_HOST"] = self.config.langfuse_host

        # Verify we can get the client
        try:
            client = get_client()
            print(f"[Langfuse] Client initialized successfully")
            self._initialized = True
        except Exception as e:
            print(f"[Langfuse] Failed to get client: {e}")
            raise

        return self

    def shutdown(self) -> None:
        """Flush and shutdown Langfuse client."""
        if LANGFUSE_AVAILABLE and get_client:
            try:
                client = get_client()
                client.flush()
            except Exception as e:
                print(f"[Langfuse] Error during flush: {e}")

    def trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Create a Langfuse trace using the new API.

        Returns a trace object that can be used as a context manager or
        have observations added to it.
        """
        if not self._initialized:
            self.initialize()

        if not LANGFUSE_AVAILABLE or not get_client:
            print(f"[Langfuse] Cannot create trace '{name}' - SDK not available")
            return None

        try:
            client = get_client()
            # Include user_id and session_id in metadata if provided
            full_metadata = metadata or {}
            if user_id:
                full_metadata["user_id"] = user_id
            if session_id:
                full_metadata["session_id"] = session_id

            # Use the context manager to create a span/trace
            trace_obj = client.start_as_current_observation(
                name=name,
                as_type="span",
                metadata=full_metadata if full_metadata else None,
            )
            print(f"[Langfuse] Trace/span created: {name}")
            return trace_obj
        except Exception as e:
            print(f"[Langfuse] Error creating trace '{name}': {e}")
            import traceback
            traceback.print_exc()
            return None

    def generation(
        self,
        trace,
        name: str,
        model: str,
        input_data: Any,
        output_data: Any = None,
        usage: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        """Record an LLM generation.

        With the new API, this creates a generation observation.
        """
        if not LANGFUSE_AVAILABLE or not get_client:
            return None

        try:
            client = get_client()
            with client.start_as_current_observation(
                name=name,
                as_type="generation",
                model=model,
            ) as gen:
                # Update with input, output, usage, and metadata
                gen.update(
                    input=input_data,
                    output=output_data,
                    usage=usage,
                    metadata=metadata,
                )
            print(f"[Langfuse] Generation recorded: {name}")
            return gen
        except Exception as e:
            print(f"[Langfuse] Error creating generation '{name}': {e}")
            return None


# Global tracer instance
_global_tracer: Optional[Tracer] = None


def get_tracer(config: Optional[TracingConfig] = None) -> Tracer:
    """Get or create the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer(config)
    return _global_tracer


def init_tracing(config: Optional[TracingConfig] = None) -> Tracer:
    """Initialize global tracing."""
    tracer = get_tracer(config)
    return tracer.initialize()


def shutdown_tracing() -> None:
    """Shutdown global tracing."""
    global _global_tracer
    if _global_tracer:
        _global_tracer.shutdown()
        _global_tracer = None


# Global LangfuseTracer instance for simplified access
_global_langfuse_tracer: Optional[LangfuseTracer] = None


def get_langfuse_tracer(config: Optional[TracingConfig] = None) -> Optional[LangfuseTracer]:
    """Get or create the global LangfuseTracer instance.

    Returns None if Langfuse is not available or not configured.
    """
    global _global_langfuse_tracer

    if not LANGFUSE_AVAILABLE:
        print("[Langfuse] SDK not available (not installed)")
        return None

    if _global_langfuse_tracer is None:
        tracer_config = config or TracingConfig(enable_langfuse=True)

        # Debug: Print configuration
        public_key = tracer_config.langfuse_public_key
        secret_key = tracer_config.langfuse_secret_key
        host = tracer_config.langfuse_host

        print(f"[Langfuse] Attempting initialization...")
        print(f"[Langfuse] Host: {host}")
        print(f"[Langfuse] Public key: {public_key[:20] + '...' if public_key else 'NOT SET'}")
        print(f"[Langfuse] Secret key: {'SET' if secret_key else 'NOT SET'}")

        if public_key and secret_key:
            _global_langfuse_tracer = LangfuseTracer(tracer_config)
            try:
                _global_langfuse_tracer.initialize()
                print(f"[Langfuse] Successfully initialized!")
            except Exception as e:
                print(f"[Langfuse] Failed to initialize: {e}")
                import traceback
                traceback.print_exc()
                _global_langfuse_tracer = None
        else:
            print("[Langfuse] Missing public or secret key - skipping initialization")

    return _global_langfuse_tracer


def flush_langfuse() -> None:
    """Flush any pending Langfuse events."""
    global _global_langfuse_tracer
    print("[Langfuse] Flushing traces...")

    if _global_langfuse_tracer:
        try:
            _global_langfuse_tracer.shutdown()
            print("[Langfuse] Tracer shutdown complete")
        except Exception as e:
            print(f"[Langfuse] Error during tracer shutdown: {e}")


def get_observe_decorator():
    """Get the Langfuse @observe decorator if available.

    Returns a no-op decorator if Langfuse is not available.

    Usage:
        observe = get_observe_decorator()

        @observe(name="my_function")
        async def my_function():
            ...
    """
    if LANGFUSE_AVAILABLE and observe:
        return observe

    # Return a no-op decorator
    def noop_decorator(*args, **kwargs):
        def wrapper(func):
            return func
        # Handle both @observe and @observe() syntax
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return wrapper

    return noop_decorator


# Export key items for easy importing
__all__ = [
    "TracingConfig",
    "Tracer",
    "LangfuseTracer",
    "get_tracer",
    "init_tracing",
    "shutdown_tracing",
    "get_langfuse_tracer",
    "flush_langfuse",
    "get_observe_decorator",
    "LANGFUSE_AVAILABLE",
    "OTEL_AVAILABLE",
    "observe",
    "get_client",
]
