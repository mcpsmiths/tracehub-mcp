"""Abstract base backend for OpenTelemetry trace storage systems."""

from abc import ABC, abstractmethod
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from opentelemetry_mcp.attributes import HealthCheckResponse
from opentelemetry_mcp.models import FilterOperator, SpanData, SpanQuery, TraceData, TraceQuery

# Only genuine transport-level failures - a connection that never completed
# or a request that timed out - are worth retrying. An HTTP response with a
# 4xx/5xx status code is a *successful* transport exchange (a response body
# still came back), not one of these exceptions, so it is never retried here.
_RETRYABLE_TRANSPORT_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException)


class _RetryingTransport(httpx.AsyncBaseTransport):
    """Wraps another async transport and retries only transport-level
    connection failures with exponential backoff, reraising the final
    exception if every attempt fails.
    """

    def __init__(self, wrapped: httpx.AsyncBaseTransport | None = None) -> None:
        """Wrap an underlying transport (defaults to a fresh AsyncHTTPTransport)."""
        self._wrapped = wrapped if wrapped is not None else httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Delegate to the wrapped transport, retrying up to 3 total attempts
        (with exponential backoff, capped around 10s) on connect/timeout
        failures only.
        """

        # A plain `async def` closure (rather than passing
        # self._wrapped.handle_async_request straight to tenacity) ensures
        # tenacity always awaits the call correctly - some transport wrappers
        # (e.g. VCR's cassette-recording patch used in this project's own
        # integration tests) expose handle_async_request as a *sync* function
        # that returns a coroutine, which tenacity's coroutine-callable
        # detection can miss, silently returning an unawaited coroutine.
        async def _send() -> httpx.Response:
            return await self._wrapped.handle_async_request(request)

        retrying = AsyncRetrying(
            retry=retry_if_exception_type(_RETRYABLE_TRANSPORT_EXCEPTIONS),
            wait=wait_exponential(multiplier=1, max=10),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        response: httpx.Response = await retrying(_send)
        return response

    async def aclose(self) -> None:
        """Close the wrapped transport's connection pool."""
        await self._wrapped.aclose()


class BaseBackend(ABC):
    """Abstract interface for OpenTelemetry trace backends."""

    def __init__(self, url: str, api_key: str | None = None, timeout: float = 30.0):
        """Initialize backend with connection parameters.

        Args:
            url: Backend API URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling.

        All backends share this client construction, so the retry transport
        applies automatically to every backend's requests.

        Returns:
            Reusable AsyncClient instance with automatic connection pooling
            and connect/timeout retry with exponential backoff
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.url,
                headers=self._create_headers(),
                timeout=self.timeout,
                follow_redirects=True,
                transport=_RetryingTransport(),
            )
        return self._client

    @abstractmethod
    def _create_headers(self) -> dict[str, str]:
        """Create backend-specific HTTP headers.

        Returns:
            Dictionary of HTTP headers (e.g., Authorization, Content-Type)
        """
        pass

    @abstractmethod
    def get_supported_operators(self) -> set[FilterOperator]:
        """Get the set of filter operators that this backend natively supports.

        Operators not in this set will be applied via client-side filtering.

        Returns:
            Set of natively supported FilterOperator values
        """
        pass

    @abstractmethod
    async def search_traces(self, query: TraceQuery) -> list[TraceData]:
        """Search for traces matching the given query.

        Args:
            query: Trace query parameters

        Returns:
            List of matching traces with all spans

        Raises:
            Exception: If the backend query fails
        """
        pass

    @abstractmethod
    async def search_spans(self, query: SpanQuery) -> list[SpanData]:
        """Search for individual spans matching the given query.

        Args:
            query: Span query parameters

        Returns:
            List of matching spans (not grouped by trace)

        Raises:
            Exception: If the backend query fails
        """
        pass

    @abstractmethod
    async def get_trace(self, trace_id: str) -> TraceData:
        """Get a specific trace by ID.

        Args:
            trace_id: Trace identifier

        Returns:
            Complete trace data with all spans

        Raises:
            Exception: If trace not found or query fails
        """
        pass

    @abstractmethod
    async def list_services(self) -> list[str]:
        """List all available services.

        Returns:
            List of service names

        Raises:
            Exception: If query fails
        """
        pass

    @abstractmethod
    async def get_service_operations(self, service_name: str) -> list[str]:
        """Get all operations for a specific service.

        Args:
            service_name: Service name

        Returns:
            List of operation names

        Raises:
            Exception: If query fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheckResponse:
        """Check backend health and connectivity.

        Returns:
            Health status information

        Raises:
            Exception: If backend is unreachable
        """
        pass

    async def __aenter__(self) -> "BaseBackend":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close HTTP client connections."""
        if self._client:
            await self._client.aclose()
            self._client = None
