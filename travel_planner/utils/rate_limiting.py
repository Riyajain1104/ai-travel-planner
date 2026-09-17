"""
Rate limiting and API request management for external services.

This module provides utilities for rate limiting, request throttling,
exponential backoff, and API quota management.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, TypeVar, cast

import aiohttp
from aiolimiter import AsyncLimiter
from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from travel_planner.utils.error_handling import APIError


F = TypeVar("F", bound=Callable[..., Any])


HTTP_STATUS_OK = 200
HTTP_STATUS_REDIRECT = 300
HTTP_STATUS_TOO_MANY_REQUESTS = 429


@dataclass
class RateLimitConfig:
    """Configuration for a service's rate limits."""

    service_name: str
    requests_per_minute: int
    requests_per_day: int
    max_retries: int = 3
    min_wait_seconds: float = 1.0
    max_wait_seconds: float = 30.0
    retry_status_codes: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )
    cooldown_after_quota: int = 60


@dataclass
class QuotaUsage:
    """Tracks API quota usage for a service."""

    daily_count: int = 0
    last_reset: datetime = field(default_factory=datetime.now)

    def increment(self) -> None:
        """Increment the daily usage counter."""
        current_time = datetime.now()

        if current_time.date() > self.last_reset.date():
            self.daily_count = 0
            self.last_reset = current_time

        self.daily_count += 1

    def get_remaining(self, daily_quota: int) -> int:
        """Get remaining requests for the day."""
        current_time = datetime.now()

        if current_time.date() > self.last_reset.date():
            self.daily_count = 0
            self.last_reset = current_time

        return max(0, daily_quota - self.daily_count)

    def is_quota_exceeded(self, daily_quota: int) -> bool:
        """Check whether the daily quota is exceeded."""
        return self.get_remaining(daily_quota) <= 0


class ServiceRateLimiter:
    """Rate limiter for a specific service."""

    def __init__(self, config: RateLimitConfig):
        """Initialize the rate limiter."""
        self.config = config
        self.quota_usage = QuotaUsage()

        self.limiter = AsyncLimiter(
            config.requests_per_minute,
            60,
        )

        self.request_timestamps: list[float] = []

        logger.info(
            f"Initialized rate limiter for {config.service_name} "
            f"({config.requests_per_minute}/min, "
            f"{config.requests_per_day}/day)"
        )

    async def acquire(self) -> bool:
        """Acquire permission to make a request."""

        if self.quota_usage.is_quota_exceeded(
            self.config.requests_per_day
        ):
            logger.warning(
                f"Daily quota exceeded for {self.config.service_name} "
                f"({self.config.requests_per_day} requests/day)"
            )
            return False

        current_time = time.time()
        minute_ago = current_time - 60

        self.request_timestamps = [
            timestamp
            for timestamp in self.request_timestamps
            if timestamp > minute_ago
        ]

        if len(self.request_timestamps) >= self.config.requests_per_minute:
            logger.warning(
                f"Rate limit reached for {self.config.service_name} "
                f"({self.config.requests_per_minute} requests/minute)"
            )
            return False

        async with self.limiter:
            self.request_timestamps.append(current_time)
            self.quota_usage.increment()
            return True

    def get_backoff_time(self) -> float:
        """Calculate backoff time when rate limited."""

        remaining_quota = self.quota_usage.get_remaining(
            self.config.requests_per_day
        )

        quota_factor = max(
            1,
            (self.config.requests_per_day * 0.1)
            / max(1, remaining_quota),
        )

        if len(self.request_timestamps) >= self.config.requests_per_minute:
            current_time = time.time()
            oldest = min(self.request_timestamps)

            wait_time = max(
                0,
                oldest + 60 - current_time,
            )

            return min(
                wait_time * quota_factor,
                self.config.max_wait_seconds,
            )

        return min(
            self.config.min_wait_seconds * quota_factor,
            self.config.max_wait_seconds,
        )

    def should_retry_exception(
        self,
        exception: Exception,
    ) -> bool:
        """Determine whether an exception should trigger a retry.

        HTTP 429 is deliberately excluded. A 429 can represent a provider
        quota exhaustion (especially for Gemini free-tier accounts), and
        blindly retrying it only burns time and can repeat the same failure.
        """

        if isinstance(
            exception,
            (
                aiohttp.ClientConnectorError,
                aiohttp.ServerDisconnectedError,
            ),
        ):
            return True

        if isinstance(exception, APIError):
            return (
                exception.status_code in self.config.retry_status_codes
                and exception.status_code != HTTP_STATUS_TOO_MANY_REQUESTS
            )

        return False

    def get_quota_stats(self) -> dict[str, Any]:
        """Get quota usage statistics."""

        return {
            "service": self.config.service_name,
            "daily_quota": self.config.requests_per_day,
            "used_today": self.quota_usage.daily_count,
            "remaining": self.quota_usage.get_remaining(
                self.config.requests_per_day
            ),
            "minute_limit": self.config.requests_per_minute,
            "current_minute_usage": len(self.request_timestamps),
        }


# Default configurations are defined BEFORE the manager is created.
DEFAULT_RATE_LIMITS = [
    RateLimitConfig(
        service_name="gemini",
        requests_per_minute=3,
        requests_per_day=20,
        max_retries=3,
        min_wait_seconds=1.0,
        max_wait_seconds=60.0,
    ),
    RateLimitConfig(
        service_name="tavily",
        requests_per_minute=60,
        requests_per_day=1000,
        max_retries=3,
        min_wait_seconds=1.0,
        max_wait_seconds=30.0,
    ),
    RateLimitConfig(
        service_name="firecrawl",
        requests_per_minute=10,
        requests_per_day=300,
        max_retries=3,
        min_wait_seconds=1.0,
        max_wait_seconds=30.0,
    ),
    RateLimitConfig(
        service_name="supabase",
        requests_per_minute=100,
        requests_per_day=5000,
        max_retries=3,
        min_wait_seconds=0.5,
        max_wait_seconds=15.0,
    ),
]


class RateLimitManager:
    """Manager for rate limiters across multiple services."""

    def __init__(self):
        """Initialize the rate limit manager."""

        self.limiters: dict[str, ServiceRateLimiter] = {}

        self.default_config = RateLimitConfig(
            service_name="default",
            requests_per_minute=30,
            requests_per_day=1000,
            max_retries=3,
            min_wait_seconds=1.0,
            max_wait_seconds=30.0,
        )

        for config in DEFAULT_RATE_LIMITS:
            self.register_service(config)

    def register_service(
        self,
        config: RateLimitConfig,
    ) -> ServiceRateLimiter:
        """Register a service with the rate limit manager."""

        limiter = ServiceRateLimiter(config)
        self.limiters[config.service_name] = limiter

        return limiter

    def get_limiter(
        self,
        service_name: str,
    ) -> ServiceRateLimiter:
        """Get the rate limiter for a service."""

        if service_name not in self.limiters:
            logger.warning(
                f"No rate limiter configured for {service_name}, "
                f"using default configuration."
            )

            config = RateLimitConfig(
                service_name=service_name,
                requests_per_minute=(
                    self.default_config.requests_per_minute
                ),
                requests_per_day=(
                    self.default_config.requests_per_day
                ),
                max_retries=self.default_config.max_retries,
                min_wait_seconds=(
                    self.default_config.min_wait_seconds
                ),
                max_wait_seconds=(
                    self.default_config.max_wait_seconds
                ),
            )

            self.register_service(config)

        return self.limiters[service_name]

    def get_all_quota_stats(self) -> list[dict[str, Any]]:
        """Get quota statistics for all services."""

        return [
            limiter.get_quota_stats()
            for limiter in self.limiters.values()
        ]

    async def wait_if_needed(
        self,
        service_name: str,
    ) -> None:
        """Wait if the service is currently rate limited."""

        limiter = self.get_limiter(service_name)

        if not await limiter.acquire():
            backoff_time = limiter.get_backoff_time()

            logger.warning(
                f"Rate limit reached for {service_name}, "
                f"waiting {backoff_time:.2f} seconds"
            )

            await asyncio.sleep(backoff_time)


rate_limit_manager = RateLimitManager()


def configure_rate_limits(
    service_configs: list[RateLimitConfig],
) -> None:
    """Configure rate limits for multiple services."""

    for config in service_configs:
        rate_limit_manager.register_service(config)


def initialize_rate_limiting() -> None:
    """Initialize rate limiting with default configurations."""

    configure_rate_limits(DEFAULT_RATE_LIMITS)

    logger.info(
        f"Initialized rate limiting for "
        f"{len(DEFAULT_RATE_LIMITS)} services"
    )


def before_sleep_callback(
    retry_state: RetryCallState,
) -> None:
    """Callback executed before sleeping between retries."""

    exception = retry_state.outcome.exception()

    if exception:
        logger.warning(
            f"Request failed "
            f"(attempt {retry_state.attempt_number}/"
            f"{retry_state.retry_object.stop.max_attempt_number}), "
            f"retrying in "
            f"{retry_state.next_action.sleep:.2f} seconds: "
            f"{exception!s}"
        )


def retry_if_request_is_retryable(
    limiter: ServiceRateLimiter,
):
    """Build a Tenacity predicate for retryable request failures.

    Provider 429 responses are deliberately excluded because they commonly
    represent quota exhaustion and repeating the request does not recover it.
    """

    from tenacity import retry_if_exception

    return retry_if_exception(limiter.should_retry_exception)


async def with_rate_limit(
    service_name: str,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a function with rate limiting."""

    limiter = rate_limit_manager.get_limiter(service_name)

    if not await limiter.acquire():
        backoff_time = limiter.get_backoff_time()

        logger.warning(
            f"Rate limit reached for {service_name}, "
            f"waiting {backoff_time:.2f} seconds"
        )

        await asyncio.sleep(backoff_time)

        if not await limiter.acquire():
            raise RuntimeError(
                f"Rate limit quota exceeded for {service_name}. "
                f"Please try again later."
            )

    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_request_is_retryable(limiter),
            stop=stop_after_attempt(
                limiter.config.max_retries
            ),
            wait=wait_exponential(
                multiplier=1,
                min=limiter.config.min_wait_seconds,
                max=limiter.config.max_wait_seconds,
            ),
            reraise=True,
            before_sleep=before_sleep_callback,
        ):
            with attempt:
                return await func(*args, **kwargs)

    except Exception as exc:
        logger.error(
            f"Request to {service_name} failed after "
            f"{limiter.config.max_retries} attempts: {exc!s}"
        )
        raise


def rate_limited(service_name: str) -> Callable[[F], F]:
    """Decorator to apply rate limiting to a function."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return await with_rate_limit(
                service_name,
                func,
                *args,
                **kwargs,
            )

        return cast(F, wrapper)

    return decorator


class APIClient:
    """Base client for API requests with rate limiting."""

    def __init__(
        self,
        service_name: str,
        base_url: str,
        api_key: str | None = None,
    ):
        """Initialize the API client."""

        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.limiter = rate_limit_manager.get_limiter(
            service_name
        )

    async def request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make an API request with rate limiting."""

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        request_headers: dict[str, str] = {}

        if headers:
            request_headers.update(headers)

        if self.api_key:
            request_headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        async def do_request() -> dict[str, Any]:
            async with aiohttp.ClientSession() as session:
                request_method = getattr(
                    session,
                    method.lower(),
                )

                async with request_method(
                    url,
                    params=params,
                    json=json_data,
                    headers=request_headers,
                ) as response:
                    status_code = response.status
                    response_text = await response.text()

                    if (
                        status_code
                        == HTTP_STATUS_TOO_MANY_REQUESTS
                    ):
                        raise APIError(
                            "Rate limit exceeded",
                            self.service_name,
                            status_code=status_code,
                        )

                    if not (
                        HTTP_STATUS_OK
                        <= status_code
                        < HTTP_STATUS_REDIRECT
                    ):
                        raise APIError(
                            f"API request failed: {response_text}",
                            self.service_name,
                            status_code=status_code,
                        )

                    try:
                        return await response.json()
                    except aiohttp.ContentTypeError:
                        return {"text": response_text}

        return await with_rate_limit(
            self.service_name,
            do_request,
        )

    @dataclass
    class RequestConfig:
        """Configuration for a generic HTTP request."""

        url: str
        method: str = "GET"
        params: dict[str, Any] | None = None
        json_data: dict[str, Any] | None = None
        headers: dict[str, str] | None = None
        service_name: str | None = None

    @rate_limited("UNKNOWN")
    async def generic_request(
        self,
        config: RequestConfig,
    ) -> dict[str, Any]:
        """Make a generic HTTP request."""

        actual_service = (
            config.service_name or self.service_name
        )

        request_headers: dict[str, str] = {}

        if config.headers:
            request_headers.update(config.headers)

        if (
            self.api_key
            and "Authorization" not in request_headers
        ):
            request_headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        async with aiohttp.ClientSession() as session:
            request_method = getattr(
                session,
                config.method.lower(),
            )

            async with request_method(
                config.url,
                params=config.params,
                json=config.json_data,
                headers=request_headers,
            ) as response:
                status_code = response.status
                response_text = await response.text()

                if not (
                    HTTP_STATUS_OK
                    <= status_code
                    < HTTP_STATUS_REDIRECT
                ):
                    raise APIError(
                        f"API request failed: {response_text}",
                        actual_service,
                        status_code=status_code,
                    )

                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    return {"text": response_text}


def update_rate_limits_from_config(
    config_dict: dict[str, dict[str, Any]],
) -> None:
    """Update rate limits from a configuration dictionary."""

    for service_name, service_config in config_dict.items():
        config = RateLimitConfig(
            service_name=service_name,
            requests_per_minute=service_config.get(
                "requests_per_minute",
                30,
            ),
            requests_per_day=service_config.get(
                "requests_per_day",
                1000,
            ),
            max_retries=service_config.get(
                "max_retries",
                3,
            ),
            min_wait_seconds=service_config.get(
                "min_wait_seconds",
                1.0,
            ),
            max_wait_seconds=service_config.get(
                "max_wait_seconds",
                30.0,
            ),
            retry_status_codes=service_config.get(
                "retry_status_codes",
                [429, 500, 502, 503, 504],
            ),
            cooldown_after_quota=service_config.get(
                "cooldown_after_quota",
                60,
            ),
        )

        rate_limit_manager.register_service(config)

        logger.info(
            f"Updated rate limits for {service_name} "
            f"from configuration"
        )