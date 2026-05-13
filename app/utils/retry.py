import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

T = TypeVar("T")


def retry_delay(attempt: int, base_delay: float, retry_after: float | None = None) -> float:
    if retry_after is not None and retry_after > 0:
        return retry_after
    return base_delay * (2 ** attempt)


def is_retryable_openai_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code == 429 or error.status_code >= 500
    return False


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_delay: float,
    is_retryable: Callable[[Exception], bool],
) -> T:
    last_error: Exception | None = None
    attempts = max(max_attempts, 1)
    for attempt in range(attempts):
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001 - caller decides retryability
            last_error = exc
            if attempt >= attempts - 1 or not is_retryable(exc):
                raise
            await asyncio.sleep(retry_delay(attempt, base_delay))
    raise last_error or RuntimeError("Retry operation failed without exception")
