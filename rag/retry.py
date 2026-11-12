"""Retry temporary API failures (rate limit, overloaded server, timeouts) with waiting that grows each time."""
import random
import time

RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
RETRYABLE_WORDS = ("timeout", "timed out", "temporarily", "unavailable", "overloaded", "rate limit", "connection")


def is_retryable(exc):
    """True for temporary errors. Wrong key (401/403) or unknown model (404) are NOT retried."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code in RETRYABLE_CODES
    message = str(exc).lower()
    return any(word in message for word in RETRYABLE_WORDS)


def call_with_retries(fn, max_attempts=4, base_delay=1.0, sleep=time.sleep, on_retry=None):
    """Call fn(); on a temporary error wait 1s, 2s, 4s... (plus a little jitter) and try again."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_attempts or not is_retryable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            if on_retry:
                on_retry(attempt, exc, delay)
            sleep(delay)
