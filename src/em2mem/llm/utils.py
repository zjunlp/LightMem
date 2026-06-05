# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import asyncio
import functools
from tenacity import retry, stop_after_attempt, wait_exponential

def dynamic_retry_decorator(func):
    """Decorator that applies retry logic with exponential backoff."""
    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        max_retries = getattr(self, 'max_retries', 5)
        decorated_func = retry(
            stop=stop_after_attempt(max_retries), 
            wait=wait_exponential(multiplier=1, min=1, max=10)
        )(func)
        return decorated_func(self, *args, **kwargs)
        
    async def async_wrapper(self, *args, **kwargs):
        max_retries = getattr(self, 'max_retries', 5)
        decorated_func = retry(
            stop=stop_after_attempt(max_retries), 
            wait=wait_exponential(multiplier=1, min=1, max=10)
        )(func)
        return await decorated_func(self, *args, **kwargs)
        
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper