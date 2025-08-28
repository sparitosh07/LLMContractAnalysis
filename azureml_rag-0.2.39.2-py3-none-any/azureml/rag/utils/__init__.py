# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Utility functions."""

import random
import time
from collections import defaultdict
from typing import Set

__path__ = __import__("pkgutil").extend_path(__path__, __name__)  # type: ignore


def merge_dicts(dict1, dict2):
    """Merge two dictionaries recursively."""
    result = defaultdict(dict)

    for d in (dict1, dict2):
        for key, value in d.items():
            if isinstance(value, dict) and key in result:
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value

    return dict(result)


def backoff_retry_on_exceptions(
    max_attempts: int, initial_delay: float, max_delay: float, retry_exceptions: Set[Exception]
):
    """
    Decorator that retries a function call with Jittered delay to spread out the retries on given exceptions.

    Args:
    ----
        max_attempts (int): The maximum number of attempts to retry the function call.
        initial_delay (float): The initial delay in seconds before the first retry.
        max_delay (float): The maximum delay in seconds between retries.
        retry_exceptions (set): A set of exception types that should trigger a retry.

    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            delay = initial_delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if type(e) not in retry_exceptions:
                        raise e
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    time.sleep(delay)
                    delay = min(max_delay, delay * random.uniform(1, 2))

        return wrapper

    return decorator
