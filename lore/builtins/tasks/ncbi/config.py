"""
Python-based REST API client for NCBI Datasets
"""
import functools
from time import sleep
import logging
import httpx

import lore


@lore.config(key="ncbi", title="NCBI Datasets")
class NcbiDatasetsConfig:
    """Global settings for the NCBI Datasets API."""
    api_key = lore.ValueInput(
        str,
        default=None,
        label="NCBI API Key",
        description=(
            "API for NCBI Datasets. This can be found/created in your NCBI account settings "
            "(https://account.ncbi.nlm.nih.gov/settings/)"
        ),
    )
    email = lore.ValueInput(
        str,
        default="",
        label="Contact Email",
        description=(
            "Email address to include in API requests. NCBI recommends including an email for "
            "contact purposes, but it is not strictly required."
        ),
    )


def retry(exceptions=(httpx.RequestError, httpx.TimeoutException), tries=4, delay=2, default_logger=None):
    """
    A decorator that allows API calls to retry a set number of times before failing.

    :param exceptions: The exception(s) to catch and retry on.
    :param tries: The number of times to retry the function.
    :param delay: The delay between retries (exponentially increasing).
    :param default_logger: The logger to use for messages.
    :return: The result of the function call.
    """
    if default_logger is None:
        default_logger = logging.getLogger(__name__)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If a logger was passed, use it
            logger = kwargs.pop("logger", default_logger)
            last_exec = None
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    # Smart retry: 429 Rate Limit, 500 Internal Server Error, etc.
                    if e.response.status_code in (429, 500, 502, 503, 504):
                        last_exec = e
                    # Non-retriable errors: 404 Not Found, 400 Bad Request, 401 Unauthorized, etc.
                    else:
                        raise e
                except exceptions as e:
                    last_exec = e

                sleeptime = delay ** attempt
                msg = f"API request failed: {last_exec}. Retrying in {sleeptime} seconds..."

                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
                sleep(sleeptime)

            if logger:
                logger.error("Failed to execute %s after %s attempts.", func.__name__, tries)
            raise last_exec if last_exec else Exception("API error. Also, bug in retry decorator.")

        return wrapper
    return decorator
