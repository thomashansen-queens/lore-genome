"""
Python-based REST API client for NCBI Datasets

NCBI Datasets API documentation:
https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/rest-api/
"""

from contextlib import contextmanager
from importlib.metadata import version
import functools
from time import sleep
import logging
import httpx

import lore.dsl as lore


NCBI_DATASETS_BASE_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"


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


@contextmanager
def ncbi_client(api_key: str | None = None, timeout: float = 60.0):
    """
    Create a configured httpx client for the NCBI Datasets API.
    event_hooks allows us to raise exceptions on HTTP errors, rather than checking
    {"success": false, "error": {...}} in the JSON response.
    """
    headers = {
        "Accept": "application/json",
        "User-Agent": f"lore-genome/{version('lore-genome')}",
    }
    if api_key:
        headers["api-key"] = api_key

    def raise_on_4xx_5xx(response: httpx.Response):
        response.raise_for_status()

    with httpx.Client(
        base_url=NCBI_DATASETS_BASE_URL,
        headers=headers,
        timeout=httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=timeout),
        event_hooks={"response": [raise_on_4xx_5xx]},
        verify=False,
    ) as client:
        yield client


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
