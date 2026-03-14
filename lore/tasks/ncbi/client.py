"""
Python-based API client for NCBI Datasets
"""

import functools
import http.client
from time import sleep
import logging

try:
    from ncbi.datasets.openapi.api_client import ApiClient
    from ncbi.datasets.openapi.api.genome_api import GenomeApi
    from ncbi.datasets.openapi.configuration import Configuration as NcbiConfig
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


class MissingNcbiDatasetsError(RuntimeError):
    """Raised when NCBI Datasets SDK is not installed."""
    pass


def _ensure_ncbi_sdk():
    """Ensure that the NCBI Datasets SDK is installed."""
    if not HAS_SDK:
        raise MissingNcbiDatasetsError(
            "NCBI Datasets SDK is not installed. Please install the wheel in lore-genome/wheels"
        )


def make_genome_api(api_key: str | None = None) -> 'GenomeApi':
    """Create an NCBI Datasets GenomeApi client."""
    _ensure_ncbi_sdk()
    config = NcbiConfig()
    if api_key:
        config.api_key = {'cookieAuth': api_key}

    api_client = ApiClient(configuration=config)
    return GenomeApi(api_client=api_client)


def retry(exceptions=(http.client.IncompleteRead, TimeoutError), tries=4, delay=2, default_logger=None):
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
            logger = kwargs.pop('logger', default_logger)
            last_exec = None
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exec = e
                    sleeptime = delay ** attempt
                    msg = f'{e}, Retrying in {sleeptime} seconds...'
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
