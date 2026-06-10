"""
A client for the NCBI Datasets v2 REST API, which is used any time
tasks need to contact the public server.

https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/rest-api/
"""
from contextlib import contextmanager
import httpx
from importlib.metadata import version

NCBI_DATASETS_BASE_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"


@contextmanager
def datasets_client(api_key: str | None = None, timeout: float = 60.0):
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
