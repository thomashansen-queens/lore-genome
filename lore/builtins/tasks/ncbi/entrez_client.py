"""
A client for the NCBI Entrez /E-Utilities API, which is used any time tasks
need to contact the public server.

https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""
from contextlib import contextmanager
from enum import StrEnum
import httpx
from importlib.metadata import version

NCBI_ENTREZ_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


class retmode(StrEnum):
    """NCBI Entrez return modes."""
    JSON = "json"
    XML = "xml"


@contextmanager
def entrez_client(
    api_key: str | None = None,
    email: str | None = None,
    timeout: float = 60.0,
    ret: retmode = retmode.JSON,
):
    """
    Create a configured httpx client for the NCBI Entrez API.
    event_hooks allows us to raise exceptions on HTTP errors, rather than checking
    {"success": false, "error": {...}} in the JSON response.
    """
    params = {
        "retmode": ret.value,
        "tool": f"lore-genome/{version('lore-genome')}",
    }

    # Be a good citizen of the NCBI API
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    def raise_on_4xx_5xx(response: httpx.Response):
        response.raise_for_status()

    with httpx.Client(
        base_url=NCBI_ENTREZ_BASE_URL,
        params=params,
        timeout=httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=timeout),
        event_hooks={"response": [raise_on_4xx_5xx]},
    ) as client:
        yield client

# --- General purpose Entrez classes ---

class EntrezDb(StrEnum):
    """NCBI Entrez databases."""
    NUCCORE = "nuccore"
    PROTEIN = "protein"
    ASSEMBLY = "assembly"
    BIOPROJECT = "bioproject"
    BIOSAMPLE = "biosample"
    GENE = "gene"
    SRA = "sra"
