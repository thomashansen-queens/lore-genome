"""
Python-based REST API client for Interpro

Interpro API documentation:
https://www.ebi.ac.uk/jdispatcher/docs/webservices/#/

Under the "Choose a Tool" dropdown choose "InterProScan 6" 
"""

from contextlib import contextmanager
from importlib.metadata import version
from time import sleep
import logging
import httpx

import lore.dsl as lore

INTERPRO_BASE_URL  = "https://www.ebi.ac.uk/Tools/services/rest"

@lore.config(key="interpro", title="Interpro")
class InterproDatasetsConfig:
    """Global settings for the Interpro API."""
    api_key = lore.ValueInput(
        str,
        default=None,
        label="Email",
        description=(
            "InterPro requires your email to run. Enter your email here:"
        ),
    )
    
@contextmanager
def interpro_client(timeout: float = 60.0):
    """
    Create a configured httpx client for general web API calls.
    event_hooks allows us to raise exceptions on HTTP errors, rather than checking
    {"success": false, "error": {...}} in the JSON response.
    """
    headers = {
        "Accept": "text/plain",
        "User-Agent": f"lore-genome/{version('lore-genome')}"
    }

    def raise_on_4xx_5xx(response: httpx.Response):
        response.raise_for_status()

    with httpx.Client(
        base_url=INTERPRO_BASE_URL,
        headers=headers,
        timeout=httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=timeout),
        event_hooks={"response": [raise_on_4xx_5xx]},
        verify=False,
    ) as client:
        yield client