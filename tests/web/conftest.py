"""
Fixtures for web layer tests.
"""
import pytest
from starlette.testclient import TestClient

from lore.web.api import create_app


@pytest.fixture
def client(temp_runtime):
    """A TestClient wired to a fresh app backed by a temporary runtime."""
    app = create_app(temp_runtime)
    with TestClient(app) as c:
        yield c
