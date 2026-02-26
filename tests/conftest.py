"""
Global fixtures live here

This tells pytest how to prepare a Runtime for tests.
"""
import pytest
import shutil
from pathlib import Path
from lore.core.runtime import build_runtime, Runtime

@pytest.fixture
def test_runtime(tmp_path: Path):
    """
    Creates a temporary Runtime for testing
    """
    data_root = tmp_path / "lore_data"
    rt = build_runtime(data_root=data_root, verbose=False)
    # return the runtime to the test
    yield rt
