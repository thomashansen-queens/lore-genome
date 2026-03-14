"""
Tests for Runtime module
"""
import os
from pathlib import Path
import pytest

from lore.core.runtime import build_runtime


def test_build_runtime_with_explicit_data_root(tmp_path: Path):
    """
    Tests specifying a data_root when building the Runtime
    """
    # ACT
    rt = build_runtime(data_root=tmp_path)

    # ASSERT
    assert rt.data_root == tmp_path.resolve()
    assert rt.sessions_dir == tmp_path.resolve() / "sessions"

    # Prove that ensure_dirs() did its job!
    assert rt.sessions_dir.exists()
    assert rt.sessions_dir.is_dir()


def test_build_runtime_with_env_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Tests that the factory correctly falls back to the LORE_DATA_ROOT 
    environment variable if no explicit argument is given.
    """
    # ARRANGE: Use Pytest's built-in 'monkeypatch' to safely fake an env variable
    fake_env_path = tmp_path / "env_root"
    monkeypatch.setenv("LORE_DATA_ROOT", str(fake_env_path))

    # ACT: Call without arguments
    rt = build_runtime()

    # ASSERT
    assert rt.data_root == fake_env_path.resolve()
    assert rt.sessions_dir.exists()
    assert rt.sessions_dir.is_dir()
