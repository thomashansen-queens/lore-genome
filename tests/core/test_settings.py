"""
Tests for the settings module
"""
import pytest
from pathlib import Path

from lore.core.settings import (
    Settings, Secrets, load_settings, load_secrets, save_settings, save_secrets,
)

def test_settings_lifecycle(tmp_path: Path):
    """Proves we can save and load Settings and paths are correct"""
    # ARRANGE
    custom_root = tmp_path / "my_data_root"
    original = Settings(data_root=custom_root, verbose=True)

    # ACT
    save_settings(tmp_path, original, filename="my_settings.json")
    loaded = load_settings(tmp_path, filename="my_settings.json")

    # ASSERT
    assert loaded.verbose is True
    assert loaded.data_root == custom_root
    assert isinstance(loaded.data_root, Path)


def test_secrets_lifecycle_and_permissions(tmp_path: Path):
    """Proves we can save and load Secrets and that file permissions are secure"""
    # ARRANGE
    original = Secrets(ncbi_api_key="super_secret_key")

    # ACT
    saved_path = save_secrets(tmp_path, original, filename="my_secrets.json")
    loaded = load_secrets(tmp_path, filename="my_secrets.json")

    # ASSERT
    assert loaded.ncbi_api_key == "super_secret_key"
    assert saved_path.exists()
    # Check that the file permissions are secure (owner read/write only)
    permissions = saved_path.stat().st_mode
    assert permissions & 0o077 == 0, "Secrets file permissions are not secure (should be 600 or 400)"
