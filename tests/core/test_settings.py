"""
Tests for the settings module
"""
import pytest
from pathlib import Path

from lore.core.settings import (
    Settings, load_settings, save_settings,
)

def test_settings_lifecycle(tmp_path: Path):
    """Proves we can save and load Settings and paths are correct"""
    # ARRANGE
    custom_root = tmp_path / "my_data_root"
    original = Settings(
        data_root=custom_root,
        verbose=True,
        auto_preview_selected_artifacts=False,
    )

    # ACT
    save_settings(tmp_path, original, filename="my_settings.json")
    loaded = load_settings(tmp_path, filename="my_settings.json")

    # ASSERT
    assert loaded.verbose is True
    assert loaded.data_root == custom_root
    assert isinstance(loaded.data_root, Path)
    assert loaded.auto_preview_selected_artifacts is False


def test_auto_preview_selected_artifacts_defaults_to_enabled():
    assert Settings().auto_preview_selected_artifacts is True
