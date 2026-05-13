"""
Tests for filelock module.
"""
from pathlib import Path
import pytest

from lore.core.filelock import acquire_lock, is_file_locked, release_lock


@pytest.fixture
def dummy_lock_file(tmp_path: Path) -> Path:
    """Creates a temporary file to be used for locking tests."""
    lock_file = tmp_path / "test.lock"
    lock_file.write_text("dummy content")
    return lock_file


def test_acquire_and_release(dummy_lock_file):
    """The happy path: We can lock and unlock a file."""
    # Append mode so we don't erase the dummy content!
    with open(dummy_lock_file, "a", encoding="utf-8") as f:
        acquire_lock(f, timeout=1)
        # If we made it here without an exception, the lock worked.
        release_lock(f)


def test_is_file_locked_status(dummy_lock_file):
    """Tests our non-blocking probe function."""
    assert is_file_locked(dummy_lock_file) is False

    # Simulate Process A locking the file
    f = open(dummy_lock_file, "a", encoding="utf-8")
    try:
        acquire_lock(f, timeout=1)

        # Simulate Process B checking the file
        assert is_file_locked(dummy_lock_file) is True
        # Process A releases
        release_lock(f)
        assert is_file_locked(dummy_lock_file) is False

    finally:
        if is_file_locked(dummy_lock_file):
            release_lock(f)
        f.close()


def test_acquire_lock_timeout(dummy_lock_file):
    """Tests that a blocked lock correctly raises a RuntimeError after the timeout."""
    # 1. Process A grabs the lock
    f1 = open(dummy_lock_file, "a", encoding="utf-8")

    try:
        acquire_lock(f1, timeout=1)

        # 2. Process B desperately tries to grab the lock, but times out quickly
        f2 = open(dummy_lock_file, "a", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Timeout acquiring session lock"):
            acquire_lock(f2, timeout=0.2) # Wait 200ms then give up

    finally:
        release_lock(f1)
        f1.close()
        f2.close()
