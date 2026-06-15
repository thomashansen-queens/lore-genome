"""
Tests for the Task definition.
"""
import pytest
from typing import get_args

from lore.core.tasks import (
    PreviewMode,
    PreviewModeLiteral,
)


def test_literal_definitions_have_not_drifted():
    """
    Literals should mirror their Enum counterparts exactly, but must be
    hardcoded as Literals. This test ensures that they haven't drifted apart.
    """
    preview_mode_literals = list(get_args(PreviewModeLiteral))

    assert preview_mode_literals == list(PreviewMode)
