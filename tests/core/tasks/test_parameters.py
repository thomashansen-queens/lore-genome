"""
Tests for the Task parameters.
"""
import pytest
from typing import get_args

from lore.core.tasks import (
    Widget,
    WidgetLiteral,
    Cardinality,
    CardinalityLiteral,
    Materialization,
    MaterializationLiteral,
)


def test_literal_parameters_have_not_drifted():
    """
    Literals should mirror their Enum counterparts exactly, but must be
    hardcoded as Literals. This test ensures that they haven't drifted apart.
    """
    widget_literals = list(get_args(WidgetLiteral))
    cardinality_literals = list(get_args(CardinalityLiteral))
    materialization_literals = list(get_args(MaterializationLiteral))

    assert widget_literals == list(Widget)
    assert cardinality_literals == list(Cardinality)
    assert materialization_literals == list(Materialization)
