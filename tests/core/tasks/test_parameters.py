"""
Tests for the Task parameters.
"""
from typing import get_args

import pytest

from lore.core.adapters import CsvAdapter, JsonAdapter
from lore.core.tasks import (
    ArtifactInput,
    Cardinality,
    CardinalityLiteral,
    Materialization,
    MaterializationLiteral,
    Widget,
    WidgetLiteral,
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


def test_artifact_input_adapt_as_preserves_option_order_in_metadata():
    field_info = ArtifactInput(
        accepted_data=["json"],
        adapt_as=[JsonAdapter, CsvAdapter],
    ).to_field_info()

    assert field_info.json_schema_extra["adapt_as"] == ["JsonAdapter", "CsvAdapter"]


def test_artifact_input_adapt_as_rejects_empty_options():
    with pytest.raises(ValueError, match="at least one adapter"):
        ArtifactInput(adapt_as=[])
