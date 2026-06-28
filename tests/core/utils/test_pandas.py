"""
Tests for DataFrame filter/sort helpers (lore.core.utils.pandas).
"""
import pandas as pd

from lore.core.utils.pandas import filter_and_sort


def test_sort_mixed_numeric_column_sinks_unsortable_cells():
    """A mostly-numeric column with stray list/string cells sorts numerically and
    sinks the unsortable cells to the bottom instead of raising on str/float compares.
    (Regression: explore-view sort blew up with "'<' not supported between 'str' and 'float'".)"""
    df = pd.DataFrame({
        "len": [123, [1, 2], 5, "99", None],
        "name": ["a", "b", "c", "d", "e"],
    })

    out = filter_and_sort(df, sort_by="len", sort_asc=True)

    # Numeric cells sorted ascending; the list, the non-numeric, and the null sink.
    assert list(out["name"])[:3] == ["c", "d", "a"]  # 5, 99, 123
    # The list cell and None are last regardless of direction.
    assert out["name"].iloc[-1] == "e"  # None always last


def test_sort_unsortable_stays_last_when_descending():
    """na_position='last' must win over sort direction: unsortable cells never
    bubble to the top on a descending sort."""
    df = pd.DataFrame({"len": [1, [9, 9], 2, None], "name": ["a", "b", "c", "d"]})

    out = filter_and_sort(df, sort_by="len", sort_asc=False)

    # 2 then 1 at the top; list + None remain at the bottom.
    assert list(out["name"])[:2] == ["c", "a"]
    assert set(out["name"].iloc[-2:]) == {"b", "d"}

def test_sort_preserves_original_cell_values():
    """Sorting must not mutate displayed values (no silent str->numeric coercion)."""
    df = pd.DataFrame({"len": ["007", "42", "1"], "name": ["a", "b", "c"]})

    out = filter_and_sort(df, sort_by="len", sort_asc=True)

    # Still the original strings (e.g. "007" not 7), just reordered numerically.
    assert set(out["len"]) == {"007", "42", "1"}
    assert list(out["name"]) == ["c", "a", "b"]  # numeric order 1, 7 ("007"), 42
