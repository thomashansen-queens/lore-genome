"""
Utility functions for working with Pandas DataFrames.
"""

import re
import pandas as pd


def normalize_query(query_string: str) -> str:
    """
    Translates user-friendly or SQL-style syntax into valid Pandas query syntax.
    Useful for Explore Artifact view.
    """
    if not query_string:
        return ""

    # 1. SQL-style Logical Operators -> Python style
    # We use regex word boundaries (\b) to ensure we don't replace 'AND' inside a country name
    query_string = re.sub(r'\bAND\b', 'and', query_string, flags=re.IGNORECASE)
    query_string = re.sub(r'\bOR\b', 'or', query_string, flags=re.IGNORECASE)
    query_string = re.sub(r'\bNOT\b', 'not', query_string, flags=re.IGNORECASE)

    # 2. Cleanup whitespace
    query_string = " ".join(query_string.split())

    return query_string


def _make_query_pattern(query_string: str) -> str:
    """Convert a simple query string into a case-insensitive substring search pattern."""
    new_query = query_string.strip().replace('"', '').replace("'", "")
    if "," in new_query:
        parts = [re.escape(p.strip()) for p in new_query.split(",") if p.strip()]
        return "|".join(parts)
    return re.escape(new_query)


def filter_and_sort(
    df: pd.DataFrame,
    query: str = "",
    regex: bool = False,
    sort_by: str | None = None,
    sort_asc: bool = True,
) -> pd.DataFrame:
    """
    Logic for 3-tier filtering (regex > pandas query > substring match) and sorting of a DataFrame.
    """
    if df is None or df.empty:
        return df

    # 1. Filter (skip if no query)
    if query.strip():

        # A. Regex search (by row)
        if regex:
            try:
                mask = df.astype(str).apply(
                    lambda col: col.str.contains(
                        query, case=False, na=False, regex=True,
                    )
                ).any(axis=1)
                df = df[mask]
            except Exception as e:
                raise ValueError(f"Invalid regex pattern: {str(e)}") from e
        else:
            # B. Pandas query string
            try:
                # TODO: query allows arbitrary code execution, should sandbox or guard
                query_str = normalize_query(query.strip())
                df = df.query(query_str)
            except Exception:
                # C. Case-insensitive substring search
                try:
                    search_pattern = _make_query_pattern(query)
                    mask = df.astype(str).apply(
                        lambda col: col.str.contains(
                            search_pattern, case=False, na=False, regex=True,
                        )
                    ).any(axis=1)
                    df = df[mask]
                except Exception as e:
                    raise ValueError(f"Invalid query: {str(e)}") from e

    # 2. Sort (always runs if sort_by is set)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=sort_asc, na_position="last")

    return df
