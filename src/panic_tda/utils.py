"""Utility functions for the panic_tda package."""

import polars as pl
from typing import Optional, List, Dict, Any


def polars_to_markdown(
    df: pl.DataFrame,
    max_col_width: Optional[int] = None,
    headers: Optional[List[str]] = None,
    floatfmt: str = ".2f",
) -> str:
    """
    Convert a Polars DataFrame to a markdown table string.

    Args:
        df: Polars DataFrame to convert
        max_col_width: Maximum width for any column (characters). If a value exceeds
                      this width, it will be truncated with "..."
        headers: Optional list of header names to override the DataFrame column names
        floatfmt: Format string for floating point numbers (default: ".2f")

    Returns:
        Markdown-formatted table string
    """
    # Make a copy to avoid modifying the original
    df_copy = df.clone()

    # Truncate long strings if max_col_width is specified
    if max_col_width:
        for col in df_copy.columns:
            if df_copy[col].dtype == pl.Utf8:
                df_copy = df_copy.with_columns(
                    pl.when(pl.col(col).str.len_chars() > max_col_width)
                    .then(pl.col(col).str.slice(0, max_col_width - 3) + "...")
                    .otherwise(pl.col(col))
                    .alias(col)
                )

    # Convert to pandas for markdown formatting
    pandas_df = df_copy.to_pandas()

    # Use custom headers if provided
    if headers:
        if len(headers) != len(pandas_df.columns):
            raise ValueError(f"Number of headers ({len(headers)}) must match number of columns ({len(pandas_df.columns)})")
        pandas_df.columns = headers

    # Generate markdown
    return pandas_df.to_markdown(index=False, floatfmt=floatfmt)


def print_polars_as_markdown(
    df: pl.DataFrame,
    title: Optional[str] = None,
    max_col_width: Optional[int] = None,
    headers: Optional[List[str]] = None,
    floatfmt: str = ".2f",
) -> None:
    """
    Print a Polars DataFrame as a markdown table with optional title.

    Args:
        df: Polars DataFrame to print
        title: Optional title to print above the table
        max_col_width: Maximum width for any column (characters)
        headers: Optional list of header names to override the DataFrame column names
        floatfmt: Format string for floating point numbers (default: ".2f")
    """
    if title:
        print(f"\n{title}")
        print("=" * len(title))
        print()

    markdown_table = polars_to_markdown(df, max_col_width, headers, floatfmt)
    print(markdown_table)
