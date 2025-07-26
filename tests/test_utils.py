"""Tests for utility functions."""

import polars as pl
import pytest

from panic_tda.utils import polars_to_markdown, print_polars_as_markdown


def test_polars_to_markdown_basic():
    """Test basic conversion of Polars DataFrame to markdown."""
    # Create test data
    df = pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [95.5, 87.3, 92.8]
    })
    
    markdown = polars_to_markdown(df)
    
    # Check that it contains expected elements
    assert "|" in markdown
    assert "name" in markdown
    assert "age" in markdown
    assert "score" in markdown
    assert "Alice" in markdown
    assert "25" in markdown
    assert "95.50" in markdown  # Default float format is .2f


def test_polars_to_markdown_truncation():
    """Test that long strings are truncated properly."""
    # Create test data with long strings
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "description": [
            "Short text",
            "This is a very long description that should be truncated",
            "Another medium length text here"
        ]
    })
    
    # Test with truncation
    markdown = polars_to_markdown(df, max_col_width=20)
    
    assert "Short text" in markdown
    assert "This is a very lo..." in markdown
    assert "Another medium le..." in markdown  # This one is 31 chars, so it IS truncated at 20


def test_polars_to_markdown_custom_headers():
    """Test custom headers functionality."""
    df = pl.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    })
    
    markdown = polars_to_markdown(df, headers=["ID", "Letter"])
    
    assert "ID" in markdown
    assert "Letter" in markdown
    assert "col1" not in markdown
    assert "col2" not in markdown


def test_polars_to_markdown_float_format():
    """Test custom float formatting."""
    df = pl.DataFrame({
        "value": [1.23456, 2.34567, 3.45678]
    })
    
    # Test with different float formats
    markdown_2f = polars_to_markdown(df, floatfmt=".2f")
    assert "1.23" in markdown_2f
    assert "2.35" in markdown_2f
    
    markdown_4f = polars_to_markdown(df, floatfmt=".4f")
    assert "1.2346" in markdown_4f
    assert "2.3457" in markdown_4f


def test_polars_to_markdown_invalid_headers():
    """Test that invalid headers raise an error."""
    df = pl.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    })
    
    with pytest.raises(ValueError, match="Number of headers"):
        polars_to_markdown(df, headers=["Only One Header"])


def test_print_polars_as_markdown(capsys):
    """Test the print function."""
    df = pl.DataFrame({
        "name": ["Alice", "Bob"],
        "score": [95, 87]
    })
    
    print_polars_as_markdown(df, title="Test Results")
    
    captured = capsys.readouterr()
    assert "Test Results" in captured.out
    assert "=" in captured.out  # Title underline
    assert "|" in captured.out  # Table
    assert "Alice" in captured.out
    assert "95" in captured.out


def test_polars_to_markdown_empty_dataframe():
    """Test handling of empty DataFrames."""
    df = pl.DataFrame({"col1": [], "col2": []})
    
    markdown = polars_to_markdown(df)
    
    # Should still have headers
    assert "|" in markdown
    assert "col1" in markdown
    assert "col2" in markdown


def test_polars_to_markdown_mixed_types():
    """Test handling of mixed data types."""
    df = pl.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.1, 2.2, 3.3],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True]
    })
    
    markdown = polars_to_markdown(df)
    
    # Check all values are present
    assert "1" in markdown
    assert "2.20" in markdown  # Float with default formatting
    assert "a" in markdown
    assert "True" in markdown
    assert "False" in markdown