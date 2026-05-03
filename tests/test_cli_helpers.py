"""Tests for CLI helper behavior."""

import pytest
import typer

from latex_tools.cli import _parse_pages


def test_parse_pages_accepts_ranges_and_deduplicates():
    assert _parse_pages("1,3-5,3") == [1, 3, 4, 5]


def test_parse_pages_all_when_empty():
    assert _parse_pages(None) is None
    assert _parse_pages(" ") is None


def test_parse_pages_rejects_invalid_range():
    with pytest.raises(typer.BadParameter):
        _parse_pages("5-3")
