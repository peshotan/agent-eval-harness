"""Tests for deterministic mock-agent tools."""

import pytest

from target_agent.tools import calculate_average, query_sales_db


def test_sales_fixture_and_average_are_deterministic() -> None:
    sales = query_sales_db(quarter="Q3", year=2024)

    assert sales.total_sales == 1_200_000
    assert calculate_average(total_sales=sales.total_sales, deals=sales.deals) == 24_000


def test_unknown_period_is_explicit_error() -> None:
    with pytest.raises(ValueError, match="no sales data"):
        query_sales_db(quarter="Q1", year=2020)


def test_average_rejects_empty_population() -> None:
    with pytest.raises(ValueError, match="deals must be positive"):
        calculate_average(total_sales=100, deals=0)
