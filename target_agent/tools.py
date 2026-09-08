"""Deterministic tools exposed to the mock target agents."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SalesSummary:
    total_sales: int
    deals: int


def query_sales_db(*, quarter: str, year: int) -> SalesSummary:
    """Return deterministic fixture data for supported reporting periods."""
    records = {
        ("Q3", 2024): SalesSummary(total_sales=1_200_000, deals=50),
        ("Q2", 2024): SalesSummary(total_sales=900_000, deals=45),
    }
    try:
        return records[(quarter, year)]
    except KeyError as error:
        raise ValueError(f"no sales data for {quarter} {year}") from error


def calculate_average(*, total_sales: int, deals: int) -> float:
    """Calculate average deal size while rejecting an empty population."""
    if deals <= 0:
        raise ValueError("deals must be positive")
    return total_sales / deals
