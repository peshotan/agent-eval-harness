"""Tests for bounded, isolated asynchronous execution."""

import asyncio

import pytest

from harness.runner import AsyncRunner


@pytest.mark.asyncio
async def test_runner_preserves_input_order_and_bounds_concurrency() -> None:
    active = 0
    maximum_active = 0

    async def operation(value: int) -> int:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep((4 - value) * 0.001)
        active -= 1
        return value * 2

    outcomes = await AsyncRunner(concurrency=2).run([1, 2, 3], operation)

    assert [outcome.value for outcome in outcomes] == [2, 4, 6]
    assert [outcome.index for outcome in outcomes] == [0, 1, 2]
    assert maximum_active == 2
    assert all(outcome.succeeded for outcome in outcomes)


@pytest.mark.asyncio
async def test_runner_isolates_exceptions() -> None:
    async def operation(value: int) -> int:
        if value == 2:
            raise RuntimeError("broken case")
        return value

    outcomes = await AsyncRunner().run([1, 2, 3], operation)

    assert outcomes[0].value == 1
    assert outcomes[1].value is None
    assert outcomes[1].error == "RuntimeError: broken case"
    assert outcomes[2].value == 3


@pytest.mark.asyncio
async def test_runner_marks_timeouts_without_terminating_batch() -> None:
    async def operation(value: float) -> float:
        await asyncio.sleep(value)
        return value

    outcomes = await AsyncRunner(timeout_seconds=0.01).run([0.05, 0.0], operation)

    assert outcomes[0].timed_out
    assert outcomes[0].error is not None
    assert outcomes[1].succeeded


@pytest.mark.parametrize(
    ("concurrency", "timeout"),
    [(0, 1.0), (1, 0.0)],
)
def test_runner_rejects_invalid_limits(concurrency: int, timeout: float) -> None:
    with pytest.raises(ValueError):
        AsyncRunner(concurrency=concurrency, timeout_seconds=timeout)
