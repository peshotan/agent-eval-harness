"""Concurrency-bounded, failure-isolated asynchronous execution."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class RunOutcome(Generic[ResultT]):
    """Result envelope that keeps per-item failures inside the batch."""

    index: int
    value: ResultT | None
    latency_ms: float
    error: str | None = None
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.value is not None


class AsyncRunner:
    """Run independent work concurrently with deterministic output ordering."""

    def __init__(self, *, concurrency: int = 5, timeout_seconds: float = 60.0) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.concurrency = concurrency
        self.timeout_seconds = timeout_seconds

    async def run(
        self,
        items: Sequence[InputT],
        operation: Callable[[InputT], Awaitable[ResultT]],
    ) -> list[RunOutcome[ResultT]]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def execute(index: int, item: InputT) -> RunOutcome[ResultT]:
            started = perf_counter()
            try:
                async with semaphore, asyncio.timeout(self.timeout_seconds):
                    value = await operation(item)
                return RunOutcome(
                    index=index,
                    value=value,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            except TimeoutError:
                return RunOutcome(
                    index=index,
                    value=None,
                    latency_ms=(perf_counter() - started) * 1000,
                    error=f"execution exceeded {self.timeout_seconds:g}s timeout",
                    timed_out=True,
                )
            except Exception as error:
                return RunOutcome(
                    index=index,
                    value=None,
                    latency_ms=(perf_counter() - started) * 1000,
                    error=f"{type(error).__name__}: {error}",
                )

        outcomes = await asyncio.gather(
            *(execute(index, item) for index, item in enumerate(items))
        )
        return sorted(outcomes, key=lambda outcome: outcome.index)
