"""Common metric interface."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from harness.schemas import MetricResult

MetricInputT = TypeVar("MetricInputT", bound=BaseModel)


class Metric(ABC, Generic[MetricInputT]):
    """A stateless evaluation metric over a typed input contract."""

    @abstractmethod
    def evaluate(self, context: MetricInputT) -> list[MetricResult]:
        """Evaluate one input and return one or more normalized metric results."""
