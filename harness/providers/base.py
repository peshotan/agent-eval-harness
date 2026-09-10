"""Provider-neutral model generation interface."""

from abc import ABC, abstractmethod

from harness.schemas import ModelRequest, ModelResponse


class ModelProvider(ABC):
    """Generate normalized model responses without leaking vendor SDK types."""

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Execute one model request."""
