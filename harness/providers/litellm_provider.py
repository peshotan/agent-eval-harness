"""LiteLLM-backed implementation of the model-provider contract."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

import litellm

from harness.providers.base import ModelProvider
from harness.schemas import ModelRequest, ModelResponse, TokenUsage

Completion = Callable[..., Awaitable[Any]]
CostCalculator = Callable[..., float]


class LiteLLMProvider(ModelProvider):
    """Normalize LiteLLM completions, usage, metadata, and known pricing."""

    def __init__(
        self,
        *,
        completion: Completion | None = None,
        cost_calculator: CostCalculator | None = None,
    ) -> None:
        self._completion = completion or litellm.acompletion
        self._cost_calculator = cost_calculator or litellm.completion_cost

    async def generate(self, request: ModelRequest) -> ModelResponse:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        arguments: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.response_format is not None:
            arguments["response_format"] = request.response_format

        response = await self._completion(**arguments)
        payload = self._mapping(response)
        usage = self._usage(payload.get("usage"), response)
        choices = payload.get("choices")
        content = self._content(choices)
        response_model = payload.get("model")
        response_id = payload.get("id")

        return ModelResponse(
            model=response_model if isinstance(response_model, str) else request.model,
            content=content,
            usage=usage,
            provider_metadata={
                "provider": "litellm",
                "response_id": response_id if isinstance(response_id, str) else None,
                "cost_status": (
                    "available"
                    if usage is not None and usage.estimated_cost_usd is not None
                    else "unavailable"
                ),
            },
            error=None if content is not None else "provider response contained no text",
        )

    def _usage(self, value: object, response: object) -> TokenUsage | None:
        if value is None:
            return None
        usage = self._mapping(value)
        input_tokens = self._integer(usage.get("prompt_tokens"))
        output_tokens = self._integer(usage.get("completion_tokens"))
        if input_tokens is None or output_tokens is None:
            return None
        try:
            estimated_cost = self._cost_calculator(completion_response=response)
        except Exception:
            estimated_cost = None
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=estimated_cost,
        )

    @classmethod
    def _content(cls, choices: object) -> str | None:
        if not isinstance(choices, list) or not choices:
            return None
        first = cls._mapping(choices[0])
        message = cls._mapping(first.get("message"))
        content = message.get("content")
        return content if isinstance(content, str) else None

    @staticmethod
    def _integer(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _mapping(value: object) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return cast(Mapping[str, Any], value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, Mapping):
                return cast(Mapping[str, Any], dumped)
        return {}
