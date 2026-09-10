from typing import Any

import pytest

from harness.providers.litellm_provider import LiteLLMProvider
from harness.schemas import ModelRequest


@pytest.mark.asyncio
async def test_normalizes_content_usage_and_cost() -> None:
    captured: dict[str, Any] = {}

    async def completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "id": "response-1",
            "model": "provider/model-version",
            "choices": [{"message": {"content": '{"answer": 42}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    provider = LiteLLMProvider(
        completion=completion,
        cost_calculator=lambda **_: 0.0003,
    )
    response = await provider.generate(
        ModelRequest(
            model="provider/model",
            prompt="Question",
            system_prompt="System",
            response_format={"type": "json_object"},
        )
    )

    assert response.content == '{"answer": 42}'
    assert response.model == "provider/model-version"
    assert response.usage is not None
    assert response.usage.total_tokens == 15
    assert response.usage.estimated_cost_usd == 0.0003
    assert response.provider_metadata["cost_status"] == "available"
    assert captured["messages"][0] == {"role": "system", "content": "System"}
    assert captured["temperature"] == 0


@pytest.mark.asyncio
async def test_cost_failure_is_explicitly_unavailable() -> None:
    async def completion(**_: Any) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }

    def unavailable_cost(**_: Any) -> float:
        raise ValueError("unknown pricing")

    response = await LiteLLMProvider(
        completion=completion,
        cost_calculator=unavailable_cost,
    ).generate(ModelRequest(model="local/model", prompt="hello"))

    assert response.usage is not None
    assert response.usage.estimated_cost_usd is None
    assert response.provider_metadata["cost_status"] == "unavailable"


@pytest.mark.asyncio
async def test_missing_text_becomes_provider_error() -> None:
    async def completion(**_: Any) -> dict[str, Any]:
        return {"choices": [], "usage": None}

    response = await LiteLLMProvider(completion=completion).generate(
        ModelRequest(model="provider/model", prompt="hello")
    )

    assert response.content is None
    assert response.error == "provider response contained no text"
    assert response.usage is None
