"""Model routing over OpenRouter: primary, fallback, and the judge-family rule.

OpenRouter can fall back on its own (`models: [...]`). We route explicitly
instead, because the delivery judge must know which family actually wrote the
draft. A native fallback would hide that.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class TransientError(Exception):
    """Worth retrying: 429, 5xx, timeout. Node RetryPolicy retries on this."""


class PermanentToolError(Exception):
    """Not worth retrying: 4xx, entity not found, no access."""


class ProviderUnavailableError(TransientError):
    """The provider could not answer (429, 5xx, timeout). The router tries the fallback model."""


class Task(StrEnum):
    """One value per LLM step. Each task has its own model route."""

    EXTRACTION = "extraction"
    RESEARCH_JUDGE = "research_judge"
    SYNTHESIS = "synthesis"
    PERSONALIZATION = "personalization"
    DELIVERY_JUDGE = "delivery_judge"


@dataclass(frozen=True)
class Route:
    """A primary model and a fallback from a different provider."""

    primary: str
    fallback: str


# Chosen in section 1.2 (price shortlist + golden-dataset eval).
ROUTES: dict[Task, Route] = {
    Task.EXTRACTION: Route("openai/gpt-5-nano", "google/gemini-2.5-flash-lite"),
    Task.RESEARCH_JUDGE: Route("google/gemini-3.8-flash", "mistralai/mistral-large-2512"),
    Task.SYNTHESIS: Route("google/gemini-3.8-flash", "openai/gpt-5.4-mini"),
    Task.PERSONALIZATION: Route("google/gemini-3.8-flash", "openai/gpt-5.4-mini"),
    Task.DELIVERY_JUDGE: Route("openai/gpt-5.4-mini", "mistralai/mistral-large-2512"),
}


def model_family(model_id: str) -> str:
    """'openai/gpt-5-nano' -> 'openai'. Used by the judge-family rule."""
    return model_id.split("/", 1)[0]


@dataclass(frozen=True)
class Message:
    """One chat message, in the order the provider receives it."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMResult(Generic[T]):
    """Validated output, the model that produced it, and the cost of every attempt."""

    output: T
    model: str
    cost_usd: float


class LLMGateway(Protocol):
    """Anything that returns JSON for a schema: OpenRouter in production, a fake in tests."""

    def complete_json(
        self, *, model: str, messages: list[Message], json_schema: dict[str, object]
    ) -> tuple[str, float]:
        """Return (raw JSON text, cost in USD). Raise TransientError subclasses on 429/5xx."""
        ...


class OpenRouterGateway:
    """Real gateway. The HTTP call is pseudocode: no key in this sample."""

    def __init__(self, api_key: str, timeout_s: float = 60.0) -> None:
        self._api_key = api_key
        self._timeout_s = timeout_s

    def complete_json(
        self, *, model: str, messages: list[Message], json_schema: dict[str, object]
    ) -> tuple[str, float]:
        # response = httpx.post(
        #     "https://openrouter.ai/api/v1/chat/completions",
        #     headers={"Authorization": f"Bearer {self._api_key}"},
        #     json={
        #         "model": model,
        #         "messages": [m.__dict__ for m in messages],
        #         "response_format": {
        #             "type": "json_schema",
        #             "json_schema": {"name": "output", "strict": True, "schema": json_schema},
        #         },
        #         "usage": {"include": True},
        #     },
        #     timeout=self._timeout_s,
        # )
        # if response.status_code == 429 or response.status_code >= 500:
        #     raise ProviderUnavailableError(f"{model}: HTTP {response.status_code}")
        # response.raise_for_status()
        # body = response.json()
        # return body["choices"][0]["message"]["content"], body["usage"]["cost"]
        raise NotImplementedError("pseudocode: requires an OpenRouter API key")


class ModelRouter:
    """Calls the right model for each task, with provider fallback and schema validation."""

    def __init__(self, gateway: LLMGateway, routes: dict[Task, Route] = ROUTES) -> None:
        self._gateway = gateway
        self._routes = routes

    def call(
        self,
        task: Task,
        messages: list[Message],
        schema: type[T],
        *,
        avoid_family: str | None = None,
    ) -> LLMResult[T]:
        """Try primary, then fallback. Raise TransientError when both fail.

        `avoid_family` enforces "a judge never grades its own family": any
        candidate from that family is dropped before the call.
        """
        route = self._routes[task]
        candidates = [m for m in (route.primary, route.fallback) if model_family(m) != avoid_family]
        if not candidates:
            raise PermanentToolError(f"{task}: every model in the route is from {avoid_family}")

        spent = 0.0
        failures: list[str] = []
        for model in candidates:
            try:
                raw, cost = self._gateway.complete_json(
                    model=model, messages=messages, json_schema=schema.model_json_schema()
                )
            except ProviderUnavailableError as exc:
                failures.append(f"{model}: {exc}")
                continue
            spent += cost
            try:
                return LLMResult(schema.model_validate_json(raw), model, spent)
            except ValidationError as exc:
                # Reason: a bad schema from one model is often fine on another.
                # The failed call's cost is still counted.
                failures.append(f"{model}: schema violation ({exc.error_count()} errors)")
        raise TransientError(f"{task}: all models failed: {'; '.join(failures)}")
