from __future__ import annotations

import pytest

from account_research_agent.llm import (
    ModelRouter,
    PermanentToolError,
    Route,
    Task,
    TransientError,
    model_family,
)
from account_research_agent.schemas import ResearchJudgeVerdict

from .fakes import FakeGateway

PRIMARY, FALLBACK = "google/gemini-3.8-flash", "mistralai/mistral-large-2512"


def test_primary_answers() -> None:
    gateway = FakeGateway()
    gateway.queue(ResearchJudgeVerdict, {"notes": "fine"})
    result = ModelRouter(gateway).call(Task.RESEARCH_JUDGE, [], ResearchJudgeVerdict)
    assert (result.model, result.output.notes, result.cost_usd) == (PRIMARY, "fine", 0.001)


def test_schema_violation_falls_back_and_both_calls_are_billed() -> None:
    gateway = FakeGateway()
    gateway.responses["ResearchJudgeVerdict"] += ['{"unsupported_claim_ids": "not a list"}', '{"notes": "ok"}']
    result = ModelRouter(gateway).call(Task.RESEARCH_JUDGE, [], ResearchJudgeVerdict)
    assert result.model == FALLBACK
    assert result.cost_usd == pytest.approx(0.002)


def test_both_providers_down_raises_transient_for_node_retry() -> None:
    gateway = FakeGateway(down={PRIMARY, FALLBACK})
    with pytest.raises(TransientError, match="all models failed"):
        ModelRouter(gateway).call(Task.RESEARCH_JUDGE, [], ResearchJudgeVerdict)


def test_avoid_family_that_covers_the_whole_route_is_permanent() -> None:
    router = ModelRouter(FakeGateway(), routes={Task.RESEARCH_JUDGE: Route("openai/a", "openai/b")})
    with pytest.raises(PermanentToolError):
        router.call(Task.RESEARCH_JUDGE, [], ResearchJudgeVerdict, avoid_family="openai")
    assert model_family("openai/gpt-5.4-mini") == "openai"
