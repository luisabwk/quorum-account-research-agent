from __future__ import annotations

from dataclasses import dataclass

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from account_research_agent.llm import ModelRouter, PermanentToolError, TransientError
from account_research_agent.nodes import Deps
from account_research_agent.orchestrator import RerunLimitExceededError, compile_app, rerun, run_account
from account_research_agent.schemas import (
    AccountBrief,
    Branch,
    DeliveryJudgeVerdict,
    EvidenceStatus,
    OutreachDraft,
    RerunRequest,
    RerunScope,
    ResearchDepth,
    ResearchJudgeVerdict,
    ResearchState,
    RunOutcome,
)

from .fakes import (
    NOW,
    FakeGateway,
    FakeKB,
    FakeRepo,
    FakeReview,
    FakeTools,
    account,
    claim_json,
    document,
    stakeholder_json,
)

FAST_RETRY = RetryPolicy(max_attempts=3, initial_interval=0.001, jitter=False, retry_on=TransientError)

REG_DOC = document(Branch.REGULATORY, 1, "Specific lobbying issues: H.R. 9001, crew size requirements")
PEOPLE_DOC = document(Branch.STAKEHOLDER, 1, "Leadership: Dana Okafor, VP Government Affairs")
NEWS_DOC = document(Branch.NEWS, 1, "The company testified before committees in Ohio and Iowa")

GOOD_BRIEF = {
    "summary": "Rail operator active on federal crew size rules.",
    "commercial_hypothesis": "State and federal tracking load.",
    "outreach_angle": "Crew size bills across states.",
    "cited_claim_ids": ["regulatory:0:c1"],
}
GOOD_VERDICT = {
    "groundedness": 5,
    "approved_capabilities_only": True,
    "relevance": 4,
    "tone_and_length": 4,
    "critique": "ok",
}
BAD_VERDICT = {
    "groundedness": 2,
    "approved_capabilities_only": True,
    "relevance": 3,
    "tone_and_length": 4,
    "critique": "Sentence 2 is not supported.",
}


def draft(claim_id: str = "regulatory:0:c1") -> dict[str, object]:
    return {
        "stakeholder_id": "stk-1",
        "subject": "Crew size bills",
        "body": "Hi Dana, ...",
        "cited_claim_ids": [claim_id],
        "cited_kb_ids": ["kb-state-tracking"],
    }


@dataclass
class Harness:
    gateway: FakeGateway
    tools: FakeTools
    repo: FakeRepo
    review: FakeReview
    deps: Deps
    app: CompiledStateGraph[ResearchState]

    def queue_research(self) -> None:
        self.gateway.extraction[Branch.REGULATORY].append(
            {
                "claims": [
                    claim_json(REG_DOC, "c1", "Lobbied on H.R. 9001 in 2026.", "H.R. 9001, crew size requirements")
                ]
            }
        )
        self.gateway.extraction[Branch.STAKEHOLDER].append(
            {
                "stakeholders": [
                    stakeholder_json("stk-1", "Dana Okafor", "VP Government Affairs"),
                    stakeholder_json("stk-2", "Lee Park", "Policy Analyst"),
                ]
            }
        )
        self.gateway.extraction[Branch.NEWS].append(
            {
                "claims": [
                    claim_json(NEWS_DOC, "c1", "Testified in Ohio and Iowa committees.", "testified before committees")
                ]
            }
        )
        self.gateway.queue(ResearchJudgeVerdict, {})


def harness() -> Harness:
    gateway, repo, review = FakeGateway(), FakeRepo(), FakeReview()
    tools = FakeTools(
        documents={Branch.REGULATORY: [REG_DOC], Branch.STAKEHOLDER: [PEOPLE_DOC], Branch.NEWS: [NEWS_DOC]}
    )
    deps = Deps(router=ModelRouter(gateway), tools=tools, kb=FakeKB(), repo=repo, review=review, now=lambda: NOW)
    return Harness(gateway, tools, repo, review, deps, compile_app(deps, InMemorySaver(), retry=FAST_RETRY))


def ready_harness() -> Harness:
    h = harness()
    h.queue_research()
    return h


def test_deep_account_reaches_review_with_grounded_draft() -> None:
    h = ready_harness()
    h.gateway.queue(AccountBrief, GOOD_BRIEF)
    h.gateway.queue(OutreachDraft, draft())
    h.gateway.queue(DeliveryJudgeVerdict, GOOD_VERDICT)

    state = run_account(h.app, account(), "run-1")

    assert state["outcome"] is RunOutcome.READY_FOR_REVIEW
    assert state["depth"] is ResearchDepth.DEEP
    assert {c.claim_id for c in state["claims"] if c.status is EvidenceStatus.VERIFIED} == {
        "regulatory:0:c1",
        "news:0:c1",
    }
    assert [r.run_id for r in h.repo.saved] == ["run-1"] and len(h.review.posted) == 1
    assert h.repo.saved[0].draft_stakeholder is not None
    assert h.repo.saved[0].draft_stakeholder.full_name == "Dana Okafor"  # VP outranks analyst
    assert state["cost_usd"] == pytest.approx(0.007)


def test_customer_is_skipped_without_research_or_llm_calls() -> None:
    h = harness()
    state = run_account(h.app, account(is_customer=True), "run-skip")
    assert state["outcome"] is RunOutcome.SKIPPED
    assert h.tools.calls == [] and h.gateway.calls == []
    assert h.repo.saved[0].outcome is RunOutcome.SKIPPED


def test_fabricated_excerpt_is_rejected_and_branch_retried_once() -> None:
    h = harness()
    h.tools.documents = {Branch.REGULATORY: [REG_DOC], Branch.STAKEHOLDER: [PEOPLE_DOC]}
    fake = {"claims": [claim_json(REG_DOC, "c1", "Lobbied on a bill about rail.", "quote that is not in the source")]}
    people = {"stakeholders": [stakeholder_json("stk-1", "Dana Okafor", "VP Government Affairs")]}
    h.gateway.extraction[Branch.REGULATORY] += [fake, fake]  # attempt 0, then the retry
    h.gateway.extraction[Branch.STAKEHOLDER].append(people)
    h.gateway.queue(ResearchJudgeVerdict, {}, {})

    state = run_account(h.app, account(), "run-fake")

    regulatory_attempts = [a for b, _, a in h.tools.calls if b is Branch.REGULATORY]
    assert regulatory_attempts == [0, 1]
    rejected = [c for c in state["claims"] if c.status is EvidenceStatus.REJECTED]
    assert all("not a verbatim quote" in " ".join(c.rejection_reasons) for c in rejected)
    assert state["outcome"] is RunOutcome.NEEDS_HUMAN_RESEARCH
    assert h.gateway.models_for("OutreachDraft") == []  # never drafts on weak evidence


def test_draft_citing_unknown_claim_fails_without_judge_call_then_rewrites() -> None:
    h = ready_harness()
    h.gateway.queue(AccountBrief, GOOD_BRIEF)
    h.gateway.queue(OutreachDraft, draft("regulatory:0:invented"), draft())
    h.gateway.queue(DeliveryJudgeVerdict, GOOD_VERDICT)

    state = run_account(h.app, account(), "run-cite")

    assert state["outcome"] is RunOutcome.READY_FOR_REVIEW
    assert len(h.gateway.models_for("DeliveryJudgeVerdict")) == 1
    rewrite_prompt = [m for _, t, m in h.gateway.calls if t == "OutreachDraft"][1][-1].content
    assert "previous_draft_rejected" in rewrite_prompt and "invented" in rewrite_prompt


def test_judge_switches_family_when_generator_used_fallback() -> None:
    h = ready_harness()
    # Synthesis and personalization fall back from google to openai.
    h.gateway.down.add("google/gemini-3.8-flash")
    h.gateway.queue(AccountBrief, GOOD_BRIEF)
    h.gateway.queue(OutreachDraft, draft())
    h.gateway.queue(DeliveryJudgeVerdict, GOOD_VERDICT)

    state = run_account(h.app, account(), "run-family")

    assert state["models_used"]["personalization"] == "openai/gpt-5.4-mini"
    # The judge's primary is openai too, so it must skip straight to mistral.
    assert h.gateway.models_for("DeliveryJudgeVerdict") == ["mistralai/mistral-large-2512"]


def test_permanent_tool_error_is_recorded_and_run_continues() -> None:
    h = ready_harness()

    def news_down(_: int) -> None:
        raise PermanentToolError("Brave: 403 subscription expired")

    h.tools.before[Branch.NEWS] = news_down
    h.gateway.queue(AccountBrief, GOOD_BRIEF)
    h.gateway.queue(OutreachDraft, draft())
    h.gateway.queue(DeliveryJudgeVerdict, GOOD_VERDICT)

    state = run_account(h.app, account(), "run-perm")

    assert state["outcome"] is RunOutcome.READY_FOR_REVIEW
    assert state["branch_errors"] == ["news/attempt 0: Brave: 403 subscription expired"]


def test_transient_tool_error_is_retried_by_node_policy() -> None:
    h = ready_harness()

    def flaky(call_number: int) -> None:
        if call_number == 1:
            raise TransientError("LDA.gov 503")

    h.tools.before[Branch.REGULATORY] = flaky
    h.gateway.queue(AccountBrief, GOOD_BRIEF)
    h.gateway.queue(OutreachDraft, draft())
    h.gateway.queue(DeliveryJudgeVerdict, GOOD_VERDICT)

    state = run_account(h.app, account(), "run-flaky")

    assert [a for b, _, a in h.tools.calls if b is Branch.REGULATORY] == [0, 0]
    assert state["outcome"] is RunOutcome.READY_FOR_REVIEW


def test_two_failed_drafts_reach_sdr_flagged_low_quality() -> None:
    h = ready_harness()
    h.gateway.queue(AccountBrief, GOOD_BRIEF)
    h.gateway.queue(OutreachDraft, draft(), draft())
    h.gateway.queue(DeliveryJudgeVerdict, BAD_VERDICT, BAD_VERDICT)

    state = run_account(h.app, account(), "run-lowq")

    assert state["outcome"] is RunOutcome.LOW_QUALITY_DRAFT
    assert state["draft_attempts"] == 2
    assert h.repo.saved[0].delivery_verdict is not None
    assert h.repo.saved[0].delivery_verdict.critique == "Sentence 2 is not supported."


def _finished_parent() -> Harness:
    h = ready_harness()
    h.gateway.queue(AccountBrief, GOOD_BRIEF)
    h.gateway.queue(OutreachDraft, draft())
    h.gateway.queue(DeliveryJudgeVerdict, GOOD_VERDICT)
    run_account(h.app, account(), "parent")
    h.tools.calls.clear()
    h.gateway.calls.clear()
    return h


def test_draft_only_rerun_reuses_evidence_and_carries_feedback() -> None:
    h = _finished_parent()
    h.gateway.queue(OutreachDraft, draft())
    h.gateway.queue(DeliveryJudgeVerdict, GOOD_VERDICT)
    request = RerunRequest(
        parent_run_id="parent",
        scope=RerunScope.DRAFT_ONLY,
        reviewer_feedback="Shorter please",
        requested_by="sdr@quorum.us",
    )

    state = rerun(h.app, h.deps, request, "child")

    assert h.tools.calls == []
    assert [t for _, t, _ in h.gateway.calls] == ["OutreachDraft", "DeliveryJudgeVerdict"]
    assert "<reviewer_feedback>Shorter please</reviewer_feedback>" in h.gateway.calls[0][2][-1].content
    assert state["parent_run_id"] == "parent" and h.repo.saved[-1].run_id == "child"


def test_one_branch_rerun_only_researches_that_branch() -> None:
    h = _finished_parent()
    h.gateway.extraction[Branch.STAKEHOLDER] = [
        {"stakeholders": [stakeholder_json("stk-3", "Ari Cole", "Director, Public Policy")]}
    ]
    h.gateway.queue(ResearchJudgeVerdict, {})
    h.gateway.queue(AccountBrief, GOOD_BRIEF)
    h.gateway.queue(OutreachDraft, {**draft(), "stakeholder_id": "stk-3"})
    h.gateway.queue(DeliveryJudgeVerdict, GOOD_VERDICT)
    request = RerunRequest(
        parent_run_id="parent", scope=RerunScope.ONE_BRANCH, branch=Branch.STAKEHOLDER, requested_by="sdr"
    )

    state = rerun(h.app, h.deps, request, "child-branch")

    assert h.tools.calls == [(Branch.STAKEHOLDER, ResearchDepth.DEEP, 0)]
    assert {s.stakeholder_id for s in state["stakeholders"]} == {"stk-3"}
    assert state["outcome"] is RunOutcome.READY_FOR_REVIEW


def test_rerun_cap_blocks_fourth_rerun_of_the_day() -> None:
    h = _finished_parent()
    h.repo.reruns[account().salesforce_account_id] = 3
    request = RerunRequest(parent_run_id="parent", scope=RerunScope.DRAFT_ONLY, requested_by="sdr")
    with pytest.raises(RerunLimitExceededError):
        rerun(h.app, h.deps, request, "child-4")
    assert h.gateway.calls == []
