"""Graph nodes and routers. Nodes return partial state; routers only read state.

Every loop is bounded by a counter in state (`research_retries`,
`draft_attempts`), never by the model deciding when to stop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypeVar

from langgraph.types import Send

from . import prompts, rules
from .llm import ModelRouter, PermanentToolError, Task, TransientError, model_family
from .ports import KnowledgeBase, ResearchTools, ReviewChannel, RunRepository
from .schemas import (
    AccountBrief,
    Branch,
    BranchTask,
    Claim,
    DeliveryJudgeVerdict,
    EvidenceStatus,
    ExtractionResult,
    OutreachDraft,
    ResearchDepth,
    ResearchJudgeVerdict,
    ResearchResult,
    ResearchState,
    RunOutcome,
    Stakeholder,
)

MAX_RESEARCH_RETRIES = 1
MAX_DRAFT_ATTEMPTS = 2  # first draft + one rewrite with the judge's critique

# Reason: seniority decides who gets the email when several people are verified.
_SENIORITY = ("chief", "svp", "vp", "vice president", "head", "director", "manager")


@dataclass(frozen=True)
class Deps:
    router: ModelRouter
    tools: ResearchTools
    kb: KnowledgeBase
    repo: RunRepository
    review: ReviewChannel
    now: Callable[[], datetime] = lambda: datetime.now(UTC)


def verified_ids(claims: Sequence[Claim]) -> set[str]:
    return {c.claim_id for c in claims if c.status is EvidenceStatus.VERIFIED}


def pick_stakeholder(stakeholders: Sequence[Stakeholder]) -> Stakeholder | None:
    verified = [s for s in stakeholders if s.status is EvidenceStatus.VERIFIED]

    def rank(person: Stakeholder) -> tuple[int, str]:
        title = person.title.lower()
        level = next((i for i, word in enumerate(_SENIORITY) if word in title), len(_SENIORITY))
        return (level, person.stakeholder_id)

    return min(verified, key=rank, default=None)


def branch_sends(state: ResearchState, branches: Sequence[Branch], attempt: int) -> list[Send]:
    return [
        Send("research_branch", BranchTask(branch=b, account=state["account"], depth=state["depth"], attempt=attempt))
        for b in branches
    ]


_Evidence = TypeVar("_Evidence", Claim, Stakeholder)


def _reject(item: _Evidence, reason: str) -> _Evidence:
    return item.model_copy(
        update={"status": EvidenceStatus.REJECTED, "rejection_reasons": [*item.rejection_reasons, reason]}
    )


class ResearchNodes:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps

    def prioritize(self, state: ResearchState) -> ResearchState:
        """D1. Deterministic priority and research depth."""
        account = state["account"]
        priority = rules.execution_priority(account)
        depth = state.get("force_depth") or rules.choose_depth(account, priority)
        update: ResearchState = {"execution_priority": priority, "depth": depth, "research_retries": 0}
        if depth is ResearchDepth.SKIP:
            update["outcome"] = RunOutcome.SKIPPED
        return update

    def research_branch(self, state: BranchTask) -> ResearchState:
        """One branch, run in parallel with the others through `Send`."""
        branch, account, attempt = state["branch"], state["account"], state["attempt"]
        try:
            documents = self.deps.tools.gather(branch, account, state["depth"], attempt)
        except PermanentToolError as exc:
            return {"branch_errors": [f"{branch}/attempt {attempt}: {exc}"]}
        if not documents:
            return {"branch_errors": [f"{branch}/attempt {attempt}: no documents found"]}

        result = self.deps.router.call(
            Task.EXTRACTION, prompts.build_extraction_messages(account, documents), ExtractionResult
        )

        # Reason: the model copies provenance from the document tags. We check
        # the copy against what the tools really returned, and the excerpt
        # against the real text. A made-up quote or URL is rejected here.
        text_by_snapshot = {(d.snapshot_key, d.content_sha256): d.text for d in documents}
        claims: list[Claim] = []
        for extracted in result.output.claims:
            # Reason: the model numbers claims per call, so ids repeat across
            # branches and attempts. The prefix keeps them from overwriting.
            claim = extracted.model_copy(
                update={"branch": branch, "claim_id": f"{branch}:{attempt}:{extracted.claim_id}"}
            )
            source_text = text_by_snapshot.get((claim.snapshot_key, claim.content_sha256))
            if source_text is None:
                claim = _reject(claim, "provenance does not match any gathered document")
            elif claim.evidence_excerpt not in source_text:
                claim = _reject(claim, "evidence excerpt is not a verbatim quote of the source")
            claims.append(claim)

        return {
            "claims": claims,
            "stakeholders": result.output.stakeholders if branch is Branch.STAKEHOLDER else [],
            "models_used": {f"{Task.EXTRACTION}:{branch}": result.model},
            "cost_usd": result.cost_usd,
        }

    def validate(self, state: ResearchState) -> ResearchState:
        """D2. Schema, freshness, source class, provenance, duplicates."""
        now = self.deps.now()
        return {
            "claims": rules.validate_claims(state.get("claims", []), now),
            "stakeholders": rules.validate_stakeholders(state.get("stakeholders", []), now),
        }

    def research_judge(self, state: ResearchState) -> ResearchState:
        claims, people = state.get("claims", []), state.get("stakeholders", [])
        update: ResearchState = {"branches_to_retry": []}
        if verified_ids(claims) or pick_stakeholder(people) is not None:
            result = self.deps.router.call(
                Task.RESEARCH_JUDGE,
                prompts.build_research_judge_messages(state["account"], claims, people),
                ResearchJudgeVerdict,
            )
            unsupported = set(result.output.unsupported_claim_ids)
            wrong_company = set(result.output.wrong_company_stakeholder_ids)
            claims = [
                _reject(c, "research judge: excerpt does not support statement") if c.claim_id in unsupported else c
                for c in claims
            ]
            people = [
                _reject(p, "research judge: not at this company") if p.stakeholder_id in wrong_company else p
                for p in people
            ]
            update |= {
                "claims": claims,
                "stakeholders": people,
                "models_used": {Task.RESEARCH_JUDGE.value: result.model},
                "cost_usd": result.cost_usd,
            }

        missing = rules.incomplete_branches(claims, people)
        retries = state.get("research_retries", 0)
        if missing and retries < MAX_RESEARCH_RETRIES:
            update |= {"branches_to_retry": missing, "research_retries": retries + 1}
        return update

    def score(self, state: ResearchState) -> ResearchState:
        """D3. Two independent scores from verified evidence only."""
        claims, people = state.get("claims", []), state.get("stakeholders", [])
        confidence = rules.evidence_confidence(claims, people)
        update: ResearchState = {
            "priority_score": rules.account_priority(state["account"], claims),
            "confidence_score": confidence,
        }
        # Reason: low confidence is not a reason to discard a valuable account.
        # It goes to a human instead of to an email built on weak evidence.
        if confidence < rules.CONFIDENCE_FLOOR or pick_stakeholder(people) is None:
            update["outcome"] = RunOutcome.NEEDS_HUMAN_RESEARCH
        return update

    def synthesize(self, state: ResearchState) -> ResearchState:
        claims = state["claims"]
        known = verified_ids(claims)
        query = " ".join(c.statement for c in claims if c.claim_id in known)[:2000]
        passages = self.deps.kb.retrieve(query, top_k=5)
        result = self.deps.router.call(
            Task.SYNTHESIS, prompts.build_synthesis_messages(state["account"], claims, passages), AccountBrief
        )
        cited = [cid for cid in result.output.cited_claim_ids if cid in known]
        if not cited:
            raise TransientError("synthesis cited no verified claim")
        return {
            "brief": result.output.model_copy(update={"cited_claim_ids": cited}),
            "kb_passages": passages,
            "draft_attempts": 0,
            "models_used": {Task.SYNTHESIS.value: result.model},
            "cost_usd": result.cost_usd,
        }

    def personalize(self, state: ResearchState) -> ResearchState:
        stakeholder = pick_stakeholder(state.get("stakeholders", []))
        brief = state.get("brief")
        if stakeholder is None or brief is None:
            return {"outcome": RunOutcome.NEEDS_HUMAN_RESEARCH}
        messages = prompts.build_personalization_messages(
            account=state["account"],
            stakeholder=stakeholder,
            brief=brief,
            claims=state["claims"],
            kb_passages=state.get("kb_passages", []),
            reviewer_feedback=state.get("reviewer_feedback", ""),
            judge_verdict=state.get("delivery_verdict"),
            previous_draft=state.get("draft"),
        )
        result = self.deps.router.call(Task.PERSONALIZATION, messages, OutreachDraft)
        return {
            "draft": result.output,
            "draft_attempts": state.get("draft_attempts", 0) + 1,
            "models_used": {Task.PERSONALIZATION.value: result.model},
            "cost_usd": result.cost_usd,
        }

    def delivery_judge(self, state: ResearchState) -> ResearchState:
        draft = state["draft"]
        assert draft is not None  # Reason: route_after_personalize guarantees a draft here.
        bad_ids = (set(draft.cited_claim_ids) - verified_ids(state["claims"])) | (
            set(draft.cited_kb_ids) - {p.kb_id for p in state.get("kb_passages", [])}
        )

        update: ResearchState = {}
        if bad_ids:
            # Reason: a citation to an id that does not exist is a hard fail we
            # can prove without paying for a judge call.
            verdict = DeliveryJudgeVerdict(
                groundedness=1,
                approved_capabilities_only=not (set(draft.cited_kb_ids) & bad_ids),
                relevance=1,
                tone_and_length=1,
                critique=f"Cited ids not in the context: {sorted(bad_ids)}. Use only the given ids.",
            )
        else:
            generator = state["models_used"][Task.PERSONALIZATION.value]
            result = self.deps.router.call(
                Task.DELIVERY_JUDGE,
                prompts.build_delivery_judge_messages(draft, state["claims"], state.get("kb_passages", [])),
                DeliveryJudgeVerdict,
                avoid_family=model_family(generator),
            )
            verdict = result.output
            update |= {"models_used": {Task.DELIVERY_JUDGE.value: result.model}, "cost_usd": result.cost_usd}

        update["delivery_verdict"] = verdict
        if verdict.passed:
            update["outcome"] = RunOutcome.READY_FOR_REVIEW
        elif state.get("draft_attempts", 0) >= MAX_DRAFT_ATTEMPTS:
            update["outcome"] = RunOutcome.LOW_QUALITY_DRAFT
        return update

    def finalize(self, state: ResearchState) -> ResearchState:
        result = to_result(state, self.deps.now())
        # Reason: storage first, Slack second. Both calls are idempotent by
        # run_id, so a RetryPolicy retry after a Slack failure is safe.
        self.deps.repo.save_run_with_outbox(result)
        self.deps.review.post_review(result)
        return {}


def route_entry(state: ResearchState) -> list[Send] | str:
    entry = state.get("entry", "prioritize")
    if entry == "branch":
        branch = state.get("rerun_branch")
        if branch is None:
            raise ValueError("entry='branch' needs rerun_branch")
        return branch_sends(state, [branch], attempt=0)
    return entry


def route_after_prioritize(state: ResearchState) -> list[Send] | str:
    if state.get("outcome") is RunOutcome.SKIPPED:
        return "finalize"
    return branch_sends(state, list(Branch), attempt=0)


def route_after_research_judge(state: ResearchState) -> list[Send] | str:
    retry = state.get("branches_to_retry", [])
    if retry:
        return branch_sends(state, retry, attempt=state["research_retries"])
    return "score"


def route_after_score(state: ResearchState) -> Literal["synthesize", "finalize"]:
    return "finalize" if state.get("outcome") is RunOutcome.NEEDS_HUMAN_RESEARCH else "synthesize"


def route_after_personalize(state: ResearchState) -> Literal["delivery_judge", "finalize"]:
    return "finalize" if state.get("outcome") is RunOutcome.NEEDS_HUMAN_RESEARCH else "delivery_judge"


def route_after_delivery_judge(state: ResearchState) -> Literal["personalize", "finalize"]:
    return "finalize" if "outcome" in state else "personalize"


def to_result(state: ResearchState, now: datetime) -> ResearchResult:
    verified = sorted(
        (c for c in state.get("claims", []) if c.status is EvidenceStatus.VERIFIED),
        key=lambda c: (c.source_class, c.claim_id),
    )
    draft = state.get("draft")
    return ResearchResult(
        run_id=state["run_id"],
        parent_run_id=state.get("parent_run_id"),
        salesforce_account_id=state["account"].salesforce_account_id,
        outcome=state["outcome"],
        depth=state["depth"],
        priority_score=state.get("priority_score"),
        confidence_score=state.get("confidence_score"),
        brief=state.get("brief"),
        draft=draft,
        draft_stakeholder=next(
            (s for s in state.get("stakeholders", []) if draft and s.stakeholder_id == draft.stakeholder_id), None
        ),
        delivery_verdict=state.get("delivery_verdict"),
        top_source_urls=[c.source_url for c in verified[:5]],
        branch_errors=state.get("branch_errors", []),
        models_used=state.get("models_used", {}),
        cost_usd=round(state.get("cost_usd", 0.0), 6),
        completed_at=now,
    )
