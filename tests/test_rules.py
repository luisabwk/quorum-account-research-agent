from __future__ import annotations

from datetime import timedelta

import pytest

from account_research_agent import rules
from account_research_agent.schemas import Branch, Claim, EvidenceStatus, ResearchDepth, SourceClass, Stakeholder

from .fakes import NOW, account, claim_json, document, stakeholder_json


def make_claim(
    cid: str,
    *,
    days_old: int = 30,
    relevance: float = 0.9,
    source_class: SourceClass = SourceClass.OFFICIAL_GOVERNMENT,
    statement: str = "Lobbied on H.R. 9001 in 2026.",
    branch: Branch = Branch.REGULATORY,
) -> Claim:
    doc = document(branch, 1, "text", source_class)
    data = claim_json(doc, cid, statement, "text", relevance)
    data["published_at"] = (NOW - timedelta(days=days_old)).isoformat()
    return Claim.model_validate(data)


@pytest.mark.parametrize(
    ("overrides", "depth"),
    [
        ({}, ResearchDepth.DEEP),
        ({"icp_fit": 0.5, "intent_score": 40, "engagement_score": 40}, ResearchDepth.LIGHT),
        ({"icp_fit": 0.1, "intent_score": 10, "engagement_score": 0}, ResearchDepth.SKIP),
        ({"icp_fit": 0.1, "intent_score": 10, "engagement_score": 0, "open_opportunity": True}, ResearchDepth.SKIP),
        ({"is_customer": True}, ResearchDepth.SKIP),
    ],
)
def test_choose_depth(overrides: dict[str, object], depth: ResearchDepth) -> None:
    acc = account(**overrides)
    assert rules.choose_depth(acc, rules.execution_priority(acc)) is depth


def test_validate_claims_statuses_and_reasons() -> None:
    result = {
        c.claim_id: c
        for c in rules.validate_claims(
            [
                make_claim("ok"),
                make_claim("stale", days_old=600),
                make_claim("irrelevant", relevance=0.2, statement="Sponsors a local marathon every year."),
                make_claim(
                    "weak", source_class=SourceClass.ENRICHMENT_PROVIDER, statement="Enrichment says they lobby a lot."
                ),
                make_claim("zdup"),  # same source + statement as "ok"
            ],
            NOW,
        )
    }
    assert result["ok"].status is EvidenceStatus.VERIFIED
    assert (
        result["stale"].status is EvidenceStatus.REJECTED and "older than 540 days" in result["stale"].rejection_reasons
    )
    assert result["irrelevant"].status is EvidenceStatus.REJECTED
    assert result["weak"].status is EvidenceStatus.NEEDS_REVIEW
    assert result["zdup"].rejection_reasons == ["duplicate"]


def test_judge_rejection_survives_revalidation() -> None:
    judged = make_claim("x").model_copy(
        update={"status": EvidenceStatus.REJECTED, "rejection_reasons": ["research judge"]}
    )
    assert rules.validate_claims([judged], NOW)[0].rejection_reasons == ["research judge"]


def test_stakeholder_from_other_company_is_rejected() -> None:
    other = Stakeholder.model_validate(stakeholder_json("s1", "Pat Doe", "VP", domain_match=False))
    assert rules.validate_stakeholders([other], NOW)[0].status is EvidenceStatus.REJECTED


def test_confidence_is_zero_without_verified_evidence_and_rises_with_coverage() -> None:
    assert rules.evidence_confidence([], []) == 0.0
    claims = rules.validate_claims([make_claim("a")], NOW)
    people = rules.validate_stakeholders([Stakeholder.model_validate(stakeholder_json("s1", "Pat Doe", "VP"))], NOW)
    assert rules.evidence_confidence(claims, []) < rules.evidence_confidence(claims, people)
    assert rules.incomplete_branches(claims, people) == []
