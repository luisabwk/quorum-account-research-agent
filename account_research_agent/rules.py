"""Deterministic decision points D1, D2 and D3. No LLM touches these numbers.

Thresholds and weights are starting values. They are calibrated against the
golden dataset, and every change goes through a LangSmith experiment.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime, timedelta

from .schemas import (
    AccountInput,
    Branch,
    Claim,
    ClaimType,
    EvidenceStatus,
    ResearchDepth,
    SourceClass,
    Stakeholder,
)

# D1: research priority -> depth
SKIP_BELOW = 0.35
DEEP_FROM = 0.65

# D2: evidence policy
MIN_RELEVANCE = 0.6
FRESHNESS = {ClaimType.REGULATORY: timedelta(days=540), ClaimType.ADVOCACY: timedelta(days=120)}
# Evidence weaker than this can guide discovery, but never reaches outreach.
MAX_SOURCE_CLASS_FOR_OUTREACH = SourceClass.TRUSTED_PUBLICATION
MAX_SOURCE_CLASS_FOR_STAKEHOLDER = SourceClass.ENRICHMENT_PROVIDER
STAKEHOLDER_MAX_AGE = timedelta(days=90)

# Completeness minimums per branch, checked before the research judge.
MIN_VERIFIED = {Branch.REGULATORY: 1, Branch.STAKEHOLDER: 1, Branch.NEWS: 0}

# D3: below this confidence the account goes to a human, not to drafting.
CONFIDENCE_FLOOR = 0.3


def execution_priority(account: AccountInput) -> float:
    return round(
        0.4 * account.icp_fit + 0.3 * account.intent_score / 100 + 0.3 * account.engagement_score / 100,
        4,
    )


def choose_depth(account: AccountInput, priority: float) -> ResearchDepth:
    # Reason: customers belong to Account Management, not SDR outbound.
    if account.is_customer or priority < SKIP_BELOW:
        return ResearchDepth.SKIP
    if priority >= DEEP_FROM or account.open_opportunity:
        return ResearchDepth.DEEP
    return ResearchDepth.LIGHT


def statement_fingerprint(claim: Claim) -> str:
    normalized = " ".join(claim.statement.lower().split())
    return hashlib.sha256(f"{claim.content_sha256}|{normalized}".encode()).hexdigest()


def validate_claims(claims: Iterable[Claim], now: datetime) -> list[Claim]:
    """D2 for claims. Returns every claim with `status` and reasons set.

    Rejected claims are kept, not dropped: the SDR can see why evidence was
    excluded, and the golden dataset needs the negatives.
    """
    seen: set[str] = set()
    result: list[Claim] = []
    for claim in sorted(claims, key=lambda c: c.claim_id):
        # Reason: a judge rejection from an earlier pass is final.
        if claim.status is EvidenceStatus.REJECTED:
            result.append(claim)
            continue
        reasons: list[str] = []
        if claim.relevance is None:
            reasons.append("relevance not assessed")
        elif claim.relevance < MIN_RELEVANCE:
            reasons.append(f"relevance {claim.relevance:.2f} < {MIN_RELEVANCE}")
        if claim.published_at is None:
            reasons.append("no publication date")
        elif now - claim.published_at > FRESHNESS[claim.claim_type]:
            reasons.append(f"older than {FRESHNESS[claim.claim_type].days} days")
        if not claim.snapshot_key or not claim.content_sha256:
            reasons.append("missing provenance")
        if claim.evidence_excerpt.strip() == "":
            reasons.append("empty evidence excerpt")
        fingerprint = statement_fingerprint(claim)
        if fingerprint in seen:
            reasons.append("duplicate")
        seen.add(fingerprint)

        if reasons:
            status = EvidenceStatus.REJECTED
        elif claim.source_class > MAX_SOURCE_CLASS_FOR_OUTREACH:
            status = EvidenceStatus.NEEDS_REVIEW
            reasons.append(f"source class {claim.source_class.name} cannot reach outreach")
        else:
            status = EvidenceStatus.VERIFIED
        result.append(claim.model_copy(update={"status": status, "rejection_reasons": reasons}))
    return result


def validate_stakeholders(stakeholders: Iterable[Stakeholder], now: datetime) -> list[Stakeholder]:
    result: list[Stakeholder] = []
    for person in sorted(stakeholders, key=lambda s: s.stakeholder_id):
        if person.status is EvidenceStatus.REJECTED:
            result.append(person)
            continue
        reasons: list[str] = []
        if not person.company_domain_match:
            reasons.append("company domain does not match account")
        if now - person.last_verified_at > STAKEHOLDER_MAX_AGE:
            reasons.append(f"not verified in {STAKEHOLDER_MAX_AGE.days} days")
        if person.source_class > MAX_SOURCE_CLASS_FOR_STAKEHOLDER:
            reasons.append(f"source class {person.source_class.name} too weak")
        status = EvidenceStatus.REJECTED if reasons else EvidenceStatus.VERIFIED
        result.append(person.model_copy(update={"status": status, "rejection_reasons": reasons}))
    return result


def incomplete_branches(claims: list[Claim], stakeholders: list[Stakeholder]) -> list[Branch]:
    verified_claims = [c for c in claims if c.status is EvidenceStatus.VERIFIED]
    counts = {
        Branch.REGULATORY: sum(c.branch is Branch.REGULATORY for c in verified_claims),
        Branch.NEWS: sum(c.branch is Branch.NEWS for c in verified_claims),
        Branch.STAKEHOLDER: sum(s.status is EvidenceStatus.VERIFIED for s in stakeholders),
    }
    return [b for b in Branch if counts[b] < MIN_VERIFIED[b]]


def account_priority(account: AccountInput, claims: list[Claim]) -> float:
    """D3, commercial value. Verified regulatory exposure raises it, capped."""
    verified_regulatory = sum(
        c.status is EvidenceStatus.VERIFIED and c.claim_type is ClaimType.REGULATORY for c in claims
    )
    exposure = min(verified_regulatory, 5) / 5
    return round(0.7 * execution_priority(account) + 0.3 * exposure, 4)


def evidence_confidence(claims: list[Claim], stakeholders: list[Stakeholder]) -> float:
    """D3, trust in the evidence. Independent of commercial value.

    Returns 0.0 when nothing was verified. That is a real measurement ("no
    trustworthy evidence"), not a missing value.
    """
    verified = [c for c in claims if c.status is EvidenceStatus.VERIFIED]
    if not verified:
        return 0.0
    # Class 1 -> 1.0, class 3 -> 0.5
    source_quality = sum(1 - (int(c.source_class) - 1) * 0.25 for c in verified) / len(verified)
    coverage = sum(1 for b in Branch if b not in incomplete_branches(claims, stakeholders)) / len(Branch)
    volume = min(len(verified), 6) / 6
    return round(0.4 * source_quality + 0.4 * coverage + 0.2 * volume, 4)
