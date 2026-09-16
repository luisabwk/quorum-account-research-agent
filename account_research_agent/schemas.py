"""Typed contracts shared by every node.

Raw tool output never reaches a prompt. Each research branch converts what it
found into `Claim` / `Stakeholder` objects, and only those travel downstream.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Annotated, Literal, TypedDict, TypeVar

from pydantic import BaseModel, Field, HttpUrl, PlainSerializer

# Reason: pydantic's Url object is not msgpack-serializable, so the LangGraph
# checkpointer cannot store it. Validate as a URL, dump as a plain string.
Url = Annotated[HttpUrl, PlainSerializer(str, return_type=str)]


class SourceClass(IntEnum):
    # Reason: lower number = more trusted. The policy engine compares with `<=`.
    OFFICIAL_GOVERNMENT = 1
    OFFICIAL_COMPANY = 2
    TRUSTED_PUBLICATION = 3
    ENRICHMENT_PROVIDER = 4
    SECONDARY_UNKNOWN = 5


class Branch(StrEnum):
    REGULATORY = "regulatory"
    STAKEHOLDER = "stakeholder"
    NEWS = "news"


class ClaimType(StrEnum):
    REGULATORY = "regulatory"
    ADVOCACY = "advocacy"


class EvidenceStatus(StrEnum):
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class ResearchDepth(StrEnum):
    SKIP = "skip"
    LIGHT = "light"
    DEEP = "deep"


class RerunScope(StrEnum):
    DRAFT_ONLY = "draft_only"
    ONE_BRANCH = "one_branch"
    FULL_DEEPER = "full_deeper"


class RunOutcome(StrEnum):
    SKIPPED = "skipped"
    READY_FOR_REVIEW = "ready_for_review"
    LOW_QUALITY_DRAFT = "low_quality_draft"
    NEEDS_HUMAN_RESEARCH = "needs_human_research"


class SourceDocument(BaseModel):
    """What a tool returns after normalization, before any LLM reads it."""

    branch: Branch
    url: Url
    source_class: SourceClass
    title: str
    text: str
    published_at: datetime | None
    retrieved_at: datetime
    snapshot_key: str  # S3 key written by `store_snapshot`
    content_sha256: str


class Claim(BaseModel):
    claim_id: str
    branch: Branch
    claim_type: ClaimType
    statement: str = Field(min_length=10)
    evidence_excerpt: str
    source_url: Url
    source_class: SourceClass
    published_at: datetime | None
    retrieved_at: datetime
    snapshot_key: str
    content_sha256: str
    # Filled by the LLM relevance step (0-1). None = not assessed yet.
    relevance: float | None = Field(default=None, ge=0, le=1)
    status: EvidenceStatus | None = None
    rejection_reasons: list[str] = Field(default_factory=list)


class Stakeholder(BaseModel):
    stakeholder_id: str
    full_name: str
    title: str
    email: str | None = None
    salesforce_contact_id: str | None = None
    source_url: Url
    source_class: SourceClass
    company_domain_match: bool
    last_verified_at: datetime
    status: EvidenceStatus | None = None
    rejection_reasons: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Structured output of the extraction + relevance model for one branch."""

    claims: list[Claim] = Field(default_factory=list)
    stakeholders: list[Stakeholder] = Field(default_factory=list)


class AccountInput(BaseModel):
    """Snapshot pulled from Salesforce + 6sense + Qualified before research."""

    salesforce_account_id: str
    name: str
    domain: str
    icp_fit: float = Field(ge=0, le=1)
    intent_score: float = Field(ge=0, le=100)  # 6sense
    engagement_score: float = Field(ge=0, le=100)  # Qualified
    open_opportunity: bool
    is_customer: bool


class ResearchJudgeVerdict(BaseModel):
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    wrong_company_stakeholder_ids: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    notes: str = ""


class AccountBrief(BaseModel):
    summary: str
    commercial_hypothesis: str
    outreach_angle: str
    # Every sentence in the brief must be traceable to verified claims.
    cited_claim_ids: list[str] = Field(min_length=1)


class KBPassage(BaseModel):
    kb_id: str
    title: str
    text: str
    version: int


class OutreachDraft(BaseModel):
    stakeholder_id: str
    subject: str = Field(max_length=80)
    body: str = Field(max_length=1200)
    cited_claim_ids: list[str] = Field(min_length=1)
    cited_kb_ids: list[str] = Field(default_factory=list)


class DeliveryJudgeVerdict(BaseModel):
    groundedness: int = Field(ge=1, le=5)
    approved_capabilities_only: bool
    relevance: int = Field(ge=1, le=5)
    tone_and_length: int = Field(ge=1, le=5)
    critique: str

    @property
    def passed(self) -> bool:
        # Reason: groundedness and approved-KB are hard gates; style is not.
        return (
            self.groundedness >= 4
            and self.approved_capabilities_only
            and self.relevance >= 3
            and self.tone_and_length >= 3
        )


class RerunRequest(BaseModel):
    parent_run_id: str
    scope: RerunScope
    branch: Branch | None = None  # required when scope == ONE_BRANCH
    reviewer_feedback: str = Field(default="", max_length=1000)
    requested_by: str


_Item = TypeVar("_Item", bound=BaseModel)


def merge_by_id(key: str) -> Callable[[list[_Item] | None, list[_Item] | None], list[_Item]]:
    """Reducer factory: later versions of an item replace earlier ones by id."""

    def _merge(left: list[_Item] | None, right: list[_Item] | None) -> list[_Item]:
        merged = {getattr(item, key): item for item in left or []}
        for item in right or []:
            merged[getattr(item, key)] = item
        # Reason: sorted keeps output order stable between runs (reproducible md5).
        return [merged[k] for k in sorted(merged)]

    return _merge


class ResearchState(TypedDict, total=False):
    run_id: str
    parent_run_id: str | None
    # Where a run enters the graph. Reruns enter downstream of prioritize.
    entry: Literal["prioritize", "branch", "personalize"]
    rerun_branch: Branch | None
    force_depth: ResearchDepth | None
    account: AccountInput
    depth: ResearchDepth
    execution_priority: float
    reviewer_feedback: str
    # Reason: branches run in parallel and write to the same keys. The reducer
    # merges by id, so validation can also rewrite a claim's status in place.
    claims: Annotated[list[Claim], merge_by_id("claim_id")]
    stakeholders: Annotated[list[Stakeholder], merge_by_id("stakeholder_id")]
    branch_errors: Annotated[list[str], operator.add]
    research_retries: int
    branches_to_retry: list[Branch]
    priority_score: float
    confidence_score: float
    brief: AccountBrief | None
    kb_passages: list[KBPassage]
    draft: OutreachDraft | None
    delivery_verdict: DeliveryJudgeVerdict | None
    draft_attempts: int
    # task name -> model id that actually answered (primary or fallback)
    models_used: Annotated[dict[str, str], operator.or_]
    cost_usd: Annotated[float, operator.add]
    outcome: RunOutcome


class ResearchResult(BaseModel):
    """What a finished run hands to storage and to the Salesforce outbox."""

    run_id: str
    parent_run_id: str | None
    salesforce_account_id: str
    outcome: RunOutcome
    depth: ResearchDepth
    priority_score: float | None
    confidence_score: float | None
    brief: AccountBrief | None
    draft: OutreachDraft | None
    draft_stakeholder: Stakeholder | None
    delivery_verdict: DeliveryJudgeVerdict | None
    top_source_urls: list[Url]
    branch_errors: list[str]
    models_used: dict[str, str]
    cost_usd: float
    completed_at: datetime


class BranchTask(TypedDict):
    """Payload sent to one research branch through `Send`."""

    branch: Branch
    account: AccountInput
    depth: ResearchDepth
    attempt: int


OutboxStatus = Literal["pending", "in_progress", "synced", "dead_letter", "superseded"]
