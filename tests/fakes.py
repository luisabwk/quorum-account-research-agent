"""In-memory fakes for every port. No network, no keys."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from account_research_agent.llm import Message, ProviderUnavailableError
from account_research_agent.schemas import (
    AccountInput,
    Branch,
    ClaimType,
    KBPassage,
    ResearchDepth,
    ResearchResult,
    SourceClass,
    SourceDocument,
)

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


def account(**overrides: object) -> AccountInput:
    data: dict[str, object] = {
        "salesforce_account_id": "001000000000001AAA",
        "name": "Harbor Freight Rail Co.",
        "domain": "harborfreightrail.example",
        "icp_fit": 0.9,
        "intent_score": 80,
        "engagement_score": 60,
        "open_opportunity": False,
        "is_customer": False,
    }
    data.update(overrides)
    return AccountInput.model_validate(data)


def document(
    branch: Branch, n: int, text: str, source_class: SourceClass = SourceClass.OFFICIAL_GOVERNMENT
) -> SourceDocument:
    return SourceDocument(
        branch=branch,
        url=f"https://source.example/{branch}/{n}",
        source_class=source_class,
        title=f"{branch} doc {n}",
        text=text,
        published_at=NOW - timedelta(days=30),
        retrieved_at=NOW,
        snapshot_key=f"s3://snapshots/{branch}/{n}",
        content_sha256=f"sha-{branch}-{n}",
    )


def claim_json(
    doc: SourceDocument, local_id: str, statement: str, excerpt: str, relevance: float = 0.9
) -> dict[str, object]:
    return {
        "claim_id": local_id,
        "branch": doc.branch.value,
        "claim_type": (ClaimType.ADVOCACY if doc.branch is Branch.NEWS else ClaimType.REGULATORY).value,
        "statement": statement,
        "evidence_excerpt": excerpt,
        "source_url": str(doc.url),
        "source_class": int(doc.source_class),
        "published_at": doc.published_at.isoformat() if doc.published_at else None,
        "retrieved_at": doc.retrieved_at.isoformat(),
        "snapshot_key": doc.snapshot_key,
        "content_sha256": doc.content_sha256,
        "relevance": relevance,
    }


def stakeholder_json(sid: str, name: str, title: str, domain_match: bool = True) -> dict[str, object]:
    return {
        "stakeholder_id": sid,
        "full_name": name,
        "title": title,
        "salesforce_contact_id": "003000000000001AAA",
        "source_url": "https://harborfreightrail.example/leadership",
        "source_class": int(SourceClass.OFFICIAL_COMPANY),
        "company_domain_match": domain_match,
        "last_verified_at": NOW.isoformat(),
    }


Response = str | Exception


@dataclass
class FakeGateway:
    """Answers by output schema title, in order. `down` models raise ProviderUnavailableError.

    Branches run in parallel, so extraction answers cannot rely on call order.
    They are queued per branch and matched by the snapshot keys in the prompt.
    """

    responses: dict[str, list[Response]] = field(default_factory=lambda: defaultdict(list))
    extraction: dict[Branch, list[Mapping[str, object]]] = field(default_factory=lambda: defaultdict(list))
    down: set[str] = field(default_factory=set)
    calls: list[tuple[str, str, list[Message]]] = field(default_factory=list)

    def queue(self, schema: type[BaseModel], *payloads: Mapping[str, object] | Exception) -> None:
        for p in payloads:
            self.responses[schema.__name__].append(p if isinstance(p, Exception) else json.dumps(p))

    def complete_json(
        self, *, model: str, messages: list[Message], json_schema: dict[str, object]
    ) -> tuple[str, float]:
        title = str(json_schema["title"])
        self.calls.append((model, title, messages))
        if model in self.down:
            raise ProviderUnavailableError(f"{model} down")
        if title == "ExtractionResult":
            branch = next(b for b in Branch if f"s3://snapshots/{b}/" in messages[-1].content)
            return json.dumps(self.extraction[branch].pop(0)), 0.001
        answer = self.responses[title].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer, 0.001

    def models_for(self, title: str) -> list[str]:
        return [m for m, t, _ in self.calls if t == title]


@dataclass
class FakeTools:
    documents: dict[Branch, list[SourceDocument]] = field(default_factory=dict)
    # Optional per-branch hook that can raise (TransientError, PermanentToolError).
    before: dict[Branch, Callable[[int], None]] = field(default_factory=dict)
    calls: list[tuple[Branch, ResearchDepth, int]] = field(default_factory=list)

    def gather(self, branch: Branch, account: AccountInput, depth: ResearchDepth, attempt: int) -> list[SourceDocument]:
        self.calls.append((branch, depth, attempt))
        if branch in self.before:
            self.before[branch](len([c for c in self.calls if c[0] is branch]))
        return self.documents.get(branch, [])


@dataclass
class FakeKB:
    passages: list[KBPassage] = field(
        default_factory=lambda: [
            KBPassage(
                kb_id="kb-state-tracking", title="State tracking", text="Track bills across all 50 states.", version=4
            )
        ]
    )

    def retrieve(self, query: str, top_k: int = 5) -> list[KBPassage]:
        return self.passages[:top_k]


@dataclass
class FakeRepo:
    saved: list[ResearchResult] = field(default_factory=list)
    reruns: dict[str, int] = field(default_factory=dict)

    def reserve_rerun(self, salesforce_account_id: str, daily_limit: int) -> bool:
        used = self.reruns.get(salesforce_account_id, 0)
        if used >= daily_limit:
            return False
        self.reruns[salesforce_account_id] = used + 1
        return True

    def save_run_with_outbox(self, result: ResearchResult) -> None:
        self.saved.append(result)


@dataclass
class FakeReview:
    posted: list[ResearchResult] = field(default_factory=list)

    def post_review(self, result: ResearchResult) -> None:
        self.posted.append(result)
