"""End-to-end demo with mock data: one account, one SDR rerun, one Salesforce sync.

Runs the real orchestrator, rules, prompts and outbox worker. Only the edges are
mocked: the LLM gateway answers from `mock_data.py`, the research tools return
fictional documents, and Salesforce is an in-memory fake.

    python examples/run_mock_demo.py
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import mock_data as mock
from langgraph.checkpoint.memory import InMemorySaver

from account_research_agent.llm import Message, ModelRouter, ProviderUnavailableError
from account_research_agent.nodes import Deps
from account_research_agent.orchestrator import _config, checkpoint_serializer, compile_app, rerun
from account_research_agent.salesforce_sync import (
    CompositeRequest,
    OutboxEntry,
    OutboxWorker,
    SubResponse,
)
from account_research_agent.schemas import (
    AccountInput,
    Branch,
    EvidenceStatus,
    KBPassage,
    RerunRequest,
    RerunScope,
    ResearchDepth,
    ResearchResult,
    ResearchState,
    SourceDocument,
)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


@dataclass
class MockGateway:
    """Answers each call from mock data and prices it with real per-token prices."""

    down: dict[str, set[str]] = field(default_factory=dict)  # schema title -> models that are down
    drafts: list[str] = field(default_factory=list)
    calls: list[tuple[str, str, int, int, float]] = field(default_factory=list)

    def complete_json(
        self, *, model: str, messages: list[Message], json_schema: dict[str, object]
    ) -> tuple[str, float]:
        title = str(json_schema["title"])
        if model in self.down.get(title, set()):
            print(f"    [llm] {title:<22} {model:<30} -> provider unavailable, trying fallback")
            raise ProviderUnavailableError(f"{model}: HTTP 503")
        answer = self._answer(title, messages)
        # Reason: ~4 characters per token is a rough estimate. Real counts come from provider usage fields.
        tokens_in = sum(len(m.content) for m in messages) // 4
        tokens_out = len(answer) // 4
        price_in, price_out = mock.PRICES[model]
        cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
        self.calls.append((title, model, tokens_in, tokens_out, cost))
        print(f"    [llm] {title:<22} {model:<30} {tokens_in:>6} in / {tokens_out:>4} out  ${cost:.5f}")
        return answer, cost

    def _answer(self, title: str, messages: list[Message]) -> str:
        if title == "ExtractionResult":
            branch = next(
                b for b in Branch if f"/{mock.DOCUMENTS[b][0].snapshot_key.rsplit('/', 1)[1]}" in messages[-1].content
            )
            payload = {k: [item.model_dump(mode="json") for item in v] for k, v in mock.EXTRACTION[branch].items()}
            return json.dumps(payload)
        if title == "ResearchJudgeVerdict":
            return mock.RESEARCH_VERDICT.model_dump_json()
        if title == "AccountBrief":
            return mock.BRIEF.model_dump_json()
        if title == "OutreachDraft":
            return self.drafts.pop(0)
        if title == "DeliveryJudgeVerdict":
            return mock.PASS.model_dump_json()
        raise KeyError(title)


class MockTools:
    def gather(self, branch: Branch, account: AccountInput, depth: ResearchDepth, attempt: int) -> list[SourceDocument]:
        docs = mock.DOCUMENTS[branch]
        print(f"    [tool] {branch:<11} depth={depth} attempt={attempt} -> {len(docs)} documents snapshotted")
        return docs


class MockKB:
    def retrieve(self, query: str, top_k: int = 5) -> list[KBPassage]:
        return mock.KB[:top_k]


@dataclass
class MockRepo:
    outbox: list[OutboxEntry] = field(default_factory=list)
    reruns: Counter[str] = field(default_factory=Counter)

    def reserve_rerun(self, salesforce_account_id: str, daily_limit: int) -> bool:
        if self.reruns[salesforce_account_id] >= daily_limit:
            return False
        self.reruns[salesforce_account_id] += 1
        return True

    def save_run_with_outbox(self, result: ResearchResult) -> None:
        self.outbox.append(
            OutboxEntry(
                outbox_id=f"ob-{result.run_id}", payload=result, status="pending", next_attempt_at=result.completed_at
            )
        )
        print(f"    [mongo] research_runs + salesforce_outbox written for {result.run_id} (one transaction)")


class MockSlack:
    def post_review(self, result: ResearchResult) -> None:
        print(f"    [slack] review card posted: outcome={result.outcome}, sources={len(result.top_source_urls)}")


def describe(node: str, update: ResearchState) -> None:
    if node == "prioritize":
        print(f"  prioritize (D1): execution priority {update['execution_priority']}, depth {update['depth']}")
    elif node == "research_branch":
        errors = update.get("branch_errors", [])
        claims, people = update.get("claims", []), update.get("stakeholders", [])
        rejected = [c for c in claims if c.status is EvidenceStatus.REJECTED]
        print(
            f"  research_branch: {len(claims)} claims, {len(people)} stakeholders{', errors: ' + str(errors) if errors else ''}"
        )
        for c in rejected:
            print(f"    rejected {c.claim_id}: {'; '.join(c.rejection_reasons)}")
    elif node == "validate":
        print("  validate (D2):")
        for c in update["claims"]:
            reasons = f"  ({'; '.join(c.rejection_reasons)})" if c.rejection_reasons else ""
            print(f"    claim {c.claim_id:<17} {c.status:<13}{reasons}")
        for s in update["stakeholders"]:
            reasons = f"  ({'; '.join(s.rejection_reasons)})" if s.rejection_reasons else ""
            print(f"    person {s.full_name:<16} {s.status:<13}{reasons}")
    elif node == "research_judge":
        judged = [c for c in update.get("claims", []) if any("research judge" in r for r in c.rejection_reasons)]
        for c in judged:
            print(f"  research_judge: rejected {c.claim_id}: {c.rejection_reasons[-1]}")
        print(f"  research_judge: branches to re-research: {update.get('branches_to_retry') or 'none'}")
    elif node == "score":
        print(
            f"  score (D3): account priority {update['priority_score']}, evidence confidence {update['confidence_score']}"
            f"{', outcome ' + update['outcome'] if 'outcome' in update else ''}"
        )
    elif node == "synthesize":
        brief = update["brief"]
        assert brief is not None
        print(f"  synthesize: angle = {brief.outreach_angle!r}, cites {brief.cited_claim_ids}")
    elif node == "personalize":
        draft = update.get("draft")
        if draft:
            print(
                f"  personalize (attempt {update['draft_attempts']}): {draft.subject!r}, cites {draft.cited_claim_ids}"
            )
    elif node == "delivery_judge":
        v = update["delivery_verdict"]
        assert v is not None
        print(
            f"  delivery_judge: passed={v.passed} groundedness={v.groundedness} relevance={v.relevance} "
            f"tone={v.tone_and_length} -> {update.get('outcome', 'rewrite')}"
        )
        print(f"    critique: {v.critique}")
    elif node == "finalize":
        print("  finalize: done")


@dataclass
class FlakySalesforce:
    """Row lock on the first call, success after."""

    requests: list[CompositeRequest] = field(default_factory=list)

    def composite(self, request: CompositeRequest) -> list[SubResponse]:
        self.requests.append(request)
        if len(self.requests) == 1:
            return [
                SubResponse(
                    referenceId="account",
                    httpStatusCode=400,
                    body=[
                        {
                            "errorCode": "UNABLE_TO_LOCK_ROW",
                            "message": "unable to obtain exclusive access to this record",
                        }
                    ],
                ),
                SubResponse(
                    referenceId="draft",
                    httpStatusCode=400,
                    body=[{"errorCode": "PROCESSING_HALTED", "message": "halted"}],
                ),
            ]
        return [
            SubResponse(referenceId="account", httpStatusCode=204),
            SubResponse(referenceId="draft", httpStatusCode=201),
        ]

    def refresh_token(self) -> None:
        pass


@dataclass
class OutboxStore:
    entries: list[OutboxEntry]
    synced_at: dict[str, datetime] = field(default_factory=dict)

    def claim_due(self, now: datetime, lease: timedelta, limit: int) -> list[OutboxEntry]:
        return [e for e in self.entries if e.status == "pending" and e.next_attempt_at <= now][:limit]

    def latest_synced_completed_at(self, salesforce_account_id: str) -> datetime | None:
        return self.synced_at.get(salesforce_account_id)

    def _get(self, outbox_id: str) -> OutboxEntry:
        return next(e for e in self.entries if e.outbox_id == outbox_id)

    def mark_synced(self, outbox_id: str, warnings: list[str]) -> None:
        e = self._get(outbox_id)
        e.status = "synced"
        self.synced_at[e.payload.salesforce_account_id] = e.payload.completed_at
        print(f"    [outbox] {outbox_id}: synced")

    def reschedule(self, outbox_id: str, attempts: int, next_attempt_at: datetime, error: str) -> None:
        e = self._get(outbox_id)
        e.attempts, e.next_attempt_at, e.last_error = attempts, next_attempt_at, error
        print(f"    [outbox] {outbox_id}: transient error, retry {attempts} at {next_attempt_at:%H:%M:%S} UTC: {error}")

    def dead_letter(self, outbox_id: str, error: str) -> None:
        self._get(outbox_id).status = "dead_letter"
        print(f"    [outbox] {outbox_id}: dead letter: {error}")

    def mark_superseded(self, outbox_id: str) -> None:
        self._get(outbox_id).status = "superseded"
        print(f"    [outbox] {outbox_id}: superseded by a newer synced run")


class PrintAlerts:
    def alert(self, text: str) -> None:
        print(f"    [alert] {text}")


def main() -> None:
    gateway = MockGateway(
        down={"OutreachDraft": {"google/gemini-3.8-flash"}},
        drafts=[
            mock.DRAFT_WITH_BAD_CITATION.model_dump_json(),
            mock.GOOD_DRAFT.model_dump_json(),
            mock.SHORTER_DRAFT.model_dump_json(),
        ],
    )
    repo = MockRepo()
    deps = Deps(
        router=ModelRouter(gateway), tools=MockTools(), kb=MockKB(), repo=repo, review=MockSlack(), now=lambda: mock.NOW
    )
    app = compile_app(deps, InMemorySaver(serde=checkpoint_serializer()))

    section(f"1. Research run for {mock.ACCOUNT.name}")
    start: ResearchState = {"run_id": "run-001", "parent_run_id": None, "account": mock.ACCOUNT, "entry": "prioritize"}
    for chunk in app.stream(start, _config("run-001"), stream_mode="updates"):
        for node, update in chunk.items():
            describe(node, update or {})

    state = app.get_state(_config("run-001")).values
    draft = state["draft"]
    section("2. Result the SDR reviews")
    print(f"outcome: {state['outcome']}\nmodels: {json.dumps(state['models_used'], indent=2)}\n")
    print(f"To: {next(s.full_name for s in state['stakeholders'] if s.stakeholder_id == draft.stakeholder_id)}")
    print(
        f"Subject: {draft.subject}\n\n{draft.body}\n\ncited claims: {draft.cited_claim_ids}\ncited KB: {draft.cited_kb_ids}"
    )

    section("3. SDR asks for a draft-only rerun from Slack")
    request = RerunRequest(
        parent_run_id="run-001",
        scope=RerunScope.DRAFT_ONLY,
        requested_by="sdr@quorum.example",
        reviewer_feedback="Shorter, and lead with the Ohio and Iowa testimony.",
    )
    before = len(gateway.calls)
    child = rerun(app, deps, request, "run-002")
    print(f"  new run run-002, parent {child['parent_run_id']}, outcome {child['outcome']}")
    print(f"  tool calls in rerun: 0; LLM calls in rerun: {len(gateway.calls) - before} (draft + judge only)")
    print(f"\nSubject: {child['draft'].subject}\n\n{child['draft'].body}")

    section("4. Salesforce sync through the outbox")
    store = OutboxStore(repo.outbox)
    salesforce = FlakySalesforce()
    worker = OutboxWorker(store, salesforce, PrintAlerts(), jitter=lambda: 0.0)
    print("  batch 1:")
    worker.process_batch(mock.NOW)
    print("  batch 2 (after backoff): run-002 already synced, so the older run-001 must not overwrite it")
    worker.process_batch(mock.NOW + timedelta(minutes=1))
    print("\n  last Composite API request sent:")
    print("  " + json.dumps(salesforce.requests[-1], indent=2, default=str).replace("\n", "\n  "))

    section("5. Mock cost of run-001 (token estimate x OpenRouter prices)")
    run1 = gateway.calls[:before]
    by_step: dict[str, float] = {}
    for title, _, _, _, cost in run1:
        by_step[title] = by_step.get(title, 0.0) + cost
    for title, cost in by_step.items():
        print(f"  {title:<22} ${cost:.5f}")
    print(f"  {'total LLM':<22} ${sum(by_step.values()):.5f}  ({len(run1)} calls)")
    print("  Excludes search/enrichment APIs. Mock prompts are shorter than real ones, so real cost is higher.")


if __name__ == "__main__":
    main()
