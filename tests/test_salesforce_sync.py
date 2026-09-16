from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest

from account_research_agent.salesforce_sync import (
    ACCOUNT_FIELDS,
    MAX_ATTEMPTS,
    CompositeRequest,
    FieldSpec,
    Kind,
    MappingError,
    OutboxEntry,
    OutboxWorker,
    SalesforceAuthExpiredError,
    SalesforceTransientError,
    SubResponse,
    build_composite,
    transform,
)
from account_research_agent.schemas import (
    AccountBrief,
    DeliveryJudgeVerdict,
    OutreachDraft,
    ResearchDepth,
    ResearchResult,
    RunOutcome,
    Stakeholder,
)

from .fakes import NOW, stakeholder_json


def result(**overrides: object) -> ResearchResult:
    data: dict[str, object] = {
        "run_id": "run-1",
        "parent_run_id": None,
        "salesforce_account_id": "001000000000001AAA",
        "outcome": RunOutcome.READY_FOR_REVIEW,
        "depth": ResearchDepth.DEEP,
        "priority_score": 0.81234,
        "confidence_score": 0.7,
        "brief": AccountBrief(summary="Summary", commercial_hypothesis="H", outreach_angle="A", cited_claim_ids=["c1"]),
        "draft": OutreachDraft(stakeholder_id="stk-1", subject="Subject", body="Body", cited_claim_ids=["c1"]),
        "draft_stakeholder": Stakeholder.model_validate(stakeholder_json("stk-1", "Dana Okafor", "VP")),
        "delivery_verdict": DeliveryJudgeVerdict(
            groundedness=5, approved_capabilities_only=True, relevance=4, tone_and_length=4, critique="ok"
        ),
        "top_source_urls": ["https://lda.gov/filing/1"],
        "branch_errors": [],
        "models_used": {},
        "cost_usd": 0.05,
        "completed_at": NOW,
    }
    data.update(overrides)
    return ResearchResult.model_validate(data)


def entry(attempts: int = 0, **overrides: object) -> OutboxEntry:
    return OutboxEntry(
        outbox_id="ob-1", payload=result(**overrides), status="in_progress", attempts=attempts, next_attempt_at=NOW
    )


OK = [SubResponse(referenceId="account", httpStatusCode=204), SubResponse(referenceId="draft", httpStatusCode=201)]


def failed(code: str, status: int = 400) -> list[SubResponse]:
    return [
        SubResponse(referenceId="account", httpStatusCode=status, body=[{"errorCode": code, "message": "boom"}]),
        SubResponse(
            referenceId="draft", httpStatusCode=400, body=[{"errorCode": "PROCESSING_HALTED", "message": "halted"}]
        ),
    ]


@dataclass
class FakeStore:
    entries: list[OutboxEntry]
    newest_synced: datetime | None = None
    events: list[tuple[str, object]] = field(default_factory=list)

    def claim_due(self, now: datetime, lease: timedelta, limit: int) -> list[OutboxEntry]:
        return self.entries[:limit]

    def latest_synced_completed_at(self, salesforce_account_id: str) -> datetime | None:
        return self.newest_synced

    def mark_synced(self, outbox_id: str, warnings: list[str]) -> None:
        self.events.append(("synced", warnings))

    def reschedule(self, outbox_id: str, attempts: int, next_attempt_at: datetime, error: str) -> None:
        self.events.append(("reschedule", (attempts, next_attempt_at)))

    def dead_letter(self, outbox_id: str, error: str) -> None:
        self.events.append(("dead_letter", error))

    def mark_superseded(self, outbox_id: str) -> None:
        self.events.append(("superseded", outbox_id))


@dataclass
class FakeClient:
    replies: list[list[SubResponse] | Exception]
    refreshed: int = 0
    requests: list[CompositeRequest] = field(default_factory=list)

    def composite(self, request: CompositeRequest) -> list[SubResponse]:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def refresh_token(self) -> None:
        self.refreshed += 1


@dataclass
class FakeAlerts:
    sent: list[str] = field(default_factory=list)

    def alert(self, text: str) -> None:
        self.sent.append(text)


def run(store: FakeStore, client: FakeClient) -> tuple[FakeAlerts, object]:
    alerts = FakeAlerts()
    report = OutboxWorker(store, client, alerts, jitter=lambda: 0.0).process_batch(NOW)
    return alerts, report


def test_account_and_draft_mapping() -> None:
    request, warnings = build_composite(result())
    account_req, draft_req = request["compositeRequest"]
    assert request["allOrNone"] is True and warnings == []
    assert account_req["body"]["AI_Research_Status__c"] == "Ready for Review"
    assert account_req["body"]["AI_Priority_Score__c"] == 0.81
    assert account_req["body"]["AI_Last_Researched_At__c"] == "2026-09-16T12:00:00+00:00"
    assert draft_req["url"].endswith("/AI_Outreach_Draft__c/Run_Id__c/run-1")
    assert draft_req["body"]["Contact__c"] == "003000000000001AAA"


def test_skipped_run_clears_scores_and_sends_no_draft() -> None:
    request, _ = build_composite(
        result(outcome=RunOutcome.SKIPPED, priority_score=None, confidence_score=None, brief=None, draft=None)
    )
    (account_req,) = request["compositeRequest"]
    assert account_req["body"]["AI_Priority_Score__c"] is None
    assert account_req["body"]["AI_Research_Summary__c"] is None


def test_long_text_is_truncated_with_a_visible_warning() -> None:
    spec = FieldSpec("X__c", Kind.TEXT, lambda r: None, max_length=20)
    value, warning = transform(spec, "a" * 50)
    assert isinstance(value, str) and len(value) == 20 and value.endswith("[truncated]")
    assert warning == "X__c: truncated 50 -> 20"


@pytest.mark.parametrize(
    ("spec", "value"),
    [
        (ACCOUNT_FIELDS[0], "archived"),  # unknown picklist value
        (FieldSpec("T__c", Kind.DATETIME, lambda r: None), datetime(2026, 9, 16)),  # naive datetime
        (FieldSpec("N__c", Kind.NUMBER, lambda r: None), "0.8"),
    ],
)
def test_bad_values_raise_mapping_error(spec: FieldSpec, value: object) -> None:
    with pytest.raises(MappingError):
        transform(spec, value)


def test_success_marks_synced() -> None:
    store = FakeStore([entry()])
    alerts, _ = run(store, FakeClient([OK]))
    assert store.events == [("synced", [])] and alerts.sent == []


def test_row_lock_is_rescheduled_with_backoff() -> None:
    store = FakeStore([entry(attempts=2)])
    run(store, FakeClient([failed("UNABLE_TO_LOCK_ROW")]))
    assert store.events == [("reschedule", (3, NOW + timedelta(minutes=2)))]


def test_transient_error_on_last_attempt_goes_to_dead_letter() -> None:
    store = FakeStore([entry(attempts=MAX_ATTEMPTS - 1)])
    alerts, _ = run(store, FakeClient([SalesforceTransientError("503")]))
    assert store.events[0][0] == "dead_letter" and len(alerts.sent) == 1


def test_validation_rule_is_permanent_and_alerts_with_real_error() -> None:
    store = FakeStore([entry()])
    alerts, _ = run(store, FakeClient([failed("FIELD_CUSTOM_VALIDATION_EXCEPTION")]))
    kind, error = store.events[0]
    assert kind == "dead_letter"
    assert "FIELD_CUSTOM_VALIDATION_EXCEPTION" in str(error) and "PROCESSING_HALTED" not in str(error)
    assert "001000000000001AAA" in alerts.sent[0]


def test_expired_token_is_refreshed_once_then_succeeds() -> None:
    store = FakeStore([entry()])
    client = FakeClient([SalesforceAuthExpiredError("401"), OK])
    run(store, client)
    assert client.refreshed == 1 and store.events == [("synced", [])]


def test_auth_failing_after_refresh_is_permanent() -> None:
    store = FakeStore([entry()])
    client = FakeClient([failed("INVALID_SESSION_ID", 401), failed("INVALID_SESSION_ID", 401)])
    run(store, client)
    assert client.refreshed == 1 and store.events[0][0] == "dead_letter"


def test_older_run_does_not_overwrite_newer_sync() -> None:
    store = FakeStore([entry()], newest_synced=NOW + timedelta(hours=1))
    client = FakeClient([])
    run(store, client)
    assert store.events == [("superseded", "ob-1")] and client.requests == []
