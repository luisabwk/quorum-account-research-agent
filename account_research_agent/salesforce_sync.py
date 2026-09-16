"""Salesforce write-back: outbox worker, field mapping, CRM error handling.

The orchestrator never calls Salesforce. It writes the result and an outbox
entry in one MongoDB transaction (section 1.3). This worker drains the outbox:

    claim due entries (lease) -> skip if a newer run already synced
    -> map fields -> one Composite API call (allOrNone)
    -> synced | retry with backoff | refresh token once | dead letter + alert

A Salesforce outage delays the write. It never loses research.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol, TypedDict
from urllib.parse import quote

from pydantic import BaseModel

from .schemas import OutboxStatus, ResearchResult, RunOutcome

API_VERSION = "v66.0"  # pin to the org's API version; bump deliberately
MAX_ATTEMPTS = 8
BASE_BACKOFF = timedelta(seconds=30)
MAX_BACKOFF = timedelta(hours=1)
LEASE = timedelta(minutes=5)

JsonValue = str | int | float | bool | None
SALESFORCE_ID = re.compile(r"[a-zA-Z0-9]{15}(?:[a-zA-Z0-9]{3})?")


class SalesforceTransientError(Exception):
    """429, 5xx, network timeout, row lock. Retry later."""


class SalesforceAuthExpiredError(Exception):
    """401 / INVALID_SESSION_ID. Refresh the token and retry once, now."""


class SalesforcePermanentError(Exception):
    """Validation rule, bad field, deleted record, no access. Retrying will not help."""


class MappingError(SalesforcePermanentError):
    """A value cannot be converted for its field. Permanent: fix the mapping, retrying will not help."""


TRANSIENT_CODES = frozenset({"UNABLE_TO_LOCK_ROW", "REQUEST_LIMIT_EXCEEDED", "SERVER_UNAVAILABLE", "QUERY_TIMEOUT"})
AUTH_CODES = frozenset({"INVALID_SESSION_ID"})
# With allOrNone, every sub-request after the failing one reports this code.
HALTED = "PROCESSING_HALTED"


class CompositeSubRequest(TypedDict):
    """One sub-request of the Composite API, with Salesforce's field names."""

    method: str
    url: str
    referenceId: str
    body: dict[str, JsonValue]


class CompositeRequest(TypedDict):
    """Body of POST /composite."""

    allOrNone: bool
    compositeRequest: list[CompositeSubRequest]


class Kind(StrEnum):
    """Salesforce field types that `transform` can convert."""

    ID = "id"
    TEXT = "text"
    NUMBER = "number"
    PICKLIST = "picklist"
    DATETIME = "datetime"
    CHECKBOX = "checkbox"


@dataclass(frozen=True)
class FieldSpec:
    """One row of the mapping table: field, type, how to read the value, and its limits."""

    sf_field: str
    kind: Kind
    read: Callable[[ResearchResult], object]
    max_length: int | None = None
    scale: int | None = None
    picklist: dict[str, str] | None = None


OUTCOME_PICKLIST = {
    RunOutcome.SKIPPED.value: "Skipped",
    RunOutcome.READY_FOR_REVIEW.value: "Ready for Review",
    RunOutcome.LOW_QUALITY_DRAFT.value: "Draft Needs Work",
    RunOutcome.NEEDS_HUMAN_RESEARCH.value: "Needs Human Research",
}

# Salesforce gets summary fields only. Claims, sources and judge scores stay in
# MongoDB; the record links back to the review page instead of copying them.
ACCOUNT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("AI_Research_Status__c", Kind.PICKLIST, lambda r: r.outcome.value, picklist=OUTCOME_PICKLIST),
    FieldSpec("AI_Priority_Score__c", Kind.NUMBER, lambda r: r.priority_score, scale=2),
    FieldSpec("AI_Evidence_Confidence__c", Kind.NUMBER, lambda r: r.confidence_score, scale=2),
    FieldSpec("AI_Research_Summary__c", Kind.TEXT, lambda r: r.brief.summary if r.brief else None, max_length=32768),
    FieldSpec(
        "AI_Commercial_Hypothesis__c",
        Kind.TEXT,
        lambda r: r.brief.commercial_hypothesis if r.brief else None,
        max_length=2000,
    ),
    FieldSpec(
        "AI_Outreach_Angle__c", Kind.TEXT, lambda r: r.brief.outreach_angle if r.brief else None, max_length=1000
    ),
    FieldSpec("AI_Research_Run_Id__c", Kind.TEXT, lambda r: r.run_id, max_length=64),
    FieldSpec("AI_Last_Researched_At__c", Kind.DATETIME, lambda r: r.completed_at),
)

DRAFT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("Account__c", Kind.ID, lambda r: r.salesforce_account_id),
    FieldSpec(
        "Contact__c",
        Kind.ID,
        lambda r: r.draft_stakeholder.salesforce_contact_id if r.draft_stakeholder else None,
    ),
    FieldSpec(
        "Stakeholder_Name__c",
        Kind.TEXT,
        lambda r: r.draft_stakeholder.full_name if r.draft_stakeholder else None,
        max_length=255,
    ),
    FieldSpec(
        "Stakeholder_Title__c",
        Kind.TEXT,
        lambda r: r.draft_stakeholder.title if r.draft_stakeholder else None,
        max_length=255,
    ),
    FieldSpec("Subject__c", Kind.TEXT, lambda r: r.draft.subject if r.draft else None, max_length=255),
    FieldSpec("Body__c", Kind.TEXT, lambda r: r.draft.body if r.draft else None, max_length=32768),
    FieldSpec("Cited_Sources__c", Kind.TEXT, lambda r: "\n".join(str(u) for u in r.top_source_urls), max_length=32768),
    FieldSpec("Judge_Passed__c", Kind.CHECKBOX, lambda r: bool(r.delivery_verdict and r.delivery_verdict.passed)),
    FieldSpec(
        "Judge_Critique__c",
        Kind.TEXT,
        lambda r: r.delivery_verdict.critique if r.delivery_verdict else None,
        max_length=1000,
    ),
    FieldSpec("Parent_Run_Id__c", Kind.TEXT, lambda r: r.parent_run_id, max_length=64),
)


def transform(spec: FieldSpec, value: object) -> tuple[JsonValue, str | None]:
    """Convert one value to its Salesforce JSON form. Returns (value, warning).

    None becomes null on purpose: a newer run with no score must clear the old
    score, not leave last quarter's number looking current.
    """
    if value is None:
        return None, None
    if spec.kind is Kind.ID:
        # Reason: a truncated record id points at the wrong record or none.
        # An id is either valid or a mapping error, never shortened.
        text = str(value)
        if not SALESFORCE_ID.fullmatch(text):
            raise MappingError(f"{spec.sf_field}: {text!r} is not a 15 or 18 character Salesforce id")
        return text, None
    if spec.kind is Kind.TEXT:
        text = str(value)
        if spec.max_length is not None and len(text) > spec.max_length:
            # Reason: truncation is allowed but never silent. The warning is
            # stored on the outbox entry and shown in the sync report.
            marker = " [truncated]"
            return text[
                : spec.max_length - len(marker)
            ] + marker, f"{spec.sf_field}: truncated {len(text)} -> {spec.max_length}"
        return text, None
    if spec.kind is Kind.NUMBER:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise MappingError(f"{spec.sf_field}: expected number, got {type(value).__name__}")
        return round(float(value), spec.scale if spec.scale is not None else 2), None
    if spec.kind is Kind.PICKLIST:
        label = (spec.picklist or {}).get(str(value))
        if label is None:
            raise MappingError(f"{spec.sf_field}: no picklist value for {value!r}")
        return label, None
    if spec.kind is Kind.DATETIME:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise MappingError(f"{spec.sf_field}: expected timezone-aware datetime")
        return value.isoformat(timespec="seconds"), None
    return bool(value), None


def map_fields(specs: tuple[FieldSpec, ...], result: ResearchResult) -> tuple[dict[str, JsonValue], list[str]]:
    """Apply a mapping table to one result. Returns the request body and any truncation warnings."""
    body: dict[str, JsonValue] = {}
    warnings: list[str] = []
    for spec in specs:
        value, warning = transform(spec, spec.read(result))
        body[spec.sf_field] = value
        if warning:
            warnings.append(warning)
    return body, warnings


def build_composite(result: ResearchResult) -> tuple[CompositeRequest, list[str]]:
    """One allOrNone request: the Account summary PATCH, plus the draft upsert when there is a draft."""
    base = f"/services/data/{API_VERSION}/sobjects"
    # Reason: the account id also goes into the URL path, so it gets the same
    # check as the Account__c field; the run id is escaped for the same reason.
    account_id, _ = transform(
        FieldSpec("Account (URL)", Kind.ID, lambda r: r.salesforce_account_id), result.salesforce_account_id
    )
    run_id = quote(result.run_id, safe="")
    account_body, warnings = map_fields(ACCOUNT_FIELDS, result)
    requests: list[CompositeSubRequest] = [
        {
            "method": "PATCH",
            "url": f"{base}/Account/{account_id}",
            "referenceId": "account",
            "body": account_body,
        }
    ]
    if result.draft is not None:
        draft_body, draft_warnings = map_fields(DRAFT_FIELDS, result)
        warnings += draft_warnings
        # Reason: upsert by external id Run_Id__c. A retry after a timeout
        # updates the same draft instead of creating a duplicate.
        requests.append(
            {
                "method": "PATCH",
                "url": f"{base}/AI_Outreach_Draft__c/Run_Id__c/{run_id}",
                "referenceId": "draft",
                "body": draft_body,
            }
        )
    # Reason: allOrNone keeps the Account summary and the draft consistent.
    return CompositeRequest(allOrNone=True, compositeRequest=requests), warnings


class OutboxEntry(BaseModel):
    """One pending CRM write, stored in the MongoDB `salesforce_outbox` collection."""

    outbox_id: str
    payload: ResearchResult
    status: OutboxStatus
    attempts: int = 0
    next_attempt_at: datetime
    last_error: str | None = None
    warnings: list[str] = []


class OutboxStore(Protocol):
    """The `salesforce_outbox` operations the worker needs."""

    def claim_due(self, now: datetime, lease: timedelta, limit: int) -> list[OutboxEntry]:
        """findOneAndUpdate loop: status pending and next_attempt_at <= now
        -> in_progress with lease_until. Expired leases count as pending."""
        ...

    def latest_synced_completed_at(self, salesforce_account_id: str) -> datetime | None: ...
    def mark_synced(self, outbox_id: str, warnings: list[str]) -> None: ...
    def reschedule(self, outbox_id: str, attempts: int, next_attempt_at: datetime, error: str) -> None: ...
    def dead_letter(self, outbox_id: str, error: str) -> None: ...
    def mark_superseded(self, outbox_id: str) -> None: ...


class SubResponse(BaseModel):
    """One entry of `compositeResponse`."""

    referenceId: str  # noqa: N815 - Salesforce's field name
    httpStatusCode: int  # noqa: N815
    body: list[dict[str, str]] | dict[str, object] | None = None


class SalesforceClient(Protocol):
    """The two Salesforce calls the worker makes."""

    def composite(self, request: CompositeRequest) -> list[SubResponse]:
        """POST /composite. Raise SalesforceAuthExpiredError / SalesforceTransientError /
        SalesforcePermanentError for failures of the whole HTTP call."""
        ...

    def refresh_token(self) -> None: ...


class Alerts(Protocol):
    """Where dead letters are announced: a Slack alert channel in production."""

    def alert(self, text: str) -> None: ...


def raise_for_sub_errors(responses: list[SubResponse]) -> None:
    """Raise the exception class that matches the first real sub-request error."""
    for sub in responses:
        if sub.httpStatusCode < 400:
            continue
        errors = sub.body if isinstance(sub.body, list) else []
        codes = [e.get("errorCode", "UNKNOWN") for e in errors]
        if codes and all(c == HALTED for c in codes):
            continue  # a sibling failed first; that one carries the real error
        detail = f"{sub.referenceId}: HTTP {sub.httpStatusCode} {codes} {[e.get('message', '') for e in errors]}"
        if AUTH_CODES & set(codes) or sub.httpStatusCode == 401:
            raise SalesforceAuthExpiredError(detail)
        if TRANSIENT_CODES & set(codes) or sub.httpStatusCode >= 500:
            raise SalesforceTransientError(detail)
        raise SalesforcePermanentError(detail)


@dataclass
class SyncReport:
    """Counts for one batch, used in logs and in the outbox metrics of section 3.1."""

    synced: int = 0
    rescheduled: int = 0
    dead_lettered: int = 0
    superseded: int = 0
    warnings: list[str] = field(default_factory=list)


class OutboxWorker:
    """Drains the outbox: one Composite call per entry, with retry, token refresh and dead letter."""

    def __init__(
        self,
        store: OutboxStore,
        client: SalesforceClient,
        alerts: Alerts,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._store = store
        self._client = client
        self._alerts = alerts
        self._jitter = jitter

    def backoff(self, attempts: int) -> timedelta:
        """30 s, 60 s, 120 s and so on, capped at 1 hour, plus up to 25% jitter so retries do not arrive together."""
        delay: timedelta = min(BASE_BACKOFF * (2 ** (attempts - 1)), MAX_BACKOFF)
        return delay * (1 + 0.25 * self._jitter())

    def process_batch(self, now: datetime, limit: int = 50) -> SyncReport:
        """Claim due entries with a lease and sync each one. Several workers can run at once."""
        report = SyncReport()
        for entry in self._store.claim_due(now, LEASE, limit):
            self._process(entry, now, report)
        return report

    def _send(self, request: CompositeRequest) -> None:
        """Send one request. If the token expired, refresh it and send once more."""
        try:
            raise_for_sub_errors(self._client.composite(request))
        except SalesforceAuthExpiredError:
            # Reason: tokens expire mid-batch. One refresh, one retry; a second
            # auth failure means bad credentials, which is permanent.
            self._client.refresh_token()
            try:
                raise_for_sub_errors(self._client.composite(request))
            except SalesforceAuthExpiredError as exc:
                raise SalesforcePermanentError(f"auth failed after refresh: {exc}") from exc

    def _process(self, entry: OutboxEntry, now: datetime, report: SyncReport) -> None:
        """Sync one entry: skip it if a newer run already synced, otherwise send and record the result."""
        result = entry.payload
        newest = self._store.latest_synced_completed_at(result.salesforce_account_id)
        if newest is not None and newest >= result.completed_at:
            # Reason: an old retry must not overwrite a newer run's fields.
            self._store.mark_superseded(entry.outbox_id)
            report.superseded += 1
            return

        attempts = entry.attempts + 1
        try:
            request, warnings = build_composite(result)
            self._send(request)
        except SalesforceTransientError as exc:
            if attempts >= MAX_ATTEMPTS:
                self._fail(entry, f"gave up after {attempts} attempts: {exc}", report)
                return
            self._store.reschedule(entry.outbox_id, attempts, now + self.backoff(attempts), str(exc))
            report.rescheduled += 1
            return
        except SalesforcePermanentError as exc:
            self._fail(entry, str(exc), report)
            return

        self._store.mark_synced(entry.outbox_id, warnings)
        report.synced += 1
        report.warnings += warnings

    def _fail(self, entry: OutboxEntry, error: str, report: SyncReport) -> None:
        """Dead letter the entry and alert, so the missing CRM update is visible."""
        self._store.dead_letter(entry.outbox_id, error)
        # Reason: a dead letter means an SDR is missing research in Salesforce.
        # Someone has to know; the research itself is safe in MongoDB.
        self._alerts.alert(
            f"Salesforce sync failed for account {entry.payload.salesforce_account_id} "
            f"(run {entry.payload.run_id}): {error}"
        )
        report.dead_lettered += 1


class HttpSalesforceClient:
    """Live client. Token exchange is pseudocode: no connected app in this sample."""

    def __init__(self, instance_url: str) -> None:
        self._instance_url = instance_url
        self._token: str | None = None

    def refresh_token(self) -> None:
        # OAuth 2.0 JWT bearer flow for a server-to-server integration user:
        # assertion = jwt.encode({"iss": CLIENT_ID, "sub": INTEGRATION_USER,
        #                         "aud": "https://login.salesforce.com", "exp": now + 180},
        #                        PRIVATE_KEY, algorithm="RS256")
        # resp = httpx.post("https://login.salesforce.com/services/oauth2/token",
        #                   data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        #                         "assertion": assertion})
        # self._token = resp.json()["access_token"]
        raise NotImplementedError("pseudocode: requires a connected app and private key")

    def composite(self, request: CompositeRequest) -> list[SubResponse]:
        # resp = httpx.post(f"{self._instance_url}/services/data/{API_VERSION}/composite",
        #                   headers={"Authorization": f"Bearer {self._token}"}, json=request, timeout=30)
        # if resp.status_code == 401: raise SalesforceAuthExpiredError(resp.text)
        # if resp.status_code in (429, 502, 503, 504): raise SalesforceTransientError(resp.text)
        # if resp.status_code >= 400: raise SalesforcePermanentError(resp.text)
        # return [SubResponse.model_validate(r) for r in resp.json()["compositeResponse"]]
        raise NotImplementedError("pseudocode: requires Salesforce credentials")
