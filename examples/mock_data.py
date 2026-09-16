"""Mock data for the end-to-end demo. Every company, person, bill and quote is fictional.

The data is built to exercise the guardrails, not only the happy path:

- a stale 10-K excerpt (rejected by freshness rules, D2)
- a claim whose quote is not in its source (rejected by the provenance check)
- a claim that overstates its evidence (rejected by the research judge)
- a rumor from an unknown blog (kept as needs_review, never reaches outreach)
- a contact who left the company (rejected: domain mismatch)
- a first draft that cites a claim id that does not exist (fails without a judge call)
- the personalization model's provider is down (fallback, judge switches family)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from account_research_agent.schemas import (
    AccountBrief,
    AccountInput,
    Branch,
    Claim,
    ClaimType,
    DeliveryJudgeVerdict,
    KBPassage,
    OutreachDraft,
    ResearchJudgeVerdict,
    SourceClass,
    SourceDocument,
    Stakeholder,
)

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)

ACCOUNT = AccountInput(
    salesforce_account_id="001Hx00000FAKE1AAA",
    name="Harbor Freight Rail Co.",
    domain="harborfreightrail.example",
    icp_fit=0.9,
    intent_score=80,
    engagement_score=60,
    open_opportunity=False,
    is_customer=False,
)


def _doc(branch: Branch, slug: str, url: str, cls: SourceClass, title: str, text: str, days_old: int) -> SourceDocument:
    return SourceDocument(
        branch=branch,
        url=url,
        source_class=cls,
        title=title,
        text=text,
        published_at=NOW - timedelta(days=days_old),
        retrieved_at=NOW,
        snapshot_key=f"s3://quorum-research-snapshots/2026/09/16/{slug}.html",
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


LDA = _doc(
    Branch.REGULATORY,
    "lda-q1",
    "https://lda.example.gov/filings/HFR-2026-Q1",
    SourceClass.OFFICIAL_GOVERNMENT,
    "LD-2 report, Q1 2026",
    "Client: Harbor Freight Rail Co. Specific lobbying issues: H.R. 9001, Rail Crew Safety Act; "
    "crew size requirements; hours-of-service rules. Reporting period: Q1 2026.",
    days_old=150,
)
FED_REG = _doc(
    Branch.REGULATORY,
    "fr-crew",
    "https://federalregister.example.gov/d/2026-11873",
    SourceClass.OFFICIAL_GOVERNMENT,
    "Proposed rule: minimum train crew staffing",
    "Federal Railroad Administration. Proposed rule on minimum train crew staffing. "
    "Comments received from Harbor Freight Rail Co. and 14 other carriers.",
    days_old=105,
)
TEN_K = _doc(
    Branch.REGULATORY,
    "10k-2024",
    "https://sec.example.gov/hfr/10-K-2023",
    SourceClass.OFFICIAL_COMPANY,
    "Form 10-K, fiscal 2023",
    "Risk factors. Changes in federal safety regulation, including crew size mandates, could increase our operating costs.",
    days_old=940,
)
LEADERSHIP = _doc(
    Branch.STAKEHOLDER,
    "leadership",
    "https://harborfreightrail.example/leadership",
    SourceClass.OFFICIAL_COMPANY,
    "Leadership team",
    "Leadership. Dana Okafor, Vice President, Government Affairs. Lee Park, Senior Policy Analyst.",
    days_old=20,
)
TRADE_NEWS = _doc(
    Branch.NEWS,
    "railweek",
    "https://railweek.example/2026/07/crew-size-coalition",
    SourceClass.TRUSTED_PUBLICATION,
    "Carriers form Safe Rail Coalition",
    "Harbor Freight Rail joined the Safe Rail Coalition and testified before committees in Ohio and Iowa on crew size bills.",
    days_old=64,
)
BLOG = _doc(
    Branch.NEWS,
    "blog",
    "https://railrumors.example/hfr-budget",
    SourceClass.SECONDARY_UNKNOWN,
    "Industry gossip",
    "Rumor: Harbor Freight Rail may cut its lobbying budget next year.",
    days_old=46,
)

DOCUMENTS: dict[Branch, list[SourceDocument]] = {
    Branch.REGULATORY: [LDA, FED_REG, TEN_K],
    Branch.STAKEHOLDER: [LEADERSHIP],
    Branch.NEWS: [TRADE_NEWS, BLOG],
}


def _claim(doc: SourceDocument, cid: str, ctype: ClaimType, statement: str, excerpt: str, relevance: float) -> Claim:
    return Claim(
        claim_id=cid,
        branch=doc.branch,
        claim_type=ctype,
        statement=statement,
        evidence_excerpt=excerpt,
        source_url=doc.url,
        source_class=doc.source_class,
        published_at=doc.published_at,
        retrieved_at=doc.retrieved_at,
        snapshot_key=doc.snapshot_key,
        content_sha256=doc.content_sha256,
        relevance=relevance,
    )


# What the extraction model "returns" for each branch.
EXTRACTION: dict[Branch, dict[str, list[Claim] | list[Stakeholder]]] = {
    Branch.REGULATORY: {
        "claims": [
            _claim(
                LDA,
                "c1",
                ClaimType.REGULATORY,
                "Harbor Freight Rail lobbied on H.R. 9001, the Rail Crew Safety Act, in Q1 2026.",
                "Specific lobbying issues: H.R. 9001, Rail Crew Safety Act",
                0.95,
            ),
            _claim(
                FED_REG,
                "c2",
                ClaimType.REGULATORY,
                "Harbor Freight Rail supports the FRA proposed rule on minimum crew staffing.",
                "Comments received from Harbor Freight Rail Co.",
                0.85,
            ),
            _claim(
                TEN_K,
                "c3",
                ClaimType.REGULATORY,
                "Harbor Freight Rail lists crew size mandates as a cost risk.",
                "including crew size mandates, could increase our operating costs",
                0.8,
            ),
        ]
    },
    Branch.STAKEHOLDER: {
        "stakeholders": [
            Stakeholder(
                stakeholder_id="stk-okafor",
                full_name="Dana Okafor",
                title="Vice President, Government Affairs",
                salesforce_contact_id="003Hx00000FAKE1AAA",
                source_url=LEADERSHIP.url,
                source_class=SourceClass.OFFICIAL_COMPANY,
                company_domain_match=True,
                last_verified_at=NOW,
            ),
            Stakeholder(
                stakeholder_id="stk-park",
                full_name="Lee Park",
                title="Senior Policy Analyst",
                source_url=LEADERSHIP.url,
                source_class=SourceClass.OFFICIAL_COMPANY,
                company_domain_match=True,
                last_verified_at=NOW,
            ),
            Stakeholder(
                stakeholder_id="stk-diaz",
                full_name="Morgan Diaz",
                title="Director of Public Policy",
                email="mdiaz@otherrail.example",
                source_url="https://enrichment.example/p/mdiaz",
                source_class=SourceClass.ENRICHMENT_PROVIDER,
                company_domain_match=False,
                last_verified_at=NOW - timedelta(days=200),
            ),
        ]
    },
    Branch.NEWS: {
        "claims": [
            _claim(
                TRADE_NEWS,
                "c1",
                ClaimType.ADVOCACY,
                "Harbor Freight Rail joined the Safe Rail Coalition and testified in Ohio and Iowa on crew size bills.",
                "joined the Safe Rail Coalition and testified before committees in Ohio and Iowa",
                0.92,
            ),
            _claim(
                TRADE_NEWS,
                "c2",
                ClaimType.ADVOCACY,
                "Harbor Freight Rail pledged $2M to the Safe Rail Coalition.",
                "Harbor Freight Rail pledged $2M to the coalition",
                0.9,
            ),
            _claim(
                BLOG,
                "c3",
                ClaimType.ADVOCACY,
                "Harbor Freight Rail may cut its lobbying budget next year.",
                "may cut its lobbying budget next year",
                0.7,
            ),
        ]
    },
}

RESEARCH_VERDICT = ResearchJudgeVerdict(
    unsupported_claim_ids=["regulatory:0:c2"],
    notes="regulatory:0:c2: submitting a comment does not show support for the rule.",
)

KB = [
    KBPassage(
        kb_id="kb-state-tracking",
        title="Multi-state tracking",
        version=4,
        text="Track bills and regulations across all 50 states and Congress, with alerts by issue.",
    ),
    KBPassage(
        kb_id="kb-testimony",
        title="Hearings and testimony",
        version=2,
        text="Log hearings and testimony and link each one to the bills it relates to.",
    ),
    KBPassage(
        kb_id="kb-coalitions",
        title="Coalition management",
        version=1,
        text="Coordinate coalition partners, shared positions and joint letters in one place.",
    ),
]

BRIEF = AccountBrief(
    summary="Harbor Freight Rail lobbies federally on crew size (H.R. 9001) and advocates on the same issue in state "
    "legislatures through the Safe Rail Coalition.",
    commercial_hypothesis="The government affairs team tracks one issue across Congress and several states at once, "
    "which is manual work that grows with every new state bill.",
    outreach_angle="Crew size is moving in Congress and in Ohio and Iowa at the same time.",
    cited_claim_ids=["regulatory:0:c1", "news:0:c1"],
)

DRAFT_WITH_BAD_CITATION = OutreachDraft(
    stakeholder_id="stk-okafor",
    subject="Crew size in Congress, Ohio and Iowa",
    body="Hi Dana,\n\nYour team lobbied on H.R. 9001 and grew the coalition to 20 carriers this year. ...",
    cited_claim_ids=["regulatory:0:c1", "news:0:c9"],
    cited_kb_ids=["kb-state-tracking"],
)

GOOD_DRAFT = OutreachDraft(
    stakeholder_id="stk-okafor",
    subject="Crew size in Congress, Ohio and Iowa",
    body=(
        "Hi Dana,\n\nYour team lobbied on H.R. 9001 in Q1 and testified in Ohio and Iowa on crew size bills with the "
        "Safe Rail Coalition. Keeping the federal bill and each state version in view at once is where the hours go.\n\n"
        "Quorum tracks bills across Congress and all 50 states with alerts by issue, and links each hearing and "
        "testimony to the bill it relates to.\n\nWould a one-page view of crew size bills by state help before your "
        "next hearing?\n\n{{sdr_first_name}}"
    ),
    cited_claim_ids=["regulatory:0:c1", "news:0:c1"],
    cited_kb_ids=["kb-state-tracking", "kb-testimony"],
)

SHORTER_DRAFT = OutreachDraft(
    stakeholder_id="stk-okafor",
    subject="Your Ohio and Iowa crew size testimony",
    body=(
        "Hi Dana,\n\nYou testified in Ohio and Iowa on crew size bills while H.R. 9001 moves in Congress. Quorum links "
        "each hearing to its bill and tracks all 50 states by issue.\n\nWorth a look before the next hearing?\n\n"
        "{{sdr_first_name}}"
    ),
    cited_claim_ids=["news:0:c1", "regulatory:0:c1"],
    cited_kb_ids=["kb-testimony", "kb-state-tracking"],
)

PASS = DeliveryJudgeVerdict(
    groundedness=5,
    approved_capabilities_only=True,
    relevance=5,
    tone_and_length=4,
    critique="Grounded and specific. Could be one sentence shorter.",
)

# USD per 1M tokens (input, output), from the OpenRouter snapshot in section 1.2.
PRICES = {
    "openai/gpt-5-nano": (0.05, 0.40),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "google/gemini-3.8-flash": (0.75, 3.75),
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "mistralai/mistral-large-2512": (0.50, 1.50),
}
