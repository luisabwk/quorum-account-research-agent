"""Five mock scenarios for the flow viewer. Every company, person, bill and quote is fictional.

Each scenario is a script for the edges of the system: what each research tool
returns per branch and attempt, what each model answers, and which providers are
down. The orchestrator, rules, prompts and Salesforce mapping are the real code.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import timedelta

import mock_data as m

from account_research_agent.llm import PermanentToolError
from account_research_agent.schemas import (
    AccountBrief,
    AccountInput,
    Branch,
    ClaimType,
    DeliveryJudgeVerdict,
    OutreachDraft,
    RerunRequest,
    RerunScope,
    ResearchJudgeVerdict,
    SourceClass,
    SourceDocument,
    Stakeholder,
)

Key = tuple[Branch, int]  # (branch, attempt)


@dataclass
class Scenario:
    id: str
    title: str
    shows: str
    account: AccountInput
    documents: dict[Key, list[SourceDocument] | Exception] = field(default_factory=dict)
    extraction: dict[Key, dict[str, list[object]]] = field(default_factory=dict)
    research_verdicts: list[ResearchJudgeVerdict] = field(default_factory=list)
    brief: AccountBrief | None = None
    drafts: list[OutreachDraft] = field(default_factory=list)
    delivery_verdicts: list[DeliveryJudgeVerdict] = field(default_factory=list)
    down: dict[str, set[str]] = field(default_factory=dict)
    parent: str | None = None
    rerun: RerunRequest | None = None


# Which tool from section 1.4 produced each mock document (display only).
TOOL_BY_HOST = {
    "lda.example.gov": "lda_search_filings",
    "federalregister.example.gov": "federal_register_search",
    "sec.example.gov": "sec_edgar_filings",
    "railweek.example": "brave_news_search + fetch_article",
    "railrumors.example": "brave_news_search + fetch_article",
    "healthpolicywire.example": "brave_news_search + fetch_article",
    "enrichment.example": "zoominfo_search_contacts",
}


def tool_for(doc: SourceDocument) -> str:
    host = doc.url.host or ""
    return TOOL_BY_HOST.get(host, "fetch_company_page")


def doc(branch: Branch, slug: str, url: str, cls: SourceClass, title: str, text: str, days_old: int) -> SourceDocument:
    return SourceDocument(
        branch=branch,
        url=url,
        source_class=cls,
        title=title,
        text=text,
        published_at=m.NOW - timedelta(days=days_old),
        retrieved_at=m.NOW,
        snapshot_key=f"s3://quorum-research-snapshots/2026/09/16/{slug}.html",
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def claims(*items: object) -> dict[str, list[object]]:
    return {"claims": list(items)}


def people(*items: object) -> dict[str, list[object]]:
    return {"stakeholders": list(items)}


def draft_to(stakeholder_id: str, subject: str, body: str, claim_ids: list[str], kb_ids: list[str]) -> OutreachDraft:
    return OutreachDraft(
        stakeholder_id=stakeholder_id, subject=subject, body=body, cited_claim_ids=claim_ids, cited_kb_ids=kb_ids
    )


# 1. Guardrails ------------------------------------------------------------------
GUARDRAILS = Scenario(
    id="guardrails",
    title="Strong account, every guardrail fires",
    shows="Stale, fabricated and overstated evidence is rejected; a provider outage triggers fallback; "
    "a draft with a bad citation is rewritten.",
    account=m.ACCOUNT,
    documents={(b, 0): docs for b, docs in m.DOCUMENTS.items()},
    extraction={(b, 0): {k: list(v) for k, v in payload.items()} for b, payload in m.EXTRACTION.items()},
    research_verdicts=[m.RESEARCH_VERDICT],
    brief=m.BRIEF,
    drafts=[m.DRAFT_WITH_BAD_CITATION, m.GOOD_DRAFT],
    delivery_verdicts=[m.PASS],
    down={"OutreachDraft": {"google/gemini-3.8-flash"}},
)

# 2. Existing customer -----------------------------------------------------------
CUSTOMER = Scenario(
    id="customer",
    title="Existing customer, skipped at D1",
    shows="Customers belong to Account Management, so the run stops before any tool or model call.",
    account=AccountInput(
        salesforce_account_id="001Hx00000FAKE2AAA",
        name="Northgate Water Utility",
        domain="northgatewater.example",
        icp_fit=0.95,
        intent_score=90,
        engagement_score=70,
        open_opportunity=False,
        is_customer=True,
    ),
)

# 3. Thin evidence ---------------------------------------------------------------
PINE_ACCOUNT = AccountInput(
    salesforce_account_id="001Hx00000FAKE3AAA",
    name="Pinecrest Logistics",
    domain="pinecrestlogistics.example",
    icp_fit=0.6,
    intent_score=55,
    engagement_score=40,
    open_opportunity=False,
    is_customer=False,
)
PINE_10K = doc(
    Branch.REGULATORY,
    "pine-10k",
    "https://sec.example.gov/pinecrest/10-K-2023",
    SourceClass.OFFICIAL_COMPANY,
    "Form 10-K, fiscal 2023",
    "Risk factors. New emissions standards for trucking fleets may raise our costs.",
    900,
)
PINE_BLOG = doc(
    Branch.NEWS,
    "pine-blog",
    "https://railrumors.example/pinecrest",
    SourceClass.SECONDARY_UNKNOWN,
    "Freight gossip",
    "Pinecrest Logistics is said to be hiring a lobbyist in Sacramento.",
    30,
)
PINE_CONTACT = doc(
    Branch.STAKEHOLDER,
    "pine-zoominfo",
    "https://enrichment.example/p/jreyes",
    SourceClass.ENRICHMENT_PROVIDER,
    "Contact record",
    "Jordan Reyes, Head of Public Affairs, email jreyes@coastfreight.example",
    10,
)
THIN = Scenario(
    id="thin-evidence",
    title="Mid-fit account with thin evidence",
    shows="Nothing survives validation, so the research judge is skipped and two branches are re-researched once. "
    "Still nothing, so the account goes to a human instead of getting an email built on weak evidence.",
    account=PINE_ACCOUNT,
    documents={
        (Branch.REGULATORY, 0): [PINE_10K],
        (Branch.REGULATORY, 1): [PINE_10K],
        (Branch.STAKEHOLDER, 0): [PINE_CONTACT],
        (Branch.STAKEHOLDER, 1): [],
        (Branch.NEWS, 0): [PINE_BLOG],
    },
    extraction={
        (Branch.REGULATORY, 0): claims(
            m._claim(
                PINE_10K,
                "c1",
                ClaimType.REGULATORY,
                "Pinecrest Logistics lists fleet emissions standards as a cost risk.",
                "New emissions standards for trucking fleets may raise our costs",
                0.75,
            )
        ),
        (Branch.REGULATORY, 1): claims(
            m._claim(
                PINE_10K,
                "c1",
                ClaimType.REGULATORY,
                "Pinecrest Logistics lists fleet emissions standards as a cost risk.",
                "New emissions standards for trucking fleets may raise our costs",
                0.75,
            )
        ),
        (Branch.STAKEHOLDER, 0): people(
            Stakeholder(
                stakeholder_id="stk-reyes",
                full_name="Jordan Reyes",
                title="Head of Public Affairs",
                email="jreyes@coastfreight.example",
                source_url=PINE_CONTACT.url,
                source_class=SourceClass.ENRICHMENT_PROVIDER,
                company_domain_match=False,
                last_verified_at=m.NOW - timedelta(days=10),
            )
        ),
        (Branch.NEWS, 0): claims(
            m._claim(
                PINE_BLOG,
                "c1",
                ClaimType.ADVOCACY,
                "Pinecrest Logistics may be hiring a lobbyist in Sacramento.",
                "is said to be hiring a lobbyist in Sacramento",
                0.8,
            )
        ),
    },
)

# 4. Outage and weak drafts ------------------------------------------------------
SUMMIT = AccountInput(
    salesforce_account_id="001Hx00000FAKE4AAA",
    name="Summit Health Alliance",
    domain="summithealth.example",
    icp_fit=0.85,
    intent_score=70,
    engagement_score=65,
    open_opportunity=True,
    is_customer=False,
)
SUMMIT_LDA = doc(
    Branch.REGULATORY,
    "summit-lda",
    "https://lda.example.gov/filings/SHA-2026-Q2",
    SourceClass.OFFICIAL_GOVERNMENT,
    "LD-2 report, Q2 2026",
    "Client: Summit Health Alliance. Specific lobbying issues: S. 7102, Rural Telehealth Access Act; "
    "Medicare telehealth reimbursement. Reporting period: Q2 2026.",
    60,
)
SUMMIT_TEAM = doc(
    Branch.STAKEHOLDER,
    "summit-team",
    "https://summithealth.example/about/team",
    SourceClass.OFFICIAL_COMPANY,
    "Our team",
    "Priya Natarajan, Director of Government Relations.",
    15,
)
SUMMIT_CLAIM = m._claim(
    SUMMIT_LDA,
    "c1",
    ClaimType.REGULATORY,
    "Summit Health Alliance lobbied on S. 7102, the Rural Telehealth Access Act, in Q2 2026.",
    "Specific lobbying issues: S. 7102, Rural Telehealth Access Act",
    0.93,
)
WEAK = DeliveryJudgeVerdict(
    groundedness=3,
    approved_capabilities_only=True,
    relevance=2,
    tone_and_length=3,
    critique="Generic: the email never connects telehealth reimbursement to a Quorum capability.",
)
OUTAGE = Scenario(
    id="outage-weak-drafts",
    title="News API down, drafts too generic",
    shows="A permanent tool error is recorded and the run continues without news. The research judge falls back to "
    "Mistral. Both drafts fail the delivery judge, so the SDR gets the draft flagged as needing work.",
    account=SUMMIT,
    documents={
        (Branch.REGULATORY, 0): [SUMMIT_LDA],
        (Branch.STAKEHOLDER, 0): [SUMMIT_TEAM],
        (Branch.NEWS, 0): PermanentToolError("Brave Search API: HTTP 403, subscription expired"),
    },
    extraction={
        (Branch.REGULATORY, 0): claims(SUMMIT_CLAIM),
        (Branch.STAKEHOLDER, 0): people(
            Stakeholder(
                stakeholder_id="stk-natarajan",
                full_name="Priya Natarajan",
                title="Director of Government Relations",
                source_url=SUMMIT_TEAM.url,
                source_class=SourceClass.OFFICIAL_COMPANY,
                company_domain_match=True,
                last_verified_at=m.NOW,
            )
        ),
    },
    research_verdicts=[ResearchJudgeVerdict()],
    brief=AccountBrief(
        summary="Summit Health Alliance lobbies on rural telehealth reimbursement (S. 7102).",
        commercial_hypothesis="Telehealth policy spans Congress and CMS rulemaking.",
        outreach_angle="S. 7102 and Medicare telehealth reimbursement.",
        cited_claim_ids=["regulatory:0:c1"],
    ),
    drafts=[
        draft_to(
            "stk-natarajan",
            "Quick question",
            "Hi Priya,\n\nQuorum helps teams like yours stay on top of policy. Open to a chat?\n\n{{sdr_first_name}}",
            ["regulatory:0:c1"],
            ["kb-state-tracking"],
        ),
        draft_to(
            "stk-natarajan",
            "Policy tracking for Summit",
            "Hi Priya,\n\nSummit lobbied on S. 7102 this quarter. Quorum tracks bills. Worth a call?\n\n{{sdr_first_name}}",
            ["regulatory:0:c1"],
            ["kb-state-tracking"],
        ),
    ],
    delivery_verdicts=[WEAK, WEAK],
    down={"ResearchJudgeVerdict": {"google/gemini-3.8-flash"}},
)

# 5. SDR rerun of one branch -----------------------------------------------------
NEW_LEADERSHIP = doc(
    Branch.STAKEHOLDER,
    "leadership-sept",
    "https://harborfreightrail.example/leadership",
    SourceClass.OFFICIAL_COMPANY,
    "Leadership team (updated)",
    "Leadership. Lee Park, Vice President, Government Affairs.",
    3,
)
RERUN = Scenario(
    id="rerun-stakeholder",
    title="SDR rerun: the contact left the company",
    shows="The SDR flags that Dana Okafor left. Only the stakeholder branch runs again; regulatory and news evidence "
    "come from the parent run's checkpoint. The new draft goes to Lee Park.",
    account=m.ACCOUNT,
    parent="guardrails",
    rerun=RerunRequest(
        parent_run_id="run-guardrails",
        scope=RerunScope.ONE_BRANCH,
        branch=Branch.STAKEHOLDER,
        reviewer_feedback="Dana Okafor left in August. Lee Park now leads government affairs.",
        requested_by="sdr@quorum.example",
    ),
    documents={(Branch.STAKEHOLDER, 0): [NEW_LEADERSHIP]},
    extraction={
        (Branch.STAKEHOLDER, 0): people(
            Stakeholder(
                stakeholder_id="stk-park",
                full_name="Lee Park",
                title="Vice President, Government Affairs",
                salesforce_contact_id="003Hx00000FAKE2AAA",
                source_url=NEW_LEADERSHIP.url,
                source_class=SourceClass.OFFICIAL_COMPANY,
                company_domain_match=True,
                last_verified_at=m.NOW,
            )
        )
    },
    research_verdicts=[ResearchJudgeVerdict()],
    brief=m.BRIEF,
    drafts=[
        draft_to(
            "stk-park",
            "Crew size in Congress, Ohio and Iowa",
            m.GOOD_DRAFT.body.replace("Hi Dana,", "Hi Lee,"),
            ["regulatory:0:c1", "news:0:c1"],
            ["kb-state-tracking", "kb-testimony"],
        )
    ],
    delivery_verdicts=[m.PASS],
)

SCENARIOS = [GUARDRAILS, CUSTOMER, THIN, OUTAGE, RERUN]
