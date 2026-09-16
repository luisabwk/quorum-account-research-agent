"""Boundaries between the orchestrator and the outside world.

The graph depends on these Protocols only. Production wires the live
implementations; tests wire in-memory fakes. API calls without keys in this
sample are pseudocode comments.
"""

from __future__ import annotations

from typing import Protocol

from .llm import PermanentToolError
from .schemas import (
    AccountInput,
    Branch,
    KBPassage,
    ResearchDepth,
    ResearchResult,
    SourceDocument,
)


class ResearchTools(Protocol):
    def gather(self, branch: Branch, account: AccountInput, depth: ResearchDepth, attempt: int) -> list[SourceDocument]:
        """Call the branch's tools and return normalized, snapshotted documents.

        Raise TransientError on 429/5xx/timeouts, PermanentToolError otherwise.
        """
        ...


class KnowledgeBase(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[KBPassage]:
        """Approved passages only (Chroma filter `status == "approved"`)."""
        ...


class RunRepository(Protocol):
    def reserve_rerun(self, salesforce_account_id: str, daily_limit: int) -> bool:
        """Atomically count a rerun against today's cap. False if the cap is reached."""
        ...

    def save_run_with_outbox(self, result: ResearchResult) -> None:
        """One MongoDB transaction: upsert `research_runs` by run_id and insert the
        `salesforce_outbox` entry (unique index on run_id), so retries are idempotent."""
        ...


class ReviewChannel(Protocol):
    def post_review(self, result: ResearchResult) -> None:
        """Post brief, draft, sources and approve/edit/reject/rerun buttons to Slack."""
        ...


class LiveResearchTools:
    """Maps each branch to the tools in section 1.4. Calls are pseudocode."""

    def gather(self, branch: Branch, account: AccountInput, depth: ResearchDepth, attempt: int) -> list[SourceDocument]:
        # Reason: a retry after an incomplete branch widens the search window
        # instead of repeating the exact same queries.
        lookback_days = (365 if depth is ResearchDepth.DEEP else 180) * (attempt + 1)
        max_items = 15 if depth is ResearchDepth.DEEP else 5

        if branch is Branch.REGULATORY:
            # filings = lda_search_filings(client_name=account.name, since_days=lookback_days)
            # bills = [congress_search_bills(bill_id=b) for b in filings.bill_ids[:max_items]]
            # rules = federal_register_search(term=account.name, since_days=lookback_days)
            # risk = sec_edgar_filings(domain=account.domain, form="10-K")  # None if private
            # return [store_snapshot(doc) for doc in normalize(filings, bills, rules, risk)]
            pass
        elif branch is Branch.STAKEHOLDER:
            # contacts = salesforce_get_contacts(account.salesforce_account_id, titles=GA_TITLES)
            # if len(contacts) < 2:  # paid enrichment only when Salesforce has gaps
            #     contacts += zoominfo_search_contacts(domain=account.domain, titles=GA_TITLES)
            # lobbyists = lda_search_filings(client_name=account.name).in_house_lobbyists
            # team_page = fetch_company_page(f"https://{account.domain}/leadership")
            # return [store_snapshot(doc) for doc in normalize(contacts, lobbyists, team_page)]
            pass
        else:
            # hits = brave_news_search(q=f'"{account.name}" policy OR advocacy', freshness=lookback_days)
            # return [store_snapshot(fetch_article(h.url)) for h in hits[:max_items]]
            pass
        raise PermanentToolError(f"pseudocode: {branch} tools need API keys ({max_items=}, {lookback_days=})")
