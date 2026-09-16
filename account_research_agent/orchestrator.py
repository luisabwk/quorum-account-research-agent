"""Account Research Agent: the LangGraph orchestrator.

Reasoning loop, in graph form (section 1.1):

    START -> prioritize (D1) -> [regulatory | stakeholder | news] in parallel
          -> validate (D2) -> research_judge --incomplete, once--> branch again
          -> score (D3) -> synthesize -> personalize -> delivery_judge
          --fail, once--> personalize with critique
          -> finalize (MongoDB + outbox, Slack) -> END

Error handling has three layers:
  1. TransientError (429, 5xx, timeout): the node's RetryPolicy retries with
     exponential backoff and jitter.
  2. PermanentToolError in a branch: recorded in `branch_errors`; the run
     continues with lower confidence instead of failing.
  3. Anything that still escapes fails the run. The MongoDB checkpointer keeps
     the finished parallel branches, so resuming re-executes only what failed.
"""

from __future__ import annotations

from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from . import nodes, schemas
from .llm import TransientError
from .nodes import Deps
from .schemas import AccountInput, Branch, BranchTask, RerunRequest, RerunScope, ResearchDepth, ResearchState

MAX_RERUNS_PER_ACCOUNT_PER_DAY = 3
# Backstop only. The counters in `nodes` end every loop well before this.
RECURSION_LIMIT = 40

TRANSIENT_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0, retry_on=TransientError)


class RerunLimitExceededError(Exception):
    pass


# Types stored in checkpoints. Reason: LangGraph deserializes checkpoints by
# importing the class named in the data. An explicit allowlist means a tampered
# checkpoint cannot make it import anything else; the default only warns, and
# a future LangGraph release blocks unlisted types, which would break reruns.
CHECKPOINT_TYPES = (
    schemas.AccountInput,
    schemas.AccountBrief,
    schemas.Branch,
    schemas.Claim,
    schemas.ClaimType,
    schemas.DeliveryJudgeVerdict,
    schemas.EvidenceStatus,
    schemas.KBPassage,
    schemas.OutreachDraft,
    schemas.ResearchDepth,
    schemas.RunOutcome,
    schemas.SourceClass,
    schemas.Stakeholder,
)


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=[(t.__module__, t.__name__) for t in CHECKPOINT_TYPES])


def build_graph(deps: Deps, retry: RetryPolicy = TRANSIENT_RETRY) -> StateGraph[ResearchState]:
    n = nodes.ResearchNodes(deps)
    graph = StateGraph(ResearchState)
    # Deterministic nodes (prioritize, validate, score) have no retry: they
    # cannot fail transiently. Nodes that call a model or a tool can.
    graph.add_node("prioritize", n.prioritize)
    graph.add_node("research_branch", n.research_branch, input_schema=BranchTask, retry_policy=retry)
    graph.add_node("validate", n.validate)
    graph.add_node("research_judge", n.research_judge, retry_policy=retry)
    graph.add_node("score", n.score)
    graph.add_node("synthesize", n.synthesize, retry_policy=retry)
    graph.add_node("personalize", n.personalize, retry_policy=retry)
    graph.add_node("delivery_judge", n.delivery_judge, retry_policy=retry)
    graph.add_node("finalize", n.finalize, retry_policy=retry)

    graph.add_conditional_edges(START, nodes.route_entry, ["prioritize", "research_branch", "personalize"])
    graph.add_conditional_edges("prioritize", nodes.route_after_prioritize, ["research_branch", "finalize"])
    graph.add_edge("research_branch", "validate")
    graph.add_edge("validate", "research_judge")
    graph.add_conditional_edges("research_judge", nodes.route_after_research_judge, ["research_branch", "score"])
    graph.add_conditional_edges("score", nodes.route_after_score, ["synthesize", "finalize"])
    graph.add_edge("synthesize", "personalize")
    graph.add_conditional_edges("personalize", nodes.route_after_personalize, ["delivery_judge", "finalize"])
    graph.add_conditional_edges("delivery_judge", nodes.route_after_delivery_judge, ["personalize", "finalize"])
    graph.add_edge("finalize", END)
    return graph


def compile_app(
    deps: Deps, checkpointer: BaseCheckpointSaver[str], retry: RetryPolicy = TRANSIENT_RETRY
) -> CompiledStateGraph[ResearchState]:
    # Production: MongoDBSaver(MongoClient(MONGODB_URI), db_name="agent_checkpoints", serde=checkpoint_serializer())
    return build_graph(deps, retry).compile(checkpointer=checkpointer)


def _config(run_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}


def run_account(app: CompiledStateGraph[ResearchState], account: AccountInput, run_id: str) -> ResearchState:
    start: ResearchState = {"run_id": run_id, "parent_run_id": None, "account": account, "entry": "prioritize"}
    return cast(ResearchState, app.invoke(start, _config(run_id)))


def rerun(app: CompiledStateGraph[ResearchState], deps: Deps, request: RerunRequest, run_id: str) -> ResearchState:
    """Start a new run seeded from the parent run's last checkpoint (section 1.5)."""
    parent = cast(ResearchState, app.get_state(_config(request.parent_run_id)).values)
    if not parent:
        raise ValueError(f"no checkpoint for parent run {request.parent_run_id}")
    account: AccountInput = parent["account"]
    if not deps.repo.reserve_rerun(account.salesforce_account_id, MAX_RERUNS_PER_ACCOUNT_PER_DAY):
        raise RerunLimitExceededError(f"{account.salesforce_account_id}: {MAX_RERUNS_PER_ACCOUNT_PER_DAY} reruns today")

    seed: ResearchState = {
        "run_id": run_id,
        "parent_run_id": request.parent_run_id,
        "account": account,
        "reviewer_feedback": request.reviewer_feedback,
    }
    if request.scope is RerunScope.DRAFT_ONLY:
        # Reason: the SDR's feedback changes the email, never the evidence.
        keys = ("depth", "claims", "stakeholders", "brief", "kb_passages", "priority_score", "confidence_score")
        seed |= cast(ResearchState, {k: parent[k] for k in keys if k in parent})  # type: ignore[literal-required]
        seed |= {"entry": "personalize", "draft_attempts": 0}
    elif request.scope is RerunScope.ONE_BRANCH:
        if request.branch is None:
            raise ValueError("ONE_BRANCH rerun needs a branch")
        seed |= {
            "entry": "branch",
            "rerun_branch": request.branch,
            "depth": ResearchDepth.DEEP,
            "research_retries": 0,
            "claims": [c for c in parent.get("claims", []) if c.branch is not request.branch],
            # Reason: stakeholders only come from the stakeholder branch.
            "stakeholders": [] if request.branch is Branch.STAKEHOLDER else parent.get("stakeholders", []),
        }
    else:
        seed |= {"entry": "prioritize", "force_depth": ResearchDepth.DEEP}

    return cast(ResearchState, app.invoke(seed, _config(run_id)))
