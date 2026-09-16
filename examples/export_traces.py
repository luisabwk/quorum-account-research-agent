"""Run every scenario through the real graph and export a step-by-step trace for the flow viewer.

    python examples/export_traces.py        -> examples/flow_viewer/traces.json

Each step records what the node received, what it returned, the documents its
tools found, and every model call it made, including the full prompt.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mock_data as m
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from scenarios import SCENARIOS, Scenario, tool_for

from account_research_agent import nodes as nodes_module
from account_research_agent.llm import Message, ModelRouter, ProviderUnavailableError
from account_research_agent.nodes import Deps
from account_research_agent.orchestrator import checkpoint_serializer, compile_app, rerun_seed, run_config
from account_research_agent.salesforce_sync import MappingError, build_composite
from account_research_agent.schemas import (
    AccountInput,
    Branch,
    KBPassage,
    ResearchDepth,
    ResearchResult,
    ResearchState,
    SourceDocument,
)

OUT = Path(__file__).parent / "flow_viewer" / "traces.json"
LOCAL = threading.local()  # the step currently running on this thread


def plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


@dataclass
class Recorder:
    steps: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def begin(self, node: str, label: str, node_input: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            step = {
                "seq": len(self.steps) + 1,
                "node": node,
                "label": label,
                "input": plain(node_input),
                "output": None,
                "error": None,
                "tools": [],
                "llm": [],
            }
            self.steps.append(step)
        LOCAL.step = step
        return step


REC = Recorder()


def curated_input(name: str, state: Any) -> dict[str, Any]:
    if name == "research_branch":
        return {
            "branch": state["branch"],
            "depth": state["depth"],
            "attempt": state["attempt"],
            "account": state["account"].name,
        }
    claims, people = state.get("claims", []), state.get("stakeholders", [])
    if name == "prioritize":
        return {"account": state["account"], "force_depth": state.get("force_depth")}
    if name in ("validate", "research_judge", "score"):
        return {"claims": claims, "stakeholders": people, "research_retries": state.get("research_retries", 0)}
    if name == "synthesize":
        return {"verified_claims": [c for c in claims if c.status == "verified"]}
    if name == "personalize":
        return {
            "brief": state.get("brief"),
            "reviewer_feedback": state.get("reviewer_feedback", ""),
            "previous_draft": state.get("draft"),
            "judge_critique": state["delivery_verdict"].critique if state.get("delivery_verdict") else None,
            "draft_attempts": state.get("draft_attempts", 0),
        }
    if name == "delivery_judge":
        return {"draft": state.get("draft"), "draft_attempts": state.get("draft_attempts", 0)}
    return {"outcome": state.get("outcome"), "run_id": state.get("run_id"), "parent_run_id": state.get("parent_run_id")}


class TracingNodes(nodes_module.ResearchNodes):
    """Wraps every node to record its input, output and error. Example-only instrumentation."""


def _wrap(name: str) -> None:
    original = getattr(nodes_module.ResearchNodes, name)

    def traced(self: nodes_module.ResearchNodes, state: Any) -> Any:
        label = (
            f"{name} · {state['branch']}" + (f" (retry {state['attempt']})" if state["attempt"] else "")
            if name == "research_branch"
            else name
        )
        step = REC.begin(name, label, curated_input(name, state))
        try:
            update = original(self, state)
        except Exception as exc:
            step["error"] = f"{type(exc).__name__}: {exc}"
            raise
        step["output"] = plain(update)
        return update

    setattr(TracingNodes, name, traced)


for _name in (
    "prioritize",
    "research_branch",
    "validate",
    "research_judge",
    "score",
    "synthesize",
    "personalize",
    "delivery_judge",
    "finalize",
):
    _wrap(_name)
# Reason: build_graph instantiates nodes.ResearchNodes; point it at the traced subclass for this script only.
nodes_module.ResearchNodes = TracingNodes  # type: ignore[misc]


@dataclass
class ScenarioGateway:
    scenario: Scenario
    served: Counter[str] = field(default_factory=Counter)

    def complete_json(
        self, *, model: str, messages: list[Message], json_schema: dict[str, object]
    ) -> tuple[str, float]:
        title = str(json_schema["title"])
        call: dict[str, Any] = {
            "schema": title,
            "model": model,
            "messages": [{"role": x.role, "content": x.content} for x in messages],
        }
        LOCAL.step["llm"].append(call)
        if model in self.scenario.down.get(title, set()):
            call["status"] = "provider unavailable (HTTP 503)"
            raise ProviderUnavailableError(f"{model}: HTTP 503")
        answer = self._answer(title, messages)
        tokens_in, tokens_out = sum(len(x.content) for x in messages) // 4, len(answer) // 4
        price_in, price_out = m.PRICES[model]
        cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
        call |= {
            "status": "ok",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost, 6),
            "response": json.loads(answer),
        }
        return answer, cost

    def _answer(self, title: str, messages: list[Message]) -> str:
        s = self.scenario
        if title == "ExtractionResult":
            step = LOCAL.step
            branch, attempt = Branch(step["input"]["branch"]), step["input"]["attempt"]
            return json.dumps(plain(s.extraction.get((branch, attempt), {})))
        index = self.served[title]
        self.served[title] += 1
        source: dict[str, list[BaseModel]] = {
            "ResearchJudgeVerdict": list(s.research_verdicts),
            "AccountBrief": [s.brief] if s.brief else [],
            "OutreachDraft": list(s.drafts),
            "DeliveryJudgeVerdict": list(s.delivery_verdicts),
        }
        return source[title][index].model_dump_json()


@dataclass
class ScenarioTools:
    scenario: Scenario

    def gather(self, branch: Branch, account: AccountInput, depth: ResearchDepth, attempt: int) -> list[SourceDocument]:
        found = self.scenario.documents.get((branch, attempt), [])
        if isinstance(found, Exception):
            LOCAL.step["tools"].append({"tool": "brave_news_search", "error": str(found)})
            raise found
        for d in found:
            LOCAL.step["tools"].append({"tool": tool_for(d), **plain(d)})
        return found


class MockKB:
    def retrieve(self, query: str, top_k: int = 5) -> list[KBPassage]:
        LOCAL.step["tools"].append({"tool": "kb_retrieve", "query": query[:200], "passages": plain(m.KB[:top_k])})
        return m.KB[:top_k]


@dataclass
class Sink:
    results: list[ResearchResult] = field(default_factory=list)

    def reserve_rerun(self, salesforce_account_id: str, daily_limit: int) -> bool:
        return True

    def save_run_with_outbox(self, result: ResearchResult) -> None:
        self.results.append(result)
        LOCAL.step["tools"].append(
            {"tool": "outbox_enqueue", "note": "MongoDB transaction: research_runs + salesforce_outbox"}
        )

    def post_review(self, result: ResearchResult) -> None:
        LOCAL.step["tools"].append({"tool": "slack_post_review", "note": f"review card for {result.run_id}"})


def run_scenario(
    s: Scenario, apps: dict[str, tuple[CompiledStateGraph[ResearchState], ScenarioGateway, ScenarioTools, Sink]]
) -> dict[str, Any]:
    if s.parent:
        app, gateway, tools, sink = apps[s.parent]
        gateway.scenario, tools.scenario = s, s
        gateway.served.clear()
    else:
        gateway, tools, sink = ScenarioGateway(s), ScenarioTools(s), Sink()
        deps = Deps(router=ModelRouter(gateway), tools=tools, kb=MockKB(), repo=sink, review=sink, now=lambda: m.NOW)
        app = compile_app(deps, InMemorySaver(serde=checkpoint_serializer()))
        apps[s.id] = (app, gateway, tools, sink)

    REC.steps = []
    run_id = f"run-{s.id}"
    if s.rerun:
        deps = Deps(router=ModelRouter(gateway), tools=tools, kb=MockKB(), repo=sink, review=sink, now=lambda: m.NOW)
        start = rerun_seed(app, deps, s.rerun, run_id)
    else:
        start = {"run_id": run_id, "parent_run_id": None, "account": s.account, "entry": "prioritize"}
    app.invoke(start, run_config(run_id))

    result = sink.results[-1]
    try:
        salesforce: Any = build_composite(result)[0]
    except MappingError as exc:
        salesforce = {"mapping_error": str(exc)}
    steps = sorted(REC.steps, key=lambda st: st["seq"])
    cost = sum(c.get("cost_usd", 0) for st in steps for c in st["llm"])
    return {
        "id": s.id,
        "title": s.title,
        "shows": s.shows,
        "account": plain(s.account),
        "run_id": run_id,
        "parent_run_id": result.parent_run_id,
        "rerun_request": plain(s.rerun),
        "steps": steps,
        "result": plain(result),
        "salesforce_request": plain(salesforce),
        "llm_cost_usd": round(cost, 6),
        "llm_calls": sum(1 for st in steps for c in st["llm"] if c["status"] == "ok"),
        "tool_calls": sum(
            1 for st in steps for t in st["tools"] if t["tool"] not in ("outbox_enqueue", "slack_post_review")
        ),
    }


def main() -> None:
    apps: dict[str, Any] = {}
    traces = [run_scenario(s, apps) for s in SCENARIOS]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"generated_for": "mock data, fictional accounts", "scenarios": traces}, indent=1))
    for t in traces:
        print(
            f"{t['id']:<20} {t['result']['outcome']:<22} steps={len(t['steps']):<3} llm={t['llm_calls']:<2} tools={t['tool_calls']:<2} ${t['llm_cost_usd']:.5f}"
        )


if __name__ == "__main__":
    main()
