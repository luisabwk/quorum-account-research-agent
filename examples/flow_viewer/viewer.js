(() => {
  const DATA = JSON.parse(document.getElementById("traces").textContent).scenarios;
  const $ = (sel) => document.querySelector(sel);
  const esc = (v) => String(v ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const money = (v) => "$" + Number(v || 0).toFixed(5);
  const W = 150, H = 80;

  // Node layout: row 1 is research, row 2 is writing. Positions are the top-left corner.
  const NODES = [
    { id: "trigger", x: 20, y: 190, kind: "Trigger", name: "Salesforce account", what: "The account enters the queue with its CRM, 6sense and Qualified fields." },
    { id: "prioritize", x: 210, y: 190, kind: "Rule · D1", name: "Prioritize", what: "Deterministic priority score. Sets research depth: skip, light or deep." },
    { id: "research_branch:regulatory", x: 400, y: 70, kind: "Tools + LLM", name: "Regulatory research", what: "LDA filings, Congress.gov, Federal Register, SEC EDGAR. A cheap model extracts claims; code checks every quote against the source." },
    { id: "research_branch:stakeholder", x: 400, y: 190, kind: "Tools + LLM", name: "Stakeholder research", what: "Salesforce contacts, company pages, ZoomInfo when Salesforce has gaps." },
    { id: "research_branch:news", x: 400, y: 310, kind: "Tools + LLM", name: "News & advocacy", what: "Brave news search and full articles, for recent public-affairs activity." },
    { id: "validate", x: 590, y: 190, kind: "Rule · D2", name: "Validate evidence", what: "Freshness, source class, provenance, relevance and duplicates. No model involved." },
    { id: "research_judge", x: 780, y: 190, kind: "LLM judge", name: "Research judge", what: "Does each excerpt support its claim? Is each person at this company? Incomplete branches are re-researched once." },
    { id: "score", x: 970, y: 190, kind: "Rule · D3", name: "Score", what: "Account priority and evidence confidence, from verified evidence only. Low confidence goes to a human." },
    { id: "synthesize", x: 210, y: 470, kind: "LLM + KB", name: "Synthesize brief", what: "Account brief, commercial hypothesis and outreach angle, grounded in verified claims and the approved Quorum KB." },
    { id: "personalize", x: 400, y: 470, kind: "LLM", name: "Personalize draft", what: "Writes the email. Must cite a claim id for every fact and a KB id for every Quorum capability." },
    { id: "delivery_judge", x: 590, y: 470, kind: "LLM judge", name: "Delivery judge", what: "Scores groundedness, approved capabilities, relevance and tone. A model from a different family than the writer." },
    { id: "finalize", x: 780, y: 470, kind: "Write", name: "Save & review", what: "Result and Salesforce outbox entry in one MongoDB transaction, then a Slack review card." },
    { id: "salesforce", x: 970, y: 470, kind: "Outbox", name: "Salesforce update", what: "The outbox worker sends one Composite API request: Account summary fields plus the draft record." },
  ];
  const BY_ID = Object.fromEntries(NODES.map((n) => [n.id, n]));
  const BRANCHES = ["regulatory", "stakeholder", "news"];
  const port = (id, side) => {
    const n = BY_ID[id];
    return { top: [n.x + W / 2, n.y], bottom: [n.x + W / 2, n.y + H], left: [n.x, n.y + H / 2], right: [n.x + W, n.y + H / 2] }[side];
  };
  const line = (a, b) => `M${a[0]} ${a[1]} C${a[0] + 60} ${a[1]}, ${b[0] - 60} ${b[1]}, ${b[0]} ${b[1]}`;

  const state = { run: DATA[0], nodeId: "trigger", exec: 0, tab: "data" };

  function nodeIdOf(step) {
    return step.node === "research_branch" ? `research_branch:${step.input.branch}` : step.node;
  }
  function execsOf(run, nodeId) {
    return run.steps.filter((s) => nodeIdOf(s) === nodeId);
  }

  function edgesFor(run) {
    const ran = (id) => id === "trigger" || id === "salesforce" ? true : execsOf(run, id).length > 0;
    const first = run.steps[0] ? nodeIdOf(run.steps[0]) : "prioritize";
    const retried = run.steps.some((s) => s.node === "research_branch" && s.input.attempt > 0);
    const rewrites = execsOf(run, "personalize").length > 1;
    const e = [];
    if (first === "prioritize") e.push({ d: line(port("trigger", "right"), port("prioritize", "left")), on: true });
    else e.push({ d: `M${port("trigger", "bottom")[0]} ${port("trigger", "bottom")[1]} C95 420, 475 150, ${port(first, "left")[0] - 4} ${port(first, "left")[1]}`, on: true, label: ["SDR rerun from Slack", 40, 440] });
    BRANCHES.forEach((b) => {
      const id = `research_branch:${b}`;
      const fromPrior = execsOf(run, id).some((s) => s.input.attempt === 0) && ran("prioritize");
      e.push({ d: line(port("prioritize", "right"), port(id, "left")), on: fromPrior });
      e.push({ d: line(port(id, "right"), port("validate", "left")), on: ran(id) });
    });
    e.push({ d: line(port("validate", "right"), port("research_judge", "left")), on: ran("research_judge") });
    e.push({ d: line(port("research_judge", "right"), port("score", "left")), on: ran("score") });
    e.push({ d: `M855 190 C855 20, 475 10, 475 70`, on: retried, label: ["re-research once", 600, 36] });
    const [sx, sy] = port("score", "bottom"), [tx, ty] = port("synthesize", "top");
    e.push({ d: `M${sx} ${sy} C${sx} 400, ${tx} 360, ${tx} ${ty}`, on: ran("synthesize") });
    e.push({ d: line(port("synthesize", "right"), port("personalize", "left")), on: ran("personalize") });
    e.push({ d: line(port("personalize", "right"), port("delivery_judge", "left")), on: ran("delivery_judge") });
    e.push({ d: `M665 550 C665 640, 475 640, 475 550`, on: rewrites, label: ["rewrite once with critique", 480, 650] });
    e.push({ d: line(port("delivery_judge", "right"), port("finalize", "left")), on: ran("delivery_judge") && ran("finalize") });
    const skipped = ran("prioritize") && !BRANCHES.some((b) => ran(`research_branch:${b}`));
    e.push({ d: `M285 270 C285 380, 830 360, 850 470`, on: skipped, label: ["skip", 330, 360] });
    const human = ran("score") && !ran("synthesize");
    e.push({ d: `M1030 270 C1030 400, 870 380, 870 470`, on: human, label: ["needs human", 960, 440] });
    e.push({ d: line(port("finalize", "right"), port("salesforce", "left")), on: ran("finalize") });
    return e;
  }

  function nodeFacts(run, id) {
    const ex = execsOf(run, id);
    const last = ex[ex.length - 1];
    const res = run.result;
    if (id === "trigger") return { cls: "ran", d: `${run.account.name}${run.parent_run_id ? " · rerun" : ""}` };
    if (id === "salesforce") return run.salesforce_request.mapping_error ? { cls: "bad", d: "mapping error" } : { cls: "ran", d: `${run.salesforce_request.compositeRequest.length} records queued` };
    if (!last) return { cls: "idle", d: run.parent_run_id && ["prioritize", "research_branch:regulatory", "research_branch:news"].includes(id) ? "reused from parent run" : "did not run" };
    const out = last.output || {};
    if (last.error) return { cls: "bad", d: "error" };
    if (id === "prioritize") return { cls: out.outcome === "skipped" ? "warn" : "ran", d: `priority ${out.execution_priority} · ${out.depth}` };
    if (id.startsWith("research_branch")) {
      if (out.branch_errors?.length) return { cls: "warn", d: out.branch_errors[0].split(": ").slice(1).join(": ").slice(0, 42) };
      const n = (out.claims || []).length + (out.stakeholders || []).length;
      return { cls: "ran", d: `${last.tools.length} docs → ${n} items` };
    }
    if (id === "validate") {
      const c = count([...(out.claims || []), ...(out.stakeholders || [])]);
      return { cls: "ran", d: `${c.verified || 0} ok · ${c.needs_review || 0} review · ${c.rejected || 0} rejected` };
    }
    if (id === "research_judge") return last.llm.length ? { cls: "ran", d: `${judged(out)} rejected${out.branches_to_retry?.length ? " · retry" : ""}` } : { cls: "warn", d: `skipped: nothing to audit${out.branches_to_retry?.length ? " · retry" : ""}` };
    if (id === "score") return { cls: out.outcome ? "warn" : "ran", d: `conf. ${out.confidence_score}${out.outcome ? " · to human" : ""}` };
    if (id === "synthesize") return { cls: "ran", d: "brief written" };
    if (id === "personalize") return { cls: "ran", d: `${ex.length} draft${ex.length > 1 ? "s" : ""} · ${modelOk(last)}` };
    if (id === "delivery_judge") return { cls: out.outcome === "ready_for_review" ? "good" : out.outcome ? "bad" : "warn", d: out.outcome === "ready_for_review" ? "passed" : out.outcome ? "failed twice" : "failed, rewrite" };
    if (id === "finalize") return { cls: res.outcome === "ready_for_review" ? "good" : res.outcome === "skipped" ? "ran" : "warn", d: res.outcome.replaceAll("_", " ") };
    return { cls: "ran", d: "" };
  }
  const count = (items) => items.reduce((acc, i) => ((acc[i.status] = (acc[i.status] || 0) + 1), acc), {});
  const judged = (out) => (out.claims || []).concat(out.stakeholders || []).filter((i) => (i.rejection_reasons || []).some((r) => r.startsWith("research judge"))).length;
  const modelOk = (step) => (step.llm.find((c) => c.status === "ok")?.model || "").split("/")[1] || "no model call";

  function renderRuns() {
    $("#runs").innerHTML = '<h2>Runs</h2>' + DATA.map((r) => `
      <button class="run" data-run="${r.id}" aria-current="${r === state.run}">
        <span class="t">${esc(r.title)}</span>
        <span class="a">${esc(r.account.name)}</span>
        ${outcomePill(r.result.outcome)}
        <span class="m">${r.steps.length} steps · ${r.llm_calls} model · ${r.tool_calls} tool · ${money(r.llm_cost_usd)}</span>
      </button>`).join("");
  }
  function outcomePill(o) {
    const cls = { ready_for_review: "s-ok", skipped: "s-idle", needs_human_research: "s-warn", low_quality_draft: "s-bad" }[o] || "s-info";
    return `<span class="pill ${cls}">${esc(o.replaceAll("_", " "))}</span>`;
  }

  function renderHead() {
    const r = state.run;
    $("#scenario").innerHTML = `
      <div class="row"><span class="t">${esc(r.title)}</span>${outcomePill(r.result.outcome)}
      ${r.parent_run_id ? `<span class="chip">rerun of ${esc(r.parent_run_id)}</span>` : ""}
      <span class="mono muted">${esc(r.run_id)}</span></div><p>${esc(r.shows)}</p>`;
  }

  function renderCanvas() {
    const r = state.run;
    const svg = edgesFor(r).map((e) => `<path class="edge ${e.on ? "on" : ""}" d="${e.d}"/>` +
      (e.label ? `<text class="edge-label ${e.on ? "on" : ""}" x="${e.label[1]}" y="${e.label[2]}">${esc(e.label[0])}</text>` : "")).join("");
    const nodes = NODES.map((n) => {
      const f = nodeFacts(r, n.id);
      const times = execsOf(r, n.id).length;
      return `<button class="node ${f.cls}" style="left:${n.x}px;top:${n.y}px" data-node="${n.id}" aria-pressed="${state.nodeId === n.id}">
        <span class="k"><span>${esc(n.kind)}</span></span><span class="n">${esc(n.name)}</span><span class="d">${esc(f.d)}</span>
        ${times > 1 ? `<span class="x">×${times}</span>` : ""}</button>`;
    }).join("");
    $("#stage").innerHTML = `<svg width="1140" height="670" viewBox="0 0 1140 670" aria-hidden="true">${svg}</svg>${nodes}`;
    $("#stage").style.width = "1140px";
    $("#stage").style.height = "670px";
    fit();
  }
  function fit() {
    const wrap = $("#canvas"), stage = $("#stage");
    const mode = document.querySelector(".zoom [aria-pressed=true]")?.dataset.zoom || "fit";
    const scale = mode === "fit" ? Math.max(0.72, Math.min(1, (wrap.clientWidth - 30) / 1140)) : 1;
    stage.style.transform = `scale(${scale})`;
    stage.style.marginBottom = `${-670 * (1 - scale)}px`;
    stage.style.marginRight = `${-1140 * (1 - scale)}px`;
  }

  function renderTimeline() {
    $("#timeline").innerHTML = '<span class="panel-label">Execution order</span>' + state.run.steps.map((s) => {
      const id = nodeIdOf(s);
      const idx = execsOf(state.run, id).indexOf(s);
      const on = state.nodeId === id && state.exec === idx;
      return `<button class="tick ${s.error ? "err" : ""}" data-node="${id}" data-exec="${idx}" aria-pressed="${on}"><b>${s.seq}</b>${esc(s.label)}</button>`;
    }).join("");
  }

  function renderInspector() {
    const r = state.run, n = BY_ID[state.nodeId];
    const ex = execsOf(r, n.id);
    const step = ex[Math.min(state.exec, ex.length - 1)];
    const tabs = step ? ["data", "model calls", "input", "output"] : ["data"];
    if (!tabs.includes(state.tab)) state.tab = "data";
    const execButtons = ex.length > 1 ? `<div class="execs">${ex.map((s, i) => `<button class="tick" data-exec="${i}" aria-pressed="${i === state.exec}"><b>${s.seq}</b>run ${i + 1}</button>`).join("")}</div>` : "";
    const calls = step ? step.llm.length : 0;
    $("#inspector").innerHTML = `
      <div><span class="panel-label">${esc(n.kind)}</span><h2>${esc(n.name)}</h2></div>
      <p class="what">${esc(n.what)}</p>
      <div class="summary">${window.FlowViews.summary(r, n.id, step, ex)}</div>
      ${execButtons}
      <div class="tabs" role="tablist">${tabs.map((t) => `<button class="tab" role="tab" data-tab="${t}" aria-selected="${t === state.tab}">${t === "model calls" ? `model calls (${calls})` : t}</button>`).join("")}</div>
      <div>${window.FlowViews.tab(state.tab, r, n.id, step)}</div>`;
  }

  function render() { renderRuns(); renderHead(); renderCanvas(); renderTimeline(); renderInspector(); }

  document.addEventListener("click", (ev) => {
    const t = ev.target.closest("button");
    if (!t) return;
    if (t.dataset.run) { state.run = DATA.find((r) => r.id === t.dataset.run); state.nodeId = "trigger"; state.exec = 0; state.tab = "data"; render(); return; }
    if (t.dataset.zoom) { document.querySelectorAll(".zoom button").forEach((b) => b.setAttribute("aria-pressed", String(b === t))); fit(); return; }
    if (t.dataset.tab) { state.tab = t.dataset.tab; renderInspector(); return; }
    if (t.dataset.node) { state.nodeId = t.dataset.node; state.exec = Number(t.dataset.exec || 0); renderCanvas(); renderTimeline(); renderInspector(); return; }
    if (t.dataset.exec) { state.exec = Number(t.dataset.exec); renderTimeline(); renderInspector(); }
  });
  window.addEventListener("resize", fit);
  window.FlowApp = { state, execsOf, nodeIdOf, esc, money, outcomePill, render };
  document.addEventListener("DOMContentLoaded", render);
  if (document.readyState !== "loading") render();
})();
