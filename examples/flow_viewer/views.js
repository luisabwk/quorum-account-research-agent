// Inspector content: a plain-language summary and the tabs for the selected step.
window.FlowViews = (() => {
  const A = () => window.FlowApp;
  const esc = (v) => A().esc(v);
  const CLASS = { 1: "official government", 2: "official company", 3: "trusted publication", 4: "enrichment provider", 5: "unknown source" };
  const STATUS = { verified: "s-ok", needs_review: "s-warn", rejected: "s-bad" };
  const pill = (s) => s ? `<span class="pill ${STATUS[s] || "s-idle"}">${esc(s.replace("_", " "))}</span>` : "";
  const cls = (n) => `<span class="cls" title="Source class ${n}: ${CLASS[n]}">class ${n} · ${CLASS[n]}</span>`;
  const day = (iso) => (iso || "").slice(0, 10) || "no date";
  const json = (v) => `<pre>${esc(JSON.stringify(v, null, 2))}</pre>`;
  const block = (title, html) => html ? `<section class="block"><h3>${esc(title)}</h3>${html}</section>` : "";
  const reasons = (list, warn) => list?.length ? `<ul class="reasons ${warn ? "warn" : ""}">${list.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : "";

  function claimCard(c) {
    return `<div class="item"><div class="top"><span class="id">${esc(c.claim_id)}</span>${pill(c.status)}</div>
      <div>${esc(c.statement)}</div><q>${esc(c.evidence_excerpt)}</q>
      <div class="top"><span class="mono muted">${day(c.published_at)} · relevance ${c.relevance ?? "–"}</span>${cls(c.source_class)}</div>
      ${reasons(c.rejection_reasons, c.status === "needs_review")}</div>`;
  }
  function personCard(p) {
    return `<div class="item"><div class="top"><strong>${esc(p.full_name)}</strong>${pill(p.status)}</div>
      <div class="muted">${esc(p.title)}${p.email ? ` · <span class="mono">${esc(p.email)}</span>` : ""}</div>
      <div class="top"><span class="mono muted">${p.company_domain_match ? "domain matches account" : "domain does not match"} · verified ${day(p.last_verified_at)}</span>${cls(p.source_class)}</div>
      ${reasons(p.rejection_reasons)}</div>`;
  }
  function docCard(d) {
    if (d.error) return `<div class="doc"><div class="top"><span class="tool">${esc(d.tool)}</span><span class="pill s-bad">error</span></div><div>${esc(d.error)}</div></div>`;
    if (d.passages) return d.passages.map((p) => `<div class="doc"><div class="top"><span class="tool">kb_retrieve</span><span class="mono muted">${esc(p.kb_id)} v${p.version}</span></div><div class="title">${esc(p.title)}</div><blockquote>${esc(p.text)}</blockquote></div>`).join("");
    if (d.note) return `<div class="doc"><div class="top"><span class="tool">${esc(d.tool)}</span></div><div class="muted">${esc(d.note)}</div></div>`;
    return `<div class="doc"><div class="top"><span class="tool">${esc(d.tool)}</span>${cls(d.source_class)}</div>
      <div class="title">${esc(d.title)}</div><blockquote>${esc(d.text)}</blockquote>
      <div class="meta">${esc(d.url)}<br>published ${day(d.published_at)} · snapshot ${esc(d.snapshot_key)}</div></div>`;
  }
  function email(draft, run, invalid) {
    const people = run.steps.flatMap((s) => (s.output?.stakeholders || [])).concat(run.result.draft_stakeholder ? [run.result.draft_stakeholder] : []);
    const to = people.find((p) => p.stakeholder_id === draft.stakeholder_id);
    const chip = (id) => `<span class="${invalid.includes(id) ? "bad" : ""}">${esc(id)}</span>`;
    return `<div class="email"><div class="h"><span>To</span><span>${esc(to ? `${to.full_name}, ${to.title}` : draft.stakeholder_id)}</span><span>Subject</span><span>${esc(draft.subject)}</span></div>
      <div class="b">${esc(draft.body)}</div></div>
      <div class="cites">${draft.cited_claim_ids.map(chip).join("")}${(draft.cited_kb_ids || []).map(chip).join("")}</div>`;
  }
  const meter = (label, v, floor) => `<div class="meter"><span>${esc(label)}</span><span class="bar"><span class="fill" style="width:${Math.round(v * 100)}%"></span>${floor != null ? `<span class="mark" style="left:${floor * 100}%" title="floor ${floor}"></span>` : ""}</span><span class="v">${v}</span></div>`;
  const score5 = (label, v) => `<div class="meter"><span>${esc(label)}</span><span class="bar"><span class="fill" style="width:${v * 20}%"></span></span><span class="v">${v}/5</span></div>`;

  function summary(run, id, step, execs) {
    const out = step?.output || {};
    const res = run.result;
    if (id === "trigger") {
      const a = run.account;
      return run.parent_run_id
        ? `The SDR asked for a <strong>${esc(run.rerun_request.scope.replace("_", " "))}</strong> rerun of <span class="mono">${esc(run.parent_run_id)}</span>: “${esc(run.rerun_request.reviewer_feedback)}”`
        : `<strong>${esc(a.name)}</strong> · ICP fit ${a.icp_fit} · 6sense intent ${a.intent_score} · Qualified engagement ${a.engagement_score}${a.is_customer ? " · <strong>existing customer</strong>" : ""}${a.open_opportunity ? " · open opportunity" : ""}`;
    }
    if (id === "salesforce") return run.salesforce_request.mapping_error ? esc(run.salesforce_request.mapping_error) : `One Composite API call with allOrNone: Account summary fields${res.draft ? " plus the outreach draft record" : ""}. Claims and sources stay in MongoDB.`;
    if (!step) return run.parent_run_id ? "Not re-run: this rerun reuses the parent run's evidence from its checkpoint." : "This step did not run in this scenario.";
    if (step.error) return `Failed: ${esc(step.error)}`;
    if (id === "prioritize") return out.outcome === "skipped" ? `Priority ${out.execution_priority}. <strong>Skipped</strong>: existing customers go to Account Management, so no tool or model is called.` : `Priority ${out.execution_priority}, so research depth is <strong>${out.depth}</strong>.`;
    if (id.startsWith("research_branch")) {
      if (out.branch_errors?.length) return `No data: ${esc(out.branch_errors.join("; "))}. The run continues without this branch.`;
      const rej = (out.claims || []).filter((c) => c.status === "rejected").length;
      return `${step.tools.length} document${step.tools.length === 1 ? "" : "s"} found; the model extracted ${(out.claims || []).length} claims and ${(out.stakeholders || []).length} people.${rej ? ` <strong>${rej} rejected immediately</strong>: the quote is not in the source.` : ""}`;
    }
    if (id === "validate") {
      const all = (out.claims || []).concat(out.stakeholders || []);
      const n = (s) => all.filter((i) => i.status === s).length;
      return `<strong>${n("verified")}</strong> verified, <strong>${n("needs_review")}</strong> kept for human review, <strong>${n("rejected")}</strong> rejected. Only verified evidence can reach the email.`;
    }
    if (id === "research_judge") {
      const retry = out.branches_to_retry?.length ? ` Re-researching <strong>${out.branches_to_retry.join(", ")}</strong> once.` : "";
      return step.llm.length ? `The judge rejected ${countJudged(out)} item(s) the evidence does not support.${retry}` : `Skipped: nothing was verified, so there is nothing to audit.${retry}`;
    }
    if (id === "score") return out.outcome ? `Evidence confidence ${out.confidence_score} is below the 0.3 floor or no stakeholder is verified. <strong>Sent to a human</strong> instead of drafting.` : `Account priority ${out.priority_score} · evidence confidence ${out.confidence_score}. Good enough to draft.`;
    if (id === "synthesize") return `Angle: “${esc(out.brief?.outreach_angle)}”`;
    if (id === "personalize") {
      const failed = step.llm.filter((c) => c.status !== "ok").map((c) => c.model);
      return `Draft ${execs.indexOf(step) + 1} written by <span class="mono">${esc(step.llm.find((c) => c.status === "ok")?.model)}</span>${failed.length ? ` after <span class="mono">${esc(failed.join(", "))}</span> was unavailable` : ""}.`;
    }
    if (id === "delivery_judge") {
      const v = out.delivery_verdict;
      if (!step.llm.length) return `<strong>Failed without a model call</strong>: ${esc(v.critique)}`;
      return `${v.groundedness >= 4 && v.approved_capabilities_only && v.relevance >= 3 && v.tone_and_length >= 3 ? "<strong>Passed</strong>" : "<strong>Failed</strong>"} (judge <span class="mono">${esc(step.llm.find((c) => c.status === "ok")?.model)}</span>). ${esc(v.critique)}`;
    }
    if (id === "finalize") return `Saved as <strong>${esc(res.outcome.replaceAll("_", " "))}</strong>, queued for Salesforce and posted to Slack for SDR review.`;
    return "";
  }
  const countJudged = (out) => (out.claims || []).concat(out.stakeholders || []).filter((i) => (i.rejection_reasons || []).some((r) => r.startsWith("research judge"))).length;

  function dataTab(run, id, step) {
    const out = step?.output || {}, inp = step?.input || {};
    if (id === "trigger") return run.parent_run_id ? json(run.rerun_request) : json(run.account);
    if (id === "salesforce") {
      const req = run.salesforce_request;
      if (req.mapping_error) return json(req);
      return req.compositeRequest.map((r) => block(`${r.method} ${r.referenceId}`, `<div class="mono muted">${esc(r.url)}</div><dl class="kv">${Object.entries(r.body).map(([k, v]) => `<dt class="mono">${esc(k)}</dt><dd>${v === null ? '<span class="muted">null (clears old value)</span>' : esc(v)}</dd>`).join("")}</dl>`)).join("");
    }
    if (!step) return "";
    if (id === "prioritize") return `<dl class="kv"><dt>execution priority</dt><dd>${out.execution_priority}</dd><dt>depth</dt><dd>${out.depth}</dd><dt>outcome</dt><dd>${out.outcome || "continue"}</dd></dl><p class="muted">0.4 × ICP fit + 0.3 × intent/100 + 0.3 × engagement/100. Below 0.35 skip, from 0.65 deep.</p>`;
    if (id.startsWith("research_branch")) {
      return block("Found by the tools", step.tools.map(docCard).join("") || '<p class="muted">No documents.</p>')
        + block("Extracted claims", (out.claims || []).map(claimCard).join(""))
        + block("Extracted people", (out.stakeholders || []).map(personCard).join(""))
        + block("Errors", (out.branch_errors || []).map((e) => `<div class="item">${esc(e)}</div>`).join(""));
    }
    if (["validate", "research_judge", "score"].includes(id)) {
      const claims = out.claims || inp.claims || [], people = out.stakeholders || inp.stakeholders || [];
      const scores = id === "score" ? meter("Priority", out.priority_score) + meter("Confidence", out.confidence_score, 0.3) : "";
      return scores + block("Claims", claims.map(claimCard).join("")) + block("People", people.map(personCard).join(""));
    }
    if (id === "synthesize") {
      const b = out.brief;
      return `<dl class="kv"><dt>summary</dt><dd>${esc(b.summary)}</dd><dt>hypothesis</dt><dd>${esc(b.commercial_hypothesis)}</dd><dt>angle</dt><dd>${esc(b.outreach_angle)}</dd><dt>cites</dt><dd class="mono">${esc(b.cited_claim_ids.join(", "))}</dd></dl>`
        + block("Approved Quorum knowledge retrieved", step.tools.map(docCard).join(""));
    }
    if (id === "personalize") {
      if (!out.draft) return json(out);
      const valid = run.steps.flatMap((s) => (s.output?.claims || []).filter((c) => c.status === "verified").map((c) => c.claim_id));
      const kb = run.steps.flatMap((s) => s.tools.flatMap((t) => (t.passages || []).map((p) => p.kb_id)));
      const invalid = [...out.draft.cited_claim_ids, ...(out.draft.cited_kb_ids || [])].filter((x) => !valid.includes(x) && !kb.includes(x));
      return (inp.reviewer_feedback ? block("SDR feedback in the prompt", `<div class="item">${esc(inp.reviewer_feedback)}</div>`) : "")
        + (inp.judge_critique ? block("Judge critique from the previous draft", `<div class="item">${esc(inp.judge_critique)}</div>`) : "")
        + block("Draft", email(out.draft, run, invalid));
    }
    if (id === "delivery_judge") {
      const v = out.delivery_verdict;
      return score5("Groundedness", v.groundedness) + score5("Relevance", v.relevance) + score5("Tone & length", v.tone_and_length)
        + `<dl class="kv"><dt>approved KB only</dt><dd>${v.approved_capabilities_only}</dd><dt>critique</dt><dd>${esc(v.critique)}</dd><dt>next</dt><dd>${esc(out.outcome || "rewrite")}</dd></dl>`;
    }
    if (id === "finalize") return block("Saved and posted", step.tools.map(docCard).join("")) + json(run.result);
    return json(out);
  }

  function callsTab(step) {
    if (!step || !step.llm.length) return '<p class="muted">No model call in this step. This part is deterministic code.</p>';
    return step.llm.map((c) => `<details class="call"><summary><span class="model">${esc(c.model)}</span>
        <span class="pill ${c.status === "ok" ? "s-ok" : "s-bad"}">${esc(c.status)}</span>
        <span class="nums">${c.status === "ok" ? `${c.tokens_in} in · ${c.tokens_out} out · ${A().money(c.cost_usd)}` : ""}</span></summary>
      <div class="body"><span class="msg-role">schema</span><span class="mono">${esc(c.schema)}</span>
      ${c.messages.map((mm) => `<span class="msg-role">${esc(mm.role)}</span><pre>${esc(mm.content)}</pre>`).join("")}
      ${c.response ? `<span class="msg-role">response</span>${json(c.response)}` : ""}</div></details>`).join("")
      + '<p class="muted">Token counts are estimates (4 characters per token) priced with the OpenRouter snapshot.</p>';
  }

  function tab(name, run, id, step) {
    if (name === "model calls") return callsTab(step);
    if (name === "input") return json(step.input);
    if (name === "output") return json(step.output ?? { error: step.error });
    return dataTab(run, id, step);
  }

  return { summary, tab };
})();
