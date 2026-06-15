const state = {
  runId: null,
  eventSource: null,
  steps: [],
  graph: null,
  report: null,
  streaming: null,
  reportStream: "",
  reportStreamEl: null,
  streamUpdateTimer: null,
  reportStreamTimer: null,
  traceAutoFollow: true,
  tracePointerInside: false,
  traceProgrammaticScroll: false,
  activeIds: new Set(),
  lockedIds: new Set(),
  latestAddedIds: new Set(),
  latestGraphTurn: null,
  graphView: { scale: 1, tx: 0, ty: 0, dragging: false, lastX: 0, lastY: 0 },
  done: false,
  sampleFile: "merchant_fee_transactions.csv",
  sampleCases: [],
};

const els = {
  query: document.getElementById("queryInput"),
  fileInput: document.getElementById("fileInput"),
  fileSummary: document.getElementById("fileSummary"),
  sampleToggle: document.getElementById("sampleToggle"),
  dataMode: document.getElementById("dataMode"),
  runButton: document.getElementById("runButton"),
  loadZhFinanceSample: document.getElementById("loadZhFinanceSample"),
  loadNvidiaSample: document.getElementById("loadNvidiaSample"),
  loadTencentSample: document.getElementById("loadTencentSample"),
  traceList: document.getElementById("traceList"),
  turnCounter: document.getElementById("turnCounter"),
  graphSvg: document.getElementById("graphSvg"),
  zoomOutButton: document.getElementById("zoomOutButton"),
  zoomResetButton: document.getElementById("zoomResetButton"),
  zoomInButton: document.getElementById("zoomInButton"),
  graphStats: document.getElementById("graphStats"),
  nodeInspector: document.getElementById("nodeInspector"),
  reportPanel: document.getElementById("reportPanel"),
  reportTitle: document.getElementById("reportTitle"),
  reportBody: document.getElementById("reportBody"),
  claimCounter: document.getElementById("claimCounter"),
  modelPill: document.getElementById("modelPill"),
  endpointPill: document.getElementById("endpointPill"),
  statusPill: document.getElementById("statusPill"),
  tooltip: document.getElementById("tooltip"),
  stageItems: Array.from(document.querySelectorAll(".stage-item")),
};

const SVG_NS = "http://www.w3.org/2000/svg";

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function shortText(value, limit = 72) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length > limit ? `${text.slice(0, limit - 1)}...` : text;
}

function setStatus(text, mode = "idle") {
  els.statusPill.textContent = text;
  els.statusPill.classList.toggle("running", mode === "running");
  els.statusPill.classList.toggle("error", mode === "error");
  els.statusPill.classList.toggle("done", mode === "done");
  document.body.dataset.runState = mode;
}

function setStage(stage) {
  const indexByStage = { setup: 0, react: 1, graph: 2, report: 3, done: 3 };
  const activeIndex = indexByStage[stage] ?? 0;
  document.body.dataset.stage = stage;
  els.stageItems.forEach((item, index) => {
    item.classList.toggle("active", index === activeIndex);
    item.classList.toggle("complete", index < activeIndex);
  });
}

function updateFileSummary() {
  if (!els.fileSummary) return;
  const files = Array.from(els.fileInput.files || []).map((file) => file.name);
  const mode = els.dataMode?.value || "local";
  const fileOptional = mode === "web" || mode === "finance";
  if (els.sampleToggle) {
    els.sampleToggle.disabled = fileOptional;
    if (fileOptional) els.sampleToggle.checked = false;
  }
  if (files.length) {
    els.fileSummary.textContent = `${files.length} selected: ${files.slice(0, 2).join(", ")}${files.length > 2 ? `, +${files.length - 2} more` : ""}`;
  } else if (els.sampleToggle.checked) {
    els.fileSummary.textContent = `Sample ready: ${state.sampleFile}`;
  } else if (fileOptional) {
    els.fileSummary.textContent = mode === "web" ? "No file needed: web evidence tools enabled." : "No file needed: financial API tools enabled.";
  } else {
    els.fileSummary.textContent = "No data file selected.";
  }
}

function setText(el, value) {
  if (el && el.textContent !== String(value ?? "")) el.textContent = String(value ?? "");
}

function scheduleStreamingTraceUpdate() {
  if (state.streamUpdateTimer) return;
  state.streamUpdateTimer = window.setTimeout(() => {
    state.streamUpdateTimer = null;
    updateStreamingTrace();
  }, 32);
}

function scheduleReportStreamUpdate() {
  if (state.reportStreamTimer) return;
  state.reportStreamTimer = window.setTimeout(() => {
    state.reportStreamTimer = null;
    updateReportStream();
  }, 32);
}

function isTraceNearBottom(threshold = 90) {
  const remaining = els.traceList.scrollHeight - els.traceList.scrollTop - els.traceList.clientHeight;
  return remaining <= threshold;
}

function scrollTraceToBottom(force = false) {
  if (!force && (!state.traceAutoFollow || state.tracePointerInside)) return;
  state.traceProgrammaticScroll = true;
  els.traceList.scrollTop = els.traceList.scrollHeight;
  requestAnimationFrame(() => {
    state.traceProgrammaticScroll = false;
  });
}

function renderStatusNotice(data = {}) {
  if (state.steps.length || !els.traceList.classList.contains("empty-state")) return;
  const message = data.message || "running";
  const detail = data.detail || "The backend accepted the run and is waiting for the model's next action.";
  const turn = data.turn ? `Turn ${data.turn}` : "Run accepted";
  els.traceList.innerHTML = `
    <div class="empty-title">${escapeHtml(turn)} · ${escapeHtml(message)}</div>
    <div class="empty-copy">${escapeHtml(detail)}</div>`;
}

function showTooltip(content, event, options = {}) {
  if (!content) return;
  if (options.html) {
    els.tooltip.innerHTML = content;
  } else {
    els.tooltip.textContent = content;
  }
  els.tooltip.hidden = false;
  moveTooltip(event);
}

function moveTooltip(event) {
  if (els.tooltip.hidden || !event) return;
  const pad = 14;
  const rect = els.tooltip.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
  els.tooltip.style.left = `${Math.max(8, x)}px`;
  els.tooltip.style.top = `${Math.max(8, y)}px`;
}

function hideTooltip() {
  els.tooltip.hidden = true;
}

async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    els.query.value = cfg.default_query || "";
    state.sampleFile = cfg.sample_file || state.sampleFile;
    state.sampleCases = Array.isArray(cfg.sample_cases) ? cfg.sample_cases : [];
    els.modelPill.textContent = `model: ${cfg.model || "unknown"}`;
    els.endpointPill.textContent = `endpoint: ${cfg.base_url || "unknown"}`;
    if (cfg.mock) setStatus("mock mode", "running");
  } catch (err) {
    els.query.value = "Analyze the selected data file and produce an evidence-grounded report.";
  } finally {
    updateFileSummary();
  }
}

function loadSampleCase(caseId) {
  const sample = state.sampleCases.find((item) => item.id === caseId) || state.sampleCases[0];
  if (!sample) return;
  els.query.value = sample.query || "";
  if (els.dataMode && sample.mode) els.dataMode.value = sample.mode;
  if (els.sampleToggle) els.sampleToggle.checked = false;
  if (els.fileInput) els.fileInput.value = "";
  updateFileSummary();
}

function resetRun() {
  if (state.eventSource) state.eventSource.close();
  state.runId = null;
  state.eventSource = null;
  state.steps = [];
  state.graph = null;
  state.report = null;
  state.reportReceived = false;
  state.streaming = null;
  state.reportStream = "";
  state.reportStreamEl = null;
  if (state.streamUpdateTimer) window.clearTimeout(state.streamUpdateTimer);
  if (state.reportStreamTimer) window.clearTimeout(state.reportStreamTimer);
  state.streamUpdateTimer = null;
  state.reportStreamTimer = null;
  state.traceAutoFollow = true;
  state.tracePointerInside = false;
  state.traceProgrammaticScroll = false;
  state.activeIds = new Set();
  state.lockedIds = new Set();
  state.latestAddedIds = new Set();
  state.latestGraphTurn = null;
  resetGraphView();
  state.done = false;
  setStage("react");
  els.traceList.className = "trace-list empty-state";
  els.traceList.innerHTML = `<div class="empty-title">Run starting</div><div class="empty-copy">Waiting for the first model action.</div>`;
  els.turnCounter.textContent = "0 turns";
  els.reportPanel.hidden = true;
  els.reportBody.innerHTML = "";
  els.claimCounter.textContent = "0 final claims";
  renderGraph(null);
}

async function startRun() {
  resetRun();
  setStatus("starting", "running");
  updateFileSummary();
  els.runButton.disabled = true;

  const form = new FormData();
  form.append("query", els.query.value.trim());
  form.append("use_sample", els.sampleToggle.checked ? "1" : "0");
  form.append("data_mode", els.dataMode?.value || "local");
  for (const file of els.fileInput.files) form.append("files", file);

  try {
    const res = await fetch("/api/run", { method: "POST", body: form });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || "Run request failed.");
    state.runId = payload.run_id;
    try { localStorage.setItem("verigraph:lastRunId", payload.run_id); } catch (_) {}
    openEvents(payload.run_id);
  } catch (err) {
    setStatus("error", "error");
    els.runButton.disabled = false;
    renderError(err.message || String(err));
  }
}

function openEvents(runId) {
  const source = new EventSource(`/api/runs/${runId}/events`);
  state.eventSource = source;

  source.addEventListener("status", (event) => {
    const data = JSON.parse(event.data);
    setStatus(data.message || "running", "running");
    renderStatusNotice(data);
    if (data.message === "waiting for model" && state.streaming && (!data.turn || state.streaming.turn === data.turn)) {
      state.streaming.result = `Waiting for model stream... ${data.elapsed_seconds || 0}s elapsed, ${data.idle_seconds || 0}s idle`;
      scheduleStreamingTraceUpdate();
    }
    if (data.model) els.modelPill.textContent = `model: ${data.model}`;
    if (data.base_url) els.endpointPill.textContent = `endpoint: ${data.base_url}`;
  });

  source.addEventListener("assistant", () => {
    setStatus("thinking", "running");
  });

  source.addEventListener("assistant_start", (event) => {
    const data = JSON.parse(event.data);
    state.streaming = { turn: data.turn, assistant_text: "", thought: "", code: "", result: "Waiting for model tokens...", streaming: true };
    setStage("react");
    setStatus("streaming", "running");
    renderTrace();
  });

  source.addEventListener("assistant_delta", (event) => {
    const data = JSON.parse(event.data);
    if (!state.streaming || state.streaming.turn !== data.turn) {
      state.streaming = { turn: data.turn, assistant_text: "", result: "Waiting for code block...", streaming: true };
    }
    state.streaming.assistant_text = data.text || `${state.streaming.assistant_text || ""}${data.delta || ""}`;
    scheduleStreamingTraceUpdate();
  });

  source.addEventListener("assistant_reasoning_delta", (event) => {
    const data = JSON.parse(event.data);
    if (!state.streaming || state.streaming.turn !== data.turn) {
      state.streaming = { turn: data.turn, assistant_text: "", reasoning_summary: "", result: "Streaming reasoning...", streaming: true };
    }
    state.streaming.reasoning_summary = `${state.streaming.reasoning_summary || ""}${data.delta || ""}`;
    if (!state.streaming.thought) state.streaming.thought = state.streaming.reasoning_summary;
    scheduleStreamingTraceUpdate();
  });

  source.addEventListener("assistant_done", (event) => {
    const data = JSON.parse(event.data);
    if (!state.streaming || state.streaming.turn !== data.turn) {
      state.streaming = { turn: data.turn, assistant_text: data.assistant_text || "", streaming: true };
    }
    state.streaming.assistant_text = data.assistant_text || state.streaming.assistant_text || "";
    state.streaming.thought = data.thought || parseAssistantStream(state.streaming.assistant_text).thought;
    state.streaming.code = data.code || parseAssistantStream(state.streaming.assistant_text).code;
    state.streaming.result = "Executing code and updating graph...";
    renderTrace();
  });

  source.addEventListener("report_start", () => {
    state.reportStream = "";
    setStage("report");
    els.reportPanel.hidden = false;
    els.reportTitle.textContent = "Composing cited report";
    els.claimCounter.textContent = "streaming";
    els.reportBody.innerHTML = `
      <div class="report-loading">
        <div class="loading-line strong"></div>
        <div class="loading-line"></div>
        <div class="loading-line short"></div>
      </div>`;
    state.reportStreamEl = els.reportBody.querySelector(".report-loading");
  });

  source.addEventListener("report_delta", (event) => {
    const data = JSON.parse(event.data);
    state.reportStream = data.text || `${state.reportStream}${data.delta || ""}`;
    els.reportPanel.hidden = false;
    scheduleReportStreamUpdate();
  });

  source.addEventListener("step", (event) => {
    const step = JSON.parse(event.data);
    state.steps.push(step);
    if (state.streaming && state.streaming.turn === step.turn) state.streaming = null;
    state.graph = step.graph || state.graph;
    state.latestAddedIds = new Set((step.added_claim_ids || []).map(String));
    state.latestGraphTurn = step.turn || null;
    setStage("graph");
    setStatus(step.error ? "tool error" : "running", step.error ? "error" : "running");
    renderTrace();
    renderGraph(state.graph);
  });

  source.addEventListener("report", (event) => {
    const data = JSON.parse(event.data);
    state.report = data.report;
    state.reportReceived = true;
    setStage("report");
    state.reportStreamEl = null;
    state.graph = data.graph || state.graph;
    state.latestAddedIds = new Set();
    renderGraph(state.graph);
    renderReport(data.report, data.final_claims || [], data.output || null);
    setStatus("report ready", "running");
  });

  source.addEventListener("error", (event) => {
    if (event.data) {
      const data = JSON.parse(event.data);
      setStatus("error", "error");
      renderError(`${data.message || "Run failed."}\n\n${data.hint || ""}\n\n${data.traceback || ""}`);
    }
  });

  source.addEventListener("done", () => {
    state.done = true;
    source.close();
    els.runButton.disabled = false;
    if (!els.statusPill.classList.contains("error")) {
      setStage("done");
      setStatus("done", "done");
    }
    if (!state.reportReceived) {
      void recoverResultFromDisk(runId);
    }
  });

  source.onerror = () => {
    if (state.done || state.reportReceived) return;
    setStatus("reconnecting", "running");
    // SSE dropped before the report event arrived (proxy idle timeout, browser sleep, etc).
    // Poll the persisted result so the user still gets the report when the run finishes.
    pollResultUntilReady(runId);
  };
}

async function recoverResultFromDisk(runId) {
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/result`, { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();
    if (data && data.ready && data.report) {
      state.reportReceived = true;
      state.report = data.report;
      state.graph = data.graph || state.graph;
      state.latestAddedIds = new Set();
      renderGraph(state.graph);
      renderReport(data.report, data.final_claims || [], data.output || null);
      setStage("done");
      setStatus("done", "done");
    }
  } catch (err) {
    console.warn("recoverResultFromDisk failed", err);
  }
}

function pollResultUntilReady(runId, attempt = 0) {
  if (state.reportReceived) return;
  const delayMs = Math.min(15000, 3000 + attempt * 1500);
  setTimeout(async () => {
    if (state.reportReceived) return;
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/result`, { cache: "no-store" });
      if (res.ok) {
        const data = await res.json();
        if (data && data.ready && data.report) {
          state.reportReceived = true;
          state.report = data.report;
          state.graph = data.graph || state.graph;
          renderGraph(state.graph);
          renderReport(data.report, data.final_claims || [], data.output || null);
          setStage("done");
          setStatus("done (recovered)", "done");
          els.runButton.disabled = false;
          return;
        }
      }
    } catch (err) {
      // ignore and keep polling
    }
    if (attempt > 240) return; // ~hours, safety cap
    pollResultUntilReady(runId, attempt + 1);
  }, delayMs);
}

function renderError(message) {
  els.traceList.className = "trace-list";
  els.traceList.innerHTML = `<div class="error-box">${escapeHtml(message)}</div>`;
}

function tagContent(text, startTag, endTag) {
  const start = text.indexOf(startTag);
  if (start < 0) return "";
  const bodyStart = start + startTag.length;
  const end = text.indexOf(endTag, bodyStart);
  return (end >= 0 ? text.slice(bodyStart, end) : text.slice(bodyStart)).trim();
}

function parseAssistantStream(text) {
  const thought = tagContent(text, "<think>", "</think>");
  const code = tagContent(text, "<code_interpreter>", "</code_interpreter>");
  let result = "Streaming model output...";
  if (text.includes("<code_interpreter>") && !text.includes("</code_interpreter>")) result = "Streaming code block...";
  if (text.includes("</code_interpreter>")) result = "Waiting to execute code...";
  return { thought: thought || text, code, result };
}

function renderTrace() {
  const shouldFollow = state.traceAutoFollow && !state.tracePointerInside;
  els.traceList.className = "trace-list";
  const visibleSteps = [...state.steps];
  if (state.streaming) visibleSteps.push(state.streaming);
  els.turnCounter.textContent = `${visibleSteps.length} turn${visibleSteps.length === 1 ? "" : "s"}`;
  els.traceList.innerHTML = visibleSteps
    .map((step, index) => {
      const added = (step.added_claim_ids || [])
        .map((cid) => `<span class="claim-badge" data-claim="${escapeHtml(cid)}">${escapeHtml(cid)}</span>`)
        .join("");
      if (step.streaming) hydrateStreamingStep(step);
      const thought = step.thought || step.reasoning_summary || "No visible reasoning summary for this turn.";
      const status = step.streaming ? "Streaming" : step.error ? "Failed" : "Success";
      return `
        <article class="trace-step ${index === visibleSteps.length - 1 ? "active" : ""}" data-turn="${escapeHtml(step.turn)}">
          <div class="step-head">
            <div class="step-title"><span class="step-dot"></span>Turn ${escapeHtml(step.turn)} · ${status}</div>
            <div class="step-claims">${added}</div>
          </div>
          <div class="trace-section">
            <div class="trace-label">Think</div>
            <p class="trace-text stream-thought">${escapeHtml(thought)}</p>
          </div>
          <div class="trace-io-grid">
            <div class="trace-section">
              <div class="trace-label">Code</div>
              <pre><code class="stream-code">${escapeHtml(step.code || "")}</code></pre>
            </div>
            <div class="trace-section">
              <div class="trace-label">Result</div>
              <pre class="result"><code class="stream-result">${escapeHtml(step.result || "")}</code></pre>
            </div>
          </div>
        </article>`;
    })
    .join("");

  els.traceList.querySelectorAll(".claim-badge").forEach((badge) => {
    const cid = badge.dataset.claim;
    badge.addEventListener("mouseenter", (event) => activateClaims([cid], event));
    badge.addEventListener("mousemove", moveTooltip);
    badge.addEventListener("mouseleave", clearTransientHighlight);
  });
  scrollTraceToBottom(shouldFollow);
}

function hydrateStreamingStep(step) {
  if (!step) return step;
  if (step.assistant_text) {
    const parsed = parseAssistantStream(step.assistant_text);
    step.thought = parsed.thought || step.thought || step.reasoning_summary || "";
    step.code = parsed.code || step.code || "";
    step.result = parsed.result || step.result || "Streaming model output...";
  } else if (step.reasoning_summary) {
    step.thought = step.reasoning_summary;
    step.result = step.result || "Streaming reasoning...";
  }
  return step;
}

function updateStreamingTrace() {
  const step = hydrateStreamingStep(state.streaming);
  if (!step) return;
  const article = els.traceList.querySelector(".trace-step.active");
  if (!article || article.dataset.turn !== String(step.turn)) {
    renderTrace();
    return;
  }
  setText(article.querySelector(".stream-thought"), step.thought || step.reasoning_summary || "No visible reasoning summary for this turn.");
  setText(article.querySelector(".stream-code"), step.code || "");
  setText(article.querySelector(".stream-result"), step.result || "");
  scrollTraceToBottom();
}

function updateReportStream() {
  if (!state.reportStreamEl) {
    els.reportBody.innerHTML = `<div class="report-loading"><div class="loading-line strong"></div><div class="loading-line"></div><div class="loading-line short"></div></div>`;
    state.reportStreamEl = els.reportBody.querySelector(".report-loading");
  }
  state.reportStreamEl.setAttribute("data-chars", String((state.reportStream || "").length));
}

function claimMap() {
  const map = new Map();
  for (const claim of state.graph?.claims || []) map.set(String(claim.id), claim);
  for (const node of state.graph?.nodes || []) {
    if (node.kind === "claim" && !map.has(String(node.id))) map.set(String(node.id), node);
  }
  return map;
}

function nodeMap() {
  const map = new Map();
  for (const node of state.graph?.nodes || []) map.set(String(node.id), node);
  return map;
}

function reverseLinks() {
  const reverse = new Map();
  for (const link of state.graph?.links || []) {
    const target = String(link.target);
    const source = String(link.source);
    if (!reverse.has(target)) reverse.set(target, []);
    reverse.get(target).push(source);
  }
  return reverse;
}

function supportClosure(ids) {
  const reverse = reverseLinks();
  const visited = new Set();
  const stack = [...ids.map(String)];
  while (stack.length) {
    const id = stack.pop();
    if (!id || visited.has(id)) continue;
    visited.add(id);
    for (const source of reverse.get(id) || []) stack.push(source);
  }
  return visited;
}

function forwardLinks() {
  const forward = new Map();
  for (const link of state.graph?.links || []) {
    const source = String(link.source);
    const target = String(link.target);
    if (!forward.has(source)) forward.set(source, []);
    forward.get(source).push(target);
  }
  return forward;
}

function chainText(ids) {
  const claims = claimMap();
  const nodes = nodeMap();
  const reverse = reverseLinks();
  const seen = new Set();
  const lines = [];

  function visit(id, depth) {
    if (seen.has(`${depth}:${id}`)) return;
    seen.add(`${depth}:${id}`);
    const indent = "  ".repeat(depth);
    const claim = claims.get(id);
    const node = nodes.get(id);
    if (claim) {
      lines.push(`${indent}[${id}] ${claim.content || node?.content || ""}`);
      if (claim.reasoning) lines.push(`${indent}reasoning: ${claim.reasoning}`);
    } else if (node) {
      lines.push(`${indent}${node.label || id}: ${node.content || node.kind || ""}`);
    } else {
      lines.push(`${indent}${id}`);
    }
    for (const source of reverse.get(id) || []) visit(source, depth + 1);
  }

  for (const id of ids) visit(String(id), 0);
  return lines.join("\n");
}

function relationLabel(type) {
  if (type === "retrieve") return "retrieve";
  if (type === "cite") return "cite";
  if (type === "contain") return "contain";
  if (type === "infer") return "infer";
  if (type === "bind") return "bind";
  if (type === "compute") return "compute";
  return type || "support";
}

function nodeTitle(id) {
  const claims = claimMap();
  const nodes = nodeMap();
  const claim = claims.get(id);
  const node = nodes.get(id);
  if (claim) return `[${id}] ${claim.type === "composite" ? "Derived claim" : "Atomic claim"}`;
  if (node?.kind === "data" && node.data_role === "external_source") return `External source: ${node.label || id}`;
  if (node?.kind === "data" && node.data_role === "external") return `Fetched artifact: ${node.label || id}`;
  if (node?.kind === "data" && node.data_role === "evidence") return `Evidence span/cell: ${node.label || id}`;
  if (node?.kind === "data") return `${node.data_role === "raw" ? "Raw data" : node.data_role === "bound" ? "Bound data" : "Computed data"}: ${node.label || id}`;
  if (node?.kind === "operation") return `Computation: ${node.label || id}`;
  return node?.label || id;
}

function nodeContent(id) {
  const claims = claimMap();
  const nodes = nodeMap();
  const claim = claims.get(id);
  const node = nodes.get(id);
  return claim?.content || node?.content || node?.kind || "";
}

function nodeReasoning(id) {
  const claims = claimMap();
  const nodes = nodeMap();
  const claim = claims.get(id);
  const node = nodes.get(id);
  return claim?.reasoning || node?.reasoning || "";
}

function rootToTargetPaths(targetIds, maxPaths = 4, maxDepth = 8) {
  const reverse = reverseLinks();
  const nodes = nodeMap();
  const links = state.graph?.links || [];
  const linkByPair = new Map(links.map((link) => [`${link.source}->${link.target}`, link]));
  const paths = [];

  function walk(id, path, seen) {
    if (!id || seen.has(id) || path.length > maxDepth || paths.length >= maxPaths) return;
    const nextPath = [id, ...path];
    const sources = (reverse.get(id) || []).filter((source) => nodes.has(source));
    if (!sources.length || nodes.get(id)?.data_role === "raw") {
      paths.push(nextPath);
      return;
    }
    const sorted = sources.slice().sort((a, b) => {
      const la = linkByPair.get(`${a}->${id}`)?.type || "";
      const lb = linkByPair.get(`${b}->${id}`)?.type || "";
      const rank = { compute: 0, bind: 1, infer: 2 };
      return (rank[la] ?? 9) - (rank[lb] ?? 9) || a.localeCompare(b, undefined, { numeric: true });
    });
    for (const source of sorted) walk(source, nextPath, new Set([...seen, id]));
  }

  for (const id of targetIds.map(String).filter(Boolean).slice(0, 3)) walk(id, [], new Set());
  return paths;
}

function edgeTypeBetween(source, target) {
  const link = (state.graph?.links || []).find((item) => String(item.source) === String(source) && String(item.target) === String(target));
  return link?.type || "support";
}

function chainItems(ids, maxDepth = 5, maxItems = 18) {
  const claims = claimMap();
  const nodes = nodeMap();
  const reverse = reverseLinks();
  const linkByPair = new Map((state.graph?.links || []).map((link) => [`${link.source}->${link.target}`, link]));
  const seen = new Set();
  const items = [];

  function describe(id, depth, relation) {
    const claim = claims.get(id);
    const node = nodes.get(id);
    const kind = claim ? `claim ${claim.type || node?.claim_type || ""}`.trim() : node?.kind || "node";
    const title = claim ? `[${id}] ${claim.type === "composite" ? "Derived claim" : "Atomic claim"}` : node?.label || id;
    const content = claim?.content || node?.content || node?.kind || "";
    const reasoning = claim?.reasoning || node?.reasoning || "";
    return { id, depth, relation, kind, title, content, reasoning, final: Boolean(claim?.final_node || node?.final) };
  }

  function visit(id, depth, relation) {
    if (!id || seen.has(id) || items.length >= maxItems) return;
    seen.add(id);
    items.push(describe(id, depth, relation));
    if (depth >= maxDepth) return;
    const sources = (reverse.get(id) || []).slice().sort((a, b) => {
      const la = linkByPair.get(`${a}->${id}`)?.type || "";
      const lb = linkByPair.get(`${b}->${id}`)?.type || "";
      const rank = { infer: 0, bind: 1, compute: 2 };
      return (rank[la] ?? 9) - (rank[lb] ?? 9) || a.localeCompare(b, undefined, { numeric: true });
    });
    for (const source of sources) {
      const link = linkByPair.get(`${source}->${id}`);
      visit(source, depth + 1, link?.type || "support");
    }
  }

  for (const id of ids.map(String).filter(Boolean).slice(0, 4)) visit(id, 0, "selected");
  return items;
}

function graphLinks() {
  return (state.graph?.links || []).map((link) => ({ ...link, source: String(link.source), target: String(link.target) }));
}

function incomingLinks(id, type = null) {
  const target = String(id);
  return graphLinks().filter((link) => link.target === target && (!type || link.type === type));
}

function upstreamDataForOperation(operationId) {
  const nodes = nodeMap();
  return incomingLinks(operationId)
    .map((link) => nodes.get(String(link.source)))
    .filter((node) => node?.kind === "data")
    .map((node) => node.label || node.id);
}

function computeDescriptions(dataId, compact = false) {
  const nodes = nodeMap();
  const rows = [];
  for (const link of incomingLinks(dataId, "compute")) {
    const source = nodes.get(String(link.source));
    if (source?.kind === "operation") {
      const inputs = upstreamDataForOperation(source.id);
      const inputText = inputs.length ? ` from ${inputs.slice(0, compact ? 2 : 4).join(", ")}` : "";
      rows.push(`${source.label || "computation"}${inputText}`);
    } else if (source?.kind === "data") {
      rows.push(`${source.label || source.id} -> ${nodes.get(String(dataId))?.label || dataId}`);
    } else if (link.label) {
      rows.push(link.label);
    }
  }
  return [...new Set(rows)].slice(0, compact ? 2 : 4);
}

function boundDataIdsForClaim(claimId) {
  return incomingLinks(claimId, "bind").map((link) => String(link.source));
}

function premiseIdsForClaim(claimId, claim = null) {
  const ids = new Set();
  for (const link of incomingLinks(claimId, "infer")) ids.add(String(link.source));
  for (const pid of claim?.premise_ids || []) ids.add(String(pid));
  return [...ids];
}

function logicCardHtml({ role, title, content, detail = "", final = false }) {
  return `
    <div class="logic-card ${escapeHtml(role)} ${final ? "final" : ""}">
      <div class="logic-card-head">
        <span class="logic-rel ${escapeHtml(role)}">${escapeHtml(role)}</span>
        <strong>${escapeHtml(title)}</strong>
      </div>
      ${content ? `<div class="logic-content">${escapeHtml(content)}</div>` : ""}
      ${detail ? `<div class="logic-reason">${escapeHtml(detail)}</div>` : ""}
    </div>`;
}

function claimEvidenceHtml(claimId, compact = false, nested = false) {
  const claims = claimMap();
  const nodes = nodeMap();
  const claim = claims.get(String(claimId));
  const node = nodes.get(String(claimId));
  if (!claim && !node) return "";

  const title = claim ? `[${claimId}] ${claim.type === "composite" ? "Inferred claim" : "Bound claim"}` : nodeTitle(claimId);
  const role = claim?.type === "composite" ? "infer" : claim ? "claim" : node?.kind || "node";
  const contentLimit = compact ? 130 : nested ? 150 : 240;
  const boundIds = boundDataIdsForClaim(claimId);
  const detailParts = [];
  if (claim?.reasoning) detailParts.push(`Reasoning: ${shortText(claim.reasoning, compact ? 130 : 220)}`);
  if (nested && boundIds.length) {
    const nodesForDetail = nodeMap();
    const boundText = boundIds.slice(0, 2).map((dataId) => {
      const dataNode = nodesForDetail.get(dataId) || {};
      const computes = computeDescriptions(dataId, true);
      return `${dataNode.label || dataId}${computes.length ? ` (${computes[0]})` : ""}`;
    }).join("; ");
    detailParts.push(`Bind evidence: ${boundText}`);
  }
  const blocks = [
    logicCardHtml({
      role,
      title: shortText(title, compact ? 46 : 70),
      content: shortText(claim?.content || node?.content || "", contentLimit),
      detail: detailParts.join("\n"),
      final: Boolean(claim?.final_node || node?.final),
    }),
  ];

  const premises = premiseIdsForClaim(claimId, claim).filter((pid) => pid !== String(claimId));
  if (premises.length && !nested) {
    const premiseCards = premises.slice(0, compact ? 3 : 5).map((pid) => claimEvidenceHtml(pid, true, true)).join("");
    blocks.push(`<section class="logic-block"><div class="logic-block-title">Premises used by infer</div>${premiseCards}</section>`);
  }

  if (boundIds.length) {
    const dataCards = boundIds.slice(0, compact ? 2 : 4).map((dataId) => {
      const dataNode = nodes.get(dataId) || {};
      const computes = computeDescriptions(dataId, compact);
      const detail = computes.length ? `Computed by: ${computes.join("; ")}` : "";
      return logicCardHtml({
        role: "data",
        title: shortText(dataNode.label || dataId, compact ? 44 : 64),
        content: shortText(dataNode.content || "Bound data used by bind.", compact ? 120 : 180),
        detail,
      });
    }).join("");
    blocks.push(`<section class="logic-block"><div class="logic-block-title">Data interpreted by bind</div>${dataCards}</section>`);
  }

  if (nested) return blocks[0];
  return `<section class="logic-block">${blocks.join("")}</section>`;
}

function chainHtml(ids, compact = false) {
  const selected = ids.map(String).filter(Boolean).slice(0, compact ? 2 : 4);
  if (!selected.length) return "";
  const rows = selected.map((id) => claimEvidenceHtml(id, compact)).filter(Boolean).join("");
  if (!rows) return "";
  const target = selected.join(", ");
  return `
    <div class="logic-chain ${compact ? "compact" : ""}">
      <div class="logic-title">
        <span>Evidence Behind Citation</span>
        <strong>${escapeHtml(target)}</strong>
      </div>
      <div class="logic-stack">${rows}</div>
    </div>`;
}

function activateClaims(ids, event, lock = false) {
  const closure = supportClosure(ids);
  state.activeIds = closure;
  if (lock) state.lockedIds = closure;
  applyGraphHighlight();
  updateInspector(ids);
  showTooltip(chainHtml(ids), event, { html: true });
}

function clearTransientHighlight() {
  if (state.lockedIds.size) {
    state.activeIds = new Set(state.lockedIds);
  } else {
    state.activeIds = new Set();
    hideTooltip();
  }
  applyGraphHighlight();
}

function updateInspector(ids) {
  const html = chainHtml(ids, true);
  els.nodeInspector.innerHTML = html || `<span class="inspector-label">Supporting chain</span><p>No supporting chain selected.</p>`;
}

function applyGraphHighlight() {
  const active = state.activeIds || new Set();
  const hasActive = active.size > 0;
  els.graphSvg.querySelectorAll(".graph-node").forEach((node) => {
    const id = node.getAttribute("data-node");
    if (!id) return;
    const highlighted = hasActive && active.has(id);
    node.classList.toggle("dimmed", hasActive && !highlighted);
    node.classList.toggle("highlight", highlighted);
  });
  els.graphSvg.querySelectorAll(".graph-link").forEach((link) => {
    const source = link.getAttribute("data-source");
    const target = link.getAttribute("data-target");
    if (!source || !target) return;
    const highlighted = hasActive && active.has(source) && active.has(target);
    link.classList.toggle("dimmed", hasActive && !highlighted);
    link.classList.toggle("highlight", highlighted);
  });
  els.graphSvg.querySelectorAll(".graph-edge-label").forEach((label) => {
    const source = label.getAttribute("data-source");
    const target = label.getAttribute("data-target");
    if (!source || !target) return;
    const highlighted = hasActive && active.has(source) && active.has(target);
    label.classList.toggle("dimmed", hasActive && !highlighted);
    label.classList.toggle("highlight", highlighted);
  });
}

function nodeLevel(node, levels, linksByTarget, nodeById) {
  if (levels.has(node.id)) return levels.get(node.id);
  if (node.kind === "file") return 0;
  if (node.id.startsWith("evidence:")) return 2;
  if (node.kind === "variable") return 1;
  const incoming = linksByTarget.get(node.id) || [];
  if (!incoming.length) return node.claim_type === "composite" ? 4 : 3;
  let level = node.claim_type === "composite" ? 4 : 3;
  for (const link of incoming) {
    const sourceNode = nodeById.get(String(link.source));
    if (!sourceNode) continue;
    level = Math.max(level, nodeLevel(sourceNode, levels, linksByTarget, nodeById) + 1);
  }
  levels.set(node.id, level);
  return level;
}

function svgEl(name, attrs = {}) {
  const el = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
}

function resetGraphView() {
  state.graphView = { scale: 1, tx: 0, ty: 0, dragging: false, lastX: 0, lastY: 0 };
  applyGraphViewTransform();
}

function applyGraphViewTransform() {
  const viewport = els.graphSvg?.querySelector(".graph-viewport");
  if (!viewport) return;
  const view = state.graphView;
  viewport.setAttribute("transform", `translate(${view.tx} ${view.ty}) scale(${view.scale})`);
  if (els.zoomResetButton) els.zoomResetButton.textContent = `${Math.round(view.scale * 100)}%`;
}

function graphCursorPoint(event) {
  const rect = els.graphSvg.getBoundingClientRect();
  const width = Number(els.graphSvg.dataset.viewWidth || 900);
  const height = Number(els.graphSvg.dataset.viewHeight || 430);
  return {
    x: ((event.clientX - rect.left) / Math.max(1, rect.width)) * width,
    y: ((event.clientY - rect.top) / Math.max(1, rect.height)) * height,
  };
}

function setGraphZoom(nextScale, anchor) {
  const view = state.graphView;
  const scale = Math.max(0.45, Math.min(3.5, nextScale));
  const point = anchor || { x: Number(els.graphSvg.dataset.viewWidth || 900) / 2, y: Number(els.graphSvg.dataset.viewHeight || 430) / 2 };
  const graphX = (point.x - view.tx) / view.scale;
  const graphY = (point.y - view.ty) / view.scale;
  view.scale = scale;
  view.tx = point.x - graphX * scale;
  view.ty = point.y - graphY * scale;
  applyGraphViewTransform();
}

function displayGraph(graph) {
  if (!graph) return graph;
  const sourceNodes = (graph.nodes || []).map((node) => ({ ...node, id: String(node.id) }));
  const sourceLinks = (graph.links || []).map((link) => ({ ...link, source: String(link.source), target: String(link.target) }));
  const nodeById = new Map(sourceNodes.map((node) => [node.id, node]));
  const claimIdsWithExternalEvidence = new Set(
    sourceNodes
      .filter((node) => node.kind === "data" && node.data_role === "evidence" && String(node.id).startsWith("evidence:"))
      .map((node) => String(node.id).split(":")[1])
      .filter(Boolean),
  );
  const hiddenBoundIds = new Set([...claimIdsWithExternalEvidence].map((cid) => `data:bound:${cid}`));
  const nodes = sourceNodes.filter((node) => (node.kind === "data" || node.kind === "claim") && !hiddenBoundIds.has(node.id));
  const visibleIds = new Set(nodes.map((node) => node.id));
  const incomingByTarget = new Map();
  for (const link of sourceLinks) {
    if (!incomingByTarget.has(link.target)) incomingByTarget.set(link.target, []);
    incomingByTarget.get(link.target).push(link);
  }

  const edgeMap = new Map();
  function addLink(source, target, type, attrs = {}) {
    if (!visibleIds.has(source) || !visibleIds.has(target) || source === target) return;
    const label = attrs.label || relationLabel(type);
    const id = `${source}->${target}:${type}:${hashString(label)}`;
    if (!edgeMap.has(id)) edgeMap.set(id, { ...attrs, id, source, target, type, label });
  }

  for (const link of sourceLinks) {
    const sourceNode = nodeById.get(link.source);
    const targetNode = nodeById.get(link.target);
    if (link.type === "compute" && sourceNode?.kind === "operation" && visibleIds.has(link.target)) {
      const operationLabel = sourceNode.label || "computation";
      const sources = (incomingByTarget.get(sourceNode.id) || []).filter((incoming) => visibleIds.has(incoming.source));
      for (const sourceLink of sources) {
        addLink(sourceLink.source, link.target, "compute", {
          label: link.label || `compute ${targetNode?.label || link.target}`,
          operation: operationLabel,
        });
      }
      continue;
    }
    if (visibleIds.has(link.source) && visibleIds.has(link.target)) {
      addLink(link.source, link.target, link.type || "support", link);
    }
  }

  return { ...graph, nodes, links: [...edgeMap.values()] };
}

function nodeRadius(node) {
  if (node.kind === "claim" && node.final) return 30;
  if (node.kind === "claim") return 26;
  if (node.kind === "data" && node.source_type === "local_file") return 24;
  if (node.kind === "data" && node.source_type === "web") return 22;
  if (node.kind === "data" && node.source_type === "finance_api") return 22;
  if (node.kind === "data" && node.source_type === "evidence") return 18;
  if (node.kind === "data" && node.data_role === "raw") return 24;
  if (node.kind === "data" && node.data_role === "evidence") return 18;
  return 20;
}

function nodeIcon(node) {
  if (node.kind === "claim") return node.final ? "★" : "";
  const st = node.source_type || node.data_role;
  if (st === "local_file" || st === "raw") return "📄";
  if (st === "web") return "🌐";
  if (st === "finance_api") return "📈";
  if (st === "evidence") return "🔖";
  if (st === "computed" || st === "bound") return "⚙";
  return "";
}

function showNodeLabel(node) {
  if (node.kind === "claim") return true;
  if (node.final) return true;
  if (node.kind === "data") return true;
  return false;
}

function nodeSize(node) {
  const r = nodeRadius(node);
  const labelWidth = showNodeLabel(node) ? Math.min(132, 28 + shortText(node.label || node.id, 18).length * 7.2) : 0;
  return { w: r * 2 + labelWidth + 44, h: Math.max(r * 2 + 34, 68) };
}

function nodeEdgePoint(from, to, node) {
  const r = nodeRadius(node) + 5;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  if (dx === 0 && dy === 0) return { ...from };
  const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
  return { x: from.x + (dx / dist) * r, y: from.y + (dy / dist) * r };
}

function hashString(value) {
  let hash = 2166136261;
  const text = String(value || "");
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hashUnit(value) {
  return (hashString(value) % 10000) / 10000;
}

function nodeKindRank(node) {
  const st = node.source_type || node.data_role;
  if (node.kind === "claim" && (node.final || node.claim_type === "composite")) return 5;
  if (node.kind === "claim") return 4;
  if (node.kind === "data" && (st === "evidence" || st === "bound")) return 3;
  if (node.kind === "data" && st === "computed") return 2;
  if (node.kind === "data" && (st === "web" || st === "finance_api" || st === "external")) return 1;
  if (node.kind === "data" && (st === "local_file" || st === "raw")) return 0;
  return 0;
}

function computeForceLayout(nodes, links, width, viewportHeight) {
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  const outgoing = new Map(nodes.map((n) => [n.id, []]));
  const incoming = new Map(nodes.map((n) => [n.id, []]));
  for (const l of links) {
    if (!nodeById.has(l.source) || !nodeById.has(l.target) || l.source === l.target) continue;
    outgoing.get(l.source).push(l.target);
    incoming.get(l.target).push(l.source);
  }

  // Initial layer = longest path from any source via topological relax.
  const layer = new Map();
  for (const n of nodes) layer.set(n.id, incoming.get(n.id).length === 0 ? 0 : -1);
  let changed = true;
  let safety = 0;
  while (changed && safety < nodes.length + 4) {
    changed = false;
    safety += 1;
    for (const n of nodes) {
      const parents = incoming.get(n.id);
      if (!parents.length) continue;
      let best = -1;
      for (const pid of parents) {
        const pl = layer.get(pid);
        if (pl >= 0 && pl > best) best = pl;
      }
      const candidate = best + 1;
      if (candidate > (layer.get(n.id) ?? -1)) {
        layer.set(n.id, candidate);
        changed = true;
      }
    }
  }
  for (const n of nodes) {
    if ((layer.get(n.id) ?? -1) < 0) layer.set(n.id, 0);
    layer.set(n.id, Math.max(layer.get(n.id), nodeKindRank(n)));
  }
  const maxLayer = Math.max(0, ...nodes.map((n) => layer.get(n.id)));
  for (const n of nodes) if (n.final) layer.set(n.id, maxLayer);

  // Group nodes per layer, then renumber layers contiguously.
  const groupMap = new Map();
  for (const n of nodes) {
    const L = layer.get(n.id);
    if (!groupMap.has(L)) groupMap.set(L, []);
    groupMap.get(L).push(n);
  }
  const layerKeys = [...groupMap.keys()].sort((a, b) => a - b);
  const layered = layerKeys.map((k) => groupMap.get(k));
  const layerIdx = new Map();
  layered.forEach((lyr, idx) => lyr.forEach((n) => layerIdx.set(n.id, idx)));

  // Initial within-layer order: by parent kind then id.
  for (const lyr of layered) {
    lyr.sort((a, b) => nodeKindRank(a) - nodeKindRank(b) || String(a.id).localeCompare(String(b.id)));
  }

  // Barycenter sweeps to reduce edge crossings.
  function indexInLayer(layerArr) {
    const m = new Map();
    layerArr.forEach((n, i) => m.set(n.id, i));
    return m;
  }
  function barycenter(node, refIndex, neighbors) {
    const vals = neighbors.map((id) => refIndex.get(id)).filter((v) => v !== undefined);
    if (!vals.length) return refIndex.size / 2;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }
  for (let pass = 0; pass < 12; pass += 1) {
    const forward = pass % 2 === 0;
    const range = forward ? layered.slice(1) : layered.slice(0, -1).reverse();
    range.forEach((lyr) => {
      const refIdx = layered.indexOf(lyr) + (forward ? -1 : 1);
      if (refIdx < 0 || refIdx >= layered.length) return;
      const refIndex = indexInLayer(layered[refIdx]);
      const bary = new Map();
      for (const n of lyr) {
        const neighbors = forward ? incoming.get(n.id) : outgoing.get(n.id);
        bary.set(n.id, barycenter(n, refIndex, neighbors));
      }
      lyr.sort((a, b) => {
        const diff = bary.get(a.id) - bary.get(b.id);
        if (Math.abs(diff) > 1e-6) return diff;
        return nodeKindRank(a) - nodeKindRank(b) || String(a.id).localeCompare(String(b.id));
      });
    });
  }

  // Pixel coordinates.
  const numLayers = layered.length;
  const minRowSpacing = 130;
  const labelGutter = 140;
  const xMargin = 110;
  const yMargin = 80;
  const tallestLayer = layered.reduce((m, l) => Math.max(m, l.length), 1);
  const requiredHeight = yMargin * 2 + (tallestLayer - 1) * minRowSpacing + 120;
  const targetHeight = Math.max(viewportHeight || 0, 640);
  const height = Math.max(640, Math.min(1400, Math.max(targetHeight, requiredHeight)));
  const requiredWidth = xMargin * 2 + Math.max(numLayers - 1, 1) * (labelGutter + 220);
  const layoutWidth = Math.max(width, requiredWidth);
  const xStep = numLayers > 1 ? (layoutWidth - xMargin * 2) / (numLayers - 1) : 0;
  const usableHeight = height - yMargin * 2;

  const positions = new Map();
  const anchors = new Map();
  layered.forEach((lyr, layerI) => {
    const x = xMargin + layerI * xStep;
    const count = lyr.length;
    if (count === 1) {
      const jx = (hashUnit(`${lyr[0].id}:x`) - 0.5) * 110;
      const jy = (hashUnit(`${lyr[0].id}:y`) - 0.5) * 180;
      positions.set(lyr[0].id, { x: x + jx, y: height / 2 + jy, vx: 0, vy: 0 });
      anchors.set(lyr[0].id, { x, y: height / 2, layerI });
      return;
    }
    const span = Math.min(usableHeight, (count - 1) * Math.max(minRowSpacing, usableHeight / (count - 1)));
    const startY = height / 2 - span / 2;
    const step = span / (count - 1);
    lyr.forEach((node, idx) => {
      const jx = (hashUnit(`${node.id}:x`) - 0.5) * 130;
      const jy = (hashUnit(`${node.id}:y`) - 0.5) * Math.max(60, step * 0.7);
      const targetY = startY + idx * step;
      positions.set(node.id, { x: x + jx, y: targetY + jy, vx: 0, vy: 0 });
      anchors.set(node.id, { x, y: targetY, layerI });
    });
  });

  // Force-directed relaxation with strong repulsion: organic, dispersed knowledge-graph feel.
  const validLinks = links.filter((l) => positions.has(l.source) && positions.has(l.target));
  const ITER = 200;
  for (let tick = 0; tick < ITER; tick += 1) {
    const alpha = 1 - tick / ITER;
    for (let i = 0; i < nodes.length; i += 1) {
      const a = positions.get(nodes[i].id);
      for (let j = i + 1; j < nodes.length; j += 1) {
        const b = positions.get(nodes[j].id);
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dist2 = dx * dx + dy * dy;
        if (dist2 < 0.5) {
          dx = (hashUnit(`${nodes[i].id}:${nodes[j].id}`) - 0.5) || 0.1;
          dy = (hashUnit(`${nodes[j].id}:${nodes[i].id}`) - 0.5) || 0.1;
          dist2 = dx * dx + dy * dy;
        }
        // Strong long-range Coulomb-style repulsion in addition to hard collision avoidance
        const dist = Math.sqrt(dist2);
        const longRange = Math.min(2.6, 9000 / Math.max(900, dist2));
        let fx = (dx / dist) * longRange;
        let fy = (dy / dist) * longRange;
        const minGap = (nodeRadius(nodes[i]) + nodeRadius(nodes[j])) + 70;
        if (dist < minGap) {
          const overlap = ((minGap - dist) / minGap) * 3.5;
          fx += (dx / dist) * overlap;
          fy += (dy / dist) * overlap;
        }
        a.vx -= fx;
        a.vy -= fy;
        b.vx += fx;
        b.vy += fy;
      }
    }
    for (const link of validLinks) {
      const s = positions.get(link.source);
      const t = positions.get(link.target);
      const dx = t.x - s.x;
      const dy = t.y - s.y;
      const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const desired = linkDistance(link.type) * 1.4;
      const force = ((dist - desired) / dist) * 0.05 * alpha;
      s.vx += dx * force * 0.30;
      s.vy += dy * force * 0.45;
      t.vx -= dx * force * 0.30;
      t.vy -= dy * force * 0.45;
    }
    for (const node of nodes) {
      const pos = positions.get(node.id);
      const anc = anchors.get(node.id);
      // Soft anchors so points can drift but don't fly away
      pos.vx += (anc.x - pos.x) * 0.022;
      pos.vy += (anc.y - pos.y) * 0.012;
      pos.x += pos.vx;
      pos.y += pos.vy;
      pos.vx *= 0.82;
      pos.vy *= 0.84;
      const r = nodeRadius(node);
      pos.x = Math.max(r + 28, Math.min(layoutWidth - r - 28, pos.x));
      pos.y = Math.max(r + 24, Math.min(height - r - 24, pos.y));
    }
  }

  return { positions, height, width: layoutWidth };
}

function linkDistance(type) {
  if (type === "retrieve") return 90;
  if (type === "cite") return 90;
  if (type === "contain") return 90;
  if (type === "bind") return 110;
  if (type === "infer") return 130;
  if (type === "compute") return 110;
  return 110;
}

function edgeText(link) {
  const label = link.type === "infer" && link.reasoning ? `infer: ${link.reasoning}` : link.label || relationLabel(link.type);
  return shortText(label, link.type === "infer" ? 22 : 18);
}

function isNewGraphLink(link, addedIds) {
  return addedIds.has(String(link.target)) || addedIds.has(String(link.source));
}

function appendEdgeLabel(layer, link, start, end, control, options = {}) {
  const text = edgeText(link);
  if (!text) return;
  const midX = (start.x + 2 * control.x + end.x) / 4;
  const midY = (start.y + 2 * control.y + end.y) / 4;
  const width = Math.max(28, Math.min(118, text.length * 5.5 + 12));
  const group = svgEl("g", {
    class: `graph-edge-label ${link.type || ""} ${options.highlighted ? "highlight" : ""} ${options.dimmed ? "dimmed" : ""} ${options.isNew ? "new" : ""}`,
    transform: `translate(${midX} ${midY})`,
    "data-source": link.source,
    "data-target": link.target,
  });
  group.append(svgEl("rect", { x: -width / 2, y: -8, width, height: 16, rx: 8, class: "edge-label-bg" }));
  const textEl = svgEl("text", { x: 0, y: 3, "text-anchor": "middle", class: "edge-label" });
  textEl.textContent = text;
  group.append(textEl);
  layer.append(group);
}

function appendGraphLegend(svg, width) {
  const items = [
    { kind: "node", label: "local data", className: "data local_file", icon: "📄" },
    { kind: "node", label: "web", className: "data web", icon: "🌐" },
    { kind: "node", label: "finance API", className: "data finance_api", icon: "📈" },
    { kind: "node", label: "computed", className: "data computed", icon: "⚙" },
    { kind: "node", label: "evidence span", className: "data evidence", icon: "🔖" },
    { kind: "node", label: "claim", className: "claim", icon: "★" },
    { kind: "edge", label: "compute", className: "compute" },
    { kind: "edge", label: "bind", className: "bind" },
    { kind: "edge", label: "infer", className: "infer" },
    { kind: "edge", label: "cite", className: "cite" },
  ];
  const legend = svgEl("g", { transform: `translate(${Math.max(16, width - 880)} 20)`, class: "graph-legend" });
  let x = 0;
  for (const item of items) {
    const group = svgEl("g", { transform: `translate(${x} 0)`, class: item.kind === "node" ? `graph-node ${item.className}` : `graph-legend-edge ${item.className}` });
    if (item.kind === "node") {
      group.append(svgEl("circle", { cx: 0, cy: 0, r: 7, class: "node-shape" }));
      if (item.icon) {
        const ic = svgEl("text", { x: 0, y: 3.4, "text-anchor": "middle", class: "node-icon" });
        ic.textContent = item.icon;
        group.append(ic);
      }
    } else {
      group.append(svgEl("line", { x1: -2, y1: 0, x2: 24, y2: 0, class: `graph-link ${item.className}` }));
    }
    const text = svgEl("text", { x: item.kind === "node" ? 14 : 30, y: 4, class: "graph-legend-label" });
    text.textContent = item.label;
    group.append(text);
    legend.append(group);
    x += item.label.length * 6 + (item.kind === "node" ? 38 : 50);
  }
  svg.append(legend);
}

function renderGraph(graph) {
  els.graphSvg.replaceChildren();
  state.graph = graph;
  const viewGraph = displayGraph(graph);
  const stats = graph?.stats || { claims: 0, variables: 0, atomic: 0, composite: 0, final: 0 };
  const dataCount = viewGraph?.nodes?.filter((node) => node.kind === "data").length || 0;
  const edgeCount = viewGraph?.links?.length || 0;
  els.graphStats.innerHTML = `
    ${state.latestGraphTurn ? `<span>turn ${escapeHtml(state.latestGraphTurn)}</span>` : ""}
    <span>${stats.claims || 0} claims</span>
    <span>${dataCount} data</span>
    ${stats.artifacts ? `<span>${stats.artifacts} artifacts</span>` : ""}
    <span>${edgeCount} edges</span>
    <span>${stats.final || 0} final</span>
    ${stats.hidden_variables ? `<span>${stats.hidden_variables} hidden vars</span>` : ""}`;

  if (!viewGraph || !viewGraph.nodes || !viewGraph.nodes.length) {
    const svg = els.graphSvg;
    svg.setAttribute("viewBox", "0 0 900 430");
    svg.dataset.viewWidth = "900";
    svg.dataset.viewHeight = "430";
    const text = svgEl("text", { x: "450", y: "215", "text-anchor": "middle", class: "node-sub" });
    text.textContent = "Evidence graph will appear here";
    svg.append(text);
    return;
  }

  const frame = els.graphSvg.parentElement.getBoundingClientRect();
  const width = Math.max(760, frame.width || 900);
  const viewportHeight = Math.max(420, frame.height || 0);
  const nodes = viewGraph.nodes.map((n) => ({ ...n, id: String(n.id) }));
  const links = (viewGraph.links || []).map((l) => ({ ...l, source: String(l.source), target: String(l.target) }));
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const addedIds = new Set([...(graph.added_claim_ids || []), ...state.latestAddedIds].map(String));
  const newNodeIds = new Set(addedIds);
  for (const link of links) {
    if (addedIds.has(link.target)) newNodeIds.add(link.source);
  }
  for (const link of links) {
    if (newNodeIds.has(link.target) && String(link.target).startsWith("data:bound:")) newNodeIds.add(link.source);
  }
  const { positions: nodePositions, height, width: layoutWidth } = computeForceLayout(nodes, links, width, viewportHeight);
  const finalWidth = layoutWidth || width;

  els.graphSvg.setAttribute("viewBox", `0 0 ${finalWidth} ${height}`);
  els.graphSvg.dataset.viewWidth = String(finalWidth);
  els.graphSvg.dataset.viewHeight = String(height);

  const defs = svgEl("defs");
  const marker = svgEl("marker", { id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "5", markerHeight: "5", orient: "auto-start-reverse" });
  marker.append(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "context-stroke" }));
  defs.append(marker);
  const shadow = svgEl("filter", { id: "node-shadow", x: "-40%", y: "-40%", width: "180%", height: "180%" });
  shadow.append(svgEl("feGaussianBlur", { in: "SourceAlpha", stdDeviation: "2.4" }));
  shadow.append(svgEl("feOffset", { dx: "0", dy: "1.4", result: "off" }));
  const merge = svgEl("feMerge");
  merge.append(svgEl("feMergeNode", { in: "off" }));
  merge.append(svgEl("feMergeNode", { in: "SourceGraphic" }));
  shadow.append(merge);
  defs.append(shadow);
  els.graphSvg.append(defs);

  const active = state.activeIds;
  const hasActive = active && active.size > 0;
  const viewport = svgEl("g", { class: "graph-viewport" });
  const linkLayer = svgEl("g", { class: "links" });
  const labelLayer = svgEl("g", { class: "edge-labels" });
  const nodeLayer = svgEl("g", { class: "nodes" });
  viewport.append(linkLayer, labelLayer, nodeLayer);
  els.graphSvg.append(viewport);

  for (const link of links) {
    const source = nodePositions.get(link.source);
    const target = nodePositions.get(link.target);
    const sourceNode = nodeById.get(link.source);
    const targetNode = nodeById.get(link.target);
    if (!source || !target) continue;
    const start = sourceNode ? nodeEdgePoint(source, target, sourceNode) : source;
    const end = targetNode ? nodeEdgePoint(target, source, targetNode) : target;
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
    const curve = (hashUnit(`${link.source}->${link.target}`) - 0.5) * 42;
    const control = {
      x: (start.x + end.x) / 2 + (-dy / dist) * curve,
      y: (start.y + end.y) / 2 + (dx / dist) * curve,
    };
    const path = svgEl("path", {
      d: `M ${start.x} ${start.y} Q ${control.x} ${control.y} ${end.x} ${end.y}`,
      class: `graph-link ${link.type || ""}`,
      "data-source": link.source,
      "data-target": link.target,
      "marker-end": "url(#arrow)",
    });
    const highlighted = hasActive && active.has(link.source) && active.has(link.target);
    const isNew = isNewGraphLink(link, addedIds) || (newNodeIds.has(link.source) && newNodeIds.has(link.target));
    if (hasActive && !highlighted) path.classList.add("dimmed");
    if (highlighted) path.classList.add("highlight");
    if (isNew) path.classList.add("new");
    linkLayer.append(path);
    const shouldShowLabel = highlighted || isNew || links.length <= 80;
    if (shouldShowLabel) {
      appendEdgeLabel(labelLayer, link, start, end, control, {
        highlighted,
        dimmed: hasActive && !highlighted,
        isNew,
      });
    }
  }

  for (const node of nodes) {
    const pos = nodePositions.get(node.id);
    if (!pos) continue;
    const group = svgEl("g", {
      class: `graph-node ${node.kind || ""} ${node.source_type || node.data_role || ""} ${node.claim_type || ""} ${node.final ? "final" : ""} ${newNodeIds.has(node.id) ? "new" : ""}`,
      transform: `translate(${pos.x} ${pos.y})`,
      tabindex: "0",
      "data-node": node.id,
    });
    const highlighted = hasActive && active.has(node.id);
    if (hasActive && !highlighted) group.classList.add("dimmed");
    if (highlighted) group.classList.add("highlight");

    const r = nodeRadius(node);
    group.append(svgEl("circle", { cx: 0, cy: 0, r, class: "node-shape" }));
    const icon = nodeIcon(node);
    if (icon) {
      const iconEl = svgEl("text", { x: 0, y: r > 18 ? 5 : 4, "text-anchor": "middle", class: "node-icon" });
      iconEl.textContent = icon;
      group.append(iconEl);
    }
    const title = svgEl("title");
    title.textContent = `${nodeTitle(node.id)}\n${nodeContent(node.id)}`.trim();
    group.append(title);
    if (showNodeLabel(node)) {
      const label = svgEl("text", { x: r + 5, y: 3, class: "node-label" });
      label.textContent = shortText(node.kind === "claim" ? node.id : node.label || node.id, 18);
      group.append(label);
    }

    group.addEventListener("mouseenter", (event) => {
      const ids = node.kind === "claim" ? [node.id] : [node.id];
      state.activeIds = supportClosure(ids);
      applyGraphHighlight();
      showTooltip(chainHtml(ids), event, { html: true });
      updateInspector(ids);
    });
    group.addEventListener("mousemove", moveTooltip);
    group.addEventListener("mouseleave", clearTransientHighlight);
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      state.lockedIds = supportClosure([node.id]);
      state.activeIds = new Set(state.lockedIds);
      applyGraphHighlight();
      updateInspector([node.id]);
    });
    nodeLayer.append(group);
  }
  applyGraphViewTransform();
}

function renderReport(report, finalClaims, output) {
  els.reportPanel.hidden = false;
  els.reportTitle.textContent = report?.title || "Report";
  els.claimCounter.textContent = `${finalClaims.length} final claim${finalClaims.length === 1 ? "" : "s"}`;

  function sentenceItems(value) {
    if (Array.isArray(value)) return value;
    if (typeof value === "string") return [{ text: value, claim_ids: [] }];
    if (value && typeof value === "object") return [value];
    return [];
  }

  function sectionItems(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object") return Object.entries(value).map(([heading, sentences]) => ({ heading, sentences: sentenceItems(sentences) }));
    return [];
  }

  function sentenceHtml(sentence) {
    const ids = (sentence.claim_ids || []).map(String).filter(Boolean);
    const cites = ids.length ? `<sup class="cite-group">[${ids.map((id) => escapeHtml(id)).join(", ")}]</sup>` : "";
    return `<span class="report-sentence" data-claims="${escapeHtml(ids.join(","))}">${escapeHtml(sentence.text || "")}${cites}</span>`;
  }

  function paragraphHtml(sentences = []) {
    return sentences.length ? `<p>${sentences.map(sentenceHtml).join(" ")}</p>` : "";
  }

  const summary = paragraphHtml(sentenceItems(report?.summary));
  const sections = sectionItems(report?.sections)
    .map((section) => `
      <section class="report-section">
        <h3>${escapeHtml(section.heading || "Findings")}</h3>
        ${paragraphHtml(section.sentences || [])}
      </section>`)
    .join("");
  const outputMeta = output?.output_dir ? `<div class="report-artifact">Saved to ${escapeHtml(output.output_dir)}</div>` : "";
  els.reportBody.innerHTML = `<article class="report-text">${summary}${sections}</article>${outputMeta}`;

  els.reportBody.querySelectorAll(".report-sentence").forEach((sentence) => {
    const ids = sentence.dataset.claims.split(",").filter(Boolean);
    sentence.addEventListener("mouseenter", (event) => {
      sentence.classList.add("active");
      activateClaims(ids, event);
    });
    sentence.addEventListener("mousemove", moveTooltip);
    sentence.addEventListener("mouseleave", () => {
      sentence.classList.remove("active");
      clearTransientHighlight();
    });
    sentence.addEventListener("click", (event) => activateClaims(ids, event, true));
  });
}

function handleGraphWheel(event) {
  if (!state.graph || !els.graphSvg.querySelector(".graph-viewport")) return;
  event.preventDefault();
  const point = graphCursorPoint(event);
  const factor = Math.exp(-event.deltaY * 0.0012);
  setGraphZoom(state.graphView.scale * factor, point);
}

function handleGraphPointerDown(event) {
  if (!state.graph || event.button !== 0 || event.target.closest(".graph-node")) return;
  state.graphView.dragging = true;
  state.graphView.lastX = event.clientX;
  state.graphView.lastY = event.clientY;
  els.graphSvg.classList.add("dragging");
  els.graphSvg.setPointerCapture?.(event.pointerId);
}

function handleGraphPointerMove(event) {
  if (!state.graphView.dragging) return;
  const rect = els.graphSvg.getBoundingClientRect();
  const width = Number(els.graphSvg.dataset.viewWidth || 900);
  const height = Number(els.graphSvg.dataset.viewHeight || 430);
  const dx = ((event.clientX - state.graphView.lastX) / Math.max(1, rect.width)) * width;
  const dy = ((event.clientY - state.graphView.lastY) / Math.max(1, rect.height)) * height;
  state.graphView.tx += dx;
  state.graphView.ty += dy;
  state.graphView.lastX = event.clientX;
  state.graphView.lastY = event.clientY;
  applyGraphViewTransform();
}

function handleGraphPointerUp(event) {
  if (!state.graphView.dragging) return;
  state.graphView.dragging = false;
  els.graphSvg.classList.remove("dragging");
  els.graphSvg.releasePointerCapture?.(event.pointerId);
}

els.runButton.addEventListener("click", startRun);
els.loadZhFinanceSample?.addEventListener("click", () => loadSampleCase("zh_finance_aapl_sp500"));
els.loadNvidiaSample?.addEventListener("click", () => loadSampleCase("nvda_outlook"));
els.loadTencentSample?.addEventListener("click", () => loadSampleCase("tencent_moat"));
els.fileInput.addEventListener("change", updateFileSummary);
els.sampleToggle.addEventListener("change", updateFileSummary);
els.dataMode?.addEventListener("change", updateFileSummary);
els.graphSvg.addEventListener("wheel", handleGraphWheel, { passive: false });
els.graphSvg.addEventListener("pointerdown", handleGraphPointerDown);
els.graphSvg.addEventListener("pointermove", handleGraphPointerMove);
els.graphSvg.addEventListener("pointerup", handleGraphPointerUp);
els.graphSvg.addEventListener("pointercancel", handleGraphPointerUp);
els.zoomOutButton?.addEventListener("click", () => setGraphZoom(state.graphView.scale / 1.22));
els.zoomResetButton?.addEventListener("click", resetGraphView);
els.zoomInButton?.addEventListener("click", () => setGraphZoom(state.graphView.scale * 1.22));
els.traceList.addEventListener("pointerenter", () => {
  state.tracePointerInside = true;
});
els.traceList.addEventListener("pointerleave", () => {
  state.tracePointerInside = false;
  state.traceAutoFollow = isTraceNearBottom();
});
els.traceList.addEventListener("scroll", () => {
  if (state.traceProgrammaticScroll) return;
  state.traceAutoFollow = isTraceNearBottom();
});
document.addEventListener("click", (event) => {
  if (!event.target.closest("#graphSvg") && !event.target.closest(".report-sentence")) {
    state.lockedIds = new Set();
    state.activeIds = new Set();
    applyGraphHighlight();
    hideTooltip();
  }
});
window.addEventListener("resize", () => renderGraph(state.graph));

loadConfig();
restorePendingRun();

async function restorePendingRun() {
  let lastId;
  try { lastId = localStorage.getItem("verigraph:lastRunId"); } catch (_) { return; }
  if (!lastId) return;
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(lastId)}/result`, { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();
    if (data && data.ready && data.report) {
      state.runId = lastId;
      state.reportReceived = true;
      state.report = data.report;
      state.graph = data.graph || state.graph;
      renderGraph(state.graph);
      renderReport(data.report, data.final_claims || [], data.output || null);
      setStage("done");
      setStatus("done (restored from last run)", "done");
    } else if (data && data.running) {
      state.runId = lastId;
      setStatus("recovering previous run…", "running");
      pollResultUntilReady(lastId);
    }
  } catch (_) { /* ignore */ }
}
setStage("setup");
renderGraph(null);
