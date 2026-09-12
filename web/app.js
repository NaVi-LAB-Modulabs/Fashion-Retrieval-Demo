"use strict";

const $ = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
const score = (value) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "N/A";
const percentage = (value) => typeof value === "number" && Number.isFinite(value) ? Math.min(100, Math.max(0, value * 100)) : 0;
const label = (value) => String(value || "").replaceAll("_", " ");
const state = { config: null, preview: [], run: null, view: "grid", busy: false, lastAttemptFailed: false };

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, signal: options.signal || AbortSignal.timeout(15000) });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(typeof data?.detail === "string" ? data.detail : `The server could not process this request (${response.status}). Please try again.`);
  if (!data) throw new Error("The server returned an unreadable response. Please try again.");
  return data;
}

function notice(message = "", kind = "") {
  $("notice").textContent = message;
  $("notice").className = `notice ${kind}`;
  $("notice").hidden = !message;
}

const imageCache = new Map();
const imageRequests = new Map();
const refreshQueue = new Set();
let refreshTimer;

function safeImageSource(value) {
  if (typeof value !== "string") return null;
  if (value.startsWith("/images/")) return value;
  try {
    const url = new URL(value);
    const allowed = ["huggingface.co", "hf.co"].some((host) => url.hostname === host || url.hostname.endsWith(`.${host}`));
    return url.protocol === "https:" && !url.username && !url.password && (!url.port || url.port === "443") && allowed ? value : null;
  } catch { return null; }
}

function imageMarkup(item, className = "") {
  const src = safeImageSource(item.image_url);
  const remoteId = typeof item.image_id === "string" && item.image_id ? item.image_id : null;
  if (!src && !remoteId) return '<span class="image-missing">Image unavailable</span>';
  return `<img class="${className}" src="${escapeHTML(src || "/assets/image-placeholder.svg")}"${remoteId ? ` data-image-id="${escapeHTML(remoteId)}"` : ""} alt="${escapeHTML(item.type_name || "Garment")} ${escapeHTML(item.id)}" loading="lazy" referrerpolicy="no-referrer">`;
}

async function resolveImageIds(ids, refresh = false) {
  const wanted = [...new Set(ids)];
  const needed = wanted.filter((id) => !imageRequests.has(id) && (refresh || !imageCache.has(id) || imageCache.get(id).expires_at * 1000 <= Date.now()));
  for (let offset = 0; offset < needed.length; offset += 50) {
    const batch = needed.slice(offset, offset + 50);
    const request = api("/api/images", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({item_ids: batch, refresh}),
    }).catch(() => ({images: {}}));
    for (const id of batch) {
      const itemRequest = request.then((data) => {
        const entry = data.images?.[id];
        const url = safeImageSource(entry?.url);
        const expires = typeof entry?.expires_at === "number" ? entry.expires_at : 0;
        const value = url && expires * 1000 > Date.now()
          ? {url, expires_at: expires}
          : {url: null, expires_at: Date.now() / 1000 + 15};
        imageCache.delete(id);
        imageCache.set(id, value);
        while (imageCache.size > 500) imageCache.delete(imageCache.keys().next().value);
        return value;
      }).finally(() => imageRequests.delete(id));
      imageRequests.set(id, itemRequest);
    }
  }
  return new Map(await Promise.all(wanted.map(async (id) => [id, imageRequests.has(id) ? await imageRequests.get(id) : imageCache.get(id)])));
}

function missingImage(img) {
  if (!img.isConnected) return;
  const fallback = document.createElement("span");
  fallback.className = "image-missing";
  fallback.textContent = "Image unavailable";
  img.replaceWith(fallback);
}

async function loadRemoteImages(images, refresh = false) {
  if (!images.length) return;
  const resolved = await resolveImageIds(images.map((img) => img.dataset.imageId), refresh);
  for (const img of images) {
    if (!img.isConnected) continue;
    const url = resolved.get(img.dataset.imageId)?.url;
    if (url) img.src = url;
    else missingImage(img);
  }
}

function handleImageErrors(container) {
  const remoteImages = [];
  container.querySelectorAll("img").forEach((img) => {
    if (img.dataset.imageBound) return;
    img.dataset.imageBound = "true";
    img.addEventListener("error", () => {
      if (!img.dataset.imageId || img.dataset.imageRetried) { missingImage(img); return; }
      // Batch simultaneous failures and bypass the short URL cache once.
      img.dataset.imageRetried = "true";
      refreshQueue.add(img);
      clearTimeout(refreshTimer);
      refreshTimer = setTimeout(() => {
        const pending = [...refreshQueue].filter((node) => node.isConnected);
        refreshQueue.clear();
        loadRemoteImages(pending, true);
      }, 50);
    });
    if (img.dataset.imageId) remoteImages.push(img);
    else if (img.complete && !img.naturalWidth) missingImage(img);
  });
  loadRemoteImages(remoteImages);
}

function renderItems() {
  const items = state.run ? state.run.items : state.preview;
  const ranked = Boolean(state.run);
  $("results-eyebrow").textContent = ranked ? "RANKED RESULTS" : "UNRANKED";
  $("results-title").firstChild.textContent = ranked ? "Search results" : "Catalog preview";
  $("result-count").textContent = items.length;
  $("results-caption").textContent = ranked
    ? `${items.length} ranked ${items.length === 1 ? "match" : "matches"} from ${state.run.candidate_count} fetched graph candidates · “${state.run.query}”`
    : "Preview images. Run a query to retrieve and rank matching items.";
  $("catalog-footnote").innerHTML = '<span class="status-dot"></span>' + (ranked
    ? " Select an item for matched attributes and score details. G = graph · T = semantic · S = style."
    : " Preview only · No ranking scores.");
  $("results-grid").innerHTML = items.map((item, index) => {
    const matches = (item.matched_filters || []).slice(0, 2).map((m) => m.value).filter(Boolean).join(" · ");
    return `<button type="button" class="item-card" data-item="${index}" aria-label="Inspect ${escapeHTML(item.id)}${ranked ? `, rank ${index + 1}, score ${score(item.score)}` : ", catalog preview"}">
      <span class="card-image">${ranked ? `<span class="card-index">${String(index + 1).padStart(2, "0")}</span>` : ""}${imageMarkup(item)}<span class="card-arrow" aria-hidden="true">↗</span></span>
      <span class="card-title"><span>${escapeHTML(item.type_name || item.type_code || "Garment")}</span><span class="card-id">${escapeHTML(item.id)}</span></span>
      <span class="card-subtitle">${ranked ? escapeHTML(matches || "Inspect ranking details") : "Not ranked"}</span>
      ${ranked ? `<span class="card-score"><span class="score-track" aria-hidden="true"><span style="width:${percentage(item.score)}%"></span></span><strong>${score(item.score)}</strong><span class="muted">final</span></span><span class="card-components"><span>G ${item.score_components?.includes("graph") ? score(item.graph_score) : "N/A"}</span><span>T ${score(item.text_score)}</span><span>S ${score(item.style_score)}</span></span>` : ""}
    </button>`;
  }).join("");
  handleImageErrors($("results-grid"));
  $("ranking-table").innerHTML = `<table><caption class="sr-only">${ranked ? "Retrieval scores; N/A means unavailable or inactive" : "Unranked catalog preview"}</caption><thead><tr>${ranked ? '<th scope="col">Rank</th>' : ""}<th scope="col">Item</th><th scope="col">Type</th>${ranked ? '<th scope="col">Final</th><th scope="col">Graph</th><th scope="col">Semantic</th><th scope="col">Style</th>' : ""}</tr></thead><tbody>${items.map((item, index) => `<tr>${ranked ? `<td>${index + 1}</td>` : ""}<td><button type="button" data-item="${index}" aria-label="Inspect ${escapeHTML(item.id)}">${escapeHTML(item.id)}</button></td><td>${escapeHTML(item.type_name || item.type_code || "Garment")}</td>${ranked ? `<td><strong>${score(item.score)}</strong></td><td>${item.score_components?.includes("graph") ? score(item.graph_score) : "N/A"}</td><td>${score(item.text_score)}</td><td>${score(item.style_score)}</td>` : ""}</tr>`).join("")}</tbody></table>`;
  $("empty-state").hidden = items.length > 0;
  $("empty-state").querySelector("h3").textContent = !ranked && !items.length ? "No preview images" : "No matching items";
  $("empty-state").querySelector("p").textContent = !ranked && !items.length
    ? "No preview images are available. You can still search a configured catalog and inspect its retrieval evidence."
    : "Try fewer constraints or lower the minimum score and edge confidence in Search settings.";
  applyView();
}

function applyView() {
  $("results-grid").hidden = state.view !== "grid";
  $("ranking-table").hidden = state.view !== "table" || !(state.run ? state.run.items.length : state.preview.length);
  for (const mode of ["grid", "table"]) {
    $(`${mode}-view`).classList.toggle("active", mode === state.view);
    $(`${mode}-view`).setAttribute("aria-pressed", String(mode === state.view));
  }
}

function tags(filters, excluded = false) {
  return `<div class="tags">${filters.map((f) => `<span class="tag${excluded ? " excluded" : ""}"><small>${escapeHTML(label(f.group))}</small>${excluded ? "− " : ""}${escapeHTML(f.value)}</span>`).join("")}</div>`;
}

function axisMarkup(target, itemValue) {
  const spec = state.config?.style_axes?.[target.axis];
  const poles = String(target.axis).split("_");
  const left = spec?.left || poles[0];
  const right = spec?.right || poles.at(-1);
  const numericTarget = typeof target.target === "number" ? target.target.toFixed(2) : "N/A";
  return `<div class="axis-row"><div class="axis-labels"><span>${escapeHTML(left)}</span><span>${escapeHTML(right)}</span></div><div class="axis-track" role="img" aria-label="${escapeHTML(left)} to ${escapeHTML(right)}, target ${numericTarget}"><span class="axis-dot" style="left:${percentage(target.target)}%"></span>${typeof itemValue === "number" ? `<span class="axis-dot item-value" style="left:${percentage(itemValue)}%"></span>` : ""}</div><p class="axis-evidence">Target ${numericTarget}${typeof itemValue === "number" ? ` · Item ${itemValue.toFixed(2)} (square)` : ""}${target.evidence ? ` · ${escapeHTML(target.evidence)}` : ""}</p></div>`;
}

// Highlight tokens after escaping every non-token segment; copied/exported text stays exact.
function highlightCypher(query) {
  const tokens = /('(?:[^'\\]|\\.)*'|\$[A-Za-z_][A-Za-z_0-9]*|\b(?:OPTIONAL|MATCH|WHERE|WITH|RETURN|ORDER|BY|DESC|ASC|LIMIT|CALL|AS|AND|OR|NOT|EXISTS|IN|CASE|WHEN|THEN|ELSE|END|IS|NULL|DISTINCT)\b)/g;
  let rendered = "", offset = 0;
  for (const match of query.matchAll(tokens)) {
    rendered += escapeHTML(query.slice(offset, match.index));
    const type = match[0].startsWith("$") ? "param" : match[0].startsWith("'") ? "string" : "keyword";
    rendered += `<span class="cypher-${type}">${escapeHTML(match[0])}</span>`;
    offset = match.index + match[0].length;
  }
  return rendered + escapeHTML(query.slice(offset));
}

function resetEvidence(running = false) {
  $("inspector-intro").hidden = false;
  $("search-insights").hidden = true;
  $("query-plan").hidden = false;
  $("query-state").textContent = running ? "Awaiting response" : "Not executed";
  $("cypher-code").textContent = running
    ? "Waiting for the retrieval response.\n\nThe executed query will appear here when the run finishes."
    : "No query has been generated.\n\nRun retrieval to inspect the executed Cypher here.";
  $("cypher-code").parentElement.classList.add("empty-code");
  $("parameter-list").innerHTML = `<p class="muted">${running ? "Waiting for parameters..." : "No parameters yet."}</p>`;
  $("params-code").textContent = running ? "Waiting for parameters..." : "No parameters yet.";
  $("extraction-code").textContent = running ? "Waiting for parser output..." : "No parser output yet.";
  $("inspector-intro").querySelectorAll("dd").forEach((el) => { el.textContent = running ? "Waiting for response" : "Not extracted"; });
  $("inspector-intro").querySelector("p").textContent = running
    ? "The parsed constraints will appear when retrieval completes."
    : "Run a query to inspect the parser output. No constraints have been inferred from the preview images.";
  $("run-summary").textContent = running ? "Search in progress. Settings are locked for this run." : "No completed run. Controls above apply to your next search.";
  $("run-state").textContent = running ? "Running" : state.lastAttemptFailed ? "Failed" : "Not run";
  for (const id of ["parse-time", "graph-time", "rerank-time", "candidate-count", "total-time"]) $(id).textContent = "—";
  for (const id of ["export-run", "copy-cypher", "copy-params", "expand-cypher"]) $(id).disabled = true;
}

function renderInsights() {
  const run = state.run;
  if (!run) { resetEvidence(); return; }
  $("inspector-intro").hidden = true;
  $("search-insights").hidden = false;
  $("query-plan").hidden = false;
  const extracted = run.extraction;
  const hard = [...(extracted.common_filters || []), ...(extracted.category_filters || [])];
  const excluded = [...(extracted.excluded_common_filters || []), ...(extracted.excluded_category_filters || [])];
  const types = (extracted.item_type_codes || []).map((code) => ({group: code, value: state.config?.type_names?.[code] || code}));
  const axes = extracted.style_axis_targets || [];
  $("search-insights").innerHTML = `
    <div class="insight-section"><h3 class="insight-label">Item types</h3>${types.length ? tags(types) : '<p class="muted">Any type</p>'}</div>
    <div class="insight-section"><h3 class="insight-label">Required attributes (${hard.length})</h3>${hard.length ? tags(hard) : '<p class="muted">None extracted</p>'}</div>
    <div class="insight-section"><h3 class="insight-label">Excluded attributes (${excluded.length})</h3>${excluded.length ? tags(excluded, true) : '<p class="muted">None extracted</p>'}</div>
    <div class="insight-section"><h3 class="insight-label">Semantic description</h3><p${extracted.description_query ? "" : ' class="muted"'}>${escapeHTML(extracted.description_query || "Not requested")}</p></div>
    <div class="insight-section"><h3 class="insight-label">Style targets (${axes.length})</h3>${axes.length ? axes.map((t) => axisMarkup(t)).join("") : '<p class="muted">Not requested</p>'}</div>`;
  $("cypher-code").innerHTML = highlightCypher(run.cypher);
  $("cypher-code").parentElement.classList.remove("empty-code");
  $("query-state").textContent = state.lastAttemptFailed ? "Previous run · executed query" : "Executed query";
  $("params-code").textContent = JSON.stringify(run.params, null, 2);
  $("parameter-list").innerHTML = `<dl>${Object.entries(run.params).map(([key, value]) => `<div class="parameter-row"><dt>$${escapeHTML(key)}</dt><dd>${escapeHTML(JSON.stringify(value))}</dd></div>`).join("")}</dl>`;
  $("extraction-code").textContent = JSON.stringify(extracted, null, 2);
  const settings = run.settings;
  $("run-summary").textContent = `Parser: ${settings.model} · Results: ${settings.limit} · Min. score: ${settings.min_score ?? "off"} · Min. edge confidence: ${settings.min_confidence ?? "off"}. ${run.reranked ? "Soft reranking requested." : "Graph ranking only."}`;
  $("run-state").textContent = state.lastAttemptFailed ? "Previous run" : "Completed";
  $("parse-time").textContent = `${run.timings.parse_ms} ms`;
  $("graph-time").textContent = `${run.timings.graph_ms} ms`;
  $("rerank-time").textContent = `${run.timings.rerank_ms} ms`;
  $("candidate-count").textContent = run.candidate_count;
  $("total-time").textContent = `${(run.timings.total_ms / 1000).toFixed(2)} s`;
  for (const id of ["export-run", "copy-cypher", "copy-params", "expand-cypher"]) $(id).disabled = false;
}

function attributeList(attributes) {
  return `<div class="attribute-list">${attributes.map((a) => `<div class="attribute-entry"><span><small>${escapeHTML(label(a.group))}</small><br>${escapeHTML(a.value)}</span><strong>${score(a.score)}</strong></div>`).join("")}</div>`;
}

function openItem(index) {
  const item = (state.run ? state.run.items : state.preview)[index];
  if (!item || state.busy) return;
  const ranked = Boolean(state.run);
  const components = item.score_components || [];
  const rows = [["Final", item.score, true], ["Graph", item.graph_score, components.includes("graph")], ["Semantic", item.text_score, components.includes("text")], ["Style", item.style_score, components.includes("style")]];
  $("item-dialog-content").innerHTML = `<div class="dialog-layout"><div>${imageMarkup(item, "dialog-image")}</div><div class="dialog-copy"><p class="eyebrow">${ranked ? `RANK ${String(index + 1).padStart(2, "0")} / MATCH EVIDENCE` : "CATALOG PREVIEW"}</p><h2 id="item-dialog-title">${escapeHTML(item.type_name || item.type_code || "Garment")}</h2><p>${escapeHTML(item.id)}</p>
    ${ranked ? `<div class="score-breakdown">${rows.map(([name, value, active]) => `<div class="score-row${active ? "" : " inactive"}"><span>${name}</span><span class="score-bar" aria-hidden="true"><span style="width:${active ? percentage(value) : 0}%"></span></span><strong>${active ? score(value) : "N/A"}</strong></div>`).join("")}</div><p class="dialog-note">${components.length ? `Final = mean of ${escapeHTML(components.map((c) => c === "text" ? "semantic" : c).join(" + "))}.` : "No active scoring components; final score falls back to the graph score."} N/A means unavailable or inactive. Scores are not probabilities.</p>
      <h3>Matched graph attributes</h3>${item.matched_filters?.length ? attributeList(item.matched_filters) : '<p class="muted">No scored attribute matches for this query.</p>'}
      ${item.style_axis_matches?.length ? `<h3>Style-axis evidence</h3>${item.style_axis_matches.map((t) => axisMarkup(t, t.value)).join("")}` : ""}
      ${item.mapped_attributes?.length ? `<h3>All mapped attributes</h3>${attributeList(item.mapped_attributes)}` : ""}`
    : '<p>Run retrieval to inspect matched attributes and component scores for this item.</p><div class="method-note">Preview only. This item has not been ranked for a query.</div>'}
  </div></div>`;
  handleImageErrors($("item-dialog-content"));
  $("item-dialog").showModal();
}

function setBusy(busy) {
  state.busy = busy;
  $("search-button").disabled = busy;
  $("search-button-text").textContent = busy ? "Running..." : "Run retrieval";
  $("results-section").setAttribute("aria-busy", String(busy));
  document.querySelectorAll("#search-form input, #search-form select, [data-query], .view-switch button").forEach((el) => { el.disabled = busy; });
  if (busy) {
    resetEvidence(true);
    $("results-grid").hidden = false;
    $("ranking-table").hidden = true;
    $("empty-state").hidden = true;
    $("results-eyebrow").textContent = "RETRIEVING";
    $("results-title").firstChild.textContent = "Search results";
    $("result-count").textContent = "—";
    $("results-caption").textContent = "Waiting for graph retrieval and ranking to complete.";
    $("catalog-footnote").textContent = "Search in progress. No results from this run are available yet.";
    $("results-grid").innerHTML = Array.from({length: 6}, () => '<div class="skeleton" aria-hidden="true"><div class="card-image"></div><div class="skeleton-line"></div><div class="skeleton-line short"></div></div>').join("");
  }
}

$("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  const query = $("query").value.trim();
  if (!query) { notice("Describe the fashion items you are looking for.", "error"); $("query").focus(); return; }
  if (!state.config?.ready) {
    notice(state.config?.preview
      ? "You’re viewing the catalog preview. Start the live web server with configured OpenAI and Neo4j credentials to run a search."
      : "Live search is not ready. Check the server’s OpenAI and Neo4j configuration, then reload the page.", "error");
    return;
  }
  const payload = {query, model: $("model").value.trim(), limit: Number($("limit").value), min_score: Number($("min-score").value), min_confidence: Number($("min-confidence").value)};
  state.lastAttemptFailed = false;
  const started = Date.now();
  setBusy(true);
  notice("Running query parsing, graph retrieval and reranking...", "loading");
  const timer = setInterval(() => notice(`Search in progress · ${Math.round((Date.now() - started) / 1000)}s elapsed. Waiting for the retrieval response.`, "loading"), 1000);
  try {
    state.run = await api("/api/search", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload), signal: AbortSignal.timeout(240000)});
    notice(`Found ${state.run.result_count} ranked ${state.run.result_count === 1 ? "match" : "matches"} in ${(state.run.timings.total_ms / 1000).toFixed(1)}s. Constraints, Cypher and ranking evidence are available below.`);
  } catch (error) {
    state.lastAttemptFailed = true;
    const message = error.name === "TimeoutError" ? "The search took too long. Please try a simpler query." : error.message || "Could not connect to the search server.";
    notice(`${message}${state.run ? " Your previous search results are shown below." : " The unranked catalog preview is shown below."}`, "error");
  } finally {
    clearInterval(timer);
    setBusy(false);
    renderItems();
    renderInsights();
  }
});

document.querySelectorAll("[data-query]").forEach((button) => button.addEventListener("click", () => {
  $("query").value = button.dataset.query;
  $("query").focus();
}));
document.querySelectorAll("[data-open-method]").forEach((button) => button.addEventListener("click", () => $("method-dialog").showModal()));
document.querySelectorAll("[data-close-dialog]").forEach((button) => button.addEventListener("click", () => button.closest("dialog").close()));
document.querySelectorAll("dialog").forEach((dialog) => dialog.addEventListener("click", (event) => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
}));
$("settings-toggle").addEventListener("click", () => {
  $("settings").hidden = !$("settings").hidden;
  $("settings-toggle").setAttribute("aria-expanded", String(!$("settings").hidden));
  $("settings-toggle").textContent = $("settings").hidden ? "Show settings" : "Hide settings";
});
for (const name of ["min-score", "min-confidence"]) $(name).addEventListener("input", () => { $(`${name}-value`).value = Number($(name).value).toFixed(2); });
for (const mode of ["grid", "table"]) $(`${mode}-view`).addEventListener("click", () => { state.view = mode; applyView(); });
for (const id of ["results-grid", "ranking-table"]) $(id).addEventListener("click", (event) => {
  const button = event.target.closest("[data-item]");
  if (button) openItem(Number(button.dataset.item));
});
$("revise-query").addEventListener("click", () => { $("query").focus(); $("query").scrollIntoView({block: "center"}); });
async function copyRunText(buttonId, codeId, text) {
  const button = $(buttonId);
  const originalLabel = button.textContent;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied";
  } catch {
    const code = $(codeId);
    const details = code.closest("details");
    if (details) details.open = true;
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges(); selection.addRange(range);
    code.scrollIntoView({block: "nearest"});
    button.textContent = "Selected: Ctrl+C";
  }
  setTimeout(() => { button.textContent = originalLabel; }, 2500);
}
$("copy-cypher").addEventListener("click", () => {
  if (state.run && !state.busy) copyRunText("copy-cypher", "cypher-code", state.run.cypher);
});
$("copy-params").addEventListener("click", () => {
  if (state.run && !state.busy) copyRunText("copy-params", "params-code", JSON.stringify(state.run.params, null, 2));
});
$("wrap-cypher").addEventListener("change", () => {
  $("cypher-code").parentElement.classList.toggle("no-wrap", !$("wrap-cypher").checked);
});
$("expand-cypher").addEventListener("click", () => {
  if (!state.run || state.busy) return;
  $("expanded-cypher-code").innerHTML = highlightCypher(state.run.cypher);
  $("cypher-dialog").showModal();
});
$("export-run").addEventListener("click", () => {
  if (!state.run || state.busy) return;
  const blob = new Blob([JSON.stringify(state.run, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `fashion-search-run-${new Date().toISOString().replaceAll(":", "-")}.json`;
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

async function init() {
  const results = await Promise.allSettled([api("/api/config"), api("/api/preview")]);
  if (results[0].status === "fulfilled") {
    state.config = results[0].value;
    $("model").value = state.config.default_model;
    $("model-options").innerHTML = state.config.models.map((model) => `<option value="${escapeHTML(model)}"></option>`).join("");
    $("runtime-label").innerHTML = '<span class="status-dot"></span>' + (state.config.ready ? " Live search configured" : state.config.preview ? " Catalog preview · Search offline" : " Catalog preview · Search not configured");
  } else {
    $("runtime-label").textContent = "Server unavailable";
    notice("Could not connect to the server. Reload the page after checking the server is running.", "error");
  }
  if (results[1].status === "fulfilled") {
    state.preview = results[1].value.items;
    for (const item of state.preview) {
      const url = safeImageSource(item.image_url);
      if (url && item.image_expires_at * 1000 > Date.now()) {
        imageCache.set(item.image_id, {url, expires_at: item.image_expires_at});
      }
    }
    if (results[1].value.status === "unavailable") notice("Preview images could not be loaded. You can still run a search, or reload to retry the preview.", "error");
  }
  else notice("The collection could not be loaded. Reload the page to try again.", "error");
  renderItems();
  renderInsights();
}

init();
