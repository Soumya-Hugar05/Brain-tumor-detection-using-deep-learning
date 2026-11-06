/* NeuroInsight – Reports page (vanilla JavaScript). Server text is inserted with textContent only. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const el = {
    q: $("q"), filterBtn: $("filterBtn"), filterLabel: $("filterLabel"), filters: $("filters"), exportBtn: $("exportBtn"),
    fClass: $("fClass"), fFrom: $("fFrom"), fTo: $("fTo"), fSort: $("fSort"), fSize: $("fSize"), clearFilters: $("clearFilters"),
    statTotal: $("statTotal"), statToday: $("statToday"), statMonth: $("statMonth"),
    loading: $("loading"), errorBox: $("errorBox"), emptyAll: $("emptyAll"), emptyFiltered: $("emptyFiltered"), emptyClear: $("emptyClear"),
    tableWrap: $("tableWrap"), rows: $("rows"), sortDate: $("sortDate"), sortArrow: $("sortArrow"), dateTh: $("dateTh"),
    pager: $("pager"), pageInfo: $("pageInfo"), pageNum: $("pageNum"), prevPage: $("prevPage"), nextPage: $("nextPage"),
    pvDialog: $("previewDialog"), pvTitle: $("pvTitle"), pvBody: $("pvBody"), pvClose: $("pvClose"),
    pvPdf: $("pvPdf"), pvJson: $("pvJson"), pvPng: $("pvPng"), pvDelete: $("pvDelete"),
    cfDialog: $("confirmDialog"), cfText: $("cfText"), cfCancel: $("cfCancel"), cfOk: $("cfOk"),
  };
  const CLASS_LABEL = { glioma: "Glioma", meningioma: "Meningioma", notumor: "No tumor", pituitary: "Pituitary tumor" };
  const nice = (c) => CLASS_LABEL[c] || c;
  const pct = (x, d = 2) => (x * 100).toFixed(d) + "%";
  const TZ = new Date().getTimezoneOffset();             // minutes, UTC minus local: "today" and date filters use your local days
  const state = { q: "", cls: "", from: "", to: "", sort: "newest", page: 1, pageSize: 10 };
  let reqId = 0, lastTotal = 0, preview = null, previewOpener = null;

  function node(tag, props, children) {
    const n = document.createElement(tag);
    Object.entries(props || {}).forEach(([k, v]) => { if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v); });
    (children || []).forEach((c) => n.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return n;
  }
  const localTime = (iso) => new Date(iso).toLocaleString([], { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
  const utcTime = (iso) => iso.replace("T", " ").replace("Z", " UTC");
  const hasFilters = () => !!(state.q || state.cls || state.from || state.to);

  /* ---------- loading the list ---------- */
  function queryString(extra) {
    const p = new URLSearchParams({ sort: state.sort, page: String(state.page), page_size: String(state.pageSize), tz_offset: String(TZ) });
    if (state.q) p.set("q", state.q);
    if (state.cls) p.set("cls", state.cls);
    if (state.from) p.set("date_from", state.from);
    if (state.to) p.set("date_to", state.to);
    Object.entries(extra || {}).forEach(([k, v]) => p.set(k, v));
    return p.toString();
  }
  function show(which) {                                  // "loading" | "table" | "emptyAll" | "emptyFiltered" | "error"
    el.loading.hidden = which !== "loading"; el.tableWrap.hidden = which !== "table"; el.emptyAll.hidden = which !== "emptyAll";
    el.emptyFiltered.hidden = which !== "emptyFiltered"; el.errorBox.hidden = which !== "error";
    if (which !== "table") el.pager.hidden = true;
  }
  async function load() {
    const mine = ++reqId;
    show("loading");
    try {
      const resp = await fetch("/api/reports?" + queryString());
      const data = await resp.json().catch(() => null);
      if (mine !== reqId) return;                         // a newer request has been made
      if (!resp.ok || !data || !data.ok) throw new Error((data && data.error) || "Could not load the reports.");
      state.page = data.page; lastTotal = data.total;
      el.statTotal.textContent = String(data.stats.total); el.statToday.textContent = String(data.stats.today); el.statMonth.textContent = String(data.stats.this_month);
      if (data.total === 0) { show(hasFilters() ? "emptyFiltered" : "emptyAll"); return; }
      renderRows(data.items);
      show("table");
      el.pager.hidden = false;
      const from = (data.page - 1) * data.page_size + 1, to = from + data.items.length - 1;
      el.pageInfo.textContent = "Showing " + from + "–" + to + " of " + data.total;
      el.pageNum.textContent = "Page " + data.page + " of " + data.pages;
      el.prevPage.disabled = data.page <= 1; el.nextPage.disabled = data.page >= data.pages;
    } catch (err) {
      if (mine !== reqId) return;
      show("error");
      el.errorBox.textContent = err instanceof TypeError ? "Could not reach the server. Check that the app is running." : err.message;
    }
  }

  function renderRows(items) {
    el.rows.replaceChildren(...items.map((r) => {
      const act = (label, action, cls, extra) => node("button", Object.assign({ type: "button", class: "act " + (cls || ""), "data-action": action, "data-id": r.report_id,
        "aria-label": label + " " + r.report_id }, extra || {}), []);
      const btn = (text, action, cls, aria) => { const b = act(aria, action, cls); b.textContent = text; return b; };
      const conf = node("span", { class: "conf", "data-level": r.warning_level, text: pct(r.confidence) });
      if (r.warning_level !== "none") conf.setAttribute("title", r.warning_level === "low" ? "Low confidence" : "Review recommended");
      return node("tr", {}, [
        node("td", { "data-label": "Report ID" }, [node("span", { class: "mono", text: r.report_id })]),
        node("td", { "data-label": "Case ID" }, [node("span", { class: "muted", title: "No case or patient information is collected", text: "—" })]),
        node("td", { "data-label": "MRI image" }, [node("span", { class: "cell-file", title: r.file_name, text: r.file_name })]),
        node("td", { "data-label": "Predicted class" }, [node("span", { class: "class-tag", "data-class": r.predicted_class, text: nice(r.predicted_class) })]),
        node("td", { class: "num", "data-label": "Confidence" }, [conf]),
        node("td", { "data-label": "Date generated" }, [node("time", { datetime: r.created_at, title: utcTime(r.created_at), text: localTime(r.created_at) })]),
        node("td", { class: "cell-actions" }, [node("div", { class: "row-actions", role: "group", "aria-label": "Actions for " + r.report_id }, [
          btn("View", "view", "act-primary", "View report"), btn("PDF", "pdf", "", "Download PDF for report"),
          btn("JSON", "json", "", "Download JSON for report"), btn("PNG", "png", "", "Download PNG for report"),
          btn("Delete", "delete", "act-danger", "Delete report")])]),
      ]);
    }));
  }

  /* ---------- search, filters, sort, pagination ---------- */
  let debounce = null;
  el.q.addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(() => { state.q = el.q.value.trim(); state.page = 1; load(); }, 300); });
  el.q.addEventListener("keydown", (e) => { if (e.key === "Enter") { clearTimeout(debounce); state.q = el.q.value.trim(); state.page = 1; load(); } });
  el.filterBtn.addEventListener("click", () => {
    const open = el.filters.hidden; el.filters.hidden = !open; el.filterBtn.setAttribute("aria-expanded", String(open));
  });
  function updateFilterLabel() {
    const n = [state.cls, state.from, state.to].filter(Boolean).length;
    el.filterLabel.textContent = n ? "Filter (" + n + ")" : "Filter";
  }
  function applyControls() {
    const prevFrom = state.from, prevTo = state.to;
    state.cls = el.fClass.value; state.from = el.fFrom.value; state.to = el.fTo.value; state.sort = el.fSort.value;
    state.pageSize = parseInt(el.fSize.value, 10) || 10; state.page = 1;
    if (state.from && state.to && state.from > state.to) {
      NI.toast("The start date is after the end date.", "error");
      state.from = prevFrom; state.to = prevTo; el.fFrom.value = prevFrom; el.fTo.value = prevTo; return;
    }
    syncSortUi(); updateFilterLabel(); load();
  }
  [el.fClass, el.fFrom, el.fTo, el.fSort, el.fSize].forEach((c) => c.addEventListener("change", applyControls));
  function syncSortUi() {
    el.fSort.value = state.sort;
    el.sortArrow.textContent = state.sort === "newest" ? "↓" : "↑";
    el.dateTh.setAttribute("aria-sort", state.sort === "newest" ? "descending" : "ascending");
  }
  el.sortDate.addEventListener("click", () => { state.sort = state.sort === "newest" ? "oldest" : "newest"; state.page = 1; syncSortUi(); load(); });
  function clearAll() {
    Object.assign(state, { q: "", cls: "", from: "", to: "", page: 1 });
    el.q.value = ""; el.fClass.value = ""; el.fFrom.value = ""; el.fTo.value = "";
    updateFilterLabel(); load();
  }
  el.clearFilters.addEventListener("click", clearAll);
  el.emptyClear.addEventListener("click", clearAll);
  el.prevPage.addEventListener("click", () => { if (state.page > 1) { state.page -= 1; load(); } });
  el.nextPage.addEventListener("click", () => { state.page += 1; load(); });

  /* ---------- downloads ---------- */
  async function download(id, fmt, button) {
    if (button) { button.setAttribute("aria-busy", "true"); button.disabled = true; }
    NI.toast("Preparing " + fmt.toUpperCase() + " for " + id + "…", "info", 1800);
    try {
      await NI.downloadReport(id, fmt);
      NI.toast(fmt.toUpperCase() + " downloaded: " + id, "success");
    } catch (err) {
      NI.toast(err instanceof TypeError ? "Could not reach the server." : err.message, "error");
    } finally {
      if (button) { button.removeAttribute("aria-busy"); button.disabled = false; }
    }
  }
  el.exportBtn.addEventListener("click", async () => {
    el.exportBtn.disabled = true;
    try {
      const resp = await fetch("/api/reports/export?" + queryString());
      if (!resp.ok) { const d = await resp.json().catch(() => null); throw new Error((d && d.error) || "Export failed."); }
      NI.saveBlob(await resp.blob(), "neuroinsight-reports.csv");
      NI.toast("CSV exported (metadata only; no images).", "success");
    } catch (err) { NI.toast(err.message, "error"); } finally { el.exportBtn.disabled = false; }
  });

  /* ---------- delete with confirmation ---------- */
  function confirmDelete(id, opener) {
    el.cfText.textContent = "Report " + id + " will be removed permanently from this list. Files you already downloaded are not affected.";
    el.cfOk.onclick = async () => {
      el.cfOk.disabled = true;
      try {
        const resp = await fetch("/api/reports/" + encodeURIComponent(id), { method: "DELETE" });
        const d = await resp.json().catch(() => null);
        if (!resp.ok || !d || !d.ok) throw new Error((d && d.error) || "Could not delete the report.");
        el.cfDialog.close(); if (el.pvDialog.open) el.pvDialog.close();
        NI.toast("Report deleted: " + id, "success");
        if (state.page > 1 && el.rows.children.length === 1) state.page -= 1;
        load();
      } catch (err) { NI.toast(err.message, "error"); } finally { el.cfOk.disabled = false; }
    };
    el.cfDialog.showModal();
    el.cfCancel.focus();
  }
  el.cfCancel.addEventListener("click", () => el.cfDialog.close());

  /* ---------- report preview ---------- */
  function kv(pairs) {
    return node("dl", { class: "rs-kv" }, pairs.flatMap(([k, v, mono]) => [node("dt", { text: k }), node("dd", {}, [mono ? node("span", { class: "mono", text: v }) : document.createTextNode(v)])]));
  }
  function section(title, child) { return node("section", { class: "rs-section" }, [node("h3", { text: title }), child]); }

  function renderPreview(r) {
    const p = r.prediction, w = r.warning || r.confidence_warning || { level: "none" }, m = r.model || {};
    const logo = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    logo.setAttribute("viewBox", "0 0 40 40"); logo.setAttribute("class", "logo"); logo.setAttribute("aria-hidden", "true");
    const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect"); bg.setAttribute("width", "40"); bg.setAttribute("height", "40"); bg.setAttribute("rx", "11"); bg.setAttribute("class", "logo-bg");
    const ln = document.createElementNS("http://www.w3.org/2000/svg", "path"); ln.setAttribute("d", "M7 21h6l3-8 5 15 3-9 2 2h7"); ln.setAttribute("class", "logo-line");
    logo.append(bg, ln);
    const warn = w.level !== "none" && w.message ? [node("div", { class: "rs-warn " + w.level, text: w.message })] : [];
    const sheet = node("article", { class: "report-sheet" }, [
      node("div", { class: "rs-head" }, [
        node("div", { class: "rs-brand" }, [logo, node("div", {}, [node("b", { text: "NeuroInsight" }), node("span", { text: "Brain MRI Analysis Report" })])]),
        node("span", { class: "rs-badge", text: "RESEARCH PROTOTYPE" })]),
      node("div", { class: "rs-strip", text: "Research prototype – not a medical diagnostic report. Not for clinical use." }),
      node("div", { class: "rs-body" }, [
        section("Report information", kv([["Report ID", r.report_id, true], ["Generated", utcTime(r.created_at) + "  (" + localTime(r.created_at) + " local)"],
          ["Case ID", "Not available (no patient information is collected)"], ["Image file", r.file_name], ["Prediction ID", r.prediction_id, true]])),
        section("MRI analysis", node("div", { class: "rs-analysis" }, [
          node("div", { class: "rs-image" }, [node("span", { text: "Original MRI image is not stored with reports" })]),
          node("div", {}, [node("div", { class: "rs-eyebrow", text: "Predicted category" }), node("div", { class: "rs-class", text: nice(p.label) }),
            node("div", { class: "rs-eyebrow", text: "Confidence", style: "margin-top:10px" }), node("div", { class: "rs-conf", text: pct(p.confidence) }),
            ...warn, node("div", { class: "rs-note", text: "Model output only – not a diagnosis. No clinical interpretation or recommendation is provided." })])])),
        section("Class probabilities", node("div", { class: "rs-prob" }, p.ranked.map((it, i) => {
          const fill = node("div"); fill.style.width = (it.prob * 100).toFixed(2) + "%";
          return node("div", { class: "rs-prob-row" + (i === 0 ? " best" : ""), "data-class": it.label }, [
            node("div", { class: "rs-prob-head" }, [node("span", { text: nice(it.label) }), node("span", { text: pct(it.prob) })]), node("div", { class: "rs-bar" }, [fill])]);
        }))),
        section("Model information", kv([["Model", m.name || ""], ["Bundle", m.description || ""], ["Feature extractor", m.feature || ""], ["Classifier", m.classifier || ""],
          ["Decision rule", m.decision_rule || ""], ["Preprocessing", m.preprocessing_detail || m.preprocessing || ""],
          ["Explainability", (r.explanation && r.explanation.reason) || "Grad-CAM is disabled until validated."],
          ["Estimator SHA-256", m.estimator_sha256 || "", true], ["Manifest SHA-256", m.manifest_sha256 || "", true]])),
        node("div", { class: "rs-disclaimer" }, [node("b", { text: "Disclaimer" }), document.createTextNode(r.disclaimer)]),
      ]),
    ]);
    el.pvBody.replaceChildren(sheet);
    el.pvTitle.textContent = "Report " + r.report_id;
  }

  async function openPreview(id, opener) {
    previewOpener = opener || null;
    try {
      const resp = await fetch("/api/reports/" + encodeURIComponent(id));
      const d = await resp.json().catch(() => null);
      if (!resp.ok || !d || !d.ok) throw new Error((d && d.error) || "Could not load the report.");
      preview = d.report;
      renderPreview(preview);
      if (!el.pvDialog.open) el.pvDialog.showModal();
      el.pvBody.scrollTop = 0;
    } catch (err) { NI.toast(err instanceof TypeError ? "Could not reach the server." : err.message, "error"); }
  }
  el.pvClose.addEventListener("click", () => el.pvDialog.close());
  el.pvDialog.addEventListener("click", (e) => { if (e.target === el.pvDialog) el.pvDialog.close(); });   // click on the backdrop
  el.pvDialog.addEventListener("close", () => { preview = null; if (previewOpener && document.contains(previewOpener)) previewOpener.focus(); });
  el.pvPdf.addEventListener("click", () => preview && download(preview.report_id, "pdf", el.pvPdf));
  el.pvJson.addEventListener("click", () => preview && download(preview.report_id, "json", el.pvJson));
  el.pvPng.addEventListener("click", () => preview && download(preview.report_id, "png", el.pvPng));
  el.pvDelete.addEventListener("click", () => preview && confirmDelete(preview.report_id, el.pvDelete));

  el.rows.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-action]");
    if (!b) return;
    const id = b.dataset.id, action = b.dataset.action;
    if (action === "view") openPreview(id, b);
    else if (action === "delete") confirmDelete(id, b);
    else download(id, action, b);
  });

  /* ---------- start ---------- */
  syncSortUi();
  load().then(() => {
    const view = new URLSearchParams(location.search).get("view");   // deep link from the analysis page: /reports?view=NI-...
    if (view && /^NI-\d{8}-[0-9A-F]{8}$/.test(view)) {
      openPreview(view, null);
      history.replaceState(null, "", location.pathname);
    }
  });
})();
