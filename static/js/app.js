/* NeuroInsight front end (vanilla JavaScript). Server text is always inserted with textContent, never innerHTML.
   Sections: helpers · theme · image viewer · file selection · prediction · results · samples · downloads · about */
(function () {
  "use strict";
  const CFG = window.NEUROINSIGHT || { maxBytes: 10 * 1024 * 1024, allowedExt: ["jpg", "jpeg", "png", "bmp"] };
  const $ = (id) => document.getElementById(id);
  const el = {
    dropTarget: $("dropTarget"), drop: $("drop"), file: $("file"), previewEmpty: $("previewEmpty"),
    viewer: $("viewer"), viewport: $("viewport"), preview: $("preview"),
    zoomIn: $("zoomIn"), zoomOut: $("zoomOut"), zoomFit: $("zoomFit"), zoomReset: $("zoomReset"), zoomLabel: $("zoomLabel"),
    fullscreen: $("fullscreen"),
    fileInfo: $("fileInfo"), fileName: $("fileName"), fileDims: $("fileDims"), fileSize: $("fileSize"),
    btn: $("btn"), btnLabel: $("btnLabel"), clear: $("clear"), error: $("error"), sampleGrid: $("sampleGrid"),
    empty: $("empty"), loading: $("loading"), result: $("result"),
    rLabel: $("rLabel"), rConf: $("rConf"), gauge: $("gauge"), rWarn: $("rWarn"), rSample: $("rSample"),
    probList: $("probList"), explainText: $("explainText"), dlJson: $("dlJson"), dlPng: $("dlPng"), dlPdf: $("dlPdf"), saveReport: $("saveReport"), saveLabel: $("saveLabel"),
    viewSaved: $("viewSaved"), aboutBody: $("aboutBody"),
  };

  const state = { file: null, previewUrl: null, sample: null, result: null, controller: null, reportId: null, saving: null };

  /* ---------- helpers ---------- */
  function node(tag, props, children) {
    const n = document.createElement(tag);
    Object.entries(props || {}).forEach(([k, v]) => {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    });
    (children || []).forEach((c) => n.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return n;
  }
  const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const CLASS_LABEL = { glioma: "Glioma", meningioma: "Meningioma", notumor: "No tumor", pituitary: "Pituitary tumor" };
  const nice = (c) => CLASS_LABEL[c] || cap(c);
  const fmtBytes = (n) => (n < 1024 ? n + " B" : n < 1048576 ? (n / 1024).toFixed(0) + " KB" : (n / 1048576).toFixed(1) + " MB");
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  function showError(msg) { el.error.textContent = msg; el.error.hidden = false; }
  function resetError() { el.error.hidden = true; el.error.textContent = ""; }

  /* ---------- image viewer (zoom / pan / fullscreen; acts on the preview only, never on the uploaded file) ---------- */
  const view = { scale: 1, x: 0, y: 0, fit: true };
  const MAX_SCALE = 8, STEP = 1.25;
  const dims = () => ({ vw: el.viewport.clientWidth, vh: el.viewport.clientHeight, iw: el.preview.naturalWidth, ih: el.preview.naturalHeight });
  function fitScale() { const d = dims(); return d.iw && d.vw ? Math.min(d.vw / d.iw, d.vh / d.ih) : 1; }
  function minScale() { return Math.max(0.05, fitScale() * 0.5); }
  function clampPan() {
    const d = dims(), m = 48;
    view.x = clamp(view.x, m - d.iw * view.scale, d.vw - m);
    view.y = clamp(view.y, m - d.ih * view.scale, d.vh - m);
  }
  function applyView() {
    el.preview.style.transform = "translate(" + view.x + "px," + view.y + "px) scale(" + view.scale + ")";
    el.zoomLabel.textContent = Math.round(view.scale * 100) + "%";
    el.viewport.classList.toggle("pannable", view.scale > fitScale() * 1.001);
  }
  function centerAt(scale) {
    const d = dims();
    view.scale = scale; view.x = (d.vw - d.iw * scale) / 2; view.y = (d.vh - d.ih * scale) / 2;
  }
  function viewFit() { centerAt(fitScale()); view.fit = true; applyView(); }
  function viewActual() { centerAt(1); view.fit = false; applyView(); }
  function zoomAround(factor, cx, cy) {
    const old = view.scale, s = clamp(old * factor, minScale(), MAX_SCALE);
    view.x = cx - (cx - view.x) * (s / old); view.y = cy - (cy - view.y) * (s / old);
    view.scale = s; view.fit = false; clampPan(); applyView();
  }
  const zoomCentre = (f) => zoomAround(f, el.viewport.clientWidth / 2, el.viewport.clientHeight / 2);

  el.zoomIn.addEventListener("click", () => zoomCentre(STEP));
  el.zoomOut.addEventListener("click", () => zoomCentre(1 / STEP));
  el.zoomFit.addEventListener("click", viewFit);
  el.zoomReset.addEventListener("click", viewActual);
  el.viewport.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "+" || e.key === "=") { e.preventDefault(); zoomCentre(STEP); }
    else if (e.key === "-" || e.key === "_") { e.preventDefault(); zoomCentre(1 / STEP); }
    else if (e.key === "0") { e.preventDefault(); viewActual(); }
    else if (e.key === "f" || e.key === "F") { e.preventDefault(); viewFit(); }
  });
  el.viewport.addEventListener("wheel", (e) => {          // Ctrl/Cmd + wheel zooms; a plain wheel still scrolls the page
    if (!(e.ctrlKey || e.metaKey) || !el.preview.naturalWidth) return;
    e.preventDefault();
    const r = el.viewport.getBoundingClientRect();
    zoomAround(e.deltaY < 0 ? STEP : 1 / STEP, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  let pan = null;                                         // mouse drag to pan when zoomed in
  el.viewport.addEventListener("pointerdown", (e) => {
    if (e.pointerType !== "mouse" || e.button !== 0 || !el.viewport.classList.contains("pannable")) return;
    pan = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
    el.viewport.setPointerCapture(e.pointerId); el.viewport.classList.add("panning");
  });
  el.viewport.addEventListener("pointermove", (e) => {
    if (!pan) return;
    view.x = pan.vx + (e.clientX - pan.x); view.y = pan.vy + (e.clientY - pan.y); clampPan(); applyView();
  });
  const endPan = () => { pan = null; el.viewport.classList.remove("panning"); };
  el.viewport.addEventListener("pointerup", endPan);
  el.viewport.addEventListener("pointercancel", endPan);

  if (el.viewer.requestFullscreen && document.fullscreenEnabled) {
    el.fullscreen.addEventListener("click", () => {
      if (document.fullscreenElement) document.exitFullscreen(); else el.viewer.requestFullscreen().catch(() => {});
    });
    document.addEventListener("fullscreenchange", () => {
      const on = document.fullscreenElement === el.viewer;
      el.fullscreen.setAttribute("aria-pressed", String(on));
      el.fullscreen.setAttribute("aria-label", on ? "Exit fullscreen" : "View fullscreen");
      if (view.fit) viewFit();
    });
  } else {
    el.fullscreen.hidden = true;                          // e.g. browsers without the Fullscreen API
  }
  if (window.ResizeObserver) new ResizeObserver(() => { if (!el.viewer.hidden && view.fit && el.preview.naturalWidth) viewFit(); }).observe(el.viewport);
  el.preview.addEventListener("load", () => {
    el.fileDims.textContent = el.preview.naturalWidth + " × " + el.preview.naturalHeight + " px";
    viewFit();
  });

  /* ---------- file selection ---------- */
  function validateClient(f) {
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    if (!CFG.allowedExt.includes(ext)) return "Unsupported file type. Please choose a JPG, JPEG, PNG or BMP image.";
    if (f.size === 0) return "The file is empty.";
    if (f.size > CFG.maxBytes) return "The file is too large (limit " + Math.round(CFG.maxBytes / 1048576) + " MB).";
    return null;
  }

  function showResultPanel(which) {                       // which: "empty" | "loading" | "result"
    el.empty.hidden = which !== "empty"; el.loading.hidden = which !== "loading"; el.result.hidden = which !== "result";
  }

  function setFile(f, sampleInfo) {
    if (state.controller) state.controller.abort();       // a pending request belongs to the previous file
    resetError();
    const problem = f ? validateClient(f) : null;
    if (problem) { clearSelection(); showError(problem); return; }
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = f; state.sample = sampleInfo || null; state.result = null; state.reportId = null;
    if (f) {
      state.previewUrl = URL.createObjectURL(f);
      el.fileDims.textContent = "";
      el.preview.src = state.previewUrl;
      el.preview.hidden = false; el.viewer.hidden = false; el.drop.hidden = true; el.previewEmpty.hidden = true;
      el.fileName.textContent = f.name + (sampleInfo ? "  (demo sample)" : "");
      el.fileSize.textContent = fmtBytes(f.size) + " · " + (f.type ? f.type.replace("image/", "").toUpperCase() : "image");
      el.fileInfo.hidden = false; el.clear.hidden = false;
    }
    el.btn.disabled = !f;
    showResultPanel("empty");
    document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-pressed", String(!!sampleInfo && b.dataset.file === sampleInfo.file)));
  }

  function clearSelection() {
    if (state.controller) state.controller.abort();
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    Object.assign(state, { file: null, previewUrl: null, sample: null, result: null, reportId: null });
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    el.preview.removeAttribute("src"); el.preview.hidden = true; el.viewer.hidden = true;
    el.drop.hidden = false; el.previewEmpty.hidden = false; el.fileInfo.hidden = true; el.clear.hidden = true;
    el.fileName.textContent = ""; el.fileDims.textContent = ""; el.fileSize.textContent = "";
    el.btn.disabled = true; el.file.value = "";
    showResultPanel("empty"); resetError();
    document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-pressed", "false"));
  }

  const openPicker = () => el.file.click();
  el.drop.addEventListener("click", openPicker);
  el.drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPicker(); } });
  el.file.addEventListener("change", () => { if (el.file.files && el.file.files[0]) setFile(el.file.files[0]); });
  el.clear.addEventListener("click", clearSelection);
  ["dragenter", "dragover"].forEach((ev) => el.dropTarget.addEventListener(ev, (e) => { e.preventDefault(); el.dropTarget.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => el.dropTarget.addEventListener(ev, (e) => { e.preventDefault(); el.dropTarget.classList.remove("dragover"); }));
  el.dropTarget.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  /* ---------- prediction ---------- */
  async function analyse() {
    if (!state.file) return;
    if (state.controller) state.controller.abort();
    const controller = (state.controller = new AbortController());
    resetError();
    showResultPanel("loading"); el.btn.disabled = true; el.btnLabel.textContent = "Analysing…";
    const fd = new FormData();
    fd.append("file", state.file, state.file.name);        // the original file, untouched by any viewer zoom or pan
    try {
      const resp = await fetch("/api/predict", { method: "POST", body: fd, signal: controller.signal });
      let data = null;
      try { data = await resp.json(); } catch (_) { /* non-JSON error page */ }
      if (!resp.ok || !data || !data.ok) {
        throw new Error((data && data.error) || "The server returned an error (" + resp.status + "). Please try again.");
      }
      state.result = data;
      renderResult(data);
    } catch (err) {
      if (err.name === "AbortError") return;
      showResultPanel("empty");
      showError(err instanceof TypeError ? "Could not reach the server. Check that the app is running and try again." : err.message);
    } finally {
      if (state.controller === controller) {
        el.loading.hidden = true; el.btn.disabled = !state.file; el.btnLabel.textContent = "Analyse image";
      }
    }
  }
  el.btn.addEventListener("click", analyse);

  /* ---------- results ---------- */
  function renderResult(d) {
    const p = d.prediction;
    el.rLabel.textContent = nice(p.label);
    el.rConf.textContent = pct(p.confidence, 2);
    el.gauge.style.setProperty("--p", (p.confidence * 100).toFixed(2));
    el.gauge.setAttribute("data-tier", d.warning.level);
    el.gauge.setAttribute("aria-label", "Model confidence " + pct(p.confidence, 2));

    el.rWarn.className = "alert " + d.warning.level;
    if (d.warning.level === "none") { el.rWarn.hidden = true; el.rWarn.textContent = ""; }
    else { el.rWarn.textContent = d.warning.message; el.rWarn.hidden = false; }

    if (state.sample) {
      const agree = state.sample.dataset_label === p.label;
      el.rSample.textContent = "Demo sample from the validation split. Dataset label: " + nice(state.sample.dataset_label) +
        (agree ? " (matches the prediction)." : " (does not match the prediction).");
      el.rSample.className = "sample-note " + (agree ? "agree" : "disagree");
      el.rSample.hidden = false;
    } else el.rSample.hidden = true;

    el.probList.replaceChildren(...p.ranked.map((item, i) => {
      const name = node("span", { class: "prob-name" }, [node("span", { text: nice(item.label) })]);
      if (i === 0) name.appendChild(node("span", { class: "prob-chip", text: "Predicted" }));
      const fill = node("div"); fill.style.width = (item.prob * 100).toFixed(2) + "%";
      return node("div", { class: "prob-item" + (i === 0 ? " best" : ""), "data-class": item.label }, [
        node("div", { class: "prob-head" }, [name, node("span", { class: "prob-pct", text: pct(item.prob, 2) })]),
        node("div", { class: "bar", role: "presentation" }, [fill]),
      ]);
    }));

    // Grad-CAM stays disabled; this is the hook where a validated explanation can be rendered later.
    renderExplanation(d.explanation);
    resetSaveUi();
    showResultPanel("result");
    // In the single-column (tablet/phone) layout the result sits below the upload card: bring it into view.
    if (window.matchMedia("(max-width: 1020px)").matches) {
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      document.querySelector(".result-card").scrollIntoView({ block: "start", behavior: reduce ? "auto" : "smooth" });
    }
  }

  function renderExplanation(x) {
    if (x && x.available) { el.explainText.textContent = "Explanation available."; return; }
    el.explainText.textContent = (x && x.reason) || "Explanations are currently unavailable.";
  }

  /* ---------- demo samples ---------- */
  async function loadSamples() {
    try {
      const r = await fetch("/api/samples"); const d = await r.json();
      if (!d.samples || !d.samples.length) { el.sampleGrid.closest(".card").hidden = true; return; }
      const counts = {};
      el.sampleGrid.replaceChildren(...d.samples.map((s) => {
        counts[s.dataset_label] = (counts[s.dataset_label] || 0) + 1;
        const b = node("button", { type: "button", class: "sample", "data-file": s.file, "aria-pressed": "false",
          "aria-label": "Load demo sample: " + nice(s.dataset_label) + " " + counts[s.dataset_label] }, [
          node("img", { src: s.url, alt: "", loading: "lazy" }),
          node("span", { class: "sample-cap" }, [node("b", { text: nice(s.dataset_label) }), node("span", { text: "Demo " + counts[s.dataset_label] })]),
        ]);
        b.addEventListener("click", async () => {
          try {
            const blob = await (await fetch(s.url)).blob();
            setFile(new File([blob], s.file, { type: blob.type || "image/jpeg" }), s);
          } catch (_) { showError("Could not load the demo sample."); }
        });
        return b;
      }));
    } catch (_) { el.sampleGrid.closest(".card").hidden = true; }
  }

  /* ---------- saving a report to the Reports history (metadata only; the server stores no image) ---------- */
  function resetSaveUi() {
    el.saveReport.disabled = false; el.saveLabel.textContent = "Save to Reports"; el.viewSaved.hidden = true;
    el.dlPdf.disabled = false;
  }
  function markSaved(id) {
    el.saveReport.disabled = true; el.saveLabel.textContent = "Saved ✓";
    el.viewSaved.href = "/reports?view=" + encodeURIComponent(id); el.viewSaved.hidden = false;
  }
  async function saveReport() {
    const result = state.result;
    if (!result || !result.prediction_id) return null;
    if (state.reportId) return state.reportId;
    if (state.saving) return state.saving;
    el.saveReport.disabled = true; el.saveLabel.textContent = "Saving…";
    state.saving = (async () => {
      try {
        const resp = await fetch("/api/reports", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prediction_id: result.prediction_id }) });
        let data = null;
        try { data = await resp.json(); } catch (_) { /* non-JSON */ }
        if (!resp.ok || !data || !data.ok) throw new Error((data && data.error) || "Could not save the report.");
        const id = data.report.report_id;
        if (state.result === result) { state.reportId = id; markSaved(id); }
        NI.toast(data.created ? "Report saved: " + id : "Already saved as " + id + ".", "success");
        return id;
      } catch (err) {
        if (state.result === result) { el.saveReport.disabled = false; el.saveLabel.textContent = "Save to Reports"; }
        NI.toast(err instanceof TypeError ? "Could not reach the server." : err.message, "error");
        return null;
      } finally { state.saving = null; }
    })();
    return state.saving;
  }
  el.saveReport.addEventListener("click", saveReport);
  el.dlPdf.addEventListener("click", async () => {
    const result = state.result;
    el.dlPdf.disabled = true;
    try {
      const id = await saveReport();                       // idempotent: the same analysis never creates a second report
      if (!id) return;
      NI.toast("Preparing the PDF…", "info", 2000);
      await NI.downloadReport(id, "pdf");
      NI.toast("PDF downloaded: " + id, "success");
    } catch (err) {
      NI.toast(err.message, "error");
    } finally { if (state.result === result) el.dlPdf.disabled = false; }
  });

  /* ---------- downloads ---------- */
  function download(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }
  const stamp = () => (state.result ? state.result.timestamp : new Date().toISOString()).slice(0, 19).replace(/[:T]/g, "-");

  el.dlJson.addEventListener("click", () => {
    if (!state.result) return;
    const d = state.result;
    const out = {
      file_name: state.file ? state.file.name : null,
      demo_sample: state.sample ? { dataset_label: state.sample.dataset_label, split: "val" } : null,
      prediction: d.prediction, confidence_warning: d.warning, explanation: d.explanation,
      model: d.model, timestamp_utc: d.timestamp, disclaimer: d.disclaimer,
    };
    download(new Blob([JSON.stringify(out, null, 2)], { type: "application/json" }), "neuroinsight-result-" + stamp() + ".json");
  });

  function wrapText(ctx, text, x, y, maxW, lh) {
    let line = "";
    text.split(" ").forEach((w) => {
      const t = line ? line + " " + w : w;
      if (ctx.measureText(t).width > maxW && line) { ctx.fillText(line, x, y); y += lh; line = w; } else line = t;
    });
    if (line) { ctx.fillText(line, x, y); y += lh; }
    return y;
  }

  el.dlPng.addEventListener("click", () => {
    if (!state.result || !el.preview.complete || !el.preview.naturalWidth) return;
    const d = state.result, p = d.prediction;
    const W = 1000, M = 36, H = 760;
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const g = c.getContext("2d");
    g.fillStyle = "#0a1327"; g.fillRect(0, 0, W, H);
    g.fillStyle = "#f59e0b"; g.fillRect(0, 0, W, 34);
    g.fillStyle = "#1a1203"; g.font = "bold 14px sans-serif"; g.textBaseline = "middle";
    g.fillText("RESEARCH PROTOTYPE — not a medical device or diagnostic tool", M, 17);
    g.textBaseline = "alphabetic";
    g.fillStyle = "#eef3fb"; g.font = "bold 24px sans-serif"; g.fillText("NeuroInsight result", M, 76);

    // original MRI (no Grad-CAM); always the natural-size image, regardless of the viewer's zoom
    const box = 380, iw = el.preview.naturalWidth, ih = el.preview.naturalHeight, s = Math.min(box / iw, box / ih);
    const dw = iw * s, dh = ih * s;
    g.fillStyle = "#000"; g.fillRect(M, 100, box, box);
    g.drawImage(el.preview, M + (box - dw) / 2, 100 + (box - dh) / 2, dw, dh);
    g.strokeStyle = "rgba(255,255,255,0.25)"; g.strokeRect(M, 100, box, box);
    g.fillStyle = "rgba(238,243,251,0.7)"; g.font = "13px sans-serif";
    g.fillText("Original uploaded image" + (state.sample ? " (demo sample, validation split)" : ""), M, 100 + box + 20);

    // details
    const X = M + box + 40, RW = W - X - M;
    let y = 118;
    g.fillStyle = "rgba(238,243,251,0.7)"; g.font = "13px sans-serif"; g.fillText("PREDICTION", X, y);
    y += 30; g.fillStyle = "#eef3fb"; g.font = "bold 30px sans-serif"; g.fillText(nice(p.label), X, y);
    y += 26; g.font = "16px sans-serif"; g.fillText("Confidence " + pct(p.confidence, 2), X, y);
    y += 34;
    if (d.warning.level !== "none") {
      g.fillStyle = d.warning.level === "low" ? "#f87171" : "#fbbf24"; g.font = "bold 14px sans-serif";
      y = wrapText(g, d.warning.message, X, y, RW, 20) + 6;
    }
    p.ranked.forEach((r) => {
      g.fillStyle = "#eef3fb"; g.font = "14px sans-serif"; g.fillText(nice(r.label), X, y);
      g.textAlign = "right"; g.fillText(pct(r.prob, 2), X + RW, y); g.textAlign = "left";
      g.fillStyle = "rgba(255,255,255,0.12)"; g.fillRect(X, y + 7, RW, 8);
      g.fillStyle = r.label === p.label ? "#2dd4bf" : "#22d3ee"; g.fillRect(X, y + 7, RW * r.prob, 8);
      y += 38;
    });
    g.fillStyle = "rgba(238,243,251,0.6)"; g.font = "12px sans-serif";
    y = wrapText(g, "Grad-CAM explanation not included (disabled until validated).", X, y + 4, RW, 17);

    // footer
    const fy = 100 + box + 56;
    g.fillStyle = "rgba(238,243,251,0.7)"; g.font = "12px sans-serif";
    let ty = wrapText(g, d.disclaimer, M, fy, W - 2 * M, 17) + 4;
    ty = wrapText(g, "Model: " + d.model.name + " · preprocessing: " + d.model.preprocessing + " · estimator SHA-256 " +
      d.model.estimator_sha256.slice(0, 12) + "… · " + d.timestamp, M, ty, W - 2 * M, 17);
    wrapText(g, "Confidence warnings are research-demo thresholds (0.90 / 0.70), not clinical safety cutoffs.", M, ty, W - 2 * M, 17);

    c.toBlob((blob) => blob && download(blob, "neuroinsight-result-" + stamp() + ".png"), "image/png");
  });

  /* ---------- about the model ---------- */
  const FLOW = [
    ["MRI Image", "JPG · PNG · BMP"],
    ["Preprocessing", "RGB · 224×224 resize"],
    ["Frozen ResNet50", "ImageNet weights, not trained"],
    ["Feature Extraction", "2048-value feature vector"],
    ["RBF SVM", "Platt-scaled probabilities"],
    ["Prediction", "Most probable of 4 classes"],
  ];
  const flowDiagram = () => node("div", { class: "flow-wrap" }, [
    node("h3", { text: "Architecture" }),
    node("ol", { class: "flow", "aria-label": "Model pipeline" }, FLOW.map(([t, s]) => node("li", {}, [node("b", { text: t }), node("span", { text: s })]))),
  ]);

  async function loadAbout() {
    try {
      const r = await fetch("/api/model-info"); const d = await r.json();
      if (!r.ok || !d.ok) throw new Error(d.error || "unavailable");
      const m = d.validation_metrics, tv = m.tumor_vs_notumor, a = d.architecture;
      const metric = (value, label, sub) => node("div", { class: "metric" }, [
        node("div", { class: "metric-value", text: value }), node("div", { class: "metric-label", text: label }), node("div", { class: "metric-sub", text: sub })]);
      const metrics = node("div", { class: "metric-grid" }, [
        metric(pct(m.accuracy, 2), "Validation accuracy", "n = " + m.n + " validation images"),
        metric(m.macro_f1.toFixed(4), "Macro-F1", "average over the 4 classes"),
        metric(pct(tv.sensitivity, 2), "Tumor vs no-tumor sensitivity", tv.tp + " of " + (tv.tp + tv.fn) + " tumor images found"),
        metric(pct(tv.specificity, 2), "Tumor vs no-tumor specificity", tv.tn + " of " + (tv.tn + tv.fp) + " no-tumor images correct"),
      ]);
      const perClass = node("div", { class: "table-scroll" }, [node("table", {}, [
        node("thead", {}, [node("tr", {}, ["Class", "Recall", "Precision", "F1", "n"].map((h, i) => node("th", { class: i ? "num" : "", text: h })))]),
        node("tbody", {}, Object.entries(m.per_class).map(([k, v]) => node("tr", {}, [
          node("td", { text: nice(k) }), node("td", { class: "num", text: v.recall.toFixed(3) }),
          node("td", { class: "num", text: v.precision.toFixed(3) }), node("td", { class: "num", text: v.f1.toFixed(3) }),
          node("td", { class: "num", text: String(v.support) })]))),
      ])]);
      const LEVEL = ["none", "review", "low"];
      const tiers = node("div", { class: "tier-list" }, d.confidence_warnings.tiers.map((t, i) =>
        node("div", { class: "tier", "data-level": LEVEL[i] || "none" }, [node("b", { text: t.range }), node("span", { text: t.warning === "none" ? "No warning" : cap(t.warning) })])));
      const detail = (title, kids, cls) => node("div", { class: "detail" + (cls ? " " + cls : "") }, [node("h3", { text: title }), ...kids]);
      const grid = node("div", { class: "detail-grid" }, [
        detail("Image preprocessing", [node("p", { text: a.preprocessing }), node("p", { text: "The same steps are used for training, evaluation and this app." })]),
        detail("Feature extractor", [node("p", { text: "A ResNet50 network pre-trained on ImageNet, kept frozen (no fine-tuning). Its final convolutional maps are averaged into 2048 numbers per image." })]),
        detail("Classifier", [node("p", { text: "RBF support-vector machine (C = " + a.svm.C + ", balanced class weights) with Platt-scaled probabilities. The prediction is the class with the highest probability." })]),
        detail("Output classes", [node("div", { class: "class-chips" }, a.classes.map((c) => node("span", { class: "class-chip", "data-class": c, text: nice(c) })))]),
        detail("Confidence warning tiers", [tiers, node("p", { class: "hint", text: d.confidence_warnings.note })]),
        detail("Validation performance by class", [perClass,
          node("p", { class: "hint", text: "Validation split only (also used to choose the model); the held-out test split has not been evaluated. These are not clinical performance figures." })]),
        detail("Limitations", [node("ul", { class: "limits" }, d.limitations.map((t) => node("li", { text: t })))], "wide"),
      ]);
      el.aboutBody.className = "about-body";
      el.aboutBody.replaceChildren(metrics, flowDiagram(), grid);
    } catch (_) {
      el.aboutBody.className = "about-body";
      el.aboutBody.replaceChildren(flowDiagram(), node("p", { class: "hint", text: "Model metrics and details are currently unavailable." }));
    }
  }

  loadSamples();
  loadAbout();
})();
