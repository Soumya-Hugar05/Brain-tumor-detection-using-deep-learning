/* NeuroInsight shared front-end helpers (vanilla JavaScript): theme toggle, toast notifications, file download.
   Loaded before app.js / reports.js. Server text is only ever inserted with textContent. */
(function () {
  "use strict";
  const root = document.documentElement;

  /* ---------- theme ---------- */
  const toggle = document.getElementById("themeToggle");
  function applyTheme(t) {
    root.setAttribute("data-theme", t);
    if (toggle) toggle.setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }
  applyTheme(root.getAttribute("data-theme") === "light" ? "light" : "dark");
  if (toggle) toggle.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    applyTheme(next);
    try { localStorage.setItem("neuroinsight-theme", next); } catch (_) { /* storage unavailable */ }
    if (document.body.classList.contains("app-shell") && window.fetch)      // remember the choice on the account too
      fetch("/api/account/theme", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ theme: next }) }).catch(() => {});
  });

  /* ---------- CSRF: send the token with every same-origin state-changing fetch ---------- */
  const csrfMeta = document.querySelector('meta[name="csrf-token"]');
  if (csrfMeta && window.fetch) {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
      init = init || {};
      const method = String(init.method || (input && input.method) || "GET").toUpperCase();
      const url = typeof input === "string" ? input : (input && input.url) || "";
      const sameOrigin = url.startsWith("/") || url.startsWith(location.origin);
      if (sameOrigin && !["GET", "HEAD", "OPTIONS"].includes(method)) {
        const headers = new Headers(init.headers || (input && input.headers) || {});
        headers.set("X-CSRFToken", csrfMeta.content);
        init = Object.assign({}, init, { headers });
      }
      return nativeFetch(input, init);
    };
  }

  /* ---------- toasts ---------- */
  function toast(message, kind, ms) {
    const box = document.getElementById("toasts");
    if (!box) return;
    const t = document.createElement("div");
    t.className = "toast toast-" + (kind || "info");
    t.setAttribute("role", kind === "error" ? "alert" : "status");
    const text = document.createElement("span");
    text.textContent = message;
    const close = document.createElement("button");
    close.type = "button"; close.className = "toast-close"; close.setAttribute("aria-label", "Dismiss notification"); close.textContent = "×";
    const remove = () => { if (t.parentNode) t.parentNode.removeChild(t); };
    close.addEventListener("click", remove);
    t.append(text, close);
    box.appendChild(t);
    while (box.children.length > 4) box.removeChild(box.firstChild);
    setTimeout(remove, ms || (kind === "error" ? 7000 : 4000));
  }

  /* ---------- download a Blob as a file ---------- */
  function saveBlob(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }

  /* ---------- download a stored report (pdf | json | png) from the server ---------- */
  async function downloadReport(id, fmt) {
    const resp = await fetch("/api/reports/" + encodeURIComponent(id) + "/download/" + encodeURIComponent(fmt));
    if (!resp.ok) {
      let msg = "The download failed.";
      try { const d = await resp.json(); if (d && d.error) msg = d.error; } catch (_) { /* not JSON */ }
      throw new Error(msg);
    }
    saveBlob(await resp.blob(), "neuroinsight-report-" + id + "." + fmt);
  }

  window.NI = { toast, saveBlob, downloadReport };
})();
