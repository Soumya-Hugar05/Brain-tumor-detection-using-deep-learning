/* NeuroInsight application shell: sidebar drawer, profile menu, local dates and times. */
(function () {
  "use strict";
  const sidebar = document.getElementById("sidebar");
  const toggle = document.getElementById("sidebarToggle");
  const backdrop = document.getElementById("sidebarBackdrop");
  if (sidebar && toggle && backdrop) {
    const setOpen = (open) => {
      sidebar.classList.toggle("open", open);
      backdrop.hidden = !open;
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
      if (open) { const first = sidebar.querySelector("a, button"); if (first) first.focus(); }
    };
    toggle.addEventListener("click", () => setOpen(!sidebar.classList.contains("open")));
    backdrop.addEventListener("click", () => { setOpen(false); toggle.focus(); });
    sidebar.addEventListener("click", (e) => { if (e.target.closest("a")) setOpen(false); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && sidebar.classList.contains("open")) { setOpen(false); toggle.focus(); } });
    window.matchMedia("(min-width: 1021px)").addEventListener("change", (e) => { if (e.matches) setOpen(false); });
  }

  const btn = document.getElementById("profileBtn");
  const menu = document.getElementById("profileMenu");
  if (btn && menu) {
    const setMenu = (open) => { menu.hidden = !open; btn.setAttribute("aria-expanded", String(open)); };
    btn.addEventListener("click", (e) => { e.stopPropagation(); setMenu(menu.hidden); if (!menu.hidden) { const f = menu.querySelector("[role=menuitem]"); if (f) f.focus(); } });
    document.addEventListener("click", (e) => { if (!menu.hidden && !menu.contains(e.target)) setMenu(false); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { setMenu(false); btn.focus(); } });
    menu.addEventListener("keydown", (e) => {
      const items = [...menu.querySelectorAll("[role=menuitem]")], i = items.indexOf(document.activeElement);
      if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length].focus(); }
      if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
    });
  }

  // Logout: show a loading state and block a second submission while the request is in flight.
  document.querySelectorAll("form[data-logout]").forEach((form) => {
    form.addEventListener("submit", (e) => {
      if (form.dataset.busy) { e.preventDefault(); return; }
      if (!window.confirm("Are you sure you want to sign out?")) { e.preventDefault(); return; }
      form.dataset.busy = "1";
      try { localStorage.removeItem("neuroinsight-theme"); } catch (_) { /* storage unavailable */ }   // no per-user preference stays behind for the next person
      document.querySelectorAll("form[data-logout]").forEach((f) => { f.dataset.busy = "1"; });   // sidebar and menu share one request
      setTimeout(() => form.querySelectorAll("button").forEach((b) => {
        b.setAttribute("aria-busy", "true"); b.disabled = true;
        const label = b.querySelector("span") || b;
        label.textContent = "Signing out…";
      }), 0);
    });
  });
  window.addEventListener("pageshow", (e) => { if (e.persisted) location.reload(); });   // Back button: never show a restored authenticated page

  // Dates and times are stored in UTC; show them in the viewer's own time zone.
  const today = document.getElementById("today");
  if (today) today.textContent = new Date().toLocaleDateString([], { weekday: "long", year: "numeric", month: "long", day: "numeric" });
  document.querySelectorAll("time.js-time").forEach((t) => {
    const d = new Date(t.getAttribute("datetime"));
    if (!isNaN(d)) { t.title = t.textContent; t.textContent = d.toLocaleString([], { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" }); }
  });
})();
