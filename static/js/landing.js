/* NeuroInsight public pages: responsive menu and section highlighting. */
(function () {
  "use strict";
  const toggle = document.getElementById("navToggle");
  const menu = document.getElementById("navMenu");
  if (toggle && menu) {
    const setOpen = (open) => {
      menu.classList.toggle("open", open);
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    };
    toggle.addEventListener("click", () => setOpen(!menu.classList.contains("open")));
    menu.addEventListener("click", (e) => { if (e.target.closest("a")) setOpen(false); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && menu.classList.contains("open")) { setOpen(false); toggle.focus(); } });
    window.matchMedia("(min-width: 981px)").addEventListener("change", (e) => { if (e.matches) setOpen(false); });
  }

  // Highlight the navigation link of the section in view (landing page only).
  const links = [...document.querySelectorAll(".pnav-links a")];
  const targets = links.map((a) => {
    const h = a.getAttribute("href") || "";
    const id = h.includes("#") ? h.split("#")[1] : "";
    return id ? document.getElementById(id) : null;
  });
  if (links.length && "IntersectionObserver" in window && targets.some(Boolean)) {
    const mark = (id) => links.forEach((a, i) => {
      if (targets[i] && targets[i].id === id) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current");
    });
    const obs = new IntersectionObserver((entries) => {
      const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) mark(visible.target.id);
    }, { rootMargin: "-30% 0px -55% 0px", threshold: [0, 0.2, 0.5] });
    targets.filter(Boolean).forEach((t) => obs.observe(t));
  }
})();
