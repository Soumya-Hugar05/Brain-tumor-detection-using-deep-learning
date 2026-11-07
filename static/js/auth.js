/* NeuroInsight authentication pages: show/hide password, strength feedback, match check, single submit. */
(function () {
  "use strict";
  document.querySelectorAll(".pw-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      const input = document.getElementById(btn.dataset.toggleFor);
      if (!input) return;
      const show = input.type === "password";
      input.type = show ? "text" : "password";
      btn.setAttribute("aria-pressed", String(show));
      btn.setAttribute("aria-label", show ? "Hide password" : "Show password");
    });
  });

  const COMMON = ["password", "password1", "password123", "qwertyuiop", "1234567890", "iloveyou123", "welcome123", "letmein123", "admin12345", "neuroinsight"];
  const meter = document.querySelector(".strength");
  const strengthInput = document.querySelector("input[data-strength]");
  if (meter && strengthInput) {
    const bar = meter.querySelector(".strength-bar span"), label = meter.querySelector(".strength-label b"), rules = {};
    meter.querySelectorAll("[data-rule]").forEach((li) => { rules[li.dataset.rule] = li; });
    const update = () => {
      const v = strengthInput.value;
      const classes = [/[a-z]/, /[A-Z]/, /[0-9]/, /[^A-Za-z0-9]/].filter((r) => r.test(v)).length;
      const okLen = v.length >= 10, okMix = v.length >= 16 || classes >= 3, okCommon = v.length > 0 && !COMMON.includes(v.toLowerCase());
      rules.len.classList.toggle("ok", okLen); rules.mix.classList.toggle("ok", okMix); rules.common.classList.toggle("ok", okCommon);
      let score = 0;
      if (v.length) score = (okLen ? 1 : 0) + (okMix ? 1 : 0) + (okCommon ? 1 : 0) + (v.length >= 14 ? 1 : 0);
      const names = ["–", "Weak", "Fair", "Good", "Strong"];
      label.textContent = names[score];
      bar.style.width = (score * 25) + "%";
      bar.setAttribute("data-score", String(score));
    };
    strengthInput.addEventListener("input", update);
    update();
  }

  document.querySelectorAll("form[data-once]").forEach((form) => {
    form.addEventListener("submit", (e) => {
      const confirm = form.querySelector("input[name=confirm_password]"), pw = form.querySelector("input[name=password]");
      if (confirm && pw && confirm.value !== pw.value) {
        e.preventDefault();
        confirm.setCustomValidity("The passwords do not match.");
        confirm.reportValidity();
        confirm.addEventListener("input", () => confirm.setCustomValidity(""), { once: true });
        return;
      }
      const btn = form.querySelector("button[type=submit]");
      if (btn) setTimeout(() => { btn.disabled = true; }, 0);          // avoid double submission
    });
  });
})();
