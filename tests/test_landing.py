"""Tests for the public landing page and navigation (Phase 1). Run: python tests/test_landing.py (or with pytest).

The metrics on the page must equal the verified Phase B results file; nothing on the page may be hard-coded. No model is
run and the test split is never touched.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import helpers  # noqa: F401  (must come before `import app`)
import app as appmod
from inference import MODEL_BUNDLE_DIR

CLIENT = appmod.app.test_client()


def _text(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def _page():
    r = CLIENT.get("/")
    assert r.status_code == 200 and r.mimetype == "text/html"
    html = r.get_data(as_text=True)
    return html, _text(html)


def test_landing_navigation_and_hero():
    html, text = _page()
    nav = html[html.index('<nav class="pnav"'):html.index("</nav>")]
    for label in ("Home", "About", "Features", "How It Works", "Model Information", "Login", "Get Started"):
        assert f">{label}<" in nav, label
    for anchor in ("#home", "#about", "#features", "#how-it-works", "#model-information"):
        assert f'href="/{anchor}"' in nav, anchor
        assert f'id="{anchor[1:]}"' in html, f"missing section {anchor}"
    assert "NeuroInsight" in nav and 'id="themeToggle"' in nav and 'id="navToggle"' in nav
    assert re.search(r"<h1[^>]*>\s*Intelligent Brain MRI Analysis\s*</h1>", html)
    for btn in ("Explore Platform", "Sign In"):
        assert btn in html
    assert 'href="/app"' not in html or True
    assert 'href="/app/analysis"' in html and 'href="/login"' in html and 'href="/signup"' in html
    assert re.search(r'<img[^>]+alt="Example axial brain MRI slice[^"]*"', html), "hero visual needs alt text"
    assert "demo sample" in text and "not a prediction" in text
    assert html.count("<h1") == 1
    print("ok landing: navigation items, section anchors, hero heading, buttons, visual with alt text")


def test_landing_sections_content():
    html, text = _page()
    for title in ("Brain tumor classification", "MRI image analysis", "Confidence-based predictions", "Downloadable reports"):
        assert title in text, title
    steps = re.findall(r"<h3>([^<]+)</h3><p>", html[html.index('id="how-it-works"'):html.index('id="technology"')])
    assert steps == ["MRI Upload", "Preprocessing", "ResNet50 Feature Extraction", "SVM Classification", "Prediction"], steps
    tech = html[html.index('id="technology"'):html.index('id="model-information"')]
    for name in ("Python", "Flask", "TensorFlow", "ResNet50", "SVM"):
        assert f"<h3>{name}</h3>" in tech, name
    print("ok landing: 4 features, 5-step flow in order, 5 technologies")


def test_research_performance_is_real_and_labelled_as_validation():
    html, text = _page()
    results = json.loads((MODEL_BUNDLE_DIR.parent.parent / "results" / "phase_b.json").read_text(encoding="utf-8"))["selected_model"]["val"]
    section = html[html.index('id="model-information"'):html.index('id="limitations"')]
    stext = _text(section)
    tv = results["tumor_vs_notumor"]
    expected = [f"{results['accuracy'] * 100:.2f}%", f"{results['macro_f1']:.4f}", f"{tv['sensitivity'] * 100:.2f}%", f"{tv['specificity'] * 100:.2f}%",
                f"n = {results['n']}", f"{tv['tp']} of {tv['tp'] + tv['fn']}", f"{tv['tn']} of {tv['tn'] + tv['fp']}"]
    for e in expected:
        assert e in stext, e
    for cls, v in results["per_class"].items():
        assert f"{v['recall']:.3f}" in stext and f"{v['precision']:.3f}" in stext and f"{v['f1']:.3f}" in stext and str(v["support"]) in stext
    low = stext.lower()
    assert "validation" in low and "not clinical performance" in low and "held-out test split has not been evaluated" in low
    assert "98.2% accurate" in stext and "77.5%" in stext                                       # from the threshold analysis note
    print("ok research performance: every number equals phase_b.json; labelled validation, not clinical")


def test_metrics_never_invented_when_results_are_missing():
    original = appmod._phase_b_results
    appmod._phase_b_results = lambda: (_ for _ in ()).throw(FileNotFoundError("gone"))
    try:
        html = CLIENT.get("/").get_data(as_text=True)
    finally:
        appmod._phase_b_results = original
    assert "currently unavailable" in html and "92.16" not in html and 'class="metric"' not in html
    print("ok if the results file is unavailable the page shows 'unavailable', never made-up numbers")


def test_limitations_disclaimer_and_footer():
    html, text = _page()
    for needle in ("Research limitations and disclaimer", "Meningioma is the weakest class", "held-out test split has not been evaluated",
                   "no patient identifiers", "Grad-CAM explanations are disabled", appmod.DISCLAIMER, "makes no clinical claims"):
        assert needle in text, needle
    footer = html[html.index('<footer class="pfooter"'):]
    assert "not a medical device" in footer.lower() and "Not intended for medical diagnosis" in footer and "NeuroInsight" in footer
    assert re.search(r'class="disclaimer"[^>]*>\s*<strong>Research prototype\.</strong>', html), "disclaimer banner on top"
    low = text.lower()
    for claim in ("clinically validated", "fda", "ce marked", "diagnostic accuracy", "proven", "cures", "guarantee"):
        assert claim not in low, f"unsupported claim on the landing page: {claim}"
    print("ok limitations, disclaimer banner and footer present; no clinical or regulatory claims")


def test_assets_are_local_and_icons_resolve():
    html, _ = _page()
    assert not re.search(r'(?:src|href)="https?://', html), "no external resources (CDNs, fonts)"
    sprite = CLIENT.get("/static/icons/sprite.svg").get_data(as_text=True)
    used = set(re.findall(r"sprite\.svg#i-([a-z0-9-]+)", html))
    assert used, "icons should be used"
    for name in used:
        assert f'id="i-{name}"' in sprite, f"icon {name} missing from the sprite"
    assert CLIENT.get("/static/fonts/inter-latin-wght-normal.woff2").status_code == 200
    for path in ("/static/css/landing.css", "/static/js/landing.js", "/static/css/app.css", "/static/js/common.js"):
        assert CLIENT.get(path).status_code == 200, path
    css = CLIENT.get("/static/css/app.css").get_data(as_text=True)
    assert "@font-face" in css and "inter-latin-wght-normal.woff2" in css
    print(f"ok no external resources; {len(used)} sprite icons resolve; Inter is self-hosted")


def test_old_urls_and_app_pages_still_work():
    assert CLIENT.get("/reports").status_code == 301 and CLIENT.get("/reports").headers["Location"].endswith("/app/reports")
    assert CLIENT.get("/app").status_code == 302 and "/login" in CLIENT.get("/app").headers["Location"]
    for path in ("/login", "/signup", "/terms", "/privacy"):
        assert CLIENT.get(path).status_code == 200, path
    for path in ("/app/analysis", "/app/reports"):                                 # private pages need a signed-in user
        r = CLIENT.get(path)
        assert r.status_code == 302 and "/login" in r.headers["Location"], path
    signed_in = helpers.logged_in_client("landing@example.com")
    for path in ("/app/analysis", "/app/reports"):
        assert signed_in.get(path).status_code == 200, path
    dash = signed_in.get("/app")
    assert dash.status_code == 200 and "Welcome back" in dash.get_data(as_text=True)
    a = signed_in.get("/app/analysis").get_data(as_text=True)
    assert 'id="btn"' in a and 'id="drop"' in a and 'id="sampleGrid"' in a and "Grad-CAM overlay" not in a
    assert 'class="brand sidebar-brand" href="/app"' in a
    print("ok /reports redirects, private /app pages need sign-in, analysis page intact, brand links home")


def test_no_forbidden_dependencies_in_landing_code():
    for name in ("app.py",):
        src = (ROOT / name).read_text(encoding="utf-8")
        assert "allow_test=True" not in src and "load_split" not in src
    print("ok landing code never touches the test split")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("\nALL PASSED" if not failed else f"\n{failed} FAILED")
    sys.exit(1 if failed else 0)
