"""Tests for the application shell and dashboard (Phase 4). Run: python tests/test_dashboard.py

Statistics must come from real records; everything is scoped to the signed-in user. Only demo samples are analysed.
"""
from __future__ import annotations

import io
import json
import re
import sys
from datetime import timedelta

import helpers
import app as appmod
from webapp.extensions import db
from webapp.models import Case, Prediction, User, utcnow

ROOT = helpers.ROOT
APP = appmod.app
SAMPLE = (ROOT / "static" / "samples" / "sample_glioma_1.jpg").read_bytes()


def fresh(email="dash@example.com", name="Dana Researcher"):
    helpers.reset_db()
    import os
    os.environ["NEUROINSIGHT_REPORTS_DB"] = str(helpers.TMP / f"reports-{abs(hash(email))}.sqlite3")
    APP.config["REPORTS_DB"] = os.environ["NEUROINSIGHT_REPORTS_DB"]
    APP.extensions.pop("report_store", None)
    return helpers.logged_in_client(email, name=name)


def add_prediction(email, name="scan.jpg", cls="glioma", conf=0.9, age_minutes=0):
    with APP.app_context():
        uid = User.query.filter_by(email=email).one().id
        db.session.add(Prediction(public_id=f"{abs(hash((email, name, age_minutes))) % 10**32:032x}", user_id=uid, file_name=name, predicted_class=cls,
                                  confidence=conf, probabilities={c: (1 - conf) / 3 for c in ("glioma", "meningioma", "notumor", "pituitary")} | {cls: conf},
                                  warning_level="none" if conf >= .9 else "review", model_estimator_sha256="a" * 64,
                                  created_at=utcnow() - timedelta(minutes=age_minutes)))
        db.session.commit()


def add_case(email, code, title=None):
    with APP.app_context():
        uid = User.query.filter_by(email=email).one().id
        db.session.add(Case(user_id=uid, case_code=code, title=title))
        db.session.commit()


def text_of(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)))


def stat(html, id_):
    return int(re.search(rf'id="{id_}">(\d+)<', html).group(1))


def test_shell_pages_require_sign_in_and_a_verified_email():
    helpers.reset_db()
    anon = APP.test_client()
    for path in ("/app", "/app/model", "/app/profile", "/app/cases", "/app/history", "/app/notifications", "/app/settings", "/app/analysis", "/app/reports"):
        r = anon.get(path)
        assert r.status_code == 302 and "/login" in r.headers["Location"], path
    helpers.create_user("unv@example.com", verified=False)
    c = APP.test_client()
    helpers.login(c, "unv@example.com")
    for path in ("/app", "/app/model", "/app/analysis"):
        assert c.get(path).headers["Location"].endswith("/verify-email"), path
    print("ok every shell page redirects anonymous and unverified users")


def test_sidebar_navigation_items_links_and_active_state():
    c = fresh()
    expected = [("Dashboard", "/app"), ("MRI Analysis", "/app/analysis"), ("Patients / Cases", "/app/cases"), ("Prediction History", "/app/history"),
                ("Reports", "/app/reports"), ("Model Performance", "/app/model"), ("Notifications", "/app/notifications"), ("My Profile", "/app/profile"),
                ("Settings", "/app/settings")]
    for path, label in (("/app", "Dashboard"), ("/app/analysis", "MRI Analysis"), ("/app/reports", "Reports"), ("/app/model", "Model Performance"),
                        ("/app/profile", "My Profile"), ("/app/cases", "Patients / Cases"), ("/app/settings", "Settings")):
        r = c.get(path)
        assert r.status_code == 200, path
        html = r.get_data(as_text=True)
        nav = html[html.index('<aside id="sidebar"'):html.index("</aside>")]
        links = re.findall(r'<a href="([^"]+)"( aria-current="page")?><svg[^>]*>.*?</svg><span>([^<]+)</span>', nav, flags=re.S)
        assert [(l[2], l[0]) for l in links] == expected, links
        assert [l[2] for l in links if l[1]] == [label], (path, [l[2] for l in links if l[1]])
        assert "<span>Logout</span>" in nav and 'action="/logout"' in nav and 'name="csrf_token"' in nav
        assert 'id="sidebarToggle"' in html and 'id="sidebarBackdrop"' in html and 'aria-controls="sidebar"' in html
        assert "Research prototype." in html and "not a medical device" in html, "the disclaimer stays visible in the shell"
    print("ok sidebar: 9 pages + Logout, correct links, exactly one active item per page, drawer controls, disclaimer")


def test_dashboard_statistics_are_real_and_scoped_to_the_user():
    c = fresh("me@example.com", "Mia Student")
    helpers.create_user("other@example.com")
    html = c.get("/app").get_data(as_text=True)
    assert [stat(html, i) for i in ("statAnalyses", "statCases", "statReports", "statActivity")] == [0, 0, 0, 0]
    for t in ("No analyses yet", "No reports yet", "No cases yet", "No activity yet"):
        assert t in html, t
    for i in range(3):
        add_prediction("me@example.com", f"mine-{i}.jpg", age_minutes=i)
    add_prediction("me@example.com", "old.jpg", age_minutes=60 * 24 * 9)                      # outside the 7-day window
    add_prediction("other@example.com", "theirs.jpg")
    add_case("me@example.com", "CASE-A", "First case")
    add_case("other@example.com", "CASE-X")
    pred = c.post("/api/predict", data={"file": (io.BytesIO(SAMPLE), "real.jpg")}, content_type="multipart/form-data").get_json()
    assert c.post("/api/reports", json={"prediction_id": pred["prediction_id"]}).status_code == 201
    html = c.get("/app").get_data(as_text=True)
    assert stat(html, "statAnalyses") == 3 + 1 + 1 and stat(html, "statCases") == 1 and stat(html, "statReports") == 1
    assert stat(html, "statActivity") == 4 + 1 and "last 7 days: 4 analyses, 1 reports" in text_of(html)
    assert "theirs.jpg" not in html and "CASE-X" not in html and "mine-0.jpg" in html and "CASE-A" in html and "First case" in html
    other = helpers.logged_in_client("other@example.com").get("/app").get_data(as_text=True)
    assert stat(other, "statAnalyses") == 1 and stat(other, "statCases") == 1 and stat(other, "statReports") == 0 and "mine-0.jpg" not in other
    print("ok statistics equal the user's own rows (analyses, cases, reports, 7-day activity); other users' data never counted")


def test_recent_lists_are_newest_first_and_limited_to_five():
    c = fresh("lists@example.com")
    for i in range(8):
        add_prediction("lists@example.com", f"file-{i}.jpg", age_minutes=100 - i)               # file-7 is the newest
    for i in range(7):
        add_case("lists@example.com", f"CASE-{i:02d}")
    html = c.get("/app").get_data(as_text=True)
    analyses = re.findall(r'class="cell-file" title="([^"]+)"', html)
    assert analyses == [f"file-{i}.jpg" for i in (7, 6, 5, 4, 3)], analyses
    assert len(re.findall(r"<li data-kind=", html)) == 8 and html.count("CASE-") == 5
    assert "<table" in html and 'id="recentAnalyses"' in html
    print("ok recent analyses: newest five; recent cases limited to five; activity feed capped at eight")


def test_dashboard_escapes_user_supplied_text():
    c = fresh("xss@example.com", "<script>alert(1)</script>")
    add_prediction("xss@example.com", "<img src=x onerror=alert(1)>.jpg")
    add_case("xss@example.com", "<b>C1</b>", "<svg onload=alert(1)>")
    html = c.get("/app").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html and "<img src=x" not in html and "<svg onload" not in html and "<b>C1</b>" not in html
    assert "&lt;script&gt;" in html
    print("ok names, file names and case text are HTML-escaped")


def test_analysis_is_recorded_as_metadata_only():
    c = fresh("rec@example.com")
    r = c.post("/api/predict", data={"file": (io.BytesIO(SAMPLE), "../weird name.jpg")}, content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200 and j["prediction"]["label"] == "glioma"
    with APP.app_context():
        rows = Prediction.query.all()
        assert len(rows) == 1
        p = rows[0]
        assert p.public_id == j["prediction_id"] and p.file_name == "weird name.jpg" and p.predicted_class == "glioma"
        assert abs(p.confidence - j["prediction"]["confidence"]) < 1e-12 and p.probabilities == j["prediction"]["probabilities"]
        assert p.warning_level == "none" and p.model_estimator_sha256 == j["model"]["estimator_sha256"] and p.case_id is None
        assert {c.name for c in Prediction.__table__.columns} == {"id", "public_id", "user_id", "case_id", "file_name", "predicted_class", "confidence",
                                                                  "probabilities", "warning_level", "model_estimator_sha256", "created_at"}
    raw = (helpers.TMP / "test.sqlite3").read_bytes()
    assert SAMPLE[:200] not in raw and b"\xff\xd8\xff" not in raw, "the image must never be stored"
    c.post("/api/predict", data={"file": (io.BytesIO(b"not an image"), "x.jpg")}, content_type="multipart/form-data")
    with APP.app_context():
        assert Prediction.query.count() == 1, "rejected uploads are not recorded"
    print("ok every successful analysis is recorded once (metadata only, no image); rejected uploads are not")


def test_database_rejects_invalid_prediction_and_case_values():
    fresh("cons@example.com")
    with APP.app_context():
        uid = User.query.filter_by(email="cons@example.com").one().id
        bad = [Prediction(public_id="1" * 32, user_id=uid, file_name="x", predicted_class="tumor", confidence=.5, probabilities={}, warning_level="none", model_estimator_sha256="a"),
               Prediction(public_id="2" * 32, user_id=uid, file_name="x", predicted_class="glioma", confidence=1.5, probabilities={}, warning_level="none", model_estimator_sha256="a"),
               Case(user_id=uid, case_code="C", status="weird")]
        for obj in bad:
            try:
                db.session.add(obj)
                db.session.commit()
            except Exception:
                db.session.rollback()
            else:
                raise AssertionError("constraint missing")
        db.session.add_all([Case(user_id=uid, case_code="DUP"), ])
        db.session.commit()
        try:
            db.session.add(Case(user_id=uid, case_code="DUP"))
            db.session.commit()
        except Exception:
            db.session.rollback()
        else:
            raise AssertionError("case codes must be unique per user")
    print("ok CHECK and UNIQUE constraints protect predictions and cases")


def test_model_performance_page_shows_real_validation_numbers():
    c = fresh()
    html = c.get("/app/model").get_data(as_text=True)
    val = json.loads((helpers.ROOT / "runs/20261105-184406_phaseB_frozen_improvements/results/phase_b.json").read_text(encoding="utf-8"))["selected_model"]["val"]
    t = text_of(html)
    tv = val["tumor_vs_notumor"]
    for needle in (f"{val['accuracy'] * 100:.2f}%", f"{val['macro_f1']:.4f}", f"{tv['sensitivity'] * 100:.2f}%", f"{tv['specificity'] * 100:.2f}%", "not clinical performance", "Limitations"):
        assert needle in t, needle
    cells = re.findall(r'<td class="num cm-cell(?: diag)?" style="--v: [^"]+">(\d+)</td>', html)
    assert [int(x) for x in cells] == [v for row in val["confusion_matrix"] for v in row] and sum(map(int, cells)) == val["n"]
    print("ok model performance page equals phase_b.json (metrics and the 4x4 confusion matrix)")


def test_profile_and_placeholder_pages_are_honest():
    c = fresh("prof@example.com", "Pat Profile")
    p = text_of(c.get("/app/profile").get_data(as_text=True))
    assert "Pat Profile" in p and "prof@example.com" in p and "Verified" in p and "User" in p and "Member since" in p
    for path in ("/app/cases", "/app/history", "/app/notifications", "/app/settings"):
        h = c.get(path).get_data(as_text=True)
        assert c.get(path).status_code == 200 and "Not available yet" not in h and "Soon" not in h, path
    print("ok profile shows real account data; every sidebar area is built")


def test_inline_scripts_carry_the_csp_nonce_and_assets_exist():
    c = fresh()
    for path in ("/app", "/app/analysis", "/app/reports", "/app/model"):
        r = c.get(path)
        nonce = re.search(r"'nonce-([^']+)'", r.headers["Content-Security-Policy"]).group(1)
        html = r.get_data(as_text=True)
        assert all(f'nonce="{nonce}"' in tag for tag in re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", html)), path
        assert r.headers["Cache-Control"] == "no-store"
    for asset in ("/static/css/shell.css", "/static/js/shell.js"):
        assert c.get(asset).status_code == 200
    print("ok inline scripts use the per-request nonce; shell assets are served")


def test_analysis_and_reports_pages_keep_their_workflow_elements_inside_the_shell():
    c = fresh()
    a = c.get("/app/analysis").get_data(as_text=True)
    for needle in ('id="drop"', 'id="btn"', 'id="sampleGrid"', 'id="dlJson"', 'id="dlPng"', 'id="dlPdf"', 'id="saveReport"', 'id="viewer"', 'class="shell-main"', "js/app.js"):
        assert needle in a, needle
    r = c.get("/app/reports").get_data(as_text=True)
    for needle in ('id="q"', 'id="rows"', 'id="previewDialog"', 'id="confirmDialog"', 'id="exportBtn"', "js/reports.js"):
        assert needle in r, needle
    assert "Grad-CAM overlay" not in a
    print("ok analysis and reports pages are unchanged inside the shell")


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
