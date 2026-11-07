"""Tests for cases and prediction history (Phase 6). Run: python tests/test_cases.py

Everything is scoped to the signed-in user. Only demo samples are analysed; the test split is never touched.
"""
from __future__ import annotations

import io
import re
import sys

import helpers
import app as appmod
from webapp.extensions import db
from webapp.models import Case, Prediction, User

ROOT = helpers.ROOT
APP = appmod.app
SAMPLE = (ROOT / "static" / "samples" / "sample_glioma_1.jpg").read_bytes()


def fresh():
    helpers.reset_db()
    return helpers.logged_in_client("alice@example.com"), helpers.logged_in_client("bob@example.com")


def make_case(c, code="CASE-1", title="First", notes="n", **kw):
    r = c.post("/app/cases", data=dict(case_code=code, title=title, notes=notes, **kw))
    return r


def case_id(email, code):
    with APP.app_context():
        uid = User.query.filter_by(email=email).one().id
        return Case.query.filter_by(user_id=uid, case_code=code).one().id


def predict(c, name="scan.jpg", **extra):
    return c.post("/api/predict", data={"file": (io.BytesIO(SAMPLE), name), **extra}, content_type="multipart/form-data")


def test_pages_require_sign_in():
    helpers.reset_db()
    anon = APP.test_client()
    for path in ("/app/cases", "/app/cases/1", "/app/history"):
        assert anon.get(path).status_code == 302, path
    for path in ("/app/cases/1/edit", "/app/cases/1/delete", "/app/cases/1/status", "/app/history/abc/case"):
        assert anon.post(path, csrf=False).status_code in (302, 400), path


def test_create_validate_edit_archive_delete():
    a, _ = fresh()
    r = make_case(a)
    assert r.status_code == 302
    cid = case_id("alice@example.com", "CASE-1")
    assert "CASE-1" in a.get("/app/cases").get_data(as_text=True)
    assert make_case(a, code="case-1").status_code == 400                      # duplicate, case-insensitive
    assert make_case(a, code="").status_code == 400
    assert make_case(a, code="<script>").status_code == 400
    assert make_case(a, code="X" * 41).status_code == 400
    assert make_case(a, code="OK-2", notes="x" * 2001).status_code == 400
    assert a.post(f"/app/cases/{cid}/edit", data=dict(case_code="CASE-1b", title="T", notes="")).status_code == 302
    assert "CASE-1b" in a.get(f"/app/cases/{cid}").get_data(as_text=True)
    assert a.post(f"/app/cases/{cid}/status", data=dict(status="archived")).status_code == 302
    assert a.post(f"/app/cases/{cid}/status", data=dict(status="bogus")).status_code == 400
    with APP.app_context():
        assert db.session.get(Case, cid).status == "archived"
    assert a.post(f"/app/cases/{cid}/delete").status_code == 302
    assert a.get(f"/app/cases/{cid}").status_code == 404


def test_notes_are_escaped():
    a, _ = fresh()
    make_case(a, notes="<img src=x onerror=alert(1)>", title="<b>t</b>")
    html = a.get(f"/app/cases/{case_id('alice@example.com', 'CASE-1')}").get_data(as_text=True)
    assert "<img src=x" not in html and "&lt;img src=x" in html


def test_users_cannot_see_or_change_each_others_cases():
    a, b = fresh()
    make_case(a, code="SECRET-A", notes="private note")
    cid = case_id("alice@example.com", "SECRET-A")
    assert "SECRET-A" not in b.get("/app/cases").get_data(as_text=True)
    assert b.get(f"/app/cases/{cid}").status_code == 404
    assert b.post(f"/app/cases/{cid}/edit", data=dict(case_code="HIJACK")).status_code == 404
    assert b.post(f"/app/cases/{cid}/status", data=dict(status="archived")).status_code == 404
    assert b.post(f"/app/cases/{cid}/delete").status_code == 404
    make_case(b, code="SECRET-A")                                              # same code for another user is fine
    with APP.app_context():
        assert Case.query.filter_by(case_code="SECRET-A").count() == 2


def test_predict_into_a_case_and_history_filters():
    a, b = fresh()
    make_case(a)
    cid = case_id("alice@example.com", "CASE-1")
    assert predict(a, "in-case.jpg", case_id=str(cid)).status_code == 200
    assert predict(a, "loose.jpg").status_code == 200
    assert predict(a, "bad.jpg", case_id="999").status_code == 400
    assert predict(a, "bad2.jpg", case_id="abc").status_code == 400
    make_case(b, code="BOBS")
    assert predict(a, "steal.jpg", case_id=str(case_id("bob@example.com", "BOBS"))).status_code == 400, "cannot file into another user's case"
    h = a.get("/app/history").get_data(as_text=True)
    assert "in-case.jpg" in h and "loose.jpg" in h and "bad.jpg" not in h and "steal.jpg" not in h
    assert "loose.jpg" not in a.get(f"/app/history?case={cid}").get_data(as_text=True)
    assert "in-case.jpg" not in a.get("/app/history?case=none").get_data(as_text=True)
    assert "in-case.jpg" in a.get("/app/history?q=in-case").get_data(as_text=True)
    assert "in-case.jpg" not in a.get("/app/history?q=100%25").get_data(as_text=True)      # LIKE wildcards are escaped
    assert "in-case.jpg" not in b.get("/app/history").get_data(as_text=True)
    assert "in-case.jpg" in a.get(f"/app/cases/{cid}").get_data(as_text=True)
    assert a.get("/app/history?page=abc").status_code == 200 and a.get("/app/history?cls=zzz&case=zzz").status_code == 200


def test_archived_case_rejects_new_analyses_and_assignment_rules():
    a, b = fresh()
    make_case(a)
    cid = case_id("alice@example.com", "CASE-1")
    predict(a, "one.jpg")
    with APP.app_context():
        pid = Prediction.query.one().public_id
    assert a.post(f"/app/history/{pid}/case", data=dict(case_id=str(cid))).status_code == 302
    with APP.app_context():
        assert Prediction.query.one().case_id == cid
    assert b.post(f"/app/history/{pid}/case", data=dict(case_id="")).status_code == 404, "not Bob's analysis"
    make_case(b, code="BOBS")
    assert a.post(f"/app/history/{pid}/case", data=dict(case_id=str(case_id("bob@example.com", "BOBS")))).status_code == 404
    r = a.post(f"/app/history/{pid}/case", data=dict(case_id="", next="//evil.example"))
    assert r.status_code == 302 and "evil.example" not in r.headers["Location"]
    a.post(f"/app/cases/{cid}/status", data=dict(status="archived"))
    assert predict(a, "two.jpg", case_id=str(cid)).status_code == 400
    a.post(f"/app/history/{pid}/case", data=dict(case_id=str(cid)))
    a.post(f"/app/cases/{cid}/delete")
    with APP.app_context():
        p = Prediction.query.one()
        assert p.case_id is None, "deleting a case keeps the analysis"


def test_analysis_page_offers_only_own_open_cases_and_sidebar_is_live():
    a, b = fresh()
    make_case(a, code="MINE")
    make_case(a, code="ARCH-ZZ")
    a.post(f"/app/cases/{case_id('alice@example.com', 'ARCH-ZZ')}/status", data=dict(status="archived"))
    make_case(b, code="BOBS")
    html = a.get("/app/analysis").get_data(as_text=True)
    select = re.search(r'<select id="caseSelect">.*?</select>', html, re.S).group(0)      # flash messages elsewhere may name a case
    assert "MINE" in select and "ARCH-ZZ" not in select and "BOBS" not in select
    assert "Soon" not in html and "Patients / Cases" in html


def test_reports_follow_their_case():
    a, b = fresh()
    make_case(a, code="RC-1")
    cid = case_id("alice@example.com", "RC-1")
    pid = predict(a, "r1.jpg", case_id=str(cid)).get_json()["prediction_id"]
    rep = a.post("/api/reports", json={"prediction_id": pid}).get_json()["report"]
    assert rep["case_id"] == cid and rep["case_code"] == "RC-1"
    assert a.get(f"/api/reports/{rep['report_id']}").get_json()["report"]["case_code"] == "RC-1"
    assert rep["report_id"] in a.get(f"/app/cases/{cid}").get_data(as_text=True)
    assert a.get(f"/api/reports?case={cid}").get_json()["total"] == 1
    assert a.get("/api/reports?case=none").get_json()["total"] == 0
    assert a.get("/api/reports?case=abc").status_code == 400
    assert b.get(f"/api/reports?case={cid}").get_json()["total"] == 0
    a.post(f"/app/history/{pid}/case", data=dict(case_id=""))               # the report follows the analysis
    assert a.get(f"/api/reports/{rep['report_id']}").get_json()["report"]["case_id"] is None
    a.post(f"/app/history/{pid}/case", data=dict(case_id=str(cid)))
    assert a.get(f"/api/reports/{rep['report_id']}").get_json()["report"]["case_id"] == cid
    a.post(f"/app/cases/{cid}/delete")
    after = a.get(f"/api/reports/{rep['report_id']}")
    assert after.status_code == 200 and after.get_json()["report"]["case_id"] is None, "deleting a case keeps the report"


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
