"""Tests for the Reports module (reports.py + the /reports page and /api/reports routes).
Run: python tests/test_reports.py   (also works with pytest if installed)

Every test uses its own temporary SQLite file, so the real report history (instance/) is never touched.
Only demo samples from the validation split are analysed; the test split is never loaded. The synthetic records used to
exercise paging, filters and dates exist only inside the temporary databases of these tests.
"""
from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

import app as appmod
import reports as rpt

SAMPLES = json.loads((ROOT / "static" / "samples" / "samples.json").read_text(encoding="utf-8"))["samples"]
_FMT = "%Y-%m-%dT%H:%M:%SZ"
_predictions: dict = {}


class _Db:
    """A fresh temporary report database wired into the app for the duration of a with-block."""
    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "reports.sqlite3"
        appmod.app.config["REPORTS_DB"] = str(self.path)
        self.client = appmod.app.test_client()
        return self

    def __exit__(self, *exc):
        appmod.app.config["REPORTS_DB"] = None
        appmod.app.extensions.pop("report_store", None)
        self.tmp.cleanup()

    @property
    def store(self):
        return appmod.report_store()

    def sql(self, query, args=()):
        con = sqlite3.connect(self.path)
        try:
            with con:
                return con.execute(query, args).fetchall()
        finally:
            con.close()


def predict(sample_index=0, filename=None):
    """Run a real demo sample through /api/predict (model loaded once; results cached per sample for speed)."""
    key = (sample_index, filename)
    if key not in _predictions:
        s = SAMPLES[sample_index]
        data = (ROOT / "static" / "samples" / s["file"]).read_bytes()
        r = appmod.app.test_client().post("/api/predict", data={"file": (io.BytesIO(data), filename or s["file"])},
                                          content_type="multipart/form-data")
        assert r.status_code == 200, r.get_data(as_text=True)
        _predictions[key] = r.get_json()
    return _predictions[key]


def save(client, pred):
    return client.post("/api/reports", json={"prediction_id": pred["prediction_id"]})


def fake_prediction(i, cls="glioma", conf=0.95, name=None):
    """Synthetic prediction dict (temp-database tests only) in the shape the server stores."""
    probs = {c: (1 - conf) / 3 for c in rpt.CLASSES}
    probs[cls] = conf
    return dict(prediction_id=f"{i:032x}", file_name=name or f"scan-{i:03d}.jpg", predicted_class=cls, confidence=conf,
                probabilities=probs, warning=appmod.confidence_warning(conf),
                model=dict(name="NeuroInsight Phase B SVM (ResNet50 features)", estimator_sha256="a" * 64, manifest_sha256="b" * 64,
                           preprocessing="resize", decision_rule="argmax_proba", feature="f", classifier="c", description="d",
                           preprocessing_detail="p"))


def _set_created(db, report_id, when: datetime):
    db.sql("UPDATE reports SET created_at = ? WHERE report_id = ?", (when.strftime(_FMT), report_id))


# ---- validation ---------------------------------------------------------------------------------------
def test_id_and_filename_validation():
    good = "NI-20261106-0A1B2C3D"
    assert rpt.valid_report_id(good) == good
    for bad in ("", None, 5, "ni-20261106-0a1b2c3d", "NI-20261106-0A1B2C3", "NI-20261106-0A1B2C3DE", "NI-2026110-0A1B2C3D",
                "../../etc/passwd", "NI-20261106-0A1B2C3D/../x", "NI-20261106-0A1B2C3D\n", "NI-20261106-ZZZZZZZZ",
                "NI-٢٠٢٦١١٠٦-0A1B2C3D", "'; DROP TABLE reports;--", "x" * 5000):
        assert rpt.valid_report_id(bad) is None, repr(bad)
    assert rpt.valid_prediction_id("0" * 32) == "0" * 32
    for bad in ("", None, "0" * 31, "0" * 33, "G" * 32, "A" * 32, "../" + "0" * 29):
        assert rpt.valid_prediction_id(bad) is None, repr(bad)
    f = rpt.safe_display_filename
    assert f("../../etc/passwd") == "passwd" and f("C:\\Users\\me\\scan 1.png") == "scan 1.png"
    assert f("a/b\\c.jpg") == "c.jpg" and f("") == "image" and f(None) == "image" and f("...") == "image"
    assert f("<script>alert(1)</script>.jpg") == "script_.jpg"           # only the last path component survives
    assert f("na\u00efve\x00name.jpg") == "na_ve_name.jpg"
    long = f("x" * 400 + ".jpeg")
    assert len(long) <= 120 and long.endswith(".jpeg"), long
    print("ok report/prediction id validation and file name sanitising")


def test_list_parameter_validation():
    p = rpt.parse_list_params
    assert p({})["page_size"] == 10 and p({})["sort"] == "newest"
    for bad in ({"cls": "tumor"}, {"sort": "random"}, {"page": "0"}, {"page": "x"}, {"page_size": "7"}, {"page_size": "1000000"},
                {"tz_offset": "99999"}, {"tz_offset": "abc"}, {"date_from": "2026-02-31"}, {"date_from": "yesterday"},
                {"date_from": "2026-05-02", "date_to": "2026-05-01"}, {"q": "x" * 101}):
        try:
            p(bad)
        except rpt.ValidationError:
            continue
        raise AssertionError(f"accepted invalid parameters {bad}")
    print("ok list parameter validation")


# ---- pages ---------------------------------------------------------------------------------------------
def test_reports_page_structure_and_no_hardcoded_statistics():
    with _Db() as db:
        r = db.client.get("/reports")
        html = r.get_data(as_text=True)
    assert r.status_code == 200
    for needle in ("Manage and download MRI analysis reports.", 'id="q"', 'id="filterBtn"', 'id="exportBtn"', "Total reports",
                   "Generated today", "Generated this month", "Report ID", "Case ID", "MRI image", "Predicted class", "Confidence",
                   "Date generated", "Actions", "No reports yet", "Research prototype", "not a medical device", 'aria-current="page"',
                   'id="toasts"', "reports.js", "common.js"):
        assert needle in html, needle
    for stat in re.findall(r'id="stat(?:Total|Today|Month)">([^<]*)<', html):
        assert not stat.strip().isdigit(), "statistics must come from the API, not the HTML"
    assert 'href="/reports"' in html and 'href="/"' in html
    print("ok Reports page: structure, nav, empty state markup, no hard-coded statistics")


def test_index_page_has_nav_and_report_buttons():
    html = appmod.app.test_client().get("/").get_data(as_text=True)
    for needle in ('id="dlJson"', 'id="dlPng"', 'id="dlPdf"', 'id="saveReport"', 'href="/reports"', "common.js", "app.js"):
        assert needle in html, needle
    assert html.count("<h1") == 1
    print("ok analysis page keeps JSON/PNG buttons and adds Save + PDF")


# ---- empty state and statistics --------------------------------------------------------------------------
def test_empty_state_and_zero_statistics():
    with _Db() as db:
        j = db.client.get("/api/reports?tz_offset=-330").get_json()
    assert j["ok"] and j["items"] == [] and j["total"] == 0 and j["page"] == 1 and j["pages"] == 1
    assert j["stats"] == {"total": 0, "today": 0, "this_month": 0}
    print("ok empty database: no items, statistics all zero")


def test_statistics_are_real_counts():
    with _Db() as db:
        now = datetime.now(timezone.utc)
        for i in range(6):
            rec, _ = db.store.create(fake_prediction(i))
            when = [now, now - timedelta(minutes=1), now - timedelta(days=45), now - timedelta(days=400), now - timedelta(hours=1), now - timedelta(days=3)][i]
            _set_created(db, rec["report_id"], when)
        for tz in (0, -330, 300):
            local = now.replace(tzinfo=None) - timedelta(minutes=tz)
            day = local.replace(hour=0, minute=0, second=0, microsecond=0)
            month = day.replace(day=1)
            def local_of(d):  # created_at is stored in whole seconds
                return (d.replace(microsecond=0, tzinfo=None) - timedelta(minutes=tz))
            whens = [now, now - timedelta(minutes=1), now - timedelta(days=45), now - timedelta(days=400), now - timedelta(hours=1), now - timedelta(days=3)]
            exp_today = sum(1 for w in whens if local_of(w) >= day)
            exp_month = sum(1 for w in whens if local_of(w) >= month)
            s = db.client.get(f"/api/reports?tz_offset={tz}").get_json()["stats"]
            assert s == {"total": 6, "today": exp_today, "this_month": exp_month}, (tz, s, exp_today, exp_month)
    print("ok statistics equal independently computed counts (several time zones)")


# ---- creating reports -------------------------------------------------------------------------------------
def test_prediction_response_unchanged_and_has_prediction_id():
    j = predict(0)
    assert re.fullmatch(r"[0-9a-f]{32}", j["prediction_id"])
    assert set(j["model"]) == {"name", "preprocessing", "decision_rule", "estimator_sha256", "manifest_sha256"}
    assert set(j) >= {"ok", "prediction", "warning", "explanation", "model", "timestamp", "disclaimer", "prediction_id"}
    assert j["explanation"]["available"] is False
    print("ok /api/predict contract preserved (plus prediction_id); Grad-CAM still disabled")


def test_create_is_idempotent_and_only_accepts_server_made_predictions():
    j = predict(0)
    with _Db() as db:
        c = db.client
        assert c.post("/api/reports", data={"prediction_id": j["prediction_id"]}).status_code == 415          # not JSON
        assert c.post("/api/reports", json={}).status_code == 400
        assert c.post("/api/reports", json={"prediction_id": "../../x"}).status_code == 400
        assert c.post("/api/reports", json=["x"]).status_code == 400
        assert c.post("/api/reports", json={"prediction_id": "0" * 32}).status_code == 404                     # never produced here
        forged = c.post("/api/reports", json={"prediction_id": "1" * 32, "predicted_class": "notumor", "confidence": 1.0})
        assert forged.status_code == 404, "client-supplied results must never be stored"
        r1 = save(c, j)
        assert r1.status_code == 201 and r1.get_json()["created"] is True
        for _ in range(3):                                                                                      # refresh / double click
            again = save(c, j)
            assert again.status_code == 200 and again.get_json()["created"] is False
            assert again.get_json()["report"]["report_id"] == r1.get_json()["report"]["report_id"]
        assert c.get("/api/reports").get_json()["total"] == 1
        appmod.PREDICTIONS._items.clear()                                                                       # cache expired
        late = save(c, j)
        assert late.status_code == 200 and late.get_json()["report"]["report_id"] == r1.get_json()["report"]["report_id"]
        rec = c.get(f"/api/reports/{r1.get_json()['report']['report_id']}").get_json()["report"]
        assert rec["prediction"]["label"] == j["prediction"]["label"] and rec["prediction"]["confidence"] == j["prediction"]["confidence"]
        assert rec["prediction"]["probabilities"] == j["prediction"]["probabilities"]
    _predictions.clear()
    print("ok save is idempotent; unknown or forged predictions are refused")


def test_stored_filename_is_sanitised():
    _predictions.clear()
    j = predict(1, filename="..\\..\\Users\\me\\<patient name>.jpg")
    with _Db() as db:
        rid = save(db.client, j).get_json()["report"]["report_id"]
        name = db.client.get(f"/api/reports/{rid}").get_json()["report"]["file_name"]
    assert name == "_patient name_.jpg", name
    _predictions.clear()
    print("ok untrusted upload file name reduced to a safe base name")


# ---- listing: search / filter / sort / pagination ------------------------------------------------------------
def _populate(db, n=25):
    base = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
    ids = []
    for i in range(n):
        cls = rpt.CLASSES[i % 4]
        rec, _ = db.store.create(fake_prediction(i, cls, 0.95 if i % 3 else 0.6, name=f"scan-{i:03d}.jpg"))
        _set_created(db, rec["report_id"], base + timedelta(hours=i * 7))
        ids.append(rec["report_id"])
    return ids


def test_pagination_and_sorting():
    with _Db() as db:
        ids = _populate(db, 25)
        c = db.client
        p1 = c.get("/api/reports?page=1&page_size=10").get_json()
        p3 = c.get("/api/reports?page=3&page_size=10").get_json()
        beyond = c.get("/api/reports?page=99&page_size=10").get_json()
        assert (p1["total"], p1["pages"], len(p1["items"])) == (25, 3, 10) and (p3["page"], len(p3["items"])) == (3, 5)
        assert beyond["page"] == 3 and len(beyond["items"]) == 5
        assert [r["report_id"] for r in p1["items"]] == ids[::-1][:10]                         # newest first
        old = c.get("/api/reports?sort=oldest&page_size=10").get_json()
        assert [r["report_id"] for r in old["items"]] == ids[:10]
        seen = []
        for page in (1, 2, 3):
            seen += [r["report_id"] for r in c.get(f"/api/reports?page={page}&page_size=10").get_json()["items"]]
        assert seen == ids[::-1] and len(set(seen)) == 25                                       # no gaps, no duplicates
        assert len(c.get("/api/reports?page_size=50").get_json()["items"]) == 25
    print("ok pagination (3 pages, clamping, no gaps) and newest/oldest sorting")


def test_search_and_class_and_date_filters():
    with _Db() as db:
        ids = _populate(db, 25)
        c = db.client
        get = lambda q: c.get("/api/reports?page_size=50&" + q).get_json()
        assert [r["file_name"] for r in get("q=scan-007")["items"]] == ["scan-007.jpg"]
        assert get("q=SCAN-007")["total"] == 1                                                  # case-insensitive
        assert get("q=" + ids[3][3:12])["total"] == 25 or get("q=" + ids[3])["total"] == 1      # report id (full) matches itself
        assert get("q=" + ids[3])["items"][0]["report_id"] == ids[3]
        for evil in ("%", "_", "a%b", "scan-0%7", "' OR 1=1 --", "'; DROP TABLE reports; --", "\\"):
            r = get("q=" + evil.replace("%", "%25").replace("&", "%26"))
            assert r["ok"] and r["total"] == 0, (evil, r["total"])
        assert get("")["total"] == 25                                                           # the table survived the injection tries
        for cls in rpt.CLASSES:
            r = get("cls=" + cls)
            assert r["total"] == len([i for i in range(25) if rpt.CLASSES[i % 4] == cls]) and all(x["predicted_class"] == cls for x in r["items"])
        assert c.get("/api/reports?cls=tumor").status_code == 400
        # dates: record i is at 2026-03-10 12:00 UTC + 7h*i
        r = get("date_from=2026-03-11&date_to=2026-03-11&tz_offset=0")
        exp = [i for i in range(25) if (datetime(2026, 3, 10, 12) + timedelta(hours=7 * i)).date() == datetime(2026, 3, 11).date()]
        assert r["total"] == len(exp) and {x["file_name"] for x in r["items"]} == {f"scan-{i:03d}.jpg" for i in exp}, (r["total"], exp)
        r = get("date_from=2026-03-11&date_to=2026-03-11&tz_offset=-330")                      # IST: local days differ from UTC days
        exp = [i for i in range(25) if (datetime(2026, 3, 10, 12) + timedelta(hours=7 * i, minutes=330)).date() == datetime(2026, 3, 11).date()]
        assert r["total"] == len(exp) and exp and {x["file_name"] for x in r["items"]} == {f"scan-{i:03d}.jpg" for i in exp}
        assert get("date_from=2030-01-01")["total"] == 0 and get("date_to=2000-01-01")["total"] == 0
        combo = get("cls=glioma&q=scan-0&date_from=2026-03-10")
        assert all(x["predicted_class"] == "glioma" and x["file_name"].startswith("scan-0") for x in combo["items"]) and combo["total"] > 0
        for bad in ("date_from=2026-02-31", "date_from=abc", "date_from=2026-05-02&date_to=2026-05-01", "sort=sideways", "page=0", "page_size=13", "tz_offset=1000"):
            assert c.get("/api/reports?" + bad).status_code == 400, bad
    print("ok search (incl. wildcard and SQL-injection input), class filter, local-time date filter, combos, invalid input")


def test_confidence_flag_in_list():
    with _Db() as db:
        _populate(db, 6)
        items = db.client.get("/api/reports?page_size=50").get_json()["items"]
        by_name = {r["file_name"]: r for r in items}
        assert by_name["scan-000.jpg"]["warning_level"] == "low" and by_name["scan-001.jpg"]["warning_level"] == "none"
    print("ok warning level stored with the report")


# ---- preview and downloads ---------------------------------------------------------------------------------------
def test_preview_detail_and_all_three_downloads():
    j = predict(2)
    with _Db() as db:
        c = db.client
        rid = save(c, j).get_json()["report"]["report_id"]
        d = c.get(f"/api/reports/{rid}").get_json()["report"]
        assert d["report_id"] == rid and d["prediction"]["label"] == j["prediction"]["label"]
        assert d["prediction"]["confidence"] == j["prediction"]["confidence"] and d["confidence_warning"] == j["warning"]
        assert [x["label"] for x in d["prediction"]["ranked"]][0] == j["prediction"]["label"] and len(d["prediction"]["ranked"]) == 4
        assert d["model"]["estimator_sha256"] == j["model"]["estimator_sha256"] and d["model"]["manifest_sha256"] == j["model"]["manifest_sha256"]
        assert d["model"]["name"] == j["model"]["name"] and "ResNet50" in d["model"]["feature"] and d["model"]["classifier"].startswith("RBF SVM")
        assert d["case_id"] is None and d["original_image_stored"] is False and "not a medical device" in d["disclaimer"]
        assert d["explanation"]["available"] is False

        jr = c.get(f"/api/reports/{rid}/download/json")
        assert jr.status_code == 200 and jr.mimetype == "application/json"
        assert jr.headers["Content-Disposition"] == f"attachment; filename=neuroinsight-report-{rid}.json"
        body = json.loads(jr.data)
        assert body["schema"] == "neuroinsight.report.v1" and body["report_id"] == rid and abs(sum(body["prediction"]["probabilities"].values()) - 1) < 1e-9
        assert body["prediction"]["probabilities"] == j["prediction"]["probabilities"]

        pr = c.get(f"/api/reports/{rid}/download/pdf")
        assert pr.status_code == 200 and pr.mimetype == "application/pdf" and pr.data.startswith(b"%PDF-") and pr.data.rstrip().endswith(b"%%EOF")
        assert len(re.findall(rb"/Type /Page\b", pr.data)) == 1                                 # exactly one A4 page
        m = re.search(rb"/MediaBox \[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", pr.data)
        assert m and abs(float(m.group(1)) - 595.28) < 0.1 and abs(float(m.group(2)) - 841.89) < 0.1, "A4 page size"
        assert f"NeuroInsight report {rid}".encode() in pr.data                                 # PDF title metadata
        assert pr.headers["Content-Disposition"] == f"attachment; filename=neuroinsight-report-{rid}.pdf"

        gr = c.get(f"/api/reports/{rid}/download/png")
        assert gr.status_code == 200 and gr.mimetype == "image/png" and gr.data.startswith(b"\x89PNG\r\n\x1a\n")
        img = Image.open(io.BytesIO(gr.data)); img.load()
        assert abs(img.size[0] - 1240) <= 1 and abs(img.size[1] - 1754) <= 1, img.size        # A4 at 150 dpi
        assert gr.headers["Content-Disposition"] == f"attachment; filename=neuroinsight-report-{rid}.png"
        for resp in (jr, pr, gr):
            assert resp.headers["X-Content-Type-Options"] == "nosniff" and resp.headers["Cache-Control"] == "no-store"
        # repeating a download never creates another report
        for fmt in ("pdf", "json", "png"):
            c.get(f"/api/reports/{rid}/download/{fmt}")
        assert c.get("/api/reports").get_json()["total"] == 1
    print("ok preview detail matches the prediction; PDF (A4, 1 page), JSON and PNG download correctly")


def test_report_page_fits_on_one_page_in_the_worst_case():
    rec = dict(report_id="NI-20261106-0A1B2C3D", prediction_id="f" * 32, file_name=("w" * 100 + ".jpeg"), predicted_class="pituitary",
               confidence=0.5, probabilities={"glioma": 0.2, "meningioma": 0.2, "notumor": 0.1, "pituitary": 0.5},
               ranked=[{"label": "pituitary", "prob": 0.5}, {"label": "glioma", "prob": 0.2}, {"label": "meningioma", "prob": 0.2}, {"label": "notumor", "prob": 0.1}],
               warning=dict(level="low", message="Low confidence – expert review strongly recommended."), created_at="2026-11-06T18:21:33Z",
               model=dict(name="N" * 60, description="d " * 40, feature="f " * 40, classifier="c " * 40, decision_rule="argmax_proba",
                          preprocessing_detail="p " * 40, estimator_sha256="a" * 64, manifest_sha256="b" * 64))
    pg = rpt._draw_report(rec, appmod.DISCLAIMER, appmod.GRADCAM_STATUS["reason"])
    assert pg.page_end_y < rpt._H - 0.7, pg.page_end_y
    print(f"ok worst-case report content ends at {pg.page_end_y:.2f} in of {rpt._H} in")


# ---- security ------------------------------------------------------------------------------------------------------------
def test_invalid_ids_traversal_and_methods_are_rejected_safely():
    with _Db() as db:
        c = db.client
        bad_ids = ["abc", "NI-1", "ni-20261106-0a1b2c3d", "x" * 600, "NI-20261106-0A1B2C3D%00", "%27%3B--", "..", "%2e%2e", "NI-20261106-0A1B2C3G"]
        bodies = []
        for rid in bad_ids:
            for path in (f"/api/reports/{rid}", f"/api/reports/{rid}/download/pdf", f"/api/reports/{rid}/download/json"):
                r = c.get(path); bodies.append(r.get_data(as_text=True))
                assert r.status_code in (400, 404) and r.is_json and r.get_json()["ok"] is False, (path, r.status_code)
            r = c.delete(f"/api/reports/{rid}"); bodies.append(r.get_data(as_text=True))
            assert r.status_code in (400, 404, 405), (rid, r.status_code)
        for path in ("/api/reports/..%2F..%2Fetc%2Fpasswd", "/api/reports/NI-20261106-0A1B2C3D/download/../../app.py", "/api/reports/../../app.py",
                     "/api/reports/%2e%2e%2fapp.py/download/pdf", "/api/reports/NI-20261106-0A1B2C3D/download/pdf%00.png"):
            r = c.get(path); bodies.append(r.get_data(as_text=True))
            assert r.status_code in (400, 404) and "import " not in r.get_data(as_text=True), (path, r.status_code)
        valid_missing = "NI-20261106-0A1B2C3D"
        assert c.get(f"/api/reports/{valid_missing}").status_code == 404
        assert c.delete(f"/api/reports/{valid_missing}").status_code == 404
        assert c.get(f"/api/reports/{valid_missing}/download/pdf").status_code == 404
        assert c.get(f"/api/reports/{valid_missing}/download/exe").status_code == 400
        assert c.get(f"/api/reports/{valid_missing}/download/PDF").status_code == 400
        assert c.post(f"/api/reports/{valid_missing}").status_code == 405 and c.put("/api/reports").status_code == 405
        for b in bodies + [c.get(f"/api/reports/{valid_missing}").get_data(as_text=True)]:
            assert "Traceback" not in b and str(ROOT) not in b and "instance" not in b and "sqlite" not in b.lower(), b[:200]
    print("ok invalid IDs, traversal attempts, bad formats and wrong methods are rejected without leaking paths")


def test_storage_failure_does_not_leak_paths():
    with _Db() as db:
        c = db.client
        db.store  # create the store
        db.path.unlink()
        db.path.mkdir()                       # the database file is now a directory: every query fails
        r = c.get("/api/reports")
        text = r.get_data(as_text=True)
        assert r.status_code == 503 and r.get_json()["error"] == "Reports storage is unavailable."
        assert str(db.path) not in text and str(ROOT) not in text and "sqlite" not in text.lower()
        db.path.rmdir()
    print("ok storage errors give a generic message (no file system paths)")


def test_delete_flow():
    j = predict(3)
    with _Db() as db:
        c = db.client
        rid = save(c, j).get_json()["report"]["report_id"]
        assert c.get("/api/reports").get_json()["stats"]["total"] == 1
        assert c.delete(f"/api/reports/{rid}").status_code == 200
        assert c.get(f"/api/reports/{rid}").status_code == 404 and c.delete(f"/api/reports/{rid}").status_code == 404
        assert c.get("/api/reports").get_json()["stats"]["total"] == 0
    print("ok delete removes the report (second delete is a 404)")


def test_csv_export_metadata_only_and_formula_safe():
    with _Db() as db:
        c = db.client
        db.store.create(fake_prediction(1, "glioma", 0.91, name="-1+1.jpg"))
        db.store.create(fake_prediction(2, "notumor", 0.99, name="plain.png"))
        r = c.get("/api/reports/export?sort=oldest")
        assert r.status_code == 200 and r.mimetype == "text/csv" and "attachment" in r.headers["Content-Disposition"]
        text = r.get_data(as_text=True)
        rows = text.strip().split("\n")
        assert rows[0].startswith("report_id,file_name,predicted_class,confidence,p_glioma") and len(rows) == 3
        assert "'-1+1.jpg" in text, "a leading '-' must be neutralised against spreadsheet formula injection"
        assert rpt._csv_safe("=cmd|x") == "'=cmd|x" and rpt._csv_safe("@SUM(A1)") == "'@SUM(A1)" and rpt._csv_safe("ok") == "ok"
        assert c.get("/api/reports/export?cls=glioma").get_data(as_text=True).count("\n") == 2
        assert c.get("/api/reports/export?sort=bad").status_code == 400
    print("ok CSV export: filtered, metadata only, formula-injection safe")


def test_privacy_nothing_but_metadata_is_stored():
    j = predict(0)
    sample = (ROOT / "static" / "samples" / SAMPLES[0]["file"]).read_bytes()
    with _Db() as db:
        save(db.client, j)
        cols = [r[1] for r in db.sql("PRAGMA table_info(reports)")]
        assert cols == ["report_id", "prediction_id", "file_name", "predicted_class", "confidence", "probabilities", "warning_level",
                        "warning_message", "created_at", "model_json"], cols
        raw = db.path.read_bytes()
        assert sample[:200] not in raw and b"\xff\xd8\xff" not in raw, "image bytes must never be stored"
        for table in ("patient", "case", "image"):
            assert not db.sql("SELECT name FROM sqlite_master WHERE name LIKE ?", (f"%{table}%",))
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^instance/$", gi, re.M), "the report database folder must be ignored by Git"
    print("ok only metadata columns exist; no image bytes in the database; instance/ is git-ignored")


def test_no_authentication_exists_and_the_dependency_is_documented():
    rules = {r.rule for r in appmod.app.url_map.iter_rules()}
    assert not any(x in rules for x in ("/login", "/logout", "/api/login")), "no auth routes are expected in this app"
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "login_required" not in src and "session[" not in src
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "no authentication" in readme and "reports" in readme
    print("ok no authentication exists (so none is simulated); the README states the dependency")


def test_reports_code_does_not_touch_models_or_the_test_split():
    for name in ("reports.py", "app.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        assert "allow_test=True" not in src and "load_split" not in src and "artifacts" not in src, name
    print("ok reports code never loads the test split or the legacy artifacts")


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
