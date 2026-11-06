"""NeuroInsight web app (research prototype).

Serves the Phase B selected SVM bundle (see inference.MODEL_BUNDLE_DIR) through a small JSON API:

    GET  /                  the single-page UI
    POST /api/predict       multipart upload (field "file") -> prediction JSON
    GET  /api/model-info    verified Phase B metrics and limitations for the "About the model" panel
    GET  /api/samples       demo samples (drawn from the approved validation split)

Reports module (metadata-only history, see reports.py; no authentication exists in this app):

    GET    /reports                          the Reports page
    POST   /api/reports                      save a just-made prediction as a report (idempotent)
    GET    /api/reports                      list / search / filter / sort / paginate, plus real statistics
    GET    /api/reports/export               CSV export of the filtered list (metadata only)
    GET    /api/reports/<id>                 one report
    DELETE /api/reports/<id>                 delete one report
    GET    /api/reports/<id>/download/<fmt>  fmt = pdf | json | png

Grad-CAM is intentionally disabled: Phase D correction and validation are not complete.
This is NOT a medical device and must not be used for diagnosis.
"""
from __future__ import annotations

import io
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import sqlite3

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError

import reports as rpt
from inference import MODEL_BUNDLE_DIR, Predictor

ROOT = Path(__file__).resolve().parent

# ---- upload limits -------------------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP"}          # what the file really contains (checked, not just the name)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024               # 10 MB
MIN_SIDE_PX = 64
MAX_PIXELS = 25_000_000                           # guards against decompression bombs

# ---- research-demo confidence warnings (NOT clinical safety cutoffs; predictions are never rejected) -----
# Approved two-tier scheme. The 0.90 / 0.70 values were checked against the validation-split calibration
# analysis (runs/*_confidence_threshold_analysis); see the "About the model" panel.
CONFIDENCE_NO_WARNING = 0.90
CONFIDENCE_LOW = 0.70

DISCLAIMER = ("NeuroInsight is a research prototype. It is not a medical device or diagnostic tool and must not "
              "be used to diagnose, treat or make decisions about any patient.")

GRADCAM_STATUS = dict(
    available=False,
    reason=("Grad-CAM visualisation is temporarily disabled until the Phase D correction is complete and the "
            "explanation quality has been validated."))

MODEL_NAME = "NeuroInsight Phase B SVM (ResNet50 features)"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 64 * 1024   # headroom for multipart framing
app.config["REPORTS_DB"] = None   # default: $NEUROINSIGHT_REPORTS_DB, else instance/reports.sqlite3 (tests override this)

_PREDICTOR: Predictor | None = None
_LOAD_ERROR: str | None = None
_LOCK = threading.Lock()
_STORE_LOCK = threading.Lock()
PREDICTIONS = rpt.PredictionCache()   # recent server-produced results (no images); only these can become reports


def confidence_warning(confidence: float) -> dict:
    """Two-tier warning for a top-class probability. Never rejects the prediction."""
    if confidence >= CONFIDENCE_NO_WARNING:
        return dict(level="none", message=None)
    if confidence >= CONFIDENCE_LOW:
        return dict(level="review", message="Review recommended – prediction has uncertainty.")
    return dict(level="low", message="Low confidence – expert review strongly recommended.")


def get_predictor() -> Predictor:
    global _PREDICTOR, _LOAD_ERROR
    with _LOCK:
        if _PREDICTOR is None and _LOAD_ERROR is None:
            try:
                _PREDICTOR = Predictor()
            except Exception as exc:  # reported to the client as a clean error
                _LOAD_ERROR = f"Could not load the model bundle: {exc}"
        if _PREDICTOR is None:
            raise RuntimeError(_LOAD_ERROR)
        return _PREDICTOR


class UploadError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def validate_upload(file_storage) -> bytes:
    """Return the raw bytes of a valid upload or raise UploadError with a user-facing message."""
    if file_storage is None or not file_storage.filename:
        raise UploadError("No file was uploaded. Choose an MRI image (JPG, PNG or BMP).")
    ext = Path(file_storage.filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(f"Unsupported file type '.{ext}'. Allowed types: JPG, JPEG, PNG, BMP.", 415)
    data = file_storage.read(MAX_UPLOAD_BYTES + 1)
    if len(data) == 0:
        raise UploadError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"The file is too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).", 413)
    try:  # header-only inspection: real format and size, without decoding the pixels
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise UploadError("The file could not be read as an image. It may be corrupted or not an image.", 422)
    if fmt not in ALLOWED_FORMATS:
        raise UploadError("The file content is not a JPG, PNG or BMP image, whatever its name says.", 415)
    if min(w, h) < MIN_SIDE_PX:
        raise UploadError(f"The image is too small ({w}×{h}px). The shorter side must be at least {MIN_SIDE_PX}px.", 422)
    if w * h > MAX_PIXELS:
        raise UploadError(f"The image is too large ({w}×{h}px).", 413)
    return data


def _json_error(message: str, status: int):
    resp = jsonify(ok=False, error=message)
    resp.status_code = status
    return resp


@app.errorhandler(413)
def _too_large(_e):
    return _json_error(f"The upload is too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).", 413)


@app.get("/")
def index():
    return render_template("index.html", disclaimer=DISCLAIMER, max_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
                           accept=",".join("." + e for e in sorted(ALLOWED_EXTENSIONS)))


@app.post("/api/predict")
def api_predict():
    try:
        data = validate_upload(request.files.get("file"))
    except UploadError as exc:
        return _json_error(exc.message, exc.status)
    try:
        predictor = get_predictor()
    except RuntimeError as exc:
        return _json_error(str(exc), 503)
    try:
        with _LOCK:  # one TensorFlow forward pass at a time
            result = predictor.predict_bytes(data)
    except Exception:
        return _json_error("The image could not be decoded for analysis. It may be corrupted.", 422)

    info = predictor.describe()
    warning = confidence_warning(result.confidence)
    prediction_id = PREDICTIONS.add(dict(
        file_name=rpt.safe_display_filename(request.files["file"].filename), predicted_class=result.label,
        confidence=result.confidence, probabilities=result.probabilities, warning=warning,
        model=_model_details(predictor)))
    return jsonify(
        ok=True,
        prediction_id=prediction_id,
        prediction=dict(label=result.label, confidence=result.confidence, probabilities=result.probabilities,
                        ranked=result.ranked),
        warning=warning,
        explanation=GRADCAM_STATUS,
        model=dict(name=MODEL_NAME, preprocessing=info["preprocessing"],
                   decision_rule=info["decision_rule"], estimator_sha256=info["estimator_sha256"],
                   manifest_sha256=info["manifest_sha256"]),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        disclaimer=DISCLAIMER,
    )


def _model_details(predictor: Predictor) -> dict:
    """Model facts stored with a report, all read from the loaded bundle (nothing is invented)."""
    meta, info = predictor.meta, predictor.describe()
    svc = (meta.get("hyperparameters") or {}).get("svc") or {}
    classifier = (f"{str(svc.get('kernel', '')).upper()} SVM (C={svc.get('C')}, {svc.get('class_weight')} class weights), "
                  "Platt-scaled probabilities") if svc else "SVM"
    detail = ("Decode to RGB; bilinear resize to 224×224 (no cropping)" if meta["preprocessing"] == "resize"
              else meta["preprocessing"])
    return dict(name=MODEL_NAME, description=meta.get("description"), preprocessing=meta["preprocessing"],
                preprocessing_detail=detail, feature=meta.get("feature"), classifier=classifier,
                decision_rule=meta["decision_rule"], estimator_sha256=info["estimator_sha256"],
                manifest_sha256=info["manifest_sha256"])


def _phase_b_results() -> dict:
    return json.loads((MODEL_BUNDLE_DIR.parent.parent / "results" / "phase_b.json").read_text(encoding="utf-8"))


@app.get("/api/model-info")
def api_model_info():
    try:
        sel = _phase_b_results()["selected_model"]
        val, bun = sel["val"], sel["bundle"]
        per_class = {k: dict(recall=v["recall"], precision=v["precision"], f1=v["f1"], support=v["support"])
                     for k, v in val["per_class"].items()}
        metrics = dict(n=val["n"], accuracy=val["accuracy"], macro_f1=val["macro_f1"],
                       tumor_vs_notumor=val["tumor_vs_notumor"], per_class=per_class,
                       ece=val["calibration"]["ece"], confusion_matrix=val["confusion_matrix"])
    except Exception:
        return _json_error("The verified model metrics are not available.", 503)
    return jsonify(
        ok=True,
        architecture=dict(
            summary=("Frozen ImageNet ResNet50 (global-average-pooled conv5 features, 2048-d) feeding an RBF "
                     "support-vector machine with Platt-scaled probabilities."),
            preprocessing="Decode to RGB; bilinear resize to 224×224 (no cropping).",
            svm=bun["hyperparameters"]["svc"], decision_rule=bun["decision_rule"],
            classes=bun["class_names"]),
        data=dict(split="Approved audited split (train 4877 / val 1046 / test 1045 held out)",
                  manifest_sha256=bun["manifest_sha256"]),
        validation_metrics=metrics,
        confidence_warnings=dict(
            tiers=[dict(range=f"≥ {CONFIDENCE_NO_WARNING:.2f}", warning="none"),
                   dict(range=f"{CONFIDENCE_LOW:.2f} – {CONFIDENCE_NO_WARNING:.2f}", warning="review recommended"),
                   dict(range=f"< {CONFIDENCE_LOW:.2f}", warning="low confidence")],
            note=("Research-demo thresholds, not clinical safety cutoffs. On the validation split, predictions "
                  "with confidence ≥ 0.90 were 98.2% accurate (77.5% of images); lower-confidence predictions "
                  "were much less reliable. Predictions are never rejected.")),
        limitations=[
            "Research prototype. Not validated for clinical use; not a medical device or diagnostic tool.",
            "Metrics are from the validation split, which was also used to choose the model, so they are optimistic "
            "for new data. The held-out test split has not been evaluated.",
            "The dataset has no patient identifiers, so there is no patient-level separation between splits and "
            "some near-duplicate leakage remains.",
            "Meningioma is the weakest class (validation recall about 0.80); it is most often confused with glioma "
            "and pituitary tumors.",
            "The model cannot tell whether an uploaded image is a brain MRI. Unrelated or very different images "
            "will still receive a prediction.",
            "Trained on 2D slices from one public dataset; it may not generalise to other scanners, protocols or populations.",
            "Grad-CAM explanations are disabled until they have been validated.",
        ],
        disclaimer=DISCLAIMER,
    )


@app.get("/api/samples")
def api_samples():
    path = ROOT / "static" / "samples" / "samples.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return jsonify(ok=True, samples=[], note="No demo samples installed.")
    return jsonify(ok=True, note=meta.get("note"), samples=[
        dict(file=s["file"], dataset_label=s["dataset_label"], url=f"/static/samples/{s['file']}")
        for s in meta["samples"]])


# ===================================================================================================================
# Reports module
# ===================================================================================================================
def report_store() -> rpt.ReportStore:
    """The SQLite report store (created lazily). Location: app.config['REPORTS_DB'], else $NEUROINSIGHT_REPORTS_DB,
    else instance/reports.sqlite3 (ignored by Git). The path is never sent to clients."""
    raw = app.config.get("REPORTS_DB") or os.environ.get("NEUROINSIGHT_REPORTS_DB", "").strip() or ROOT / "instance" / "reports.sqlite3"
    path = Path(raw).expanduser()
    path = path if path.is_absolute() else ROOT / path
    with _STORE_LOCK:
        store = app.extensions.get("report_store")
        if store is None or store.path != path:
            store = app.extensions["report_store"] = rpt.ReportStore(path)
        return store


def _summary(rec: dict) -> dict:
    return dict(report_id=rec["report_id"], case_id=None, file_name=rec["file_name"], predicted_class=rec["predicted_class"],
                confidence=rec["confidence"], warning_level=rec["warning"]["level"], created_at=rec["created_at"])


def _detail(rec: dict) -> dict:
    return dict(rpt.report_json(rec, DISCLAIMER, GRADCAM_STATUS["reason"]),
                warning=rec["warning"], created_at=rec["created_at"], case_id=None)


@app.after_request
def _report_headers(resp):
    if request.path.startswith("/api/reports"):
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@app.errorhandler(404)
@app.errorhandler(405)
@app.errorhandler(500)
def _api_errors(e):
    """JSON errors for /api/ paths (no stack traces or file system paths); normal HTML errors elsewhere."""
    if request.path.startswith("/api/"):
        status = getattr(e, "code", 500) or 500
        msg = {404: "Not found.", 405: "Method not allowed."}.get(status, "Something went wrong on the server.")
        return _json_error(msg, status)
    return e


def _storage_error():
    return _json_error("Reports storage is unavailable.", 503)


def _require_report_id(report_id: str):
    rid = rpt.valid_report_id(report_id)
    if rid is None:
        raise rpt.ValidationError("Invalid report ID.")
    return rid


@app.get("/reports")
def reports_page():
    return render_template("reports.html", disclaimer=DISCLAIMER, page_sizes=rpt.PAGE_SIZES,
                           classes=[(c, rpt.CLASS_LABEL[c]) for c in rpt.CLASSES])


@app.post("/api/reports")
def api_reports_create():
    if not request.is_json:   # a cross-site HTML form cannot send JSON, which blocks simple CSRF posts
        return _json_error("Send the request as application/json.", 415)
    body = request.get_json(silent=True)
    pid = rpt.valid_prediction_id(body.get("prediction_id")) if isinstance(body, dict) else None
    if pid is None:
        return _json_error("A valid prediction_id is required.", 400)
    try:
        store = report_store()
        pred = PREDICTIONS.get(pid)
        if pred is None:
            existing = store.find_by_prediction_id(pid)       # already saved earlier (e.g. the cache entry has expired)
            if existing:
                return jsonify(ok=True, created=False, report=_summary(existing)), 200
            return _json_error("This analysis is no longer available. Run it again, then save the report.", 404)
        rec, created = store.create(dict(pred, prediction_id=pid))
    except (sqlite3.Error, OSError, RuntimeError):
        return _storage_error()
    return jsonify(ok=True, created=created, report=_summary(rec)), (201 if created else 200)


@app.get("/api/reports")
def api_reports_list():
    try:
        params = rpt.parse_list_params(request.args)
        store = report_store()
        result = store.list(params)
        stats = store.stats(params["tz_offset"])
    except rpt.ValidationError as exc:
        return _json_error(str(exc), 400)
    except (sqlite3.Error, OSError):
        return _storage_error()
    result["items"] = [_summary(r) for r in result["items"]]
    return jsonify(ok=True, stats=stats, **result)


@app.get("/api/reports/export")
def api_reports_export():
    try:
        params = rpt.parse_list_params(request.args)
        records = report_store().export_rows(params)
    except rpt.ValidationError as exc:
        return _json_error(str(exc), 400)
    except (sqlite3.Error, OSError):
        return _storage_error()
    return send_file(io.BytesIO(rpt.reports_csv(records).encode("utf-8")), mimetype="text/csv", as_attachment=True,
                     download_name="neuroinsight-reports.csv")


@app.get("/api/reports/<report_id>")
def api_reports_get(report_id):
    try:
        rec = report_store().get(_require_report_id(report_id))
    except rpt.ValidationError as exc:
        return _json_error(str(exc), 400)
    except (sqlite3.Error, OSError):
        return _storage_error()
    return jsonify(ok=True, report=_detail(rec)) if rec else _json_error("Report not found.", 404)


@app.delete("/api/reports/<report_id>")
def api_reports_delete(report_id):
    try:
        deleted = report_store().delete(_require_report_id(report_id))
    except rpt.ValidationError as exc:
        return _json_error(str(exc), 400)
    except (sqlite3.Error, OSError):
        return _storage_error()
    return jsonify(ok=True) if deleted else _json_error("Report not found.", 404)


@app.get("/api/reports/<report_id>/download/<fmt>")
def api_reports_download(report_id, fmt):
    """The file name is fixed by the server (never taken from the request or from the stored image name)."""
    if fmt not in rpt.DOWNLOAD_FORMATS:
        return _json_error("Unsupported download format. Use pdf, json or png.", 400)
    try:
        rid = _require_report_id(report_id)
        rec = report_store().get(rid)
    except rpt.ValidationError as exc:
        return _json_error(str(exc), 400)
    except (sqlite3.Error, OSError):
        return _storage_error()
    if rec is None:
        return _json_error("Report not found.", 404)
    note = GRADCAM_STATUS["reason"]
    try:
        if fmt == "json":
            data = json.dumps(rpt.report_json(rec, DISCLAIMER, note), indent=2).encode("utf-8")
        elif fmt == "pdf":
            data = rpt.report_pdf(rec, DISCLAIMER, note)
        else:
            data = rpt.report_png(rec, DISCLAIMER, note)
    except Exception:
        return _json_error("The report file could not be generated.", 500)
    return send_file(io.BytesIO(data), mimetype=rpt.DOWNLOAD_FORMATS[fmt], as_attachment=True,
                     download_name=f"neuroinsight-report-{rid}.{fmt}")


if __name__ == "__main__":
    # Set NEUROINSIGHT_DEBUG=1 for Flask's debug mode (its reloader loads TensorFlow twice).
    app.run(debug=os.environ.get("NEUROINSIGHT_DEBUG") == "1")
