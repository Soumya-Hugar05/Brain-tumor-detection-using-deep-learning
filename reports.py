"""Reports module for NeuroInsight (research prototype): storage, validation and rendering.

What is stored (metadata only, in a small SQLite file): report ID, prediction ID, a sanitised file name, predicted
class, confidence, the four probabilities, the confidence warning, the timestamp and the model description / hashes.
Original MRI images, patient identifiers and PDFs are NOT stored. PDF, PNG and JSON files are rendered on demand
from the stored record.

There is no authentication in this application, so every visitor to the server can see every report. Keep the server
bound to localhost (the default) or add authentication before exposing it; see the README.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_LABEL = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumor", "pituitary": "Pituitary tumor"}
REPORT_ID_RE = re.compile(r"^NI-[0-9]{8}-[0-9A-F]{8}$")   # ASCII digits only ("\d" would also match other scripts)
PREDICTION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
SORTS = {"newest": "created_at DESC, report_id DESC", "oldest": "created_at ASC, report_id ASC"}
PAGE_SIZES = (10, 25, 50)
MAX_QUERY_LEN = 100
EXPORT_LIMIT = 10000
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DOWNLOAD_FORMATS = {"json": "application/json", "png": "image/png", "pdf": "application/pdf"}


class ValidationError(ValueError):
    """A request value is invalid. The message is safe to show to the user."""


# ---------------------------------------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------------------------------------
def valid_report_id(value) -> str | None:
    return value if isinstance(value, str) and REPORT_ID_RE.fullmatch(value) else None


def valid_prediction_id(value) -> str | None:
    return value if isinstance(value, str) and PREDICTION_ID_RE.fullmatch(value) else None


_FILENAME_BAD = re.compile(r"[^A-Za-z0-9._ ()\-]")


def safe_display_filename(name) -> str:
    """The browser-supplied name is untrusted: keep only the last path component and a conservative character set."""
    base = re.split(r"[\\/]", str(name or ""))[-1]
    base = _FILENAME_BAD.sub("_", base).strip(" .")
    if not base:
        return "image"
    if len(base) > 120:
        stem, dot, ext = base.rpartition(".")
        ext = ext[:10] if dot else ""
        base = (stem or base)[: 119 - len(ext)] + ("." + ext if ext else "")
    return base


def _local_day_start_utc(day: str, tz_offset: int) -> datetime:
    """UTC instant of local midnight for 'YYYY-MM-DD'. tz_offset = JavaScript getTimezoneOffset() (UTC minus local)."""
    try:
        d = datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        raise ValidationError("Invalid date.") from None
    return d + timedelta(minutes=tz_offset)


def _fmt(dt: datetime) -> str:
    return dt.strftime(TS_FORMAT)


def parse_list_params(args) -> dict:
    """Validate and normalise the query parameters of the list/export endpoints (raises ValidationError)."""
    def _int(name, default, lo, hi):
        raw = args.get(name)
        if raw in (None, ""):
            return default
        try:
            v = int(raw)
        except (TypeError, ValueError):
            raise ValidationError(f"Invalid {name}.") from None
        if not lo <= v <= hi:
            raise ValidationError(f"Invalid {name}.")
        return v

    q = (args.get("q") or "").strip()
    if len(q) > MAX_QUERY_LEN:
        raise ValidationError("The search text is too long.")
    cls = (args.get("cls") or "").strip()
    if cls and cls not in CLASSES:
        raise ValidationError("Unknown predicted class.")
    case = (args.get("case") or "").strip()
    if case and case != "none" and not (case.isascii() and case.isdigit() and len(case) <= 12):
        raise ValidationError("Invalid case filter.")
    sort = (args.get("sort") or "newest").strip()
    if sort not in SORTS:
        raise ValidationError("Invalid sort order.")
    tz = _int("tz_offset", 0, -840, 840)
    dates = {}
    for key in ("date_from", "date_to"):
        v = (args.get(key) or "").strip()
        if v:
            if not DATE_RE.fullmatch(v):
                raise ValidationError("Dates must look like YYYY-MM-DD.")
            _local_day_start_utc(v, tz)  # also rejects impossible dates such as 2026-02-31
        dates[key] = v
    if dates["date_from"] and dates["date_to"] and dates["date_from"] > dates["date_to"]:
        raise ValidationError("The start date is after the end date.")
    page_size = _int("page_size", PAGE_SIZES[0], 1, 1000)
    if page_size not in PAGE_SIZES:
        raise ValidationError("Invalid page size.")
    return dict(q=q, cls=cls, case=case, sort=sort, tz_offset=tz, date_from=dates["date_from"], date_to=dates["date_to"],
                page=_int("page", 1, 1, 1_000_000), page_size=page_size)


# ---------------------------------------------------------------------------------------------------------
# prediction cache: only results that the server itself produced can become reports
# ---------------------------------------------------------------------------------------------------------
class PredictionCache:
    """Short-lived, bounded, in-memory store of recent prediction results (no images)."""

    def __init__(self, max_items: int = 256, ttl_seconds: int = 3600):
        self.max_items, self.ttl = max_items, ttl_seconds
        self._items: OrderedDict[str, tuple[float, int, dict]] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, data: dict, user_id: int) -> str:
        pid = secrets.token_hex(16)
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            self._items[pid] = (now, user_id, data)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return pid

    def get(self, pid: str, user_id: int) -> dict | None:
        """Only the user who made the prediction can retrieve it (a different user gets None, as if it never existed)."""
        with self._lock:
            self._purge(time.monotonic())
            hit = self._items.get(pid)
            return None if hit is None or hit[1] != user_id else json.loads(json.dumps(hit[2]))

    def _purge(self, now: float) -> None:
        for k in [k for k, (t, *_rest) in self._items.items() if now - t > self.ttl]:
            del self._items[k]


# ---------------------------------------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    report_id       TEXT PRIMARY KEY,
    prediction_id   TEXT NOT NULL UNIQUE,
    file_name       TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence      REAL NOT NULL,
    probabilities   TEXT NOT NULL,
    warning_level   TEXT NOT NULL,
    warning_message TEXT,
    created_at      TEXT NOT NULL,
    model_json      TEXT NOT NULL,
    user_id         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_reports_created ON reports (created_at);
CREATE INDEX IF NOT EXISTS idx_reports_class ON reports (predicted_class);
"""


def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class ReportStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)
            if "user_id" not in {r["name"] for r in con.execute("PRAGMA table_info(reports)")}:
                # reports saved before accounts existed have no owner: they stay in the file but belong to nobody
                con.execute("ALTER TABLE reports ADD COLUMN user_id INTEGER")
            con.execute("CREATE INDEX IF NOT EXISTS idx_reports_user ON reports (user_id)")

    @contextmanager
    def _connect(self):
        """A connection that commits (or rolls back) and is always closed (open handles would lock the file on Windows)."""
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    # -- records -------------------------------------------------------------------------------------------
    @staticmethod
    def _record(row: sqlite3.Row) -> dict:
        probs = json.loads(row["probabilities"])
        ranked = sorted(({"label": c, "prob": float(probs[c])} for c in CLASSES if c in probs), key=lambda d: -d["prob"])
        return dict(
            report_id=row["report_id"], prediction_id=row["prediction_id"], file_name=row["file_name"],
            predicted_class=row["predicted_class"], confidence=row["confidence"], probabilities=probs, ranked=ranked,
            warning=dict(level=row["warning_level"], message=row["warning_message"]),
            created_at=row["created_at"], model=json.loads(row["model_json"]))

    def create(self, pred: dict, user_id: int) -> tuple[dict, bool]:
        """Persist a server-produced prediction as the user's report. Idempotent per prediction_id."""
        with self._lock, self._connect() as con:
            row = con.execute("SELECT * FROM reports WHERE prediction_id = ? AND user_id = ?", (pred["prediction_id"], user_id)).fetchone()
            if row:
                return self._record(row), False
            now = datetime.now(timezone.utc)
            for _ in range(8):
                rid = f"NI-{now:%Y%m%d}-{secrets.token_hex(4).upper()}"
                try:
                    con.execute(
                        "INSERT INTO reports (report_id, prediction_id, file_name, predicted_class, confidence, probabilities,"
                        " warning_level, warning_message, created_at, model_json, user_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (rid, pred["prediction_id"], pred["file_name"], pred["predicted_class"], float(pred["confidence"]),
                         json.dumps({c: float(pred["probabilities"][c]) for c in CLASSES}), pred["warning"]["level"],
                         pred["warning"]["message"], _fmt(now), json.dumps(pred["model"]), user_id))
                    break
                except sqlite3.IntegrityError:
                    row = con.execute("SELECT * FROM reports WHERE prediction_id = ? AND user_id = ?", (pred["prediction_id"], user_id)).fetchone()
                    if row:  # lost a race with an identical request
                        return self._record(row), False
            else:
                raise RuntimeError("could not allocate a report id")
            return self._record(con.execute("SELECT * FROM reports WHERE report_id = ?", (rid,)).fetchone()), True

    def get(self, report_id: str, user_id: int) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM reports WHERE report_id = ? AND user_id = ?", (report_id, user_id)).fetchone()
        return self._record(row) if row else None

    def find_by_prediction_id(self, prediction_id: str, user_id: int) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM reports WHERE prediction_id = ? AND user_id = ?", (prediction_id, user_id)).fetchone()
        return self._record(row) if row else None

    def delete(self, report_id: str, user_id: int) -> bool:
        with self._lock, self._connect() as con:
            return con.execute("DELETE FROM reports WHERE report_id = ? AND user_id = ?", (report_id, user_id)).rowcount > 0

    # -- queries -------------------------------------------------------------------------------------------
    @staticmethod
    def _where(p: dict, user_id: int) -> tuple[str, list]:
        clauses, args = ["user_id = ?"], [user_id]
        if p["q"]:
            clauses.append("(report_id LIKE ? ESCAPE '\\' OR file_name LIKE ? ESCAPE '\\')")
            args += [_like(p["q"]), _like(p["q"])]
        if p["cls"]:
            clauses.append("predicted_class = ?")
            args.append(p["cls"])
        if p["date_from"]:
            clauses.append("created_at >= ?")
            args.append(_fmt(_local_day_start_utc(p["date_from"], p["tz_offset"])))
        if p["date_to"]:
            clauses.append("created_at < ?")
            args.append(_fmt(_local_day_start_utc(p["date_to"], p["tz_offset"]) + timedelta(days=1)))
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", args

    def list(self, p: dict, user_id: int) -> dict:
        where, args = self._where(p, user_id)
        with self._connect() as con:
            total = con.execute("SELECT COUNT(*) FROM reports" + where, args).fetchone()[0]
            pages = max(1, math.ceil(total / p["page_size"]))
            page = min(p["page"], pages)
            rows = con.execute(f"SELECT * FROM reports{where} ORDER BY {SORTS[p['sort']]} LIMIT ? OFFSET ?",
                               args + [p["page_size"], (page - 1) * p["page_size"]]).fetchall()
        return dict(items=[self._record(r) for r in rows], total=total, page=page, pages=pages, page_size=p["page_size"])

    def stats(self, tz_offset: int = 0, user_id: int = 0) -> dict:
        """Real counts: all reports, reports generated today and this month (local day / month of the client)."""
        local_now = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=tz_offset)
        day = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        month = day.replace(day=1)
        offset = timedelta(minutes=tz_offset)
        with self._connect() as con:
            def count(since=None):
                if since is None:
                    return con.execute("SELECT COUNT(*) FROM reports WHERE user_id = ?", (user_id,)).fetchone()[0]
                return con.execute("SELECT COUNT(*) FROM reports WHERE user_id = ? AND created_at >= ?", (user_id, _fmt(since + offset))).fetchone()[0]
            return dict(total=count(), today=count(day), this_month=count(month))

    def count_since(self, user_id: int, since_ts: str) -> int:
        """Number of the user's reports created at or after a UTC timestamp string (TS_FORMAT)."""
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM reports WHERE user_id = ? AND created_at >= ?", (user_id, since_ts)).fetchone()[0]

    def export_rows(self, p: dict, user_id: int) -> list[dict]:
        where, args = self._where(p, user_id)
        with self._connect() as con:
            rows = con.execute(f"SELECT * FROM reports{where} ORDER BY {SORTS[p['sort']]} LIMIT {EXPORT_LIMIT}", args).fetchall()
        return [self._record(r) for r in rows]


# ---------------------------------------------------------------------------------------------------------
# renderers: JSON, CSV, PDF and PNG are all built from the stored record
# ---------------------------------------------------------------------------------------------------------
def report_json(rec: dict, disclaimer: str, explanation_note: str) -> dict:
    return {
        "schema": "neuroinsight.report.v1",
        "report_id": rec["report_id"], "prediction_id": rec["prediction_id"],
        "generated_utc": rec["created_at"], "case_id": None, "file_name": rec["file_name"],
        "prediction": {"label": rec["predicted_class"], "confidence": rec["confidence"],
                       "probabilities": rec["probabilities"], "ranked": rec["ranked"]},
        "confidence_warning": rec["warning"],
        "explanation": {"available": False, "reason": explanation_note},
        "model": rec["model"],
        "original_image_stored": False,
        "disclaimer": disclaimer,
    }


def _csv_safe(value) -> str:
    s = str(value)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s  # neutralise spreadsheet formula injection


def reports_csv(records: list[dict]) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["report_id", "file_name", "predicted_class", "confidence", *[f"p_{c}" for c in CLASSES],
                "confidence_warning", "generated_utc", "model_estimator_sha256"])
    for r in records:
        w.writerow([r["report_id"], _csv_safe(r["file_name"]), r["predicted_class"], f"{r['confidence']:.6f}",
                    *[f"{r['probabilities'][c]:.6f}" for c in CLASSES], r["warning"]["level"], r["created_at"],
                    r["model"].get("estimator_sha256", "")])
    return out.getvalue()


_render_lock = threading.Lock()
_W, _H, _M = 210 / 25.4, 297 / 25.4, 0.6   # A4 (210 x 297 mm) in inches, page margin
_NAVY, _TEAL, _CYAN = "#0a1327", "#0f766e", "#0891b2"
_TEXT, _MUTED, _BORDER, _SURFACE = "#0d1a31", "#475a78", "#d6e0ee", "#f3f7fc"
_CLASS_COLOR = {"glioma": "#6d28d9", "meningioma": "#b45309", "notumor": "#0f766e", "pituitary": "#be123c"}


class _Page:
    """A tiny layout helper on a matplotlib Figure in page inches (y grows downwards)."""

    def __init__(self):
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(_W, _H), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, _W)
        self.ax.set_ylim(_H, 0)
        self.ax.axis("off")

    def rect(self, x, y, w, h, fc, ec="none", lw=0.8, r=0.0):
        from matplotlib.patches import FancyBboxPatch, Rectangle
        patch = (FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r},rounding_size={r}", fc=fc, ec=ec, lw=lw)
                 if r else Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
        self.ax.add_patch(patch)

    def text(self, x, y, s, size=10, weight="normal", color=_TEXT, ha="left", va="top", family="DejaVu Sans"):
        return self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, family=family)

    def width(self, s, size, weight="normal") -> float:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import text_to_path
        w, _, _ = text_to_path.get_text_width_height_descent(s, FontProperties(family="DejaVu Sans", size=size, weight=weight), False)
        return w / 72.0

    def wrap(self, s, size, maxw, weight="normal") -> list[str]:
        lines, cur = [], ""
        for word in str(s).split(" "):
            if self.width(word, size, weight) > maxw:          # an unbreakable token (e.g. a long file name): split it
                if cur:
                    lines.append(cur)
                    cur = ""
                for ch in word:
                    if cur and self.width(cur + ch, size, weight) > maxw:
                        lines.append(cur)
                        cur = ch
                    else:
                        cur += ch
                continue
            trial = f"{cur} {word}" if cur else word
            if cur and self.width(trial, size, weight) > maxw:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        return lines + ([cur] if cur else [])

    def para(self, x, y, s, size, maxw, color=_TEXT, weight="normal", lh=1.3) -> float:
        for line in self.wrap(s, size, maxw, weight):
            self.text(x, y, line, size, weight, color)
            y += size / 72.0 * lh
        return y


def _nice_time(ts: str) -> str:
    return datetime.strptime(ts, TS_FORMAT).strftime("%Y-%m-%d %H:%M:%S UTC")


def _draw_report(rec: dict, disclaimer: str, explanation_note: str) -> _Page:
    pg = _Page()
    m = rec["model"]
    cw = _W - 2 * _M

    # header band
    pg.rect(0, 0, _W, 1.1, _NAVY)
    pg.rect(_M, 0.25, 0.6, 0.6, _TEAL, r=0.12)
    pg.ax.plot([_M + 0.1, _M + 0.22, _M + 0.3, _M + 0.38, _M + 0.43, _M + 0.5], [0.57, 0.57, 0.4, 0.73, 0.55, 0.55], color="white", lw=1.8, solid_capstyle="round")
    pg.text(_M + 0.8, 0.23, "NeuroInsight", 22, "bold", "white")
    pg.text(_M + 0.8, 0.61, "Brain MRI Analysis Report", 11, "normal", "#bcd0ea")
    bw = pg.width("RESEARCH PROTOTYPE", 8.5, "bold") + 0.3
    pg.rect(_W - _M - bw, 0.33, bw, 0.34, "#f59e0b", r=0.17)
    pg.text(_W - _M - bw / 2, 0.5, "RESEARCH PROTOTYPE", 8.5, "bold", "#1a1203", ha="center", va="center")

    # disclaimer strip
    y = 1.1
    pg.rect(0, y, _W, 0.38, "#fef3c7")
    pg.text(_W / 2, y + 0.19, "Research prototype – not a medical diagnostic report. Not for clinical use.", 9.5, "bold", "#78350f", ha="center", va="center")
    y += 0.38 + 0.25

    def section(title, y):
        pg.text(_M, y, title.upper(), 8.5, "bold", _TEAL)
        pg.ax.plot([_M, _W - _M], [y + 0.19, y + 0.19], color=_BORDER, lw=0.8)
        return y + 0.3

    # report information
    y = section("Report information", y)
    info = [("Report ID", rec["report_id"]), ("Generated", _nice_time(rec["created_at"])),
            ("Case ID", "Not available (no patient information is collected)"), ("Image file", rec["file_name"]),
            ("Prediction ID", rec["prediction_id"])]
    for k, v in info:
        pg.text(_M, y, k, 9, "bold", _MUTED)
        y = pg.para(_M + 1.35, y, v, 9.5, cw - 1.35) + 0.02
    y += 0.12

    # MRI image placeholder + prediction
    y = section("MRI analysis", y)
    box = 1.7
    pg.rect(_M, y, box, box, _SURFACE, _BORDER, r=0.08)
    pg.text(_M + box / 2, y + box / 2 - 0.12, "Original MRI image", 9.5, "bold", _MUTED, ha="center", va="center")
    pg.text(_M + box / 2, y + box / 2 + 0.12, "is not stored with reports", 9, "normal", _MUTED, ha="center", va="center")
    px = _M + box + 0.4
    pw = _W - _M - px
    col = _CLASS_COLOR.get(rec["predicted_class"], _TEAL)
    pg.text(px, y, "PREDICTED CATEGORY", 8, "bold", _MUTED)
    pg.text(px, y + 0.2, CLASS_LABEL.get(rec["predicted_class"], rec["predicted_class"]), 22, "bold", col)
    pg.text(px, y + 0.68, "CONFIDENCE", 8, "bold", _MUTED)
    pg.text(px, y + 0.88, f"{rec['confidence'] * 100:.2f}%", 24, "bold", _TEXT)
    wy = y + 1.32
    if rec["warning"]["level"] != "none" and rec["warning"]["message"]:
        wc = "#b91c1c" if rec["warning"]["level"] == "low" else "#92400e"
        wy = pg.para(px, wy, rec["warning"]["message"], 9.5, pw, wc, "bold") + 0.04
    wy = pg.para(px, wy, "Model output only – not a diagnosis. No clinical interpretation or recommendation is provided.", 8.5, pw, _MUTED)
    y += max(box, wy - y) + 0.22

    # probabilities
    y = section("Class probabilities", y)
    for item in rec["ranked"]:
        c = _CLASS_COLOR.get(item["label"], _TEAL)
        pg.rect(_M, y + 0.045, 0.13, 0.13, c, r=0.02)
        pg.text(_M + 0.25, y, CLASS_LABEL.get(item["label"], item["label"]), 10, "bold" if item["label"] == rec["predicted_class"] else "normal")
        pg.text(_W - _M, y, f"{item['prob'] * 100:.2f}%", 10, "bold", ha="right")
        pg.rect(_M, y + 0.25, cw, 0.09, "#e2e8f0", r=0.045)
        if item["prob"] > 0:
            pg.rect(_M, y + 0.25, max(cw * item["prob"], 0.09), 0.09, c, r=0.045)
        y += 0.43
    y += 0.05

    # model information
    y = section("Model information", y)
    rows = [("Model", m.get("name", "")), ("Bundle", m.get("description", "")), ("Feature extractor", m.get("feature", "")),
            ("Classifier", m.get("classifier", "")), ("Decision rule", m.get("decision_rule", "")),
            ("Preprocessing", m.get("preprocessing_detail") or m.get("preprocessing", "")), ("Explainability", explanation_note)]
    for k, v in rows:
        pg.text(_M, y, k, 9, "bold", _MUTED)
        y = pg.para(_M + 1.35, y, v, 9.5, cw - 1.35) + 0.02
    for k, v in (("Estimator SHA-256", m.get("estimator_sha256", "")), ("Manifest SHA-256", m.get("manifest_sha256", ""))):
        pg.text(_M, y, k, 9, "bold", _MUTED)
        pg.text(_M + 1.35, y + 0.01, v, 7, "normal", _TEXT, family="DejaVu Sans Mono")
        y += 0.21
    y += 0.1

    # disclaimer box
    dh = 0.3 + len(pg.wrap(disclaimer, 9, cw - 0.4)) * 9 / 72 * 1.3 + 0.1
    pg.rect(_M, y, cw, dh, "#fffbeb", "#f59e0b", r=0.06)
    pg.text(_M + 0.2, y + 0.12, "Disclaimer", 9, "bold", "#78350f")
    pg.para(_M + 0.2, y + 0.3, disclaimer, 9, cw - 0.4, "#451a03")

    # footer
    pg.ax.plot([_M, _W - _M], [_H - 0.62, _H - 0.62], color=_BORDER, lw=0.8)
    pg.text(_M, _H - 0.52, f"NeuroInsight · Research prototype · {rec['report_id']}", 8, "normal", _MUTED)
    pg.text(_W - _M, _H - 0.52, "Page 1 of 1", 8, "normal", _MUTED, ha="right")
    pg.page_end_y = y + dh
    return pg


def report_pdf(rec: dict, disclaimer: str, explanation_note: str) -> bytes:
    with _render_lock:
        pg = _draw_report(rec, disclaimer, explanation_note)
        buf = io.BytesIO()
        created = datetime.strptime(rec["created_at"], TS_FORMAT).replace(tzinfo=timezone.utc)
        pg.fig.savefig(buf, format="pdf", metadata={"Title": f"NeuroInsight report {rec['report_id']}", "Author": "NeuroInsight (research prototype)",
                                                    "Subject": "Brain MRI analysis report – not a medical diagnostic report",
                                                    "Creator": "NeuroInsight", "CreationDate": created, "ModDate": created})
        return buf.getvalue()


def report_png(rec: dict, disclaimer: str, explanation_note: str) -> bytes:
    with _render_lock:
        pg = _draw_report(rec, disclaimer, explanation_note)
        buf = io.BytesIO()
        pg.fig.savefig(buf, format="png", dpi=150, metadata={"Software": "NeuroInsight"})
        return buf.getvalue()
