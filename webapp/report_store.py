"""Report storage in the main application database (SQLAlchemy). Same interface as the old SQLite file store.

Every method is scoped to one user; another user's report behaves as if it did not exist. A report inherits the case of the
analysis it was saved from.
"""
from __future__ import annotations

import math
import secrets
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from flask import has_app_context

import reports as rpt

from .extensions import db
from .models import Case, Prediction, Report

_ORDER = {"newest": lambda: (Report.created_at.desc(), Report.report_id.desc()),
          "oldest": lambda: (Report.created_at.asc(), Report.report_id.asc())}


def _escape(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class DbReportStore:
    def __init__(self, app):
        self.app = app

    def _run(self, fn):
        """Run inside the current app context, or create one (used by scripts and unit tests)."""
        if has_app_context():
            return fn()
        with self.app.app_context():
            try:
                return fn()
            finally:
                db.session.remove()

    @staticmethod
    def _record(row: Report, codes: dict) -> dict:
        probs = row.probabilities
        ranked = sorted(({"label": c, "prob": float(probs[c])} for c in rpt.CLASSES if c in probs), key=lambda d: -d["prob"])
        return dict(report_id=row.report_id, prediction_id=row.prediction_id, file_name=row.file_name,
                    predicted_class=row.predicted_class, confidence=row.confidence, probabilities=probs, ranked=ranked,
                    warning=dict(level=row.warning_level, message=row.warning_message), created_at=row.created_at,
                    model=row.model_json, case_id=row.case_id, case_code=codes.get(row.case_id))

    @staticmethod
    def _codes(rows, user_id: int) -> dict:
        ids = {r.case_id for r in rows if r.case_id}
        if not ids:
            return {}
        return dict(db.session.execute(sa.select(Case.id, Case.case_code).where(Case.user_id == user_id, Case.id.in_(ids))).all())

    def _one(self, row: Report | None, user_id: int) -> dict | None:
        return self._record(row, self._codes([row], user_id)) if row else None

    # -- records -------------------------------------------------------------------------------------------
    def create(self, pred: dict, user_id: int) -> tuple[dict, bool]:
        """Persist a server-produced prediction as the user's report. Idempotent per prediction_id."""
        def work():
            existing = Report.query.filter_by(prediction_id=pred["prediction_id"], user_id=user_id).first()
            if existing:
                return self._one(existing, user_id), False
            source = Prediction.query.filter_by(public_id=pred["prediction_id"], user_id=user_id).first()
            now = datetime.now(timezone.utc)
            for _ in range(8):
                row = Report(report_id=f"NI-{now:%Y%m%d}-{secrets.token_hex(4).upper()}", prediction_id=pred["prediction_id"],
                             user_id=user_id, case_id=source.case_id if source else None, file_name=pred["file_name"],
                             predicted_class=pred["predicted_class"], confidence=float(pred["confidence"]),
                             probabilities={c: float(pred["probabilities"][c]) for c in rpt.CLASSES},
                             warning_level=pred["warning"]["level"], warning_message=pred["warning"]["message"],
                             created_at=rpt._fmt(now), model_json=pred["model"])
                try:
                    with db.session.begin_nested():
                        db.session.add(row)
                except sa.exc.IntegrityError:
                    again = Report.query.filter_by(prediction_id=pred["prediction_id"], user_id=user_id).first()
                    if again:                                    # lost a race with an identical request
                        return self._one(again, user_id), False
                    continue                                     # report id collision: draw another
                db.session.commit()
                return self._one(row, user_id), True
            raise RuntimeError("could not allocate a report id")
        return self._run(work)

    def get(self, report_id: str, user_id: int) -> dict | None:
        return self._run(lambda: self._one(Report.query.filter_by(report_id=report_id, user_id=user_id).first(), user_id))

    def find_by_prediction_id(self, prediction_id: str, user_id: int) -> dict | None:
        return self._run(lambda: self._one(Report.query.filter_by(prediction_id=prediction_id, user_id=user_id).first(), user_id))

    def delete(self, report_id: str, user_id: int) -> bool:
        def work():
            n = Report.query.filter_by(report_id=report_id, user_id=user_id).delete()
            db.session.commit()
            return n > 0
        return self._run(work)

    # -- queries -------------------------------------------------------------------------------------------
    @staticmethod
    def _filters(p: dict, user_id: int) -> list:
        f = [Report.user_id == user_id]
        if p["q"]:
            f.append(sa.or_(Report.report_id.ilike(_escape(p["q"]), escape="\\"), Report.file_name.ilike(_escape(p["q"]), escape="\\")))
        if p["cls"]:
            f.append(Report.predicted_class == p["cls"])
        if p["date_from"]:
            f.append(Report.created_at >= rpt._fmt(rpt._local_day_start_utc(p["date_from"], p["tz_offset"])))
        if p["date_to"]:
            f.append(Report.created_at < rpt._fmt(rpt._local_day_start_utc(p["date_to"], p["tz_offset"]) + timedelta(days=1)))
        case = p.get("case", "")
        if case == "none":
            f.append(Report.case_id.is_(None))
        elif case:
            f.append(Report.case_id == int(case))
        return f

    def list(self, p: dict, user_id: int) -> dict:
        def work():
            flt = self._filters(p, user_id)
            total = db.session.scalar(sa.select(sa.func.count()).select_from(Report).where(*flt))
            pages = max(1, math.ceil(total / p["page_size"]))
            page = min(p["page"], pages)
            rows = (db.session.scalars(sa.select(Report).where(*flt).order_by(*_ORDER[p["sort"]]())
                                       .limit(p["page_size"]).offset((page - 1) * p["page_size"])).all())
            codes = self._codes(rows, user_id)
            return dict(items=[self._record(r, codes) for r in rows], total=total, page=page, pages=pages, page_size=p["page_size"])
        return self._run(work)

    @staticmethod
    def _count(user_id: int, since: str | None = None) -> int:
        q = sa.select(sa.func.count()).select_from(Report).where(Report.user_id == user_id)
        if since is not None:
            q = q.where(Report.created_at >= since)
        return db.session.scalar(q)

    def stats(self, tz_offset: int = 0, user_id: int = 0) -> dict:
        """Real counts: all reports, reports generated today and this month (local day / month of the client)."""
        local_now = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=tz_offset)
        day = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        month = day.replace(day=1)
        offset = timedelta(minutes=tz_offset)
        return self._run(lambda: dict(total=self._count(user_id), today=self._count(user_id, rpt._fmt(day + offset)),
                                      this_month=self._count(user_id, rpt._fmt(month + offset))))

    def count_since(self, user_id: int, since_ts: str) -> int:
        return self._run(lambda: self._count(user_id, since_ts))

    def export_rows(self, p: dict, user_id: int) -> list[dict]:
        def work():
            rows = db.session.scalars(sa.select(Report).where(*self._filters(p, user_id)).order_by(*_ORDER[p["sort"]]())
                                      .limit(rpt.EXPORT_LIMIT)).all()
            codes = self._codes(rows, user_id)
            return [self._record(r, codes) for r in rows]
        return self._run(work)
