"""Cases and prediction history. Every query is scoped to the signed-in user; other users' records answer 404.

A case is only a code the user chooses plus an optional title and notes. No patient fields exist, and the form tells the user
not to enter identifying information. The MRI image is never stored; history rows hold the model output only.
"""
from __future__ import annotations

import re

import sqlalchemy as sa
from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user

import reports as rpt

from .access import protected
from .dashboard import NAV
from .extensions import db
from .models import CASE_STATUSES, CLASSES, Case, Prediction, Report

bp = Blueprint("cases", __name__)

PAGE_SIZE = 15
CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._/\-]{0,39}$")
MAX_TITLE, MAX_NOTES = 150, 2000


def _ctx(active: str, **extra):
    return dict(nav=NAV, active=active, disclaimer=current_app.config["DISCLAIMER_TEXT"], class_label=rpt.CLASS_LABEL,
                max_title=MAX_TITLE, max_notes=MAX_NOTES, **extra)


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _page() -> int:
    try:
        return max(1, int(request.args.get("page", "1")))
    except ValueError:
        return 1


def _pages(total: int) -> int:
    return max(1, -(-total // PAGE_SIZE))


def _own_case(case_id: int) -> Case:
    case = Case.query.filter_by(id=case_id, user_id=current_user.id).first()
    if case is None:
        abort(404)
    return case


def _validate(form, *, exclude_id: int | None = None) -> tuple[dict, dict]:
    code = (form.get("case_code") or "").strip()
    title = (form.get("title") or "").strip()
    notes = (form.get("notes") or "").strip()
    errors: dict[str, list[str]] = {}
    if not code:
        errors["case_code"] = ["Enter a case code."]
    elif not CODE_RE.match(code):
        errors["case_code"] = ["Use up to 40 letters, digits, spaces or . _ - / characters."]
    else:
        q = Case.query.filter(Case.user_id == current_user.id, sa.func.lower(Case.case_code) == code.lower())
        if exclude_id:
            q = q.filter(Case.id != exclude_id)
        if q.first():
            errors["case_code"] = ["You already have a case with this code."]
    if len(title) > MAX_TITLE:
        errors["title"] = [f"Keep the title under {MAX_TITLE} characters."]
    if len(notes) > MAX_NOTES:
        errors["notes"] = [f"Keep the notes under {MAX_NOTES} characters."]
    return dict(case_code=code, title=title or None, notes=notes or None), errors


def _case_rows(status: str, q: str, page: int):
    query = Case.query.filter_by(user_id=current_user.id)
    if status in CASE_STATUSES:
        query = query.filter(Case.status == status)
    if q:
        like = f"%{_escape_like(q)}%"
        query = query.filter(sa.or_(Case.case_code.ilike(like, escape="\\"), Case.title.ilike(like, escape="\\")))
    total = query.count()
    items = query.order_by(Case.created_at.desc(), Case.id.desc()).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE).all()
    counts = dict(db.session.execute(
        sa.select(Prediction.case_id, sa.func.count())
        .where(Prediction.user_id == current_user.id, Prediction.case_id.is_not(None))
        .group_by(Prediction.case_id)).all())
    return items, total, counts


def _render_cases(form, errors, status="", q="", page=1, code=200):
    items, total, counts = _case_rows(status, q, page)
    return render_template("cases.html", **_ctx("cases.cases", items=items, total=total, counts=counts, status=status, q=q,
                                                page=page, pages=_pages(total), form=form, errors=errors)), code


def _case_preds(case: Case):
    return (Prediction.query.filter_by(user_id=current_user.id, case_id=case.id)
            .order_by(Prediction.created_at.desc(), Prediction.id.desc()).all())


def _case_reports(case: Case):
    return (Report.query.filter_by(user_id=current_user.id, case_id=case.id)
            .order_by(Report.created_at.desc(), Report.report_id.desc()).all())


@bp.get("/app/cases")
@protected()
def cases():
    return _render_cases({}, {}, request.args.get("status", ""), (request.args.get("q") or "").strip()[:60], _page())


@bp.post("/app/cases")
@protected()
def create_case():
    values, errors = _validate(request.form)
    if errors:
        return _render_cases(request.form, errors, code=400)
    case = Case(user_id=current_user.id, **values)
    db.session.add(case)
    db.session.commit()
    flash(f"Case {case.case_code} created.", "success")
    return redirect(url_for("cases.case_detail", case_id=case.id))


@bp.get("/app/cases/<int:case_id>")
@protected()
def case_detail(case_id: int):
    case = _own_case(case_id)
    form = dict(case_code=case.case_code, title=case.title or "", notes=case.notes or "")
    return render_template("case_detail.html", **_ctx("cases.cases", case=case, preds=_case_preds(case), reports=_case_reports(case),
                                                      form=form, errors={}))


@bp.post("/app/cases/<int:case_id>/edit")
@protected()
def edit_case(case_id: int):
    case = _own_case(case_id)
    values, errors = _validate(request.form, exclude_id=case.id)
    if errors:
        return render_template("case_detail.html", **_ctx("cases.cases", case=case, preds=_case_preds(case), reports=_case_reports(case),
                                                          form=request.form, errors=errors)), 400
    for key, value in values.items():
        setattr(case, key, value)
    db.session.commit()
    flash("Case updated.", "success")
    return redirect(url_for("cases.case_detail", case_id=case.id))


@bp.post("/app/cases/<int:case_id>/status")
@protected()
def set_case_status(case_id: int):
    case = _own_case(case_id)
    new = request.form.get("status")
    if new not in CASE_STATUSES:
        abort(400)
    case.status = new
    db.session.commit()
    flash("Case archived." if new == "archived" else "Case reopened.", "success")
    return redirect(url_for("cases.case_detail", case_id=case.id))


@bp.post("/app/cases/<int:case_id>/delete")
@protected()
def delete_case(case_id: int):
    case = _own_case(case_id)
    code = case.case_code
    Prediction.query.filter_by(user_id=current_user.id, case_id=case.id).update({Prediction.case_id: None})  # keep the analyses
    Report.query.filter_by(user_id=current_user.id, case_id=case.id).update({Report.case_id: None})           # and the reports
    db.session.delete(case)
    db.session.commit()
    flash(f"Case {code} deleted. Its analyses were kept in your history.", "success")
    return redirect(url_for("cases.cases"))


@bp.get("/app/history")
@protected()
def history():
    q = (request.args.get("q") or "").strip()[:60]
    cls = request.args.get("cls", "")
    case_arg = request.args.get("case", "")
    page = _page()
    query = Prediction.query.filter_by(user_id=current_user.id)
    if cls in CLASSES:
        query = query.filter(Prediction.predicted_class == cls)
    if case_arg == "none":
        query = query.filter(Prediction.case_id.is_(None))
    elif case_arg.isdigit():
        query = query.filter(Prediction.case_id == int(case_arg))
    if q:
        query = query.filter(Prediction.file_name.ilike(f"%{_escape_like(q)}%", escape="\\"))
    total = query.count()
    items = (query.order_by(Prediction.created_at.desc(), Prediction.id.desc())
             .offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE).all())
    all_cases = Case.query.filter_by(user_id=current_user.id).order_by(Case.case_code).all()
    return render_template("history.html", **_ctx("cases.history", items=items, total=total, page=page, pages=_pages(total), q=q,
                                                  cls=cls, case_arg=case_arg, all_cases=all_cases,
                                                  by_id={c.id: c for c in all_cases}, classes=CLASSES))


@bp.post("/app/history/<public_id>/case")
@protected()
def assign_case(public_id: str):
    pred = Prediction.query.filter_by(public_id=public_id, user_id=current_user.id).first()
    if pred is None:
        abort(404)
    raw = (request.form.get("case_id") or "").strip()
    if raw:
        if not raw.isdigit():
            abort(400)
        pred.case_id = _own_case(int(raw)).id
        flash("Analysis added to the case.", "success")
    else:
        pred.case_id = None
        flash("Analysis removed from its case.", "success")
    Report.query.filter_by(user_id=current_user.id, prediction_id=pred.public_id).update({Report.case_id: pred.case_id})  # its report follows
    db.session.commit()
    nxt = request.form.get("next", "")
    return redirect(nxt if nxt.startswith("/app/") and not nxt.startswith("//") else url_for("cases.history"))
