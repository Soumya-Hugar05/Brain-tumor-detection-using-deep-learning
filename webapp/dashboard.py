"""The signed-in area outside the analysis and report pages: dashboard, model performance, profile and placeholders.

Everything shown on the dashboard is read from real records of the signed-in user; nothing is invented. Features that do not
exist yet (cases, history, notifications, settings) render an honest placeholder page instead of fake data.
"""
from __future__ import annotations

from datetime import timedelta

from flask import Blueprint, current_app, render_template, request
from flask_login import current_user

import reports as rpt

from .access import protected
from .extensions import db
from .models import Case, Prediction, utcnow

bp = Blueprint("dashboard", __name__)

# (endpoint, label, icon, built?)  - one list drives the sidebar everywhere
NAV = (
    ("dashboard.home", "Dashboard", "layout-dashboard", True),
    ("analysis_page", "MRI Analysis", "scan-line", True),
    ("cases.cases", "Patients / Cases", "users", True),
    ("cases.history", "Prediction History", "history", True),
    ("reports_page", "Reports", "file-text", True),
    ("dashboard.model", "Model Performance", "chart-column", True),
    ("account.notifications", "Notifications", "bell", True),
    ("account.profile", "My Profile", "user", True),
    ("account.settings", "Settings", "settings", True),
)


@bp.app_context_processor
def _inject_nav():
    return dict(nav=NAV)


def _tz_offset() -> int:
    try:
        v = int(request.args.get("tz_offset", "0"))
    except ValueError:
        return 0
    return v if -840 <= v <= 840 else 0


def _report_store():
    return current_app.config["REPORT_STORE_FACTORY"]()


def dashboard_data(user) -> dict:
    """Real numbers and recent records for one user."""
    week_ago = utcnow() - timedelta(days=7)
    store = _report_store()
    reports_page = store.list(rpt.parse_list_params({"page_size": "10"}), user.id)
    analyses = Prediction.query.filter_by(user_id=user.id)
    cases = Case.query.filter_by(user_id=user.id)
    recent_analyses = analyses.order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(5).all()
    recent_cases = cases.order_by(Case.created_at.desc(), Case.id.desc()).limit(5).all()
    latest_reports = reports_page["items"][:5]
    activity = [dict(kind="analysis", when=p.created_at.strftime(rpt.TS_FORMAT), title=f"Analysis completed: {p.file_name}",
                     detail=f"{rpt.CLASS_LABEL[p.predicted_class]} · {p.confidence * 100:.2f}%", href=None)
                for p in analyses.order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(8)]
    activity += [dict(kind="report", when=r["created_at"], title=f"Report saved: {r['report_id']}",
                      detail=f"{rpt.CLASS_LABEL[r['predicted_class']]} · {r['confidence'] * 100:.2f}% · {r['file_name']}", href=None)
                 for r in reports_page["items"][:8]]
    activity.sort(key=lambda a: a["when"], reverse=True)
    week_analyses = analyses.filter(Prediction.created_at >= week_ago).count()
    week_reports = store.count_since(user.id, week_ago.strftime(rpt.TS_FORMAT))
    return dict(
        stats=dict(analyses=analyses.count(), cases=cases.count(), reports=store.stats(0, user.id)["total"],
                   activity_week=week_analyses + week_reports, analyses_week=week_analyses, reports_week=week_reports),
        recent_analyses=recent_analyses, recent_cases=recent_cases, latest_reports=latest_reports, activity=activity[:8])


def _ctx(active: str, **extra):
    return dict(nav=NAV, active=active, disclaimer=current_app.config["DISCLAIMER_TEXT"], class_label=rpt.CLASS_LABEL, **extra)


@bp.get("/app")
@protected()
def home():
    first = (current_user.full_name or "").split(" ")[0] or "there"
    return render_template("dashboard.html", **_ctx("dashboard.home", first_name=first, **dashboard_data(current_user)))


@bp.get("/app/model")
@protected()
def model():
    from flask import current_app as app
    payload = app.config["MODEL_INFO_FACTORY"]()
    return render_template("model_performance.html", **_ctx("dashboard.model", info=payload))
