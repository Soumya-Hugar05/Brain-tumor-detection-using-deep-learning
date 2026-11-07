"""Administration: user accounts only.

Admins can list accounts, deactivate or reactivate them, and end their sessions. They cannot see cases, analyses, reports or
notifications of any user: no admin query touches those tables, and the case, history and report routes are scoped to the
signed-in user whatever their role. Every action is written to the audit log. Roles are never changed here (use the
`create-admin` command), so a stolen admin session cannot mint new admins.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user

from . import security
from .access import protected
from .dashboard import NAV
from .extensions import db
from .models import AuditLog, User

bp = Blueprint("admin", __name__)
log = logging.getLogger("neuroinsight.admin")
PAGE_SIZE = 20
ACTIONS = ("deactivate", "reactivate", "revoke_sessions")


def _ctx(active: str, **extra):
    return dict(nav=NAV, active=active, disclaimer=current_app.config["DISCLAIMER_TEXT"], **extra)


def _audit(action: str, target: User | None) -> None:
    db.session.add(AuditLog(actor_id=current_user.id, target_id=target.id if target else None, action=action, ip=security.client_ip()))
    log.info("admin action=%s actor=%s target=%s", action, current_user.id, target.id if target else None)


def _page() -> int:
    try:
        return max(1, int(request.args.get("page", "1")))
    except ValueError:
        return 1


def _escape(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


@bp.get("/app/admin")
@protected("admin")
def overview():
    count = lambda *w: db.session.scalar(sa.select(sa.func.count()).select_from(User).where(*w))  # noqa: E731
    stats = dict(users=count(), active=count(User.is_active_flag.is_(True)), inactive=count(User.is_active_flag.is_(False)),
                 unverified=count(User.email_verified_at.is_(None)), admins=count(User.role == "admin"))
    recent = AuditLog.query.order_by(AuditLog.id.desc()).limit(5).all()
    return render_template("admin_overview.html", **_ctx("admin.overview", stats=stats, recent=recent, names=_names(recent)))


def _names(rows) -> dict:
    ids = {i for r in rows for i in (r.actor_id, r.target_id) if i}
    return dict(db.session.execute(sa.select(User.id, User.email).where(User.id.in_(ids))).all()) if ids else {}


@bp.get("/app/admin/users")
@protected("admin")
def users():
    q = (request.args.get("q") or "").strip()[:60]
    status = request.args.get("status", "")
    page = _page()
    query = User.query
    if q:
        query = query.filter(sa.or_(User.email.ilike(_escape(q), escape="\\"), User.full_name.ilike(_escape(q), escape="\\")))
    if status == "active":
        query = query.filter(User.is_active_flag.is_(True))
    elif status == "inactive":
        query = query.filter(User.is_active_flag.is_(False))
    elif status == "unverified":
        query = query.filter(User.email_verified_at.is_(None))
    elif status == "admin":
        query = query.filter(User.role == "admin")
    total = query.count()
    items = query.order_by(User.created_at.desc(), User.id.desc()).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE).all()
    return render_template("admin_users.html", **_ctx("admin.users", items=items, total=total, page=page,
                                                      pages=max(1, -(-total // PAGE_SIZE)), q=q, status=status))


def _target(user_id: int) -> User:
    user = db.session.get(User, user_id)
    if user is None:
        abort(404)
    return user


@bp.get("/app/admin/users/<int:user_id>")
@protected("admin")
def user_detail(user_id: int):
    user = _target(user_id)
    history = AuditLog.query.filter_by(target_id=user.id).order_by(AuditLog.id.desc()).limit(20).all()
    return render_template("admin_user.html", **_ctx("admin.users", u=user, history=history, names=_names(history),
                                                     is_self=user.id == current_user.id))


@bp.post("/app/admin/users/<int:user_id>/<action>")
@protected("admin")
def user_action(user_id: int, action: str):
    if action not in ACTIONS:
        abort(404)
    user = _target(user_id)
    if action == "deactivate":
        if user.id == current_user.id:
            flash("You cannot deactivate your own account.", "error")
            return redirect(url_for("admin.user_detail", user_id=user.id))
        if user.is_admin and user.is_active and User.query.filter_by(role="admin", is_active_flag=True).count() <= 1:
            flash("You cannot deactivate the last active administrator.", "error")
            return redirect(url_for("admin.user_detail", user_id=user.id))
        user.is_active_flag = False
        user.session_version += 1                              # also ends any session that is open right now
        msg = "Account deactivated and signed out everywhere."
    elif action == "reactivate":
        user.is_active_flag = True
        msg = "Account reactivated. The user can sign in again."
    else:
        user.session_version += 1
        msg = "All sessions of this account were ended."
    _audit(action, user)
    db.session.commit()
    flash(msg, "success")
    return redirect(url_for("admin.user_detail", user_id=user.id))


@bp.get("/app/admin/audit")
@protected("admin")
def audit():
    page = _page()
    query = AuditLog.query
    total = query.count()
    items = query.order_by(AuditLog.id.desc()).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE).all()
    return render_template("admin_audit.html", **_ctx("admin.audit", items=items, total=total, page=page,
                                                      pages=max(1, -(-total // PAGE_SIZE)), names=_names(items)))
