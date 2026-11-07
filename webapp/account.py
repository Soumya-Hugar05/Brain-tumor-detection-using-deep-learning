"""The signed-in user's own account: profile, settings (appearance, notification choices, password, sessions) and notifications.

Everything acts on current_user only. Passwords are never logged or shown. The email address cannot be changed here (a change
would need a fresh verification flow); there is no profile picture because that would mean storing uploaded files.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_user

from . import security
from .access import protected
from .auth import _send
from .dashboard import NAV
from .extensions import db, limiter
from .models import OCCUPATIONS, Notification, utcnow
from .notify import notify, unread_count

bp = Blueprint("account", __name__)
log = logging.getLogger("neuroinsight.account")
PAGE_SIZE = 20


@bp.app_context_processor
def _inject_unread():
    if current_user.is_authenticated and current_user.is_verified:
        return dict(unread_notifications=unread_count(current_user.id))
    return dict(unread_notifications=0)


def _ctx(active: str, **extra):
    return dict(nav=NAV, active=active, disclaimer=current_app.config["DISCLAIMER_TEXT"], occupations=OCCUPATIONS, **extra)


# --------------------------------------------------------------------------------------------------------- profile
def _profile_page(values, errors, status=200):
    return render_template("profile.html", **_ctx("account.profile", values=values, errors=errors)), status


@bp.get("/app/profile")
@protected()
def profile():
    return _profile_page(dict(full_name=current_user.full_name, institution=current_user.institution or "",
                              occupation=current_user.occupation or ""), {})


@bp.post("/app/profile")
@protected()
def update_profile():
    name = " ".join((request.form.get("full_name") or "").split())
    institution = (request.form.get("institution") or "").strip()
    occupation = (request.form.get("occupation") or "").strip()
    errors: dict[str, list[str]] = {}
    if not 2 <= len(name) <= 100:
        errors["full_name"] = ["Enter your full name (2 to 100 characters)."]
    if len(institution) > 150:
        errors["institution"] = ["Use at most 150 characters."]
    if occupation and occupation not in OCCUPATIONS:
        errors["occupation"] = ["Choose one of the listed options."]
    if errors:
        return _profile_page(dict(full_name=name, institution=institution, occupation=occupation), errors, 400)
    current_user.full_name, current_user.institution, current_user.occupation = name, institution or None, occupation or None
    db.session.commit()                                       # email and role are never taken from this form
    flash("Profile updated.", "success")
    return redirect(url_for("account.profile"))


# --------------------------------------------------------------------------------------------------------- settings
def _settings_page(errors=None, status=200):
    return render_template("settings.html", **_ctx("account.settings", errors=errors or {})), status


@bp.get("/app/settings")
@protected()
def settings():
    return _settings_page()


@bp.post("/app/settings/preferences")
@protected()
def save_preferences():
    theme = request.form.get("theme", "")
    if theme not in ("light", "dark"):
        abort(400)
    current_user.theme = theme
    current_user.notify_analysis = request.form.get("notify_analysis") == "on"
    current_user.notify_reports = request.form.get("notify_reports") == "on"
    db.session.commit()
    flash("Preferences saved.", "success")
    return redirect(url_for("account.settings"))


@bp.post("/api/account/theme")
@protected()
def api_theme():
    body = request.get_json(silent=True)
    theme = body.get("theme") if isinstance(body, dict) else None
    if theme not in ("light", "dark"):
        resp = jsonify(ok=False, error="Unknown theme.")
        resp.status_code = 400
        return resp
    current_user.theme = theme
    db.session.commit()
    return jsonify(ok=True)


def _refresh_session(remember: bool = False) -> None:
    """After session_version changed: keep only the current browser signed in."""
    user = current_user._get_current_object()                  # noqa: SLF001  (login_user needs the real object, not the proxy)
    user.session_version += 1
    db.session.flush()
    login_user(user, remember=remember)


@bp.post("/app/settings/password")
@protected()
@limiter.limit("10 per hour")
def change_password():
    current, new, confirm = (request.form.get(k) or "" for k in ("current_password", "password", "confirm_password"))
    errors: dict[str, list[str]] = {}
    if not security.verify_password(current_user.password_hash, current):
        errors["current_password"] = ["Your current password is incorrect."]
    else:
        problems = security.password_problems(new, email=current_user.email, full_name=current_user.full_name)
        if problems:
            errors["password"] = problems
        elif new != confirm:
            errors["confirm_password"] = ["The passwords do not match."]
        elif security.verify_password(current_user.password_hash, new):
            errors["password"] = ["Choose a password you have not used before on this account."]
    if errors:
        log.info("password change refused user_id=%s", current_user.id)
        return _settings_page(errors, 400)
    current_user.password_hash = security.hash_password(new)
    _refresh_session()                                         # every other browser is signed out
    notify(current_user, "security", "Password changed", "Other devices were signed out.", url_for("account.settings"))
    db.session.commit()
    _send(current_user.email, "Your NeuroInsight password was changed",
          f"Hello {current_user.full_name},\n\nThe password of your NeuroInsight account was just changed. If this was not you, reset it "
          "again and contact an administrator.", "password-changed")
    log.info("password changed user_id=%s", current_user.id)
    flash("Your password was changed. Other devices were signed out.", "success")
    return redirect(url_for("account.settings"))


@bp.post("/app/settings/sessions")
@protected()
def sign_out_everywhere():
    _refresh_session()
    notify(current_user, "security", "Signed out of other devices", "All other sessions were ended.", url_for("account.settings"))
    db.session.commit()
    log.info("sessions revoked user_id=%s", current_user.id)
    flash("All other sessions were signed out. You stay signed in here.", "success")
    return redirect(url_for("account.settings"))


# --------------------------------------------------------------------------------------------------------- notifications
@bp.get("/app/notifications")
@protected()
def notifications():
    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1
    query = Notification.query.filter_by(user_id=current_user.id)
    total = query.count()
    items = query.order_by(Notification.created_at.desc(), Notification.id.desc()).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE).all()
    return render_template("notifications.html", **_ctx("account.notifications", items=items, total=total, page=page,
                                                        pages=max(1, -(-total // PAGE_SIZE))))


@bp.post("/app/notifications/read")
@protected()
def mark_read():
    """Mark one notification (id=...) or all of them as read."""
    q = Notification.query.filter_by(user_id=current_user.id, read_at=None)
    raw = request.form.get("id", "")
    if raw:
        if not raw.isdigit():
            abort(400)
        q = q.filter(Notification.id == int(raw))
    q.update({Notification.read_at: utcnow()})
    db.session.commit()
    return redirect(url_for("account.notifications"))


@bp.post("/app/notifications/clear")
@protected()
def clear():
    db.session.execute(sa.delete(Notification).where(Notification.user_id == current_user.id))
    db.session.commit()
    flash("Notifications cleared.", "success")
    return redirect(url_for("account.notifications"))
