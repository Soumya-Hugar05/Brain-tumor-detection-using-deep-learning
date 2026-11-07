"""Authentication blueprint: sign up, sign in/out, email verification and password reset."""
from __future__ import annotations

import logging
import threading
from datetime import timedelta

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user

from . import mailer, security
from .extensions import db, limiter
from .models import OCCUPATIONS, User, utcnow
from .notify import notify

bp = Blueprint("auth", __name__)
log = logging.getLogger("neuroinsight.auth")

GENERIC_LOGIN_ERROR = "Incorrect email or password."


def _page(template: str, status: int = 200, **ctx):
    return render_template(template, disclaimer=current_app.config["DISCLAIMER_TEXT"], **ctx), status


def _send(to: str, subject: str, body: str, template: str) -> None:
    """Deliver an email. SMTP delivery runs in a background thread so the response time does not reveal whether an address exists."""
    app = current_app._get_current_object()                    # noqa: SLF001
    if app.config["MAIL_BACKEND"] == "smtp":
        def run():
            with app.app_context():
                mailer.send_email(to, subject, body, template=template)
        threading.Thread(target=run, daemon=True).start()
    else:
        mailer.send_email(to, subject, body, template=template)


def _send_verification(user: User) -> None:
    raw = security.issue_token(user, "verify_email", timedelta(hours=current_app.config["VERIFY_TOKEN_HOURS"]))
    url = mailer.link(url_for("auth.verify_confirm", token=raw))
    _send(user.email, "Confirm your NeuroInsight email address",
          f"Hello {user.full_name},\n\nPlease confirm your email address to activate your NeuroInsight account:\n\n{url}\n\n"
          f"The link works once and expires in {current_app.config['VERIFY_TOKEN_HOURS']} hours. If you did not create an account, "
          "ignore this message.\n\nNeuroInsight is a research prototype and not a medical device.", "verify-email")


# --------------------------------------------------------------------------------------------------------- sign in / out
@bp.route("/login", methods=["GET", "POST"])
@limiter.limit("60 per 10 minutes", methods=["POST"])
def login():
    next_url = security.safe_next_url(request.values.get("next"))
    if current_user.is_authenticated:
        return redirect(next_url or url_for("dashboard.home"))
    if request.method == "GET":
        return _page("login.html", next_url=next_url, email="")

    raw_email = (request.form.get("email") or "").strip()
    password = request.form.get("password") or ""
    remember = request.form.get("remember") == "on"
    email = security.normalize_email(raw_email) or raw_email.lower()[:254]
    if security.is_locked_out(email):
        log.warning("login blocked by throttling ip=%s", security.client_ip())
        return _page("login.html", 429, next_url=next_url, email=raw_email,
                     error="Too many failed attempts. Please wait a few minutes before trying again.")
    user = User.query.filter_by(email=email).first()
    password_ok = security.verify_password(user.password_hash if user else None, password)
    if not (user and password_ok):
        security.record_attempt(email, False)
        log.info("login failed ip=%s", security.client_ip())
        return _page("login.html", 401, next_url=next_url, email=raw_email, error=GENERIC_LOGIN_ERROR)
    if not user.is_active:
        log.info("login refused for deactivated account user_id=%s", user.id)
        return _page("login.html", 403, next_url=next_url, email=raw_email,
                     error="This account has been deactivated. Please contact an administrator.")
    security.clear_failures(email)
    if security.needs_rehash(user.password_hash):
        user.password_hash = security.hash_password(password)
    user.last_login_at = utcnow()
    db.session.commit()
    session.clear()                                           # new session (and CSRF token) at the privilege change
    login_user(user, remember=remember)
    session.permanent = True
    log.info("login ok user_id=%s", user.id)
    if not user.is_verified:
        return redirect(url_for("auth.verify_notice"))
    return redirect(next_url or url_for("dashboard.home"))


@bp.route("/logout", methods=["GET", "POST"])
def logout():
    if request.method == "POST":
        if current_user.is_authenticated:
            log.info("logout user_id=%s", current_user.id)
        logout_user()
        session.clear()
        flash("You have been signed out.", "success")
    return redirect(url_for("auth.login") if request.method == "POST" else url_for("landing"))


# --------------------------------------------------------------------------------------------------------- sign up
def _validate_signup(form) -> tuple[dict, dict]:
    values = dict(full_name=(form.get("full_name") or "").strip(), email=(form.get("email") or "").strip(),
                  institution=(form.get("institution") or "").strip(), occupation=(form.get("occupation") or "").strip())
    errors: dict[str, list[str]] = {}
    name = " ".join(values["full_name"].split())
    values["full_name"] = name
    if not 2 <= len(name) <= 100:
        errors["full_name"] = ["Enter your full name (2 to 100 characters)."]
    email = security.normalize_email(values["email"])
    if email is None or len(email) > 254:
        errors["email"] = ["Enter a valid email address."]
    if len(values["institution"]) > 150:
        errors["institution"] = ["Use at most 150 characters."]
    if values["occupation"] and values["occupation"] not in OCCUPATIONS:
        errors["occupation"] = ["Choose one of the listed options."]
    password, confirm = form.get("password") or "", form.get("confirm_password") or ""
    problems = security.password_problems(password, email=email or "", full_name=name)
    if problems:
        errors["password"] = problems
    elif password != confirm:
        errors["confirm_password"] = ["The passwords do not match."]
    if form.get("accept_terms") != "on":
        errors["accept_terms"] = ["You must accept the terms and the privacy notice."]
    values["normalized_email"] = email
    values["password"] = password
    return values, errors


@bp.route("/signup", methods=["GET", "POST"])
@limiter.limit("20 per hour", methods=["POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.home"))
    if request.method == "GET":
        return _page("signup.html", values={}, errors={}, occupations=OCCUPATIONS)
    values, errors = _validate_signup(request.form)           # the "role" field of a forged request is simply ignored
    if errors:
        return _page("signup.html", 400, values=values, errors=errors, occupations=OCCUPATIONS)
    email = values["normalized_email"]
    existing = User.query.filter_by(email=email).first()
    if existing is None:
        user = User(email=email, password_hash=security.hash_password(values["password"]), full_name=values["full_name"],
                    institution=values["institution"] or None, occupation=values["occupation"] or None, role="user",
                    terms_accepted_at=utcnow())
        db.session.add(user)
        db.session.commit()
        _send_verification(user)
        log.info("account created user_id=%s", user.id)
    elif not existing.is_verified:
        _send_verification(existing)                          # same response: never reveal whether the address is registered
    else:
        _send(existing.email, "Someone tried to create a NeuroInsight account with your address",
              "Hello,\n\nSomeone used your email address to sign up for NeuroInsight, but an account already exists. "
              f"You can sign in, or reset your password if you forgot it:\n\n{mailer.link(url_for('auth.login'))}\n\n"
              "If this was not you, no action is needed.", "signup-existing")
    return redirect(url_for("auth.check_email"))


@bp.get("/check-email")
def check_email():
    return _page("auth_message.html", title="Check your email", icon_name="mail",
                 message="If the address can be used, we have sent a link to confirm it. Open the link to activate your account. "
                         "It works once and expires in 24 hours.",
                 actions=[("Back to sign in", url_for("auth.login"))])


# --------------------------------------------------------------------------------------------------------- email verification
@bp.get("/verify-email")
@login_required
def verify_notice():
    if current_user.is_verified:
        return redirect(url_for("dashboard.home"))
    return _page("verify_notice.html", email=current_user.email)


@bp.post("/resend-verification")
@login_required
@limiter.limit("5 per hour")
def resend_verification():
    if not current_user.is_verified:
        _send_verification(current_user)
        flash("We have sent a new confirmation link.", "success")
    return redirect(url_for("auth.verify_notice"))


@bp.route("/verify-email/<token>", methods=["GET", "POST"])
def verify_confirm(token: str):
    """GET shows a confirmation button (so mail scanners that prefetch links cannot consume the token); POST verifies."""
    tok = security.find_token(token, "verify_email")
    if tok is None:
        return _page("auth_message.html", 400, title="This link is not valid", icon_name="circle-x",
                     message="The confirmation link has expired or was already used. Sign in and request a new one.",
                     actions=[("Sign in", url_for("auth.login"))])
    if request.method == "GET":
        return _page("verify_confirm.html", token=token)
    user = tok.user
    security.consume_token(tok)
    user.email_verified_at = user.email_verified_at or utcnow()
    notify(user, "account", "Email address confirmed", "Your account is active.", url_for("dashboard.home"))
    db.session.commit()
    log.info("email verified user_id=%s", user.id)
    flash("Your email address is confirmed. You can sign in.", "success")
    return redirect(url_for("dashboard.home") if current_user.is_authenticated and current_user.id == user.id else url_for("auth.login"))


# --------------------------------------------------------------------------------------------------------- password reset
@bp.route("/forgot-password", methods=["GET", "POST"])
@limiter.limit("10 per hour", methods=["POST"])
def forgot_password():
    if request.method == "GET":
        return _page("forgot_password.html", email="")
    raw = (request.form.get("email") or "").strip()
    email = security.normalize_email(raw)
    if email is None:
        return _page("forgot_password.html", 400, email=raw, error="Enter a valid email address.")
    user = User.query.filter_by(email=email).first()
    if user and user.is_active:
        token = security.issue_token(user, "reset_password", timedelta(minutes=current_app.config["RESET_TOKEN_MINUTES"]))
        url = mailer.link(url_for("auth.reset_with_token", token=token))
        _send(user.email, "Reset your NeuroInsight password",
              f"Hello {user.full_name},\n\nUse this link to choose a new password:\n\n{url}\n\nThe link works once and expires in "
              f"{current_app.config['RESET_TOKEN_MINUTES']} minutes. If you did not ask for this, ignore the message; your password "
              "has not changed.", "reset-password")
        log.info("password reset requested user_id=%s", user.id)
    return _page("auth_message.html", title="Check your email", icon_name="mail",
                 message="If an account exists for that address, we have sent a link to reset the password. It expires in 60 minutes.",
                 actions=[("Back to sign in", url_for("auth.login"))])


@bp.get("/reset-password/<token>")
def reset_with_token(token: str):
    """Validate the emailed link, keep the token in the session and move to a token-free URL (keeps it out of later Referer headers)."""
    if security.find_token(token, "reset_password") is None:
        return _page("auth_message.html", 400, title="This link is not valid", icon_name="circle-x",
                     message="The reset link has expired or was already used. Request a new one.",
                     actions=[("Request a new link", url_for("auth.forgot_password"))])
    session["reset_token"] = token
    return redirect(url_for("auth.reset_password"))


@bp.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    tok = security.find_token(session.get("reset_token"), "reset_password")
    if tok is None:
        session.pop("reset_token", None)
        return redirect(url_for("auth.forgot_password"))
    user = tok.user
    if request.method == "GET":
        return _page("reset_password.html", errors={})
    password, confirm = request.form.get("password") or "", request.form.get("confirm_password") or ""
    errors = {}
    problems = security.password_problems(password, email=user.email, full_name=user.full_name)
    if problems:
        errors["password"] = problems
    elif password != confirm:
        errors["confirm_password"] = ["The passwords do not match."]
    if not errors and security.verify_password(user.password_hash, password):
        errors["password"] = ["Choose a password you have not used before on this account."]
    if errors:
        return _page("reset_password.html", 400, errors=errors)
    user.password_hash = security.hash_password(password)
    user.session_version += 1                                 # signs the account out everywhere
    security.consume_token(tok)
    notify(user, "security", "Password changed", "Your password was reset with an emailed link. All devices were signed out.", None)
    db.session.commit()
    session.pop("reset_token", None)
    _send(user.email, "Your NeuroInsight password was changed",
          f"Hello {user.full_name},\n\nThe password of your NeuroInsight account was just changed. If this was not you, reset it again "
          "and contact an administrator.", "password-changed")
    log.info("password reset completed user_id=%s", user.id)
    flash("Your password was changed. Please sign in.", "success")
    return redirect(url_for("auth.login"))
