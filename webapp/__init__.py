"""NeuroInsight web application package: configuration, extensions, authentication and security plumbing.

create_app() builds the Flask app. The existing analysis and report routes still live in app.py (they are attached to the
object returned here); new areas (cases, history, account, admin) are added as blueprints in later phases.
"""
from __future__ import annotations

import logging
import re
import secrets

from flask import Flask, g, jsonify, render_template, request
from flask_login import current_user
from flask_wtf.csrf import CSRFError
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import INSTANCE_DIR, ROOT, build_config
from .extensions import csrf, db, limiter, login_manager, migrate

_TOKEN_PATH = re.compile(r"(/(?:reset-password|verify-email)/)[A-Za-z0-9_\-]{16,}")


class _RedactTokens(logging.Filter):
    """Keeps reset and verification tokens out of the access log (they appear in the URL of the emailed link)."""
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        if _TOKEN_PATH.search(msg):
            record.msg, record.args = _TOKEN_PATH.sub(r"\1<redacted>", msg), ()
        return True


def _wants_json() -> bool:
    return request.path.startswith("/api/") or request.is_json


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask("neuroinsight", root_path=str(ROOT))
    app.config.update(build_config())
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    if config_overrides:
        app.config.update(config_overrides)
    if app.config["TRUSTED_PROXY_COUNT"] > 0:
        n = app.config["TRUSTED_PROXY_COUNT"]
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=n, x_proto=n, x_host=n)
    logging.getLogger("werkzeug").addFilter(_RedactTokens())

    db.init_app(app)
    migrate.init_app(app, db, directory=str(ROOT / "migrations"))
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message = None
    login_manager.session_protection = "basic"

    from .models import User

    @login_manager.user_loader
    def load_user(user_id: str):
        try:
            uid, version = (int(part) for part in user_id.split("."))
        except (ValueError, AttributeError):
            return None
        user = db.session.get(User, uid)
        if user is None or not user.is_active or user.session_version != version:
            return None                                       # deactivated, or signed out everywhere: the session is void
        return user

    from . import account, admin, auth, cases, dashboard, public
    app.register_blueprint(auth.bp)
    app.register_blueprint(dashboard.bp)
    app.register_blueprint(cases.bp)
    app.register_blueprint(account.bp)
    app.register_blueprint(admin.bp)
    app.register_blueprint(public.bp)

    from .cli import register_cli
    register_cli(app)

    # ---------------------------------------------------------------- security headers and CSP (script nonce per request)
    @app.before_request
    def _nonce():
        g.csp_nonce = secrets.token_urlsafe(16)

    @app.context_processor
    def _inject():
        return dict(csp_nonce=g.get("csp_nonce", ""))

    @app.after_request
    def _headers(resp):
        h = resp.headers
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "no-referrer" if "/reset-password" in request.path or "/verify-email" in request.path
                     else "strict-origin-when-cross-origin")
        h.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
        h.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        if resp.mimetype == "text/html":
            nonce = g.get("csp_nonce", "")
            h.setdefault("Content-Security-Policy",
                         f"default-src 'self'; script-src 'self' 'nonce-{nonce}'; style-src 'self' 'unsafe-inline'; "
                         "img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; object-src 'none'; "
                         "base-uri 'self'; form-action 'self'; frame-ancestors 'none'")
        if app.config["SESSION_COOKIE_SECURE"] and request.is_secure:
            h.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        if not request.path.startswith("/static/") and (current_user.is_authenticated or request.path in ("/login", "/signup")):
            h["Cache-Control"] = "no-store"                    # private pages are never cached by the browser or proxies
        return resp

    # ---------------------------------------------------------------- errors
    def _error_page(status: int, title: str, message: str):
        return render_template("error.html", disclaimer=app.config["DISCLAIMER_TEXT"], status=status, title=title, message=message), status

    @app.errorhandler(CSRFError)
    def _csrf_error(_e):
        msg = "Your session expired or the request was not allowed. Reload the page and try again."
        if _wants_json():
            resp = jsonify(ok=False, error=msg)
            resp.status_code = 400
            return resp
        return _error_page(400, "Request not allowed", msg)

    @app.errorhandler(429)
    def _rate_limited(_e):
        msg = "Too many requests. Please wait a while and try again."
        if _wants_json():
            resp = jsonify(ok=False, error=msg)
            resp.status_code = 429
            return resp
        return _error_page(429, "Too many requests", msg)

    if app.config["AUTO_UPGRADE_DB"] and (ROOT / "migrations").is_dir():
        from flask_migrate import upgrade
        with app.app_context():
            upgrade(directory=str(ROOT / "migrations"))      # development convenience: apply pending migrations at start

    return app
