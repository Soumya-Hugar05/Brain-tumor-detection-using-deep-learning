"""Application configuration. Everything secret or deployment-specific comes from environment variables.

Environment variables (all optional in development):

    NEUROINSIGHT_ENV             development (default) | production | testing
    SECRET_KEY                   REQUIRED when NEUROINSIGHT_ENV=production. In development a random key is generated once and
                                 kept in instance/secret_key (git-ignored).
    DATABASE_URL                 SQLAlchemy URL, e.g. postgresql+psycopg://user:pass@host/db. Default: SQLite in instance/.
    SESSION_COOKIE_SECURE        1 to send cookies over HTTPS only (default 1 in production, 0 otherwise)
    SESSION_LIFETIME_HOURS       idle session lifetime (default 8)
    REMEMBER_DAYS                "remember me" lifetime (default 14)
    NEUROINSIGHT_MAIL_BACKEND    outbox (default: writes emails to instance/outbox) | smtp
    MAIL_SERVER, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD, MAIL_USE_TLS, MAIL_FROM     used by the smtp backend
    APP_BASE_URL                 public base URL used in emailed links (default: the request's URL root)
    RATELIMIT_STORAGE_URI        Flask-Limiter storage (default memory://; use redis:// when running several workers)
    TRUSTED_PROXY_COUNT          number of reverse proxies in front of the app (default 0); enables X-Forwarded-For handling
    NEUROINSIGHT_REPORTS_DB, NEUROINSIGHT_BUNDLE_DIR, NEUROINSIGHT_DEBUG      see the README
"""
from __future__ import annotations

import os
import secrets
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTANCE_DIR = ROOT / "instance"

DISCLAIMER_TEXT = ("NeuroInsight is a research prototype. It is not a medical device or diagnostic tool and must not "
                   "be used to diagnose, treat or make decisions about any patient.")


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _dev_secret_key() -> str:
    """A random development key, created once and kept (owner-only) in instance/secret_key."""
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    path = INSTANCE_DIR / "secret_key"
    if path.is_file():
        key = path.read_text(encoding="utf-8").strip()
        if len(key) >= 32:
            return key
    key = secrets.token_hex(32)
    path.write_text(key, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


def build_config() -> dict:
    env = os.environ.get("NEUROINSIGHT_ENV", "development").strip().lower()
    production = env == "production"
    testing = env == "testing"
    secret = os.environ.get("SECRET_KEY", "").strip()
    if production and len(secret) < 32:
        raise RuntimeError("SECRET_KEY must be set (at least 32 characters) when NEUROINSIGHT_ENV=production.")
    if not secret:
        secret = _dev_secret_key() if not testing else secrets.token_hex(32)
    db_url = os.environ.get("DATABASE_URL", "").strip() or f"sqlite:///{(INSTANCE_DIR / 'neuroinsight.sqlite3').as_posix()}"
    cookie_secure = _bool("SESSION_COOKIE_SECURE", production)
    lifetime = timedelta(hours=max(1, _int("SESSION_LIFETIME_HOURS", 8)))
    remember = timedelta(days=max(1, _int("REMEMBER_DAYS", 14)))
    return dict(
        ENV_NAME=env, SECRET_KEY=secret, DISCLAIMER_TEXT=DISCLAIMER_TEXT,
        SQLALCHEMY_DATABASE_URI=db_url, SQLALCHEMY_TRACK_MODIFICATIONS=False,
        # sessions and cookies
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=cookie_secure,
        SESSION_COOKIE_NAME="neuroinsight_session",
        PERMANENT_SESSION_LIFETIME=lifetime, SESSION_REFRESH_EACH_REQUEST=True,    # idle timeout: renewed on every request
        REMEMBER_COOKIE_DURATION=remember, REMEMBER_COOKIE_HTTPONLY=True, REMEMBER_COOKIE_SECURE=cookie_secure,
        REMEMBER_COOKIE_SAMESITE="Lax", REMEMBER_COOKIE_NAME="neuroinsight_remember",
        # CSRF (Flask-WTF)
        WTF_CSRF_TIME_LIMIT=None, WTF_CSRF_SSL_STRICT=production,
        # rate limiting
        RATELIMIT_STORAGE_URI=os.environ.get("RATELIMIT_STORAGE_URI", "memory://"), RATELIMIT_ENABLED=True,
        RATELIMIT_HEADERS_ENABLED=False,
        # authentication policy
        LOGIN_MAX_FAILURES=5, LOGIN_WINDOW_MINUTES=15, LOGIN_IP_MAX_FAILURES=25,
        VERIFY_TOKEN_HOURS=24, RESET_TOKEN_MINUTES=60, PASSWORD_MIN_LENGTH=10, PASSWORD_MAX_LENGTH=128,
        # mail
        MAIL_BACKEND=os.environ.get("NEUROINSIGHT_MAIL_BACKEND", "outbox").strip().lower(),
        MAIL_SERVER=os.environ.get("MAIL_SERVER", ""), MAIL_PORT=_int("MAIL_PORT", 587),
        MAIL_USERNAME=os.environ.get("MAIL_USERNAME", ""), MAIL_PASSWORD=os.environ.get("MAIL_PASSWORD", ""),
        MAIL_USE_TLS=_bool("MAIL_USE_TLS", True), MAIL_FROM=os.environ.get("MAIL_FROM", "NeuroInsight <no-reply@localhost>"),
        OUTBOX_DIR=os.environ.get("NEUROINSIGHT_OUTBOX_DIR", "").strip() or str(INSTANCE_DIR / "outbox"),
        APP_BASE_URL=os.environ.get("APP_BASE_URL", "").strip().rstrip("/"),
        TRUSTED_PROXY_COUNT=_int("TRUSTED_PROXY_COUNT", 0),
        AUTO_UPGRADE_DB=_bool("AUTO_UPGRADE_DB", not production and not testing),
    )
