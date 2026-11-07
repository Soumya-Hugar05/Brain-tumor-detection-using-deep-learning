"""Shared test helpers. Import this module BEFORE `import app`: it points the app at throw-away databases and outboxes.

    import helpers                      # sets the environment (temporary database, outbox, reports file)
    import app as appmod
    client = helpers.logged_in_client() # a verified user, signed in, CSRF token handled automatically
"""
from __future__ import annotations

import atexit
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="ni-test-"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
os.environ.update(
    NEUROINSIGHT_ENV="testing", DATABASE_URL=f"sqlite:///{(TMP / 'test.sqlite3').as_posix()}",
    NEUROINSIGHT_OUTBOX_DIR=str(TMP / "outbox"), NEUROINSIGHT_REPORTS_DB=str(TMP / "reports.sqlite3"),
    TF_CPP_MIN_LOG_LEVEL="2")
os.environ.pop("SECRET_KEY", None)
os.environ.pop("NEUROINSIGHT_BUNDLE_DIR", None)

from flask.testing import FlaskClient  # noqa: E402

PASSWORD = "Correct-Horse-9!"
_META = re.compile(r'<meta name="csrf-token" content="([^"]+)"')


class CsrfClient(FlaskClient):
    """A test client that adds the CSRF token to every state-changing request (like the front-end does)."""

    def csrf_token(self) -> str:
        html = super().open("/privacy").get_data(as_text=True)
        m = _META.search(html)
        assert m, "no csrf-token meta tag on /privacy"
        return m.group(1)

    def open(self, *args, **kwargs):
        method = (kwargs.get("method") or "GET").upper()
        if method not in ("GET", "HEAD", "OPTIONS") and kwargs.pop("csrf", True):
            headers = dict(kwargs.get("headers") or {})
            headers.setdefault("X-CSRFToken", self.csrf_token())
            kwargs["headers"] = headers
        kwargs.pop("csrf", None)
        return super().open(*args, **kwargs)


def _app():
    import app as appmod
    appmod.app.test_client_class = CsrfClient
    return appmod.app


def reset_db() -> None:
    from webapp.extensions import db, limiter
    app = _app()
    limiter.enabled = False
    with app.app_context():
        db.session.remove()
        db.drop_all()
        db.create_all()
    shutil.rmtree(TMP / "outbox", ignore_errors=True)


def create_user(email="tester@example.com", password=PASSWORD, *, verified=True, role="user", name="Test Person", active=True):
    from webapp import security
    from webapp.extensions import db
    from webapp.models import User, utcnow
    app = _app()
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if user is None:
            user = User(email=email, password_hash=security.hash_password(password), full_name=name, role=role,
                        email_verified_at=utcnow() if verified else None, terms_accepted_at=utcnow(), is_active_flag=active)
            db.session.add(user)
            db.session.commit()
        return user.id


def login(client, email="tester@example.com", password=PASSWORD, **extra):
    return client.post("/login", data=dict(email=email, password=password, **extra))


def logged_in_client(email="tester@example.com", **kwargs):
    """A fresh client for a verified user (created on first use). The database must already be initialised."""
    create_user(email, **{k: v for k, v in kwargs.items() if k in ("role", "verified", "name", "active", "password")})
    client = _app().test_client()
    r = login(client, email, kwargs.get("password", PASSWORD))
    assert r.status_code == 302, (r.status_code, r.get_data(as_text=True)[:300])
    return client


def outbox() -> list[str]:
    box = TMP / "outbox"
    return [p.read_text(encoding="utf-8") for p in sorted(box.glob("*.txt"))] if box.is_dir() else []


def last_link(path_prefix: str) -> str:
    """The most recent emailed link (path only) that starts with the given prefix, e.g. '/verify-email/'."""
    for message in reversed(outbox()):
        m = re.search(r"https?://[^/\s]+(" + re.escape(path_prefix) + r"[^\s]*)", message)
        if m:
            return m.group(1)
    raise AssertionError(f"no emailed link starting with {path_prefix}")


reset_db()
