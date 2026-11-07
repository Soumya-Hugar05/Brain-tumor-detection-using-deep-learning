"""Tests for accounts, sessions and access control (Phases 2 and 3). Run: python tests/test_auth.py

Everything uses throw-away databases and outboxes (see helpers.py). Only demo samples from the validation split are analysed;
the test split is never touched.
"""
from __future__ import annotations

import io
import json
import logging
import re
import sys
from datetime import timedelta
from pathlib import Path

import helpers
import app as appmod
from webapp import security
from webapp.extensions import db, limiter
from webapp.models import AuthToken, LoginAttempt, User, utcnow

ROOT = helpers.ROOT
PW = helpers.PASSWORD
SAMPLE = (ROOT / "static" / "samples" / "sample_glioma_1.jpg").read_bytes()
APP = appmod.app

from webapp.access import protected  # noqa: E402


@APP.get("/__admin_only")                 # registered before the first request: Flask forbids adding routes later
@protected("admin")
def _admin_only_probe():
    return "ok"


def fresh():
    helpers.reset_db()
    return APP.test_client()


def signup(c, email="new@example.com", password=PW, **extra):
    data = dict(full_name="New Person", email=email, password=password, confirm_password=password, accept_terms="on", occupation="Researcher")
    data.update(extra)
    return c.post("/signup", data=data)


def user_by_email(email):
    with APP.app_context():
        u = User.query.filter_by(email=email).first()
        return None if u is None else dict(id=u.id, role=u.role, verified=u.is_verified, hash=u.password_hash, sv=u.session_version, active=u.is_active)


# ---- registration -------------------------------------------------------------------------------------------------
def test_signup_page_and_validation():
    c = fresh()
    html = c.get("/signup").get_data(as_text=True)
    for needle in ('name="full_name"', 'name="email"', 'name="password"', 'name="confirm_password"', 'name="institution"', 'name="occupation"',
                   'name="accept_terms"', "terms of use", "privacy notice", 'data-toggle-for="f-password"', 'data-strength="1"'):
        assert needle in html, needle
    bad = [dict(full_name="A"), dict(email="not-an-email"), dict(password="short1A!", confirm_password="short1A!"),
           dict(password="password123", confirm_password="password123"), dict(confirm_password="Different-Pass-9!"),
           dict(accept_terms=""), dict(occupation="Wizard"), dict(institution="x" * 151),
           dict(password="alllowercase", confirm_password="alllowercase"),
           dict(full_name="Marianne Smith", email="m@example.com", password="Marianne-Strong-1", confirm_password="Marianne-Strong-1")]
    for extra in bad:
        r = signup(c, **extra)
        assert r.status_code == 400, (extra, r.status_code)
    assert user_by_email("new@example.com") is None
    assert "Please correct" in signup(c, email="bad").get_data(as_text=True)
    print("ok signup form fields; every invalid input is rejected server-side")


def test_signup_creates_unverified_user_hashes_password_and_ignores_forged_role():
    c = fresh()
    r = signup(c, role="admin", is_active="0", email_verified_at="2020-01-01")
    assert r.status_code == 302 and r.headers["Location"].endswith("/check-email")
    u = user_by_email("new@example.com")
    assert u and u["role"] == "user" and not u["verified"] and u["active"], u
    assert u["hash"].startswith("$argon2id$") and PW not in u["hash"]
    raw_db = Path(helpers.TMP / "test.sqlite3").read_bytes()
    assert PW.encode() not in raw_db, "the plaintext password must never reach the database file"
    mail = helpers.outbox()
    assert len(mail) == 1 and "Confirm your NeuroInsight email" in mail[0] and "/verify-email/" in mail[0]
    print("ok signup: unverified 'user' role, argon2id hash, forged role ignored, one confirmation email")


def test_signup_does_not_reveal_whether_an_email_is_registered():
    c = fresh()
    helpers.create_user("taken@example.com")
    first = signup(c, email="fresh@example.com")
    second = signup(c, email="taken@example.com")
    assert first.status_code == second.status_code == 302 and first.headers["Location"] == second.headers["Location"]
    page = c.get("/check-email").get_data(as_text=True)
    assert "If the address can be used" in page
    assert any("already exists" in m for m in helpers.outbox()), "the owner of an existing address is told by email"
    with APP.app_context():
        assert User.query.filter_by(email="taken@example.com").count() == 1
    print("ok same response for new and existing addresses (no account enumeration)")


# ---- email verification -----------------------------------------------------------------------------------------------
def test_email_verification_flow_and_gate():
    c = fresh()
    signup(c, email="v@example.com")
    link = helpers.last_link("/verify-email/")
    r = helpers.login(c, "v@example.com")
    assert r.status_code == 302 and r.headers["Location"].endswith("/verify-email")          # signed in but unverified
    assert c.get("/app/analysis").status_code == 302 and c.get("/app/analysis").headers["Location"].endswith("/verify-email")
    api = c.post("/api/predict", data={"file": (io.BytesIO(SAMPLE), "x.jpg")}, content_type="multipart/form-data")
    assert api.status_code == 403 and "Verify your email" in api.get_json()["error"]
    assert "Confirm your email address" in c.get("/verify-email").get_data(as_text=True)
    page = c.get(link)                                                                       # GET only shows a button
    assert page.status_code == 200 and "Confirm my email address" in page.get_data(as_text=True)
    assert not user_by_email("v@example.com")["verified"], "a GET must not consume the token (mail scanners prefetch links)"
    done = c.post(link)
    assert done.status_code == 302 and user_by_email("v@example.com")["verified"]
    assert c.get("/app/analysis").status_code == 200
    again = c.post(link)
    assert again.status_code == 400, "a token works once"
    print("ok verification: gate on pages and API, GET does not consume the token, POST verifies once")


def test_verification_token_expiry_and_resend():
    c = fresh()
    signup(c, email="e@example.com")
    link = helpers.last_link("/verify-email/")
    with APP.app_context():
        tok = AuthToken.query.filter_by(purpose="verify_email").one()
        assert len(tok.token_hash) == 64 and link.split("/")[-1] not in tok.token_hash
        tok.expires_at = utcnow() - timedelta(minutes=1)
        db.session.commit()
    assert c.get(link).status_code == 400 and not user_by_email("e@example.com")["verified"]
    helpers.login(c, "e@example.com")
    r = c.post("/resend-verification")
    assert r.status_code == 302
    new_link = helpers.last_link("/verify-email/")
    assert new_link != link and c.post(new_link).status_code == 302 and user_by_email("e@example.com")["verified"]
    print("ok tokens are stored hashed, expire, and a resend issues a fresh single-use link")


# ---- sign in / out ------------------------------------------------------------------------------------------------------
def test_login_logout_and_generic_errors():
    c = fresh()
    helpers.create_user("u@example.com")
    html = c.get("/login").get_data(as_text=True)
    for needle in ('name="email"', 'name="password"', 'name="remember"', "Forgot password?", "Create an account", 'data-toggle-for="f-password"'):
        assert needle in html, needle
    wrong = c.post("/login", data=dict(email="u@example.com", password="Wrong-Password-1"))
    unknown = c.post("/login", data=dict(email="nobody@example.com", password="Wrong-Password-1"))
    assert wrong.status_code == unknown.status_code == 401
    for r in (wrong, unknown):
        assert "Incorrect email or password." in r.get_data(as_text=True)
    ok = helpers.login(c, "U@Example.Com")                                                 # email is case-insensitive
    assert ok.status_code == 302 and ok.headers["Location"].endswith("/app")
    assert c.get("/login").status_code == 302                                                # already signed in
    assert c.get("/app/analysis").status_code == 200
    out = c.post("/logout")
    assert out.status_code == 302 and c.get("/app/analysis").status_code == 302
    c2 = helpers.logged_in_client("u@example.com")
    assert c2.get("/logout").status_code == 302 and c2.get("/app/analysis").status_code == 200, "GET /logout must not sign anyone out"
    print("ok login/logout; wrong password and unknown email look identical; logout needs POST")


def test_open_redirect_is_blocked_and_next_works():
    c = fresh()
    helpers.create_user("n@example.com")
    for evil in ("https://evil.test/x", "//evil.test", "/\\evil.test", "javascript:alert(1)", "http:/evil"):
        c2 = APP.test_client()
        r = c2.post("/login", data=dict(email="n@example.com", password=PW, next=evil))
        assert r.status_code == 302 and "evil" not in r.headers["Location"] and r.headers["Location"].endswith("/app"), evil
    c3 = APP.test_client()
    gate = c3.get("/app/reports")
    assert gate.status_code == 302 and "next=/app/reports" in gate.headers["Location"]
    r = c3.post("/login", data=dict(email="n@example.com", password=PW, next="/app/reports"))
    assert r.headers["Location"].endswith("/app/reports")
    print("ok post-login redirects are same-site only")


def test_login_throttling_locks_out_after_repeated_failures():
    c = fresh()
    helpers.create_user("t@example.com")
    for _ in range(5):
        assert c.post("/login", data=dict(email="t@example.com", password="Wrong-Password-1")).status_code == 401
    blocked = c.post("/login", data=dict(email="t@example.com", password=PW))
    assert blocked.status_code == 429 and "Too many failed attempts" in blocked.get_data(as_text=True), "even the right password is refused while locked"
    with APP.app_context():
        assert LoginAttempt.query.count() == 5 and all(len(a.email_hash) == 64 and "@" not in a.email_hash for a in LoginAttempt.query)
        db.session.query(LoginAttempt).update({LoginAttempt.created_at: utcnow() - timedelta(minutes=30)})
        db.session.commit()
    ok = c.post("/login", data=dict(email="t@example.com", password=PW))
    assert ok.status_code == 302, "the lock expires after the window"
    helpers.create_user("other@example.com")
    c.post("/logout")
    assert c.post("/login", data=dict(email="other@example.com", password=PW)).status_code == 302
    print("ok 5 failures lock sign-in for that address; stored attempts hold only an email hash; lock expires")


def test_rate_limiter_on_signup_and_forgot_password():
    c = fresh()
    limiter.enabled = True
    limiter.reset()
    try:
        codes = [c.post("/forgot-password", data=dict(email=f"x{i}@example.com")).status_code for i in range(12)]
    finally:
        limiter.enabled = False
    assert codes[:10] == [200] * 10 and codes[10] == 429, codes
    print("ok forgot-password is rate limited (10/hour per address)")


def test_deactivated_account_cannot_sign_in_and_existing_sessions_end():
    c = fresh()
    uid = helpers.create_user("d@example.com")
    c = helpers.logged_in_client("d@example.com")
    assert c.get("/app/analysis").status_code == 200
    with APP.app_context():
        u = db.session.get(User, uid)
        u.is_active_flag = False
        db.session.commit()
    assert c.get("/app/analysis").status_code == 302, "an open session of a deactivated user is void immediately"
    r = APP.test_client().post("/login", data=dict(email="d@example.com", password=PW))
    assert r.status_code == 403 and "deactivated" in r.get_data(as_text=True)
    print("ok deactivation ends sessions and blocks sign-in")


def test_session_cookie_flags_and_lifetime():
    c = fresh()
    helpers.create_user("s@example.com")
    r = c.post("/login", data=dict(email="s@example.com", password=PW, remember="on"))
    cookies = r.headers.getlist("Set-Cookie")
    sess = next(x for x in cookies if x.startswith("neuroinsight_session="))
    assert "HttpOnly" in sess and "SameSite=Lax" in sess and "Expires" in sess, sess
    rem = next(x for x in cookies if x.startswith("neuroinsight_remember="))
    assert "HttpOnly" in rem and "SameSite=Lax" in rem
    assert APP.config["PERMANENT_SESSION_LIFETIME"] == timedelta(hours=8) and APP.config["SESSION_REFRESH_EACH_REQUEST"] is True
    assert APP.config["REMEMBER_COOKIE_DURATION"] == timedelta(days=14)
    from webapp.config import build_config
    import os
    os.environ["NEUROINSIGHT_ENV"] = "production"
    os.environ["SECRET_KEY"] = "k" * 40
    try:
        prod = build_config()
    finally:
        os.environ["NEUROINSIGHT_ENV"] = "testing"
        os.environ.pop("SECRET_KEY", None)
    assert prod["SESSION_COOKIE_SECURE"] is True and prod["REMEMBER_COOKIE_SECURE"] is True and prod["AUTO_UPGRADE_DB"] is False
    os.environ["NEUROINSIGHT_ENV"] = "production"
    try:
        try:
            build_config()
        except RuntimeError as exc:
            assert "SECRET_KEY" in str(exc)
        else:
            raise AssertionError("production must refuse to start without SECRET_KEY")
    finally:
        os.environ["NEUROINSIGHT_ENV"] = "testing"
    print("ok cookies HttpOnly + SameSite; Secure and a required SECRET_KEY in production; 8 h idle, 14 day remember")


# ---- password reset -------------------------------------------------------------------------------------------------------
def test_password_reset_flow():
    c = fresh()
    helpers.create_user("r@example.com")
    keep = helpers.logged_in_client("r@example.com")                      # a second, already signed-in browser
    same_a = c.post("/forgot-password", data=dict(email="r@example.com"))
    same_b = c.post("/forgot-password", data=dict(email="ghost@example.com"))
    strip = lambda r: re.sub(r'(csrf-token" content=")[^"]+|nonce="[^"]+"|name="csrf_token" value="[^"]+"', "X", r.get_data(as_text=True))
    assert same_a.status_code == same_b.status_code == 200 and strip(same_a) == strip(same_b)
    assert len(helpers.outbox()) == 1, "only a real account receives an email"
    link = helpers.last_link("/reset-password/")
    go = c.get(link)
    assert go.status_code == 302 and go.headers["Location"].endswith("/reset-password") and link.split("/")[-1] not in go.headers["Location"]
    form = c.get("/reset-password").get_data(as_text=True)
    assert 'data-strength="1"' in form and 'name="confirm_password"' in form
    for pw, conf in (("short", "short"), ("password1234", "password1234"), ("Valid-Pass-123!", "Mismatch-Pass-1!"), (PW, PW)):
        assert c.post("/reset-password", data=dict(password=pw, confirm_password=conf)).status_code == 400, pw
    new_pw = "Brand-New-Pass-77!"
    done = c.post("/reset-password", data=dict(password=new_pw, confirm_password=new_pw))
    assert done.status_code == 302 and done.headers["Location"].endswith("/login")
    assert any("password was changed" in m for m in helpers.outbox())
    assert keep.get("/app/analysis").status_code == 302, "every other session is signed out after a reset"
    assert c.post(link).status_code == 405 or c.get(link).status_code == 400, "the link is single use"
    assert c.get(link).status_code == 400
    assert APP.test_client().post("/login", data=dict(email="r@example.com", password=PW)).status_code == 401
    assert APP.test_client().post("/login", data=dict(email="r@example.com", password=new_pw)).status_code == 302
    print("ok reset: same reply for unknown addresses, token-free form URL, policy enforced, single use, other sessions ended")


def test_reset_token_expiry_and_invalidation_by_a_newer_request():
    c = fresh()
    helpers.create_user("x@example.com")
    c.post("/forgot-password", data=dict(email="x@example.com"))
    old = helpers.last_link("/reset-password/")
    c.post("/forgot-password", data=dict(email="x@example.com"))
    new = helpers.last_link("/reset-password/")
    assert old != new and c.get(old).status_code == 400, "requesting again invalidates the previous link"
    with APP.app_context():
        for t in AuthToken.query.filter_by(purpose="reset_password", used_at=None):
            t.expires_at = utcnow() - timedelta(seconds=1)
        db.session.commit()
    assert c.get(new).status_code == 400
    print("ok reset links expire and only the newest works")


def test_tokens_are_redacted_from_logs():
    from webapp import _RedactTokens
    rec = logging.LogRecord("werkzeug", logging.INFO, "", 0, '127.0.0.1 - - "GET /reset-password/%s HTTP/1.1" 302 -', ("A" * 43,), None)
    _RedactTokens().filter(rec)
    assert "AAAA" not in rec.getMessage() and "/reset-password/<redacted>" in rec.getMessage()
    rec = logging.LogRecord("werkzeug", logging.INFO, "", 0, 'GET /verify-email/' + "b" * 43 + " HTTP/1.1", (), None)
    _RedactTokens().filter(rec)
    assert "bbbb" not in rec.getMessage()
    print("ok reset and verification tokens never reach the access log")


# ---- CSRF, headers -----------------------------------------------------------------------------------------------------------
def test_csrf_is_enforced_for_forms_and_json_api():
    c = fresh()
    helpers.create_user("c@example.com")
    no_token = c.post("/login", data=dict(email="c@example.com", password=PW), csrf=False)
    assert no_token.status_code == 400 and "Request not allowed" in no_token.get_data(as_text=True)
    assert c.post("/signup", data={}, csrf=False).status_code == 400
    assert c.post("/forgot-password", data=dict(email="c@example.com"), csrf=False).status_code == 400
    signed = helpers.logged_in_client("c@example.com")
    r = signed.post("/api/predict", data={"file": (io.BytesIO(SAMPLE), "x.jpg")}, content_type="multipart/form-data", csrf=False)
    assert r.status_code == 400 and r.is_json and "expired or the request was not allowed" in r.get_json()["error"]
    assert signed.post("/api/reports", json={"prediction_id": "0" * 32}, csrf=False).status_code == 400
    assert signed.delete("/api/reports/NI-20260101-0A1B2C3D", headers={}, csrf=False).status_code == 400
    assert signed.post("/logout", csrf=False).status_code == 400 and signed.get("/app/analysis").status_code == 200, "forged logout is refused"
    bad = signed.post("/api/reports", json={"prediction_id": "0" * 32}, headers={"X-CSRFToken": "forged"}, csrf=False)
    assert bad.status_code == 400
    print("ok CSRF token required on every POST/DELETE (forms and fetch APIs); a forged token is refused")


def test_security_headers_csp_and_no_store():
    c = helpers.logged_in_client("h@example.com")
    r = c.get("/app/analysis")
    csp = r.headers["Content-Security-Policy"]
    assert "script-src 'self' 'nonce-" in csp and "frame-ancestors 'none'" in csp and "object-src 'none'" in csp and "'unsafe-eval'" not in csp
    nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
    html = r.get_data(as_text=True)
    inline = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", html)
    assert inline and all(f'nonce="{nonce}"' in tag for tag in inline), inline
    assert r.headers["X-Frame-Options"] == "DENY" and r.headers["X-Content-Type-Options"] == "nosniff"
    assert "Referrer-Policy" in r.headers and "Permissions-Policy" in r.headers and r.headers["Cache-Control"] == "no-store"
    assert r.headers["Content-Security-Policy"] != APP.test_client().get("/").headers["Content-Security-Policy"], "nonce differs per request"
    link_page = APP.test_client().get("/reset-password/" + "z" * 43)
    assert link_page.headers["Referrer-Policy"] == "no-referrer"
    print("ok CSP with per-request nonce, frame denial, no-store for signed-in pages, no-referrer on token pages")


# ---- authorization ---------------------------------------------------------------------------------------------------------------
def test_protected_routes_and_api_responses_when_signed_out():
    c = fresh()
    for path in ("/app", "/app/analysis", "/app/reports"):
        r = c.get(path)
        assert r.status_code == 302 and "/login?next=" in r.headers["Location"] or path == "/app" and "/login" in r.headers["Location"], path
    for method, path in (("get", "/api/reports"), ("get", "/api/reports/export"), ("get", "/api/reports/NI-20260101-0A1B2C3D"),
                         ("get", "/api/reports/NI-20260101-0A1B2C3D/download/pdf"), ("delete", "/api/reports/NI-20260101-0A1B2C3D")):
        r = getattr(c, method)(path)
        assert r.status_code == 401 and r.get_json() == {"error": "Sign in required.", "ok": False}, path
    r = c.post("/api/predict", data={"file": (io.BytesIO(SAMPLE), "x.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 401 and r.is_json
    assert c.post("/api/reports", json={"prediction_id": "0" * 32}).status_code == 401
    for public in ("/", "/login", "/signup", "/forgot-password", "/terms", "/privacy", "/api/model-info", "/api/samples", "/static/css/app.css"):
        assert c.get(public).status_code == 200, public
    print("ok private pages redirect to sign-in, private APIs answer 401 JSON, public pages stay public")


def test_role_based_authorization():
    fresh()
    user = helpers.logged_in_client("plain@example.com")
    admin = helpers.logged_in_client("boss@example.com", role="admin")
    assert user.get("/__admin_only").status_code == 403 and admin.get("/__admin_only").status_code == 200
    assert APP.test_client().get("/__admin_only").status_code == 302
    with APP.app_context():                                                                     # a role change applies at once
        u = User.query.filter_by(email="boss@example.com").one()
        u.role = "user"
        db.session.commit()
    assert admin.get("/__admin_only").status_code == 403, "the role is read from the database, not from the cookie"
    with APP.app_context():
        try:
            db.session.add(User(email="bad@example.com", password_hash="x", full_name="Bad", role="superuser"))
            db.session.commit()
        except Exception:
            db.session.rollback()
        else:
            raise AssertionError("the database must reject unknown roles")
    print("ok role checks are server side, re-read per request, and constrained in the database")


def test_users_cannot_reach_each_others_predictions_or_reports():
    fresh()
    a = helpers.logged_in_client("alice@example.com")
    b = helpers.logged_in_client("bob@example.com")
    pred = a.post("/api/predict", data={"file": (io.BytesIO(SAMPLE), "alice-scan.jpg")}, content_type="multipart/form-data").get_json()
    assert pred["ok"] and re.fullmatch(r"[0-9a-f]{32}", pred["prediction_id"])
    stolen = b.post("/api/reports", json={"prediction_id": pred["prediction_id"]})
    assert stolen.status_code == 404, "Bob cannot save Alice's analysis"
    saved = a.post("/api/reports", json={"prediction_id": pred["prediction_id"]})
    rid = saved.get_json()["report"]["report_id"]
    assert saved.status_code == 201
    assert b.post("/api/reports", json={"prediction_id": pred["prediction_id"]}).status_code == 404
    for path in (f"/api/reports/{rid}", f"/api/reports/{rid}/download/pdf", f"/api/reports/{rid}/download/json", f"/api/reports/{rid}/download/png"):
        assert b.get(path).status_code == 404, path
        assert a.get(path).status_code == 200, path
    assert b.delete(f"/api/reports/{rid}").status_code == 404 and a.get(f"/api/reports/{rid}").status_code == 200
    lb, la = b.get("/api/reports").get_json(), a.get("/api/reports").get_json()
    assert lb["total"] == 0 and lb["stats"] == {"total": 0, "today": 0, "this_month": 0} and la["total"] == 1
    assert "alice-scan" not in b.get("/api/reports/export").get_data(as_text=True) and "alice-scan" in a.get("/api/reports/export").get_data(as_text=True)
    assert b.get(f"/api/reports?q=alice").get_json()["total"] == 0
    assert a.delete(f"/api/reports/{rid}").status_code == 200
    print("ok users are isolated: no cross-user save, view, download, delete, list, search, stats or export")


def test_legacy_sqlite_reports_are_imported_only_by_the_explicit_command():
    fresh()
    import sqlite3
    import reports as rpt
    legacy = helpers.TMP / "legacy-reports.sqlite3"
    legacy.unlink(missing_ok=True)
    rpt.ReportStore(legacy)                                   # creates the old schema
    con = sqlite3.connect(legacy)
    probs = '{"glioma":0.9,"meningioma":0.05,"notumor":0.03,"pituitary":0.02}'
    for rid, pid, owner in (("NI-20250101-AAAAAAAA", "a" * 32, None), ("NI-20250101-BBBBBBBB", "b" * 32, 99)):
        con.execute("INSERT INTO reports (report_id, prediction_id, file_name, predicted_class, confidence, probabilities, warning_level, created_at, model_json, user_id)"
                    " VALUES (?, ?, 'old.jpg', 'glioma', 0.9, ?, 'none', '2025-01-01T00:00:00Z', '{}', ?)", (rid, pid, probs, owner))
    con.commit()
    con.close()
    before = legacy.read_bytes()
    c = helpers.logged_in_client("legacy@example.com")
    assert c.get("/api/reports").get_json()["total"] == 0, "nothing is visible before the explicit import"
    runner = appmod.app.test_cli_runner()
    bad = runner.invoke(args=["import-legacy-reports", "--email", "nobody@example.com", "--file", str(legacy), "--yes"])
    assert bad.exit_code != 0
    ok = runner.invoke(args=["import-legacy-reports", "--email", "legacy@example.com", "--file", str(legacy), "--yes"])
    assert ok.exit_code == 0 and "Imported 1, skipped 0" in ok.output, ok.output
    again = runner.invoke(args=["import-legacy-reports", "--email", "legacy@example.com", "--file", str(legacy), "--yes"])
    assert "Imported 0, skipped 1" in again.output, again.output
    assert c.get("/api/reports").get_json()["total"] == 1 and c.get("/api/reports/NI-20250101-AAAAAAAA").status_code == 200
    other = helpers.logged_in_client("other@example.com")
    assert other.get("/api/reports").get_json()["total"] == 0 and other.get("/api/reports/NI-20250101-AAAAAAAA").status_code == 404
    assert legacy.read_bytes() == before, "the old file is only read"
    print("ok legacy reports are imported only by the explicit command, once, for one user; the old file is untouched")


# ---- the existing workflow still works while signed in --------------------------------------------------------------------------------
def test_existing_prediction_workflow_works_when_signed_in():
    fresh()
    c = helpers.logged_in_client("flow@example.com")
    r = c.post("/api/predict", data={"file": (io.BytesIO(SAMPLE), "sample_glioma_1.jpg")}, content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200 and j["prediction"]["label"] == "glioma" and abs(j["prediction"]["confidence"] - 0.9281720004317073) < 1e-12
    assert j["warning"]["level"] == "none" and j["explanation"]["available"] is False
    assert c.get("/api/model-info").get_json()["validation_metrics"]["n"] == 1046
    assert len(c.get("/api/samples").get_json()["samples"]) == 8
    print("ok signed-in prediction identical to the verified pipeline (glioma 92.82%, no warning)")


# ---- admin bootstrap ---------------------------------------------------------------------------------------------------------------------
def test_create_admin_command():
    fresh()
    import os
    runner = APP.test_cli_runner()
    os.environ["NEUROINSIGHT_ADMIN_PASSWORD"] = "short"
    try:
        weak = runner.invoke(args=["create-admin", "--email", "root@example.com", "--name", "Root Admin"])
        assert weak.exit_code != 0 and user_by_email("root@example.com") is None
        os.environ["NEUROINSIGHT_ADMIN_PASSWORD"] = "Quartz-Falcon-Orbit-42"
        ok = runner.invoke(args=["create-admin", "--email", "root@example.com", "--name", "Root Admin"])
    finally:
        os.environ.pop("NEUROINSIGHT_ADMIN_PASSWORD", None)
    assert ok.exit_code == 0, ok.output
    u = user_by_email("root@example.com")
    assert u["role"] == "admin" and u["verified"] and u["hash"].startswith("$argon2id$")
    assert "Quartz-Falcon" not in ok.output
    print("ok `flask create-admin`: weak passwords refused, admin created verified, password never echoed")


def test_passwords_policy_and_hashing_helpers():
    with APP.app_context():
        assert security.password_problems("Tr0ub4dor&3-horse", email="a@b.test", full_name="Al Pha") == []
        assert security.password_problems("short")
        assert any("common" in m for m in security.password_problems("Password1234"))
        assert security.password_problems("x" * 200)
        h = security.hash_password("Correct-Horse-9!")
        assert security.verify_password(h, "Correct-Horse-9!") and not security.verify_password(h, "wrong") and not security.verify_password(None, "anything")
        assert security.safe_next_url("/app/reports?x=1") and not security.safe_next_url("//x.test") and not security.safe_next_url("https://x.test")
    print("ok password policy, argon2id verification and redirect-target validation")


def test_auth_code_never_touches_models_or_the_test_split():
    for name in ("webapp/auth.py", "webapp/security.py", "webapp/access.py", "webapp/__init__.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        assert "load_split" not in src and "allow_test" not in src and "inference" not in src, name
    print("ok authentication code is independent of the model and the test split")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("\nALL PASSED" if not failed else f"\n{failed} FAILED")
    sys.exit(1 if failed else 0)
