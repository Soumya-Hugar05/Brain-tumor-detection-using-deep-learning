"""Tests for profile, settings and notifications (Phase 8). Run: python tests/test_account.py

Everything acts on the signed-in user only. Only demo samples are analysed; the test split is never touched.
"""
from __future__ import annotations

import io
import re
import sys

import helpers
import app as appmod
from webapp.extensions import db
from webapp.models import Notification, User

ROOT = helpers.ROOT
APP = appmod.app
SAMPLE = (ROOT / "static" / "samples" / "sample_glioma_1.jpg").read_bytes()
NEW_PW = "Another-Strong-Pass-42!"


def fresh():
    helpers.reset_db()
    return helpers.logged_in_client("alice@example.com"), helpers.logged_in_client("bob@example.com")


def user(email="alice@example.com") -> dict:
    with APP.app_context():
        u = User.query.filter_by(email=email).one()
        return dict(name=u.full_name, inst=u.institution, occ=u.occupation, role=u.role, email=u.email, theme=u.theme,
                    na=u.notify_analysis, nr=u.notify_reports, ver=u.session_version, pw=u.password_hash)


def notes(email="alice@example.com") -> list[str]:
    with APP.app_context():
        uid = User.query.filter_by(email=email).one().id
        return [n.title for n in Notification.query.filter_by(user_id=uid).order_by(Notification.id)]


def predict(c, name="scan.jpg"):
    return c.post("/api/predict", data={"file": (io.BytesIO(SAMPLE), name)}, content_type="multipart/form-data")


def test_pages_require_sign_in():
    helpers.reset_db()
    anon = APP.test_client()
    for path in ("/app/profile", "/app/settings", "/app/notifications"):
        assert anon.get(path).status_code == 302, path
    for path in ("/app/profile", "/app/settings/preferences", "/app/settings/password", "/app/settings/sessions",
                 "/app/notifications/read", "/app/notifications/clear"):
        assert anon.post(path, csrf=False).status_code in (302, 400), path
    assert anon.post("/api/account/theme", json={"theme": "light"}, csrf=False).status_code in (400, 401)


def test_profile_edit_validates_and_never_changes_email_or_role():
    a, _ = fresh()
    before = user()
    r = a.post("/app/profile", data=dict(full_name="Alice Q. Researcher", institution="Lab 7", occupation="Student",
                                         email="evil@example.com", role="admin"))
    assert r.status_code == 302
    after = user()
    assert (after["name"], after["inst"], after["occ"]) == ("Alice Q. Researcher", "Lab 7", "Student")
    assert after["email"] == before["email"] and after["role"] == "user"
    assert "Alice Q. Researcher" in a.get("/app/profile").get_data(as_text=True)
    for bad in (dict(full_name="A"), dict(full_name="Alice", occupation="Wizard"), dict(full_name="Alice", institution="x" * 151)):
        assert a.post("/app/profile", data=bad).status_code == 400, bad
    assert user()["name"] == "Alice Q. Researcher"
    a.post("/app/profile", data=dict(full_name="<b>Al</b>ice"))
    assert "<b>Al</b>" not in a.get("/app/profile").get_data(as_text=True)


def test_preferences_theme_and_notification_choices():
    a, _ = fresh()
    assert a.post("/app/settings/preferences", data=dict(theme="light", notify_analysis="on")).status_code == 302
    u = user()
    assert (u["theme"], u["na"], u["nr"]) == ("light", True, False)
    html = a.get("/app/settings").get_data(as_text=True)
    assert 'data-theme="light"' in html and re.search(r'value="light"\s+checked', html)
    assert a.post("/app/settings/preferences", data=dict(theme="neon")).status_code == 400
    assert a.post("/api/account/theme", json={"theme": "dark"}).get_json()["ok"] is True and user()["theme"] == "dark"
    assert a.post("/api/account/theme", json={"theme": "x"}).status_code == 400
    assert a.post("/api/account/theme", data="theme=dark", content_type="text/plain").status_code == 400


def test_notifications_follow_real_events_and_preferences():
    a, b = fresh()
    assert notes() == []
    pid = predict(a).get_json()["prediction_id"]
    assert notes() == ["Analysis completed"]
    a.post("/api/reports", json={"prediction_id": pid})
    a.post("/api/reports", json={"prediction_id": pid})                       # idempotent: no second notice
    assert notes() == ["Analysis completed", "Report saved"]
    a.post("/app/settings/preferences", data=dict(theme="dark"))              # both switched off
    predict(a, "two.jpg")
    assert notes() == ["Analysis completed", "Report saved"], "switched-off kinds are not created"
    assert notes("bob@example.com") == [], "notifications are per user"
    html = a.get("/app/notifications").get_data(as_text=True)
    assert "Analysis completed" in html and "Report saved" in html
    assert "Analysis completed" not in b.get("/app/notifications").get_data(as_text=True)
    assert re.search(r'class="bell-count"[^>]*>2<', a.get("/app").get_data(as_text=True)), "unread badge shows the real count"


def test_mark_read_clear_and_isolation():
    a, b = fresh()
    predict(a)
    predict(a, "x.jpg")
    with APP.app_context():
        ids = [n.id for n in Notification.query.all()]
    assert b.post("/app/notifications/read", data=dict(id=str(ids[0]))).status_code == 302
    with APP.app_context():
        assert Notification.query.filter_by(read_at=None).count() == 2, "Bob cannot mark Alice's notifications"
    assert a.post("/app/notifications/read", data=dict(id="abc")).status_code == 400
    a.post("/app/notifications/read", data=dict(id=str(ids[0])))
    with APP.app_context():
        assert Notification.query.filter_by(read_at=None).count() == 1
    a.post("/app/notifications/read")
    with APP.app_context():
        assert Notification.query.filter_by(read_at=None).count() == 0
    predict(b)
    a.post("/app/notifications/clear")
    assert notes() == [] and notes("bob@example.com") == ["Analysis completed"]


def test_change_password_rules_and_other_sessions_end():
    a, _ = fresh()
    other = helpers.logged_in_client("alice@example.com")                     # a second device
    old = user()["pw"]
    base = dict(current_password=helpers.PASSWORD, password=NEW_PW, confirm_password=NEW_PW)
    assert a.post("/app/settings/password", data=dict(base, current_password="wrong")).status_code == 400
    assert a.post("/app/settings/password", data=dict(base, confirm_password="different")).status_code == 400
    assert a.post("/app/settings/password", data=dict(base, password="short", confirm_password="short")).status_code == 400
    assert a.post("/app/settings/password", data=dict(base, password=helpers.PASSWORD, confirm_password=helpers.PASSWORD)).status_code == 400
    assert user()["pw"] == old
    assert NEW_PW not in a.post("/app/settings/password", data=dict(base, current_password="wrong")).get_data(as_text=True)
    assert a.post("/app/settings/password", data=base).status_code == 302
    assert user()["pw"] != old
    assert a.get("/app/settings").status_code == 200, "this browser stays signed in"
    assert other.get("/app/settings").status_code == 302, "the other device was signed out"
    assert "Password changed" in notes()
    assert any("password was changed" in m for m in helpers.outbox())
    fresh_client = APP.test_client()
    assert helpers.login(fresh_client, "alice@example.com", password=helpers.PASSWORD).status_code == 401
    assert helpers.login(fresh_client, "alice@example.com", password=NEW_PW).status_code == 302


def test_sign_out_everywhere_keeps_only_this_browser():
    a, _ = fresh()
    other = helpers.logged_in_client("alice@example.com")
    v = user()["ver"]
    assert a.post("/app/settings/sessions").status_code == 302
    assert user()["ver"] == v + 1
    assert a.get("/app/settings").status_code == 200 and other.get("/app/settings").status_code == 302
    assert "Signed out of other devices" in notes()


def test_navigation_is_fully_live_and_user_theme_applies():
    a, _ = fresh()
    html = a.get("/app").get_data(as_text=True)
    assert "Soon" not in html and "Not available yet" not in html
    for path in ("/app/profile", "/app/settings", "/app/notifications"):
        assert a.get(path).status_code == 200, path
    a.post("/app/settings/preferences", data=dict(theme="light"))
    page = a.get("/app/analysis").get_data(as_text=True)
    assert 'data-theme="light"' in page and 'var t = "light"' in page


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
