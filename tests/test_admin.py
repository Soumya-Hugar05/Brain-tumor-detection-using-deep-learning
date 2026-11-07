"""Tests for the admin area (Phase 9). Run: python tests/test_admin.py

Admins manage accounts only: they must not reach any user's cases, analyses, reports or notifications.
"""
from __future__ import annotations

import io
import re
import sys

import helpers
import app as appmod
from webapp.models import AuditLog, Case, Notification, Prediction, User

ROOT = helpers.ROOT
APP = appmod.app
SAMPLE = (ROOT / "static" / "samples" / "sample_glioma_1.jpg").read_bytes()


def fresh():
    helpers.reset_db()
    admin = helpers.logged_in_client("boss@example.com", role="admin", name="Boss Admin")
    alice = helpers.logged_in_client("alice@example.com", name="Alice Anderson")
    return admin, alice


def uid(email):
    with APP.app_context():
        return User.query.filter_by(email=email).one().id


def state(email):
    with APP.app_context():
        u = User.query.filter_by(email=email).one()
        return u.is_active_flag, u.session_version


def test_non_admins_and_anonymous_are_refused():
    admin, alice = fresh()
    anon = APP.test_client()
    target = uid("alice@example.com")
    for path in ("/app/admin", "/app/admin/users", f"/app/admin/users/{target}", "/app/admin/audit"):
        assert anon.get(path).status_code == 302, path
        assert alice.get(path).status_code == 403, path
        assert admin.get(path).status_code == 200, path
    for action in ("deactivate", "reactivate", "revoke_sessions"):
        assert alice.post(f"/app/admin/users/{target}/{action}").status_code == 403
        assert anon.post(f"/app/admin/users/{target}/{action}", csrf=False).status_code in (302, 400)
    assert state("alice@example.com") == (True, 1)
    assert "Administration" not in alice.get("/app").get_data(as_text=True), "no admin links for ordinary users"
    assert "Administration" in admin.get("/app").get_data(as_text=True)


def test_list_search_filter_and_detail():
    admin, _ = fresh()
    helpers.create_user("pending@example.com", verified=False, name="Pat Pending")
    html = admin.get("/app/admin/users").get_data(as_text=True)
    assert all(e in html for e in ("alice@example.com", "boss@example.com", "pending@example.com"))
    assert "pending@example.com" not in admin.get("/app/admin/users?q=alice").get_data(as_text=True)
    assert "pending@example.com" in admin.get("/app/admin/users?status=unverified").get_data(as_text=True)
    assert "alice@example.com" not in admin.get("/app/admin/users?status=unverified").get_data(as_text=True)
    assert "boss@example.com" in admin.get("/app/admin/users?status=admin").get_data(as_text=True)
    assert "alice@example.com" not in admin.get("/app/admin/users?q=%25").get_data(as_text=True), "LIKE wildcards are escaped"
    assert admin.get("/app/admin/users?page=abc&status=zzz").status_code == 200
    d = admin.get(f"/app/admin/users/{uid('alice@example.com')}").get_data(as_text=True)
    assert "Alice Anderson" in d and "password" not in d.lower().replace("password rules", "")
    assert admin.get("/app/admin/users/99999").status_code == 404
    assert admin.post(f"/app/admin/users/{uid('alice@example.com')}/promote").status_code == 404, "no role changes in the UI"


def test_deactivate_signs_the_user_out_and_blocks_sign_in_until_reactivated():
    admin, alice = fresh()
    target = uid("alice@example.com")
    assert alice.get("/app/settings").status_code == 200
    assert admin.post(f"/app/admin/users/{target}/deactivate").status_code == 302
    assert state("alice@example.com") == (False, 2)
    assert alice.get("/app/settings").status_code == 302, "open session is void immediately"
    assert helpers.login(APP.test_client(), "alice@example.com").status_code == 403
    assert admin.post(f"/app/admin/users/{target}/reactivate").status_code == 302
    assert helpers.login(APP.test_client(), "alice@example.com").status_code == 302
    with APP.app_context():
        assert [a.action for a in AuditLog.query.order_by(AuditLog.id)] == ["deactivate", "reactivate"]


def test_revoke_sessions_and_self_protection():
    admin, alice = fresh()
    target = uid("alice@example.com")
    assert admin.post(f"/app/admin/users/{target}/revoke_sessions").status_code == 302
    assert alice.get("/app/settings").status_code == 302
    boss = uid("boss@example.com")
    admin.post(f"/app/admin/users/{boss}/deactivate")
    assert state("boss@example.com")[0] is True, "an admin cannot deactivate themselves"
    second = helpers.logged_in_client("second@example.com", role="admin")
    second.post(f"/app/admin/users/{boss}/deactivate")
    assert state("boss@example.com")[0] is False
    second.post(f"/app/admin/users/{uid('second@example.com')}/deactivate")
    assert state("second@example.com")[0] is True, "never lock out the last administrator"


def test_audit_log_records_actor_target_and_no_content():
    admin, alice = fresh()
    admin.post(f"/app/admin/users/{uid('alice@example.com')}/revoke_sessions")
    html = admin.get("/app/admin/audit").get_data(as_text=True)
    assert "boss@example.com" in html and "alice@example.com" in html and "revoke sessions" in html
    assert "alice@example.com" in admin.get("/app/admin").get_data(as_text=True)
    assert "Alice Anderson" not in alice.get("/app/notifications").get_data(as_text=True)
    with APP.app_context():
        row = AuditLog.query.one()
        assert row.actor_id == uid("boss@example.com") and row.target_id == uid("alice@example.com")


def test_admins_cannot_reach_other_users_data():
    admin, alice = fresh()
    alice.post("/app/cases", data=dict(case_code="PRIVATE-7", title="secret title", notes="secret notes"))
    pid = alice.post("/api/predict", data={"file": (io.BytesIO(SAMPLE), "alice-private.jpg")}, content_type="multipart/form-data").get_json()["prediction_id"]
    rid = alice.post("/api/reports", json={"prediction_id": pid}).get_json()["report"]["report_id"]
    with APP.app_context():
        cid = Case.query.one().id
    assert admin.get(f"/app/cases/{cid}").status_code == 404
    assert admin.post(f"/app/cases/{cid}/delete").status_code == 404
    assert admin.post(f"/app/history/{pid}/case", data=dict(case_id="")).status_code == 404
    assert admin.get(f"/api/reports/{rid}").status_code == 404 and admin.get(f"/api/reports/{rid}/download/pdf").status_code == 404
    assert admin.delete(f"/api/reports/{rid}").status_code == 404
    for path in ("/app/cases", "/app/history", "/app/reports", "/api/reports", "/api/reports/export", "/app/notifications", "/app",
                 "/app/admin", "/app/admin/users", f"/app/admin/users/{uid('alice@example.com')}", "/app/admin/audit"):
        body = admin.get(path).get_data(as_text=True)
        for secret in ("PRIVATE-7", "secret title", "secret notes", "alice-private", rid):
            assert secret not in body, (path, secret)
    with APP.app_context():
        assert Case.query.count() == 1 and Prediction.query.count() == 1 and Notification.query.filter_by(user_id=uid("boss@example.com")).count() == 0


def test_admin_code_never_touches_content_tables():
    src = (ROOT / "webapp" / "admin.py").read_text(encoding="utf-8")
    code = re.sub(r'""".*?"""', "", src, count=1, flags=re.S)
    for banned in ("Case", "Prediction", "Report", "Notification", "report_store"):
        assert not re.search(rf"\b{banned}\b", code), banned


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
