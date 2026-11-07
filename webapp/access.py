"""Route protection: sign-in, verified email and role checks (server side; never trusted to the browser)."""
from __future__ import annotations

from functools import wraps

from flask import jsonify, redirect, render_template, request, url_for
from flask_login import current_user


def _is_api() -> bool:
    return request.path.startswith("/api/")


def _api_error(message: str, status: int):
    resp = jsonify(ok=False, error=message)
    resp.status_code = status
    return resp


def protected(role: str | None = None, *, verified: bool = True):
    """Decorator: the user must be signed in, (by default) have a verified email, and (optionally) hold a role.

    API requests get JSON 401/403 responses; page requests are redirected to the sign-in or verification page.
    Roles are read from the database on every request, never from the cookie.
    """
    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                if _is_api():
                    return _api_error("Sign in required.", 401)
                return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))
            if verified and not current_user.is_verified:
                if _is_api():
                    return _api_error("Verify your email address first.", 403)
                return redirect(url_for("auth.verify_notice"))
            if role and current_user.role != role:
                if _is_api():
                    return _api_error("You do not have permission to do that.", 403)
                return render_template("error.html", disclaimer=_disclaimer(), status=403, title="Access denied",
                                       message="You do not have permission to open this page."), 403
            return view(*args, **kwargs)
        return wrapper
    return decorator


def _disclaimer() -> str:
    from flask import current_app
    return current_app.config.get("DISCLAIMER_TEXT", "")
