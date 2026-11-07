"""Public information pages (draft terms and privacy notice)."""
from __future__ import annotations

from flask import Blueprint, current_app, render_template

bp = Blueprint("public", __name__)


@bp.get("/terms")
def terms():
    return render_template("terms.html", disclaimer=current_app.config["DISCLAIMER_TEXT"])


@bp.get("/privacy")
def privacy():
    return render_template("privacy.html", disclaimer=current_app.config["DISCLAIMER_TEXT"])
