"""Command-line tools: flask --app app create-admin"""
from __future__ import annotations

import os

import click
from flask import current_app

from . import security
from .extensions import db
from .models import CLASSES, Report, User, utcnow


def register_cli(app) -> None:
    @app.cli.command("create-admin")
    @click.option("--email", prompt="Admin email", help="Email address of the administrator.")
    @click.option("--name", prompt="Full name", help="Full name of the administrator.")
    def create_admin(email: str, name: str) -> None:
        """Create an administrator (or promote an existing verified account). There are no default credentials.

        The password is read from a hidden prompt, or from NEUROINSIGHT_ADMIN_PASSWORD for scripted setups
        (never pass it as a command-line argument: it would end up in the shell history).
        """
        normalized = security.normalize_email(email)
        if normalized is None:
            raise click.ClickException("That is not a valid email address.")
        password = os.environ.get("NEUROINSIGHT_ADMIN_PASSWORD") or click.prompt("Password", hide_input=True, confirmation_prompt=True)
        problems = security.password_problems(password, email=normalized, full_name=name)
        if problems:
            raise click.ClickException(" ".join(problems))
        user = User.query.filter_by(email=normalized).first()
        if user is None:
            user = User(email=normalized, password_hash=security.hash_password(password), full_name=" ".join(name.split())[:100],
                        role="admin", email_verified_at=utcnow(), terms_accepted_at=utcnow())
            db.session.add(user)
            action = "created"
        else:
            user.role, user.is_active_flag = "admin", True
            user.password_hash = security.hash_password(password)
            user.session_version += 1
            user.email_verified_at = user.email_verified_at or utcnow()
            action = "promoted (password replaced)"
        db.session.commit()
        current_app.logger.info("administrator %s user_id=%s", action, user.id)
        click.echo(f"Administrator {action}: user id {user.id}.")


    @app.cli.command("import-legacy-reports")
    @click.option("--email", prompt="Import the reports for which account (email)", help="Account that will own the imported reports.")
    @click.option("--file", "legacy_file", default=None, type=click.Path(exists=True, dir_okay=False),
                  help="Old reports SQLite file (default: instance/reports.sqlite3).")
    @click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
    def import_legacy_reports(email: str, legacy_file: str | None, yes: bool) -> None:
        """Copy reports saved before accounts existed (rows with no owner) into the database as one user's reports.

        The old file is only read, never changed. Rows already imported (same prediction id) and rows with an unknown
        class are skipped. Imported reports are not linked to any case.
        """
        import json
        import sqlite3
        from pathlib import Path

        from .config import INSTANCE_DIR

        user = User.query.filter_by(email=security.normalize_email(email) or "").first()
        if user is None:
            raise click.ClickException("No account with that email address.")
        path = Path(legacy_file) if legacy_file else INSTANCE_DIR / "reports.sqlite3"
        if not path.is_file():
            raise click.ClickException(f"Legacy reports file not found: {path}")
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute("SELECT * FROM reports WHERE user_id IS NULL ORDER BY created_at").fetchall()
        except sqlite3.Error as exc:
            raise click.ClickException(f"Could not read the legacy file: {exc}") from None
        finally:
            con.close()
        click.echo(f"{len(rows)} ownerless report(s) found in {path.name}.")
        if not rows or not (yes or click.confirm(f"Import them as reports of {user.email}?")):
            return
        have = {r.prediction_id for r in Report.query.filter_by(user_id=user.id)}
        taken = {r.report_id for r in Report.query.all()}
        added = skipped = 0
        for r in rows:
            if r["predicted_class"] not in CLASSES or r["prediction_id"] in have or r["report_id"] in taken:
                skipped += 1
                continue
            db.session.add(Report(report_id=r["report_id"], prediction_id=r["prediction_id"], user_id=user.id, case_id=None,
                                  file_name=r["file_name"], predicted_class=r["predicted_class"], confidence=r["confidence"],
                                  probabilities=json.loads(r["probabilities"]), warning_level=r["warning_level"],
                                  warning_message=r["warning_message"], created_at=r["created_at"], model_json=json.loads(r["model_json"])))
            added += 1
        db.session.commit()
        click.echo(f"Imported {added}, skipped {skipped}.")
