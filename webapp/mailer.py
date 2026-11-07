"""Outgoing email. Two backends, chosen with NEUROINSIGHT_MAIL_BACKEND:

* outbox (default): each message is written to instance/outbox/ as a text file so links can be opened during development.
  The folder is git-ignored. Because verification and reset links are secrets, treat the folder like a password store
  and do not use this backend in production.
* smtp: real delivery through MAIL_SERVER (STARTTLS by default), with credentials from environment variables.

Tokens and passwords are never written to the application log; only the masked recipient and the template name are.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from flask import current_app

from .security import mask_email

log = logging.getLogger("neuroinsight.mail")


def send_email(to: str, subject: str, body: str, *, template: str) -> None:
    cfg = current_app.config
    backend = cfg["MAIL_BACKEND"]
    if backend == "smtp" and cfg["MAIL_SERVER"]:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = cfg["MAIL_FROM"], to, subject
        msg.set_content(body)
        try:
            with smtplib.SMTP(cfg["MAIL_SERVER"], cfg["MAIL_PORT"], timeout=15) as smtp:
                if cfg["MAIL_USE_TLS"]:
                    smtp.starttls(context=ssl.create_default_context())
                if cfg["MAIL_USERNAME"]:
                    smtp.login(cfg["MAIL_USERNAME"], cfg["MAIL_PASSWORD"])
                smtp.send_message(msg)
            log.info("email sent template=%s to=%s", template, mask_email(to))
        except (OSError, smtplib.SMTPException):
            log.error("email delivery failed template=%s to=%s", template, mask_email(to))   # no details: they may echo secrets
        return
    outbox = Path(cfg["OUTBOX_DIR"])
    outbox.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    path = outbox / f"{stamp}-{template}.txt"
    path.write_text(f"To: {to}\nSubject: {subject}\n\n{body}\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    log.info("email written to outbox template=%s to=%s", template, mask_email(to))


def link(path: str) -> str:
    """Absolute URL for an emailed link."""
    from flask import request
    base = current_app.config["APP_BASE_URL"] or request.url_root.rstrip("/")
    return f"{base}{path}"
