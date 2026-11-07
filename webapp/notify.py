"""In-app notifications. Created only for real events; the caller commits the session."""
from __future__ import annotations

import sqlalchemy as sa

from .extensions import db
from .models import Notification, User

KEEP_PER_USER = 200
KINDS = ("analysis", "report", "security", "account")


def notify(user: User, kind: str, title: str, body: str | None = None, href: str | None = None) -> bool:
    """Queue a notification unless the user switched that kind off (security and account notices are always delivered)."""
    assert kind in KINDS
    if (kind == "analysis" and not user.notify_analysis) or (kind == "report" and not user.notify_reports):
        return False
    db.session.add(Notification(user_id=user.id, kind=kind, title=title[:150], body=(body or None) and body[:300], href=href))
    old = db.session.scalars(sa.select(Notification.id).where(Notification.user_id == user.id)
                             .order_by(Notification.id.desc()).offset(KEEP_PER_USER).limit(1)).all()
    if old:
        db.session.execute(sa.delete(Notification).where(Notification.user_id == user.id, Notification.id <= old[0]))
    return True


def unread_count(user_id: int) -> int:
    return db.session.scalar(sa.select(sa.func.count()).select_from(Notification)
                             .where(Notification.user_id == user_id, Notification.read_at.is_(None))) or 0
