"""Database models (SQLAlchemy 2.0 style). Schema changes go through Alembic migrations (see migrations/).

Only account-related tables exist so far; later phases add cases, predictions, reports, notifications and audit logs.
No patient data and no images are stored here.
"""
from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from flask_login import UserMixin
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .extensions import db

ROLES = ("user", "admin")
OCCUPATIONS = ("Researcher", "Student", "Clinician", "Engineer", "Other")
TOKEN_PURPOSES = ("verify_email", "reset_password")


def utcnow() -> datetime:
    """Naive UTC timestamps (portable across SQLite and PostgreSQL)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(UserMixin, db.Model):
    __tablename__ = "users"
    __table_args__ = (
        sa.CheckConstraint("role IN ('user', 'admin')", name="ck_users_role"),
        sa.CheckConstraint("email = lower(email)", name="ck_users_email_lower"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(sa.String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(sa.String(255))
    full_name: Mapped[str] = mapped_column(sa.String(100))
    institution: Mapped[str | None] = mapped_column(sa.String(150))
    occupation: Mapped[str | None] = mapped_column(sa.String(40))            # self-declared label, grants no permissions
    role: Mapped[str] = mapped_column(sa.String(20), default="user", server_default="user")
    is_active_flag: Mapped[bool] = mapped_column("is_active", default=True, server_default=sa.true())
    email_verified_at: Mapped[datetime | None] = mapped_column()
    session_version: Mapped[int] = mapped_column(default=1, server_default="1")  # bumped to sign out everywhere
    terms_accepted_at: Mapped[datetime | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column()
    theme: Mapped[str | None] = mapped_column(sa.String(5))                   # 'light' | 'dark' | None (default dark)
    notify_analysis: Mapped[bool] = mapped_column(default=True, server_default=sa.true())
    notify_reports: Mapped[bool] = mapped_column(default=True, server_default=sa.true())

    tokens: Mapped[list["AuthToken"]] = relationship(back_populates="user", cascade="all, delete-orphan", passive_deletes=True)

    # -- Flask-Login ---------------------------------------------------------------------------------------
    @property
    def is_active(self) -> bool:                      # Flask-Login: deactivated accounts cannot be signed in
        return bool(self.is_active_flag)

    def get_id(self) -> str:
        """'<id>.<session_version>': bumping session_version invalidates every existing session and remember-me cookie."""
        return f"{self.id}.{self.session_version}"

    @property
    def is_verified(self) -> bool:
        return self.email_verified_at is not None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def __repr__(self) -> str:  # never include the email or hash
        return f"<User id={self.id} role={self.role}>"


class AuthToken(db.Model):
    """Single-use email-verification and password-reset tokens. Only a SHA-256 hash of the token is stored."""
    __tablename__ = "auth_tokens"
    __table_args__ = (sa.CheckConstraint("purpose IN ('verify_email', 'reset_password')", name="ck_auth_tokens_purpose"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"), index=True)
    purpose: Mapped[str] = mapped_column(sa.String(20))
    token_hash: Mapped[str] = mapped_column(sa.String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column()
    used_at: Mapped[datetime | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    user: Mapped[User] = relationship(back_populates="tokens")


class LoginAttempt(db.Model):
    """Recent sign-in attempts, used for throttling. The email is stored only as a hash."""
    __tablename__ = "login_attempts"
    __table_args__ = (sa.Index("ix_login_attempts_email_created", "email_hash", "created_at"),
                      sa.Index("ix_login_attempts_ip_created", "ip", "created_at"))

    id: Mapped[int] = mapped_column(primary_key=True)
    email_hash: Mapped[str] = mapped_column(sa.String(64))
    ip: Mapped[str] = mapped_column(sa.String(45))
    succeeded: Mapped[bool] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
CASE_STATUSES = ("open", "archived")


class Case(db.Model):
    """A user's own grouping of analyses. Holds only a user-chosen code and optional free text: no patient fields."""
    __tablename__ = "cases"
    __table_args__ = (
        sa.UniqueConstraint("user_id", "case_code", name="uq_cases_user_code"),
        sa.CheckConstraint("status IN ('open', 'archived')", name="ck_cases_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"), index=True)
    case_code: Mapped[str] = mapped_column(sa.String(40))
    title: Mapped[str | None] = mapped_column(sa.String(150))
    notes: Mapped[str | None] = mapped_column(sa.Text)
    status: Mapped[str] = mapped_column(sa.String(12), default="open", server_default="open")
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


class Prediction(db.Model):
    """One analysis made by a user: the model output and its context. The MRI image itself is never stored."""
    __tablename__ = "predictions"
    __table_args__ = (
        sa.CheckConstraint("predicted_class IN ('glioma', 'meningioma', 'notumor', 'pituitary')", name="ck_predictions_class"),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_predictions_confidence"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(sa.String(32), unique=True)          # the prediction_id returned by /api/predict
    user_id: Mapped[int] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"), index=True)
    case_id: Mapped[int | None] = mapped_column(sa.ForeignKey("cases.id", ondelete="SET NULL"), index=True)
    file_name: Mapped[str] = mapped_column(sa.String(120))
    predicted_class: Mapped[str] = mapped_column(sa.String(20))
    confidence: Mapped[float] = mapped_column()
    probabilities: Mapped[dict] = mapped_column(sa.JSON)
    warning_level: Mapped[str] = mapped_column(sa.String(10))
    model_estimator_sha256: Mapped[str] = mapped_column(sa.String(64))
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)


class Report(db.Model):
    """A saved analysis report (metadata only; the PDF, PNG and JSON files are rendered on demand from this row).

    created_at keeps the 'YYYY-MM-DDTHH:MM:SSZ' string the report API has always returned, so ordering and date
    filtering behave exactly as before.
    """
    __tablename__ = "reports"
    __table_args__ = (
        sa.UniqueConstraint("user_id", "prediction_id", name="uq_reports_user_prediction"),
        sa.CheckConstraint("predicted_class IN ('glioma', 'meningioma', 'notumor', 'pituitary')", name="ck_reports_class"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    report_id: Mapped[str] = mapped_column(sa.String(24), unique=True)
    prediction_id: Mapped[str] = mapped_column(sa.String(32))
    user_id: Mapped[int] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"), index=True)
    case_id: Mapped[int | None] = mapped_column(sa.ForeignKey("cases.id", ondelete="SET NULL"), index=True)
    file_name: Mapped[str] = mapped_column(sa.String(120))
    predicted_class: Mapped[str] = mapped_column(sa.String(20), index=True)
    confidence: Mapped[float] = mapped_column()
    probabilities: Mapped[dict] = mapped_column(sa.JSON)
    warning_level: Mapped[str] = mapped_column(sa.String(10))
    warning_message: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[str] = mapped_column(sa.String(20), index=True)
    model_json: Mapped[dict] = mapped_column(sa.JSON)


class Notification(db.Model):
    """An in-app notification about a real event on the user's own account. No patient or image content."""
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(sa.String(20))                        # analysis | report | security | account
    title: Mapped[str] = mapped_column(sa.String(150))
    body: Mapped[str | None] = mapped_column(sa.String(300))
    href: Mapped[str | None] = mapped_column(sa.String(200))                # an internal path such as /app/reports
    read_at: Mapped[datetime | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)


class AuditLog(db.Model):
    """Administrative actions on accounts. Holds who did what to which account; never patient, case or report content."""
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor_id: Mapped[int | None] = mapped_column(sa.ForeignKey("users.id", ondelete="SET NULL"), index=True)
    target_id: Mapped[int | None] = mapped_column(sa.ForeignKey("users.id", ondelete="SET NULL"), index=True)
    action: Mapped[str] = mapped_column(sa.String(30))
    ip: Mapped[str | None] = mapped_column(sa.String(45))
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
