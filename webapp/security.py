"""Password hashing and policy, single-use tokens, login throttling and small request helpers."""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import timedelta
from urllib.parse import urlparse

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from email_validator import EmailNotValidError, validate_email
from flask import current_app, request

from .extensions import db
from .models import AuthToken, LoginAttempt, utcnow

_hasher = PasswordHasher()                              # argon2id with the library's current recommended parameters
_DUMMY_HASH = _hasher.hash("not-a-real-password-for-timing")

# A short deny-list of very common passwords (lower-case). It complements, not replaces, the length requirement.
COMMON_PASSWORDS = frozenset("""
password password1 password12 password123 password1234 passw0rd p@ssw0rd p@ssword 1234567890 12345678910 qwertyuiop qwerty123
qwerty12345 1q2w3e4r5t 1qaz2wsx3edc iloveyou1 iloveyou123 welcome123 welcome1234 letmein123 letmein1234 admin12345 administrator
changeme123 changeme1234 abc1234567 abcd123456 abcd1234567 monkey1234 dragon1234 football123 baseball123 superman123 trustno1234
zaq12wsxcde3 0987654321 1234512345 11111111111 1111111111 0000000000 neuroinsight neuroinsight1 neuroinsight123 braintumor123
""".split())


# ---------------------------------------------------------------------------------------------------------
# passwords
# ---------------------------------------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(stored_hash: str | None, password: str) -> bool:
    """Constant-effort check: a dummy hash is verified when there is no account, so timing does not reveal existence."""
    try:
        return _hasher.verify(stored_hash or _DUMMY_HASH, password) and stored_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except InvalidHashError:
        return False


def password_problems(password: str, *, email: str = "", full_name: str = "") -> list[str]:
    """Server-side password policy. Returns user-facing problems (empty list = acceptable)."""
    cfg = current_app.config
    lo, hi = cfg["PASSWORD_MIN_LENGTH"], cfg["PASSWORD_MAX_LENGTH"]
    problems = []
    if len(password) < lo:
        problems.append(f"Use at least {lo} characters.")
    if len(password) > hi:
        problems.append(f"Use at most {hi} characters.")
    low = password.lower()
    if low in COMMON_PASSWORDS or re.fullmatch(r"(.)\1{7,}", password):
        problems.append("This password is too common. Choose something less predictable.")
    local = email.split("@")[0].lower() if email else ""
    if local and len(local) >= 4 and local in low:
        problems.append("The password must not contain your email name.")
    for part in re.split(r"\s+", full_name.strip().lower()):
        if len(part) >= 4 and part in low:
            problems.append("The password must not contain your name.")
            break
    classes = sum(bool(re.search(p, password)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[^A-Za-z0-9]"))
    if len(password) < 16 and classes < 3:
        problems.append("Mix at least three of: lower-case, upper-case, digits and symbols (or use 16+ characters).")
    return problems


# ---------------------------------------------------------------------------------------------------------
# email and request helpers
# ---------------------------------------------------------------------------------------------------------
def normalize_email(raw: str) -> str | None:
    """Syntax-validated, lower-cased email or None. No DNS lookups (works offline)."""
    try:
        return validate_email((raw or "").strip(), check_deliverability=False).normalized.lower()
    except EmailNotValidError:
        return None


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


def email_hash(email: str) -> str:
    return hashlib.sha256(("neuroinsight-login:" + email.strip().lower()).encode()).hexdigest()


def client_ip() -> str:
    """The client address. X-Forwarded-For is only honoured behind TRUSTED_PROXY_COUNT trusted proxies (ProxyFix)."""
    return (request.remote_addr or "0.0.0.0")[:45]


def safe_next_url(target: str | None) -> str | None:
    """Only same-site relative paths are accepted for post-login redirects (prevents open redirects)."""
    if not target or len(target) > 300:
        return None
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return None
    if any(ord(c) < 32 for c in target):
        return None
    return target


# ---------------------------------------------------------------------------------------------------------
# single-use tokens (email verification, password reset)
# ---------------------------------------------------------------------------------------------------------
def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def issue_token(user, purpose: str, lifetime: timedelta) -> str:
    """Create a token for the user and return the raw value (shown only in the emailed link). Older unused tokens
    of the same purpose are invalidated."""
    now = utcnow()
    for old in AuthToken.query.filter_by(user_id=user.id, purpose=purpose, used_at=None).all():
        old.used_at = now
    raw = secrets.token_urlsafe(32)
    db.session.add(AuthToken(user_id=user.id, purpose=purpose, token_hash=_digest(raw), expires_at=now + lifetime))
    db.session.commit()
    return raw


def find_token(raw: str | None, purpose: str) -> AuthToken | None:
    """A valid (unused, unexpired) token of that purpose, or None. Does not consume it."""
    if not raw or len(raw) > 200:
        return None
    tok = AuthToken.query.filter_by(token_hash=_digest(raw), purpose=purpose).first()
    if tok is None or tok.used_at is not None or tok.expires_at < utcnow():
        return None
    return tok


def consume_token(tok: AuthToken) -> None:
    tok.used_at = utcnow()


# ---------------------------------------------------------------------------------------------------------
# login throttling (database backed, so it works across processes)
# ---------------------------------------------------------------------------------------------------------
def record_attempt(email: str, succeeded: bool) -> None:
    db.session.add(LoginAttempt(email_hash=email_hash(email), ip=client_ip(), succeeded=succeeded))
    # opportunistic cleanup of old rows keeps the table small
    db.session.query(LoginAttempt).filter(LoginAttempt.created_at < utcnow() - timedelta(days=2)).delete(synchronize_session=False)
    db.session.commit()


def is_locked_out(email: str) -> bool:
    """Too many recent failures for this email from this address, or from this address overall."""
    cfg = current_app.config
    since = utcnow() - timedelta(minutes=cfg["LOGIN_WINDOW_MINUTES"])
    ip = client_ip()
    per_email = (LoginAttempt.query.filter(LoginAttempt.email_hash == email_hash(email), LoginAttempt.ip == ip,
                                           LoginAttempt.succeeded.is_(False), LoginAttempt.created_at >= since).count())
    if per_email >= cfg["LOGIN_MAX_FAILURES"]:
        return True
    per_ip = LoginAttempt.query.filter(LoginAttempt.ip == ip, LoginAttempt.succeeded.is_(False), LoginAttempt.created_at >= since).count()
    return per_ip >= cfg["LOGIN_IP_MAX_FAILURES"]


def clear_failures(email: str) -> None:
    db.session.query(LoginAttempt).filter(LoginAttempt.email_hash == email_hash(email), LoginAttempt.ip == client_ip(),
                                          LoginAttempt.succeeded.is_(False)).delete(synchronize_session=False)
    db.session.commit()


def safe_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())
