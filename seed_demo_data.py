"""Seed clearly marked DEMO data (fictional doctors, cases, genuine model predictions, reports, notifications).

    python seed_demo_data.py            # insert (safe to run again: nothing is duplicated)
    python seed_demo_data.py --reset    # remove ONLY the demo data (accounts with demo.*@example.com and everything they own)

Safety
- Uses the same database as the app (DATABASE_URL, else instance/neuroinsight.sqlite3) and makes a backup copy first.
- Only creates or removes rows carrying the demo marker: accounts `demo.<name>@example.com`, cases `DEMO-CASE-nnn`.
  Existing accounts (including administrators) are never modified.
- Predictions are genuine Phase B model outputs for the 8 demo samples from the approved VALIDATION split. The locked test split
  is never read. Nothing about the model, the bundle or the dataset manifest is changed.
- All inserts happen in one transaction; any error rolls everything back.
- Doctor passwords are random, stored only as argon2 hashes. The plain temporary passwords are written once to
  instance/demo_credentials.txt (git-ignored) and are not printed.

The schema has no Doctor role, no specialty, no patient entity and only open/archived case status, so: doctors are ordinary `user`
accounts with occupation "Clinician" (specialty is written into the institution text); each "patient" is one case (the fictional
patient reference is in the case notes); New/In progress map to "open" and Completed to "archived" (the label is kept in the notes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEMO_DOMAIN = "example.com"
DEMO_EMAIL_PREFIX = "demo."
DEMO_CASE_PREFIX = "DEMO-CASE-"
CREDENTIALS_FILE = ROOT / "instance" / "demo_credentials.txt"

DOCTORS = (
    ("Aarav", "Mehta", "Neurologist", "NeuroCare Research Centre"),
    ("Ananya", "Rao", "Neurosurgeon", "Brain Health Institute"),
    ("Rohan", "Kapoor", "Epileptologist", "Advanced Neuro Imaging Centre"),
    ("Meera", "Sharma", "Radiologist", "NeuroCare Research Centre"),
    ("Vikram", "Nair", "Neurologist", "Brain Health Institute"),
)
N_CASES = 15
STATUS_LABELS = ("New", "In progress", "Completed")      # case i gets STATUS_LABELS[i % 3]


def doctor_email(first: str, last: str) -> str:
    return f"{DEMO_EMAIL_PREFIX}{first.lower()}.{last.lower()}@{DEMO_DOMAIN}"


def demo_hex(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def temp_password() -> str:
    return f"{secrets.token_urlsafe(9)}-{secrets.token_urlsafe(9)}!"


# ------------------------------------------------------------------------------------------------------ reset
def reset(appmod, db) -> None:
    from webapp.models import AuditLog, AuthToken, Case, Notification, Prediction, Report, User
    users = User.query.filter(User.email.like(f"{DEMO_EMAIL_PREFIX}%@{DEMO_DOMAIN}"), User.role == "user").all()
    ids = [u.id for u in users]
    counts = {}
    if ids:
        for model in (Notification, Report, Prediction, Case, AuthToken):          # children first (SQLite FK checks may be off)
            counts[model.__tablename__] = model.query.filter(model.user_id.in_(ids)).delete(synchronize_session=False)
        AuditLog.query.filter(AuditLog.target_id.in_(ids)).delete(synchronize_session=False)
        counts["users"] = User.query.filter(User.id.in_(ids)).delete(synchronize_session=False)
    db.session.commit()
    CREDENTIALS_FILE.unlink(missing_ok=True)
    print("Removed demo data:", counts or "nothing found")


# ------------------------------------------------------------------------------------------------------ seed
def seed(appmod, db) -> dict:
    import reports as rpt
    from webapp import security
    from webapp.models import Case, Notification, Prediction, Report, User, utcnow

    summary = dict(doctors=0, cases=0, predictions=0, reports=0, notifications=0)
    new_credentials: list[tuple[str, str]] = []

    # --- doctors
    doctors: list[User] = []
    for first, last, specialty, institution in DOCTORS:
        email = doctor_email(first, last)
        user = User.query.filter_by(email=email).first()
        if user is None:
            password = temp_password()
            user = User(email=email, password_hash=security.hash_password(password), full_name=f"Dr. {first} {last} (DEMO)",
                        institution=f"{institution} - {specialty} - DEMO DATA", occupation="Clinician", role="user",
                        email_verified_at=utcnow(), terms_accepted_at=utcnow())
            db.session.add(user)
            new_credentials.append((email, password))
            summary["doctors"] += 1
        doctors.append(user)
    db.session.flush()

    # --- cases (one per fictional patient)
    samples_meta = json.loads((ROOT / "static" / "samples" / "samples.json").read_text(encoding="utf-8"))
    samples = samples_meta["samples"]
    now = utcnow().replace(minute=0, second=0, microsecond=0)
    cases: list[tuple[Case, str]] = []
    for i in range(1, N_CASES + 1):
        code = f"{DEMO_CASE_PREFIX}{i:03d}"
        doctor = doctors[(i - 1) % len(doctors)]
        label = STATUS_LABELS[(i - 1) % 3]
        sample = samples[(i - 1) % len(samples)]
        created = now - timedelta(days=32 - 2 * i, hours=i)
        case = Case.query.filter_by(user_id=doctor.id, case_code=code).first()
        if case is None:
            case = Case(user_id=doctor.id, case_code=code, title=f"Patient Demo {i:03d} (DEMO DATA)", status="archived" if label == "Completed" else "open",
                        notes=(f"DEMO DATA - fictional record, not a real patient.\nPatient reference: DEMO-PAT-{i:03d}\nCase status (demo label): {label}\n"
                               f"MRI study: demo slice {sample['file']} from the approved validation split.\n"
                               f"Dataset reference label for this demo image: {sample['dataset_label']}."),
                        created_at=created, updated_at=created + timedelta(days=1 if label != "New" else 0))
            db.session.add(case)
            summary["cases"] += 1
        cases.append((case, sample["file"]))
    db.session.flush()

    # --- genuine predictions (one per case) and reports, using the real Phase B model on validation demo samples
    pending = [(i, c, f) for i, (c, f) in enumerate(cases, 1)
               if Prediction.query.filter_by(public_id=demo_hex("pred", i)[:32]).first() is None]
    results: dict[str, object] = {}
    predictor = None
    if pending or any(Report.query.filter_by(prediction_id=demo_hex("pred", i)[:32]).first() is None for i in range(1, N_CASES + 1) if (i - 1) % 3 != 0):
        predictor = appmod.get_predictor()
        info = predictor.describe()
        for _, _, fname in {(0, 0, s["file"]) for s in samples}:
            results[fname] = predictor.predict_bytes((ROOT / "static" / "samples" / fname).read_bytes())   # deterministic: infer each sample once
    for i, (case, fname) in enumerate(cases, 1):
        pid = demo_hex("pred", i)[:32]
        pred = Prediction.query.filter_by(public_id=pid).first()
        created = case.created_at + timedelta(hours=2)
        if pred is None:
            res = results[fname]
            warn = appmod.confidence_warning(res.confidence)
            pred = Prediction(public_id=pid, user_id=case.user_id, case_id=case.id, file_name=f"DEMO-{fname}", predicted_class=res.label,
                              confidence=float(res.confidence), probabilities={k: float(v) for k, v in res.probabilities.items()},
                              warning_level=warn["level"], model_estimator_sha256=info["estimator_sha256"], created_at=created)
            db.session.add(pred)
            summary["predictions"] += 1
        label = STATUS_LABELS[(i - 1) % 3]
        if label != "New" and Report.query.filter_by(prediction_id=pid, user_id=case.user_id).first() is None:
            res = results[fname]
            warn = appmod.confidence_warning(res.confidence)
            when = created + timedelta(hours=3)
            db.session.add(Report(
                report_id=f"NI-{when:%Y%m%d}-{demo_hex('report', i)[:8].upper()}", prediction_id=pid, user_id=case.user_id, case_id=case.id,
                file_name=f"DEMO-{fname}", predicted_class=res.label, confidence=float(res.confidence),
                probabilities={k: float(v) for k, v in res.probabilities.items()}, warning_level=warn["level"],
                warning_message=warn["message"], created_at=when.strftime(rpt.TS_FORMAT), model_json=appmod._model_details(predictor)))
            summary["reports"] += 1
    db.session.flush()

    # --- notifications (clearly marked, a few left unread)
    def add_note(user, kind, title, body, href, when, read):
        if Notification.query.filter_by(user_id=user.id, title=title).first() is None:
            db.session.add(Notification(user_id=user.id, kind=kind, title=title, body=body, href=href, created_at=when,
                                        read_at=when + timedelta(hours=1) if read else None))
            summary["notifications"] += 1

    for i, (case, fname) in enumerate(cases, 1):
        doctor = next(d for d in doctors if d.id == case.user_id)
        label = STATUS_LABELS[(i - 1) % 3]
        href = f"/app/cases/{case.id}"
        read = i <= 9
        add_note(doctor, "account", f"[DEMO] New demo case created: {case.case_code}", "Demo activity, not a real event.", href, case.created_at, read)
        add_note(doctor, "account", f"[DEMO] Case {case.case_code} assigned to you", "Demo activity, not a real event.", href, case.created_at + timedelta(minutes=5), read)
        add_note(doctor, "analysis", f"[DEMO] MRI analysis completed for {case.case_code}", "Demo activity: genuine model output on a validation demo sample.",
                 "/app/history", case.created_at + timedelta(hours=2), read)
        if label != "New":
            add_note(doctor, "report", f"[DEMO] Demo report generated for {case.case_code}", "Demo activity, not a real event.", "/app/reports",
                     case.created_at + timedelta(hours=5), read)

    if new_credentials:                                                    # written once, never printed
        CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = CREDENTIALS_FILE.read_text(encoding="utf-8") if CREDENTIALS_FILE.exists() else (
            "DEMO doctor accounts (fictional). Temporary random passwords. Delete this file when no longer needed.\n")
        CREDENTIALS_FILE.write_text(existing + "".join(f"{e}  {p}\n" for e, p in new_credentials), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reset", action="store_true", help="remove only the demo data")
    args = parser.parse_args()

    import app as appmod
    from webapp.extensions import db
    uri = appmod.app.config["SQLALCHEMY_DATABASE_URI"]
    print("Database:", uri.split("///", 1)[-1])
    with appmod.app.app_context():
        path = Path(db.engine.url.database) if db.engine.url.database else None
        if path and path.is_file():
            backup = path.with_name(path.stem + ".before-demo-seed" + path.suffix)
            if not backup.exists():
                shutil.copy2(path, backup)
                print("Backup:", backup.name)
        try:
            if args.reset:
                reset(appmod, db)
            else:
                summary = seed(appmod, db)
                db.session.commit()
                print("Inserted this run:", summary)
                print("Credentials file:", CREDENTIALS_FILE if CREDENTIALS_FILE.exists() else "(none created this run)")
        except Exception:
            db.session.rollback()
            print("ERROR: rolled back, nothing was changed.", file=sys.stderr)
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
