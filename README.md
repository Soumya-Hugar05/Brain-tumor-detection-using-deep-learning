# NeuroInsight – Brain MRI Classification (research prototype)

> **Research prototype.** NeuroInsight is not a medical device or diagnostic tool and must not be used to
> diagnose, treat or make decisions about any patient. Results are for research demonstration only.

NeuroInsight classifies a 2D brain MRI slice as **glioma**, **meningioma**, **pituitary tumor** or **no tumor**.
The web app (Flask) serves the **Phase B selected SVM bundle**: frozen ImageNet ResNet50 features feeding an
RBF support-vector machine.

## How the current app works

| Step | What happens |
|---|---|
| Model loading | `inference.Predictor` loads the bundle at `inference.MODEL_BUNDLE_DIR` (the single configuration constant; default `runs/20261105-184406_phaseB_frozen_improvements/models/improved_svm`, overridable with `NEUROINSIGHT_BUNDLE_DIR`, see "Getting the Phase B bundle"). The estimator's SHA-256 is checked against `bundle.json` on load. |
| Preprocessing | Exactly the bundle's `resize` mode: TensorFlow decode to RGB, bilinear resize to 224×224 (no antialiasing, no cropping). |
| Features | Frozen ImageNet ResNet50, global average of `conv5_block3_out` → 2048-d vector (`neuroinsight.features.FeatureExtractor`). No flip averaging (read from the bundle). |
| Classifier | RBF SVC (C=1, `gamma="scale"`, balanced class weights) with built-in Platt probabilities. |
| Decision rule | `argmax_proba` – the class with the highest probability (read from the bundle). |

The app code path was verified against the evaluation pipeline on all 1,046 validation images: features, probabilities
and labels were bit-for-bit identical (`verify_app_pipeline.py`).

Validation results of this model (1,046 images; see `runs/20261105-184406_phaseB_frozen_improvements/results/`):
accuracy 0.9216, macro-F1 0.918, tumor-vs-no-tumor sensitivity 98.5% and specificity 98.5%. Meningioma is the
weakest class (recall about 0.80). These numbers come from the **validation** split, which was also used to choose
the model, so they are optimistic for new data. The held-out test split has **not** been evaluated.

### Confidence warnings

The predicted class and all class probabilities are always shown; predictions are never rejected. A warning is
added based on the top-class probability:

| Top-class probability | Message |
|---|---|
| ≥ 0.90 | none |
| 0.70 to < 0.90 | "Review recommended – prediction has uncertainty." |
| < 0.70 | "Low confidence – expert review strongly recommended." |

These are research-demo thresholds, not clinical safety cutoffs. They were checked against the validation split
(`analyze_confidence_threshold.py`): predictions with confidence ≥ 0.90 were 98.2% accurate and covered 77.5% of
images; lower-confidence predictions were much less reliable.

### Grad-CAM: currently disabled

Grad-CAM is **disabled** in the app until the Phase D correction is complete and the explanation quality has been
validated. The old CNN heatmap is no longer used. The API returns `explanation.available = false` with a reason, and
`renderExplanation()` in `static/js/app.js` is the hook where a validated explanation can be added later.

### Web app features

- Upload by click or drag-and-drop, with a live image preview.
- Prediction without a page reload (`fetch`), a loading spinner and clear error messages.
- **Upload validation**, on both the browser and the server: extensions JPG/JPEG/PNG/BMP only, at most 10 MB,
  not empty. The server also checks the real file format (not just the name), that the image can be read, and that it is
  at least 64 px on its shorter side and at most 25 megapixels. Invalid or corrupt files return a JSON error message.
- **Demo samples:** 8 images (2 per class), drawn at random (seed 1337) from the approved **validation** split only
  and labelled as demo samples (`make_demo_samples.py`, `static/samples/`). They are not test data and the model may
  get some of them wrong.
- **About the model** panel: architecture, preprocessing, the verified Phase B validation metrics, the warning tiers
  and the limitations.
- **Downloads:** JSON (prediction, probabilities, warning, model hashes, timestamp, disclaimer) and PNG (original image,
  prediction details, warning and disclaimer; no Grad-CAM).
- A research-prototype disclaimer is shown at the top of every page and repeated in the footer, the JSON and the PNG.
- Dark and light themes (saved to your account), responsive down to phone width.

### Reports and downloads

After an analysis, press **Save to Reports** (or **Download PDF**, which saves first) to keep a record in the report
history. Open it from the **Reports** page (`/app/reports`, in the sidebar):

- **Statistics** (total, generated today, generated this month) are real counts from the stored reports, using your local day and month.
- **Table:** report ID, case (the case code, or "—" if the analysis is not in a case), MRI image file name, predicted class,
  confidence, date generated and actions. Search by file name or report ID, filter by class, case and date range, sort newest/oldest,
  and page through the results. Empty states, loading indicators and success/error notifications are included.
- **View report** opens a preview (report information, prediction, four probabilities, confidence warning, model information and
  the research disclaimer). **Download PDF / JSON / PNG** work from the table and the preview; **Export CSV** saves the filtered list
  (metadata only). **Delete** asks for confirmation first.
- The JSON and PNG buttons on the analysis page are unchanged (the PNG there still includes the image because it is made in your
  browser). The PDF, and the JSON/PNG downloaded from the Reports page, are rendered by the server from the stored record.

**What is stored** (the `reports` table of the main application database, `instance/neuroinsight.sqlite3` by default, ignored by Git): report ID,
prediction ID, a sanitised image file name, predicted class, confidence, the four probabilities, the confidence warning, the
timestamp and the model description and hashes. **Not stored:** the MRI image, patient or case information, PDFs or PNGs (generated on
demand). Because "original image included" is not possible without keeping images, reports show a placeholder instead; keeping
images would raise privacy and retention questions (consent, de-identification, how long to keep them) that this prototype avoids.
File names can themselves contain identifiers, so use the Delete button to remove reports you no longer need. Saving the same analysis
again never creates a second report (each analysis has one prediction ID, and only results the server itself produced can be saved).

Reports belong to the signed-in user. Other users, administrators included, cannot list, open, download, search, export or delete them. A report takes the case of the analysis it was saved from and follows it if you move the analysis to another case; deleting a case keeps its reports.

**Reports saved before accounts existed.** Earlier versions kept reports in a separate file, `instance/reports.sqlite3`, with no owner. They are visible to nobody until you import them into one account (the old file is only read, never changed; rows already imported are skipped):

```powershell
.\.venv\Scripts\python.exe -m flask --app webapp import-legacy-reports --email you@example.com
```

### API

| Route | Purpose |
|---|---|
| `GET /` | the public landing page (the analysis UI is at `/app/analysis`, signed-in users only) |
| `POST /api/predict` | multipart upload, field `file` (optional `case_id` of one of your open cases) → prediction JSON (or `{ok:false, error}`) |
| `GET /api/model-info` | verified Phase B metrics, architecture and limitations |
| `GET /api/samples` | demo samples |
| `GET /app/reports` | the Reports page (`/reports` redirects here) |
| `POST /api/reports` | save a just-made prediction (`{"prediction_id": ...}`) as a report; idempotent |
| `GET /api/reports` | list reports (`q`, `cls`, `case` = a case id or `none`, `date_from`, `date_to`, `sort`, `page`, `page_size`, `tz_offset`) plus statistics |
| `GET /api/reports/export` | CSV of the filtered list (metadata only) |
| `GET /api/reports/<id>` · `DELETE /api/reports/<id>` | one report · delete it |
| `GET /api/reports/<id>/download/<pdf\|json\|png>` | download a report file (file name fixed by the server) |
| `POST /api/account/theme` | save the signed-in user's theme (`{"theme": "light"}` or `{"theme": "dark"}`) |

All `/api/...` routes except `/api/model-info` and `/api/samples` need a signed-in user and, for writes, a CSRF token.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Required versions

The Phase B estimator is a pickled scikit-learn object and its features come from a TensorFlow model, so use the same
major libraries it was built and tested with (`runs/20261105-184406_phaseB_frozen_improvements/config.json` records the
exact versions). `requirements.txt` only gives minimum versions; for faithful results install these:

| Library | Version used |
|---|---|
| Python | 3.13.9 |
| scikit-learn | **1.8.0** (the estimator was pickled with it; other versions may fail to load or change results) |
| TensorFlow / Keras | **2.21.0 / 3.14.0** (feature extractor) |
| NumPy / SciPy | 2.4.x / 1.17.1 |
| OpenCV (`opencv-python`) | 4.13.0.92 |
| Pillow | 12.0.0 – 12.2.0 |
| joblib | 1.5.3 |
| Flask | 3.1.x |

### First run: ImageNet weights download

The first time the app (or any script) builds the feature extractor, Keras downloads the ResNet50 ImageNet weights
(about 90 MB) into `~/.keras/models/`. This needs an internet connection once; after that the app works offline.

## Getting the Phase B bundle

> **Status: the Phase B bundle release is pending.** The GitHub Release described below has **not been published
> yet**, so there is currently nothing to download from this repository. Until the maintainer publishes it, the
> bundle exists only in the maintainer's local `runs/` folder and the app cannot be started from a fresh clone. The
> steps below describe the planned procedure.

The trained Phase B model (`estimator.joblib`, about 41 MB) is **not stored in Git** and never will be. The small
bundle metadata (`bundle.json`), run configurations and result reports under `runs/` are not excluded by `.gitignore`,
so they can be committed, but at the time of writing they are **untracked** (not yet committed). The estimator file
has to be obtained separately, or the app reports "Model bundle not found" instead of starting.

**Planned distribution: GitHub Release asset (pending).**

1. Once the release is published, download `neuroinsight-phaseB-bundle.zip` from this repository's GitHub Releases
   (planned tag `phase-b-bundle-v1`).
2. Check the archive. In PowerShell, from the folder that holds the download:

   ```powershell
   Get-FileHash .\neuroinsight-phaseB-bundle.zip -Algorithm SHA256
   ```

   It must equal `c8c481e606c7aea91484bb93218881fa9155481b3a620e0608dcf9737ca047ba`. If it does not, do not use the file.
3. Extract it so that the files end up in the project's `runs/` folder (the archive already contains the
   `improved_svm/` folder):

   ```powershell
   Expand-Archive .\neuroinsight-phaseB-bundle.zip -DestinationPath <project>\runs\20261105-184406_phaseB_frozen_improvements\models
   ```

   The result must be `runs/20261105-184406_phaseB_frozen_improvements/models/improved_svm/bundle.json` and `estimator.joblib`.
4. Start the app. On load it re-checks the estimator against the hash stored in `bundle.json` and refuses a modified file.

`joblib` files are Python pickles: only load a bundle from a source you trust, and always compare the hashes below.

**Expected hashes**

| File | SHA-256 |
|---|---|
| `neuroinsight-phaseB-bundle.zip` | `c8c481e606c7aea91484bb93218881fa9155481b3a620e0608dcf9737ca047ba` |
| `improved_svm/estimator.joblib` | `f15312b456c32391457319d420501df768e44165f855538c696abedf744bb1a8` |
| `improved_svm/bundle.json` | `407c93f2bbaaf7063794b4869b0c8d36f4ea8086f6c32d7e7c367d593c894fd5` |
| data manifest `data_splits/manifest.csv` | `4d09b01d782fa076890f32bf30bcbbe1a2319c135ead2e58cbd5e1f9ec0d930f` |

The ZIP hash is that of the archive prepared locally for the (still unpublished) release (`neuroinsight-phaseB-bundle.zip`, 21,574,104 bytes, two files:
`improved_svm/bundle.json` and `improved_svm/estimator.joblib`). It is only valid for that exact file; if the archive is
ever rebuilt, update this README.

### Using a different location: `NEUROINSIGHT_BUNDLE_DIR`

By default the app loads `runs/20261105-184406_phaseB_frozen_improvements/models/improved_svm` (the constant
`DEFAULT_BUNDLE_DIR` in `inference.py`). To load the same bundle from another folder, set the environment variable to
the folder that contains `bundle.json` and `estimator.joblib` (absolute path, or relative to the project folder):

```powershell
$env:NEUROINSIGHT_BUNDLE_DIR = "D:\models\improved_svm"
python app.py
```

If the folder is missing or incomplete, the app starts but returns a clear "Model bundle not found" error and tells you
where it looked. The override is meant for the Phase B bundle: the app's "About the model" panel reads
`results/phase_b.json` from two levels above the bundle folder and the model name in the API says "Phase B", so the
panel will show "metrics unavailable" for a bundle stored elsewhere, and the labels would be wrong for any other model.
`tests/test_app.py` also assumes the Phase B bundle.

### Packaging instructions (for maintainers; not run automatically)

The archive is built once and published unchanged; do not rebuild it for each release, because the same inputs can give
different bytes with a different zlib version. To (re)create it from the project folder:

```python
import zipfile
from pathlib import Path
src = Path("runs/20261105-184406_phaseB_frozen_improvements/models/improved_svm")
with zipfile.ZipFile("neuroinsight-phaseB-bundle.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name in ("bundle.json", "estimator.joblib"):      # fixed order and timestamps
        zi = zipfile.ZipInfo(f"improved_svm/{name}", date_time=(2026, 11, 5, 21, 10, 0))
        zi.compress_type, zi.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
        z.writestr(zi, (src / name).read_bytes(), compresslevel=9)
```

Then compute the ZIP's own checksum (`Get-FileHash neuroinsight-phaseB-bundle.zip -Algorithm SHA256`), put it in the table above
and in the release notes, and upload the archive as a release asset, for example with
`gh release create phase-b-bundle-v1 neuroinsight-phaseB-bundle.zip --title "Phase B bundle" --notes "SHA-256: <zip hash>"`.
The ZIP checksum is not the estimator's checksum; they are different files and must not be mixed up.

## Run the web app

```bash
python app.py
```

Open `http://127.0.0.1:5000/`, create an account (or an administrator, see "Accounts") and confirm the email address. The app needs the Phase B bundle (see above); do not move or delete it. Set the
environment variable `NEUROINSIGHT_DEBUG=1` to use Flask's debug mode (its reloader loads TensorFlow twice).

## Dashboard and navigation

After signing in you land on the **dashboard** (`/app`) inside a fixed sidebar: Dashboard, MRI Analysis, Patients / Cases, Prediction History, Reports,
Model Performance, Notifications, My Profile, Settings and Logout (the sidebar becomes a drawer on small screens; administrators also get an
Administration section). Every number and list on the dashboard comes from your own records: total analyses (each successful analysis is recorded as
metadata only, never the image), total cases, reports generated, and activity in the last 7 days, plus your recent analyses, cases, reports and an
activity feed. Nothing on any page is invented; an area with no data shows an empty state.

### Cases and prediction history

- **Cases** (`/app/cases`): a case is a code you choose (letters, digits, spaces and `. _ - /`, up to 40 characters, unique per user), an optional title and
  optional notes, and it can be open or archived. **No patient fields exist and the forms ask you not to enter identifying information**; the notes are
  free text, so that rule is yours to keep. Create, search, edit, archive, reopen and delete cases. Deleting a case keeps its analyses and reports.
- **Prediction History** (`/app/history`): all your analyses (file name, result, confidence, time) with search, filters for result and case, paging,
  and a per-row control to put an analysis in a case or take it out. The analysis page has an optional **Add to case** list of your open cases.
- A case page lists its analyses and reports. Everything is scoped to the signed-in user; another user's case or analysis answers 404.

### Profile, settings and notifications

- **My Profile** (`/app/profile`): edit your name, institution and occupation. The email address and role cannot be changed here, and there is no profile
  picture (that would mean storing uploaded files).
- **Settings** (`/app/settings`): theme (saved to your account and applied on every device), whether you are notified about completed analyses and saved
  reports, **change password** (needs the current password; signs out every other device and sends an email), and **sign out of other devices**.
- **Notifications** (`/app/notifications`): created only for real events (analysis completed, report saved, password changed or reset, other devices
  signed out, email confirmed). The bell shows the unread count. Mark one or all as read, or clear them; the newest 200 are kept per user.
  Security notices cannot be switched off.

### Administration

Administrators (role `admin`, created with the `create-admin` command) get an **Administration** section: an overview of account counts, a searchable
**Users** list and an **audit log**. They can **deactivate** an account (the user is signed out at once and cannot sign in until reactivated; data is
kept), **reactivate** it, or **end all its sessions**. They cannot deactivate themselves or the last active administrator, and roles are never changed
in the web UI. Every action is written to the audit log (who, what, which account, when, IP address).

**Administrators manage accounts only.** They cannot see any user's cases, analyses, reports or notifications: the case, history and report routes
are scoped to the signed-in user whatever their role, no admin code touches those tables, and tests check this.

## Accounts, sessions and security

NeuroInsight has real accounts. The landing page, sign-up/sign-in pages, the draft terms and privacy notice and `/api/model-info`
are public; MRI analysis, reports and every other `/app/...` page require a signed-in user with a **confirmed email address**.

| Area | How it works |
|---|---|
| Passwords | argon2id hashes only. Server-side policy: 10+ characters, not common, not containing your name or email name, and three character classes (or 16+ characters). |
| Sessions | Signed cookie, `HttpOnly`, `SameSite=Lax`, `Secure` in production. 8-hour idle timeout; "Remember me" keeps a 14-day cookie. A per-user `session_version` lets a password change or reset, "sign out of other devices" and an administrator's deactivation end sessions at once. |
| CSRF | Every POST/PUT/DELETE needs a token (forms and the front-end's `fetch` calls). |
| Sign-in throttling | 5 failed attempts per address and email lock sign-in for 15 minutes (stored in the database; the email only as a hash). Sign-up, forgot-password and resend are rate limited per client. |
| Email links | Verification (24 h) and reset (60 min) tokens are random, single use, stored only as hashes, and never written to the access log. Opening a link first shows a confirm button, so mail scanners cannot use it up. Responses never reveal whether an address is registered. |
| Roles | `user` (default) and `admin`. Self-declared occupation at sign-up is only a label. Roles are read from the database on every request and constrained by the schema. |
| Headers | CSP with a per-request script nonce, frame denial, `no-referrer` on token pages, `no-store` for signed-in pages. |

**First run (development).** `python app.py` applies pending database migrations automatically and keeps a random development secret
key in `instance/secret_key`. Create the first administrator (no default credentials exist):

```powershell
.\.venv\Scripts\python.exe -m flask --app webapp create-admin --email you@example.com --name "Your Name"
```

The password is requested through a hidden prompt (or read from `NEUROINSIGHT_ADMIN_PASSWORD` for scripted setups).

**Email.** Without a mail server, messages are written to `instance/outbox/` (git-ignored); open the link from that text file. Treat the folder like a
password store and do not use it in production. For real delivery set `NEUROINSIGHT_MAIL_BACKEND=smtp` with `MAIL_SERVER`, `MAIL_PORT`,
`MAIL_USERNAME`, `MAIL_PASSWORD`, `MAIL_USE_TLS` and `MAIL_FROM`, and `APP_BASE_URL` for the public address used in links.

**Database and migrations.** Default: SQLite in `instance/neuroinsight.sqlite3`. Set `DATABASE_URL` for another database (for example
PostgreSQL, after installing a driver such as `psycopg`). Schema changes are Alembic migrations in `migrations/`:

```powershell
.\.venv\Scripts\python.exe -m flask --app webapp db upgrade        # apply migrations (done automatically in development)
.\.venv\Scripts\python.exe -m flask --app webapp db migrate -m "describe the change"   # after changing webapp/models.py
```

**Production checklist.** `NEUROINSIGHT_ENV=production` refuses to start without `SECRET_KEY` (32+ characters) and turns on secure cookies. Serve over HTTPS
behind a reverse proxy and set `TRUSTED_PROXY_COUNT`; use a persistent `RATELIMIT_STORAGE_URI` (for example Redis) when running several workers;
run `db upgrade` as a deployment step; configure SMTP. The prediction cache and report history of this prototype are per process/file and are not
designed for multi-server deployment. Reports now live in the main database, but the short-lived prediction cache that feeds "Save to Reports" is still in memory per process.

**Privacy.** No MRI images and no patient data are stored. Accounts hold name, email, optional institution/occupation, a password hash, display preferences and timestamps. Cases hold only the code, title and notes you type; history, reports and notifications hold model output and event text, never images. The
terms and privacy pages are **drafts** that describe the software's behaviour; they need legal or ethics review before public use.

## Tests and verification

```bash
python tests/test_app.py            # web app: validation, tiers, samples, API == evaluation pipeline
python tests/test_neuroinsight.py   # package: manifest, metrics, SVM / feature equivalence
python tests/test_reports.py       # Reports module: storage, search/filter/sort/paging, downloads, security (slow: runs the model)
python tests/test_landing.py       # public landing page
python tests/test_auth.py          # accounts: sign-up, verification, sign-in/out, reset, CSRF, roles, user isolation
python tests/test_dashboard.py     # shell, sidebar, dashboard statistics, analysis recording, model performance page
python tests/test_cases.py         # cases, prediction history, reports following their case
python tests/test_account.py       # profile, settings, password change, sessions, notifications
python tests/test_admin.py         # admin area: account management, audit log, no access to user content
python verify_app_pipeline.py       # app path vs evaluation path on the validation split
```

These use the validation split only. The test split is locked and is only read by `evaluate.py --split test --final`
after the selection is frozen and approved.

## Data and research pipeline

`dataset/` holds the original Kaggle images. Training and evaluation use the audited split in `data_splits/`
(`manifest.csv`, SHA-256 `4d09b01d…0d930f`): train 4,877 / val 1,046 / test 1,045, with 232 images excluded (label-conflict
copies and pre-augmented images). The audit is described in `data_splits/audit_report.md`. There are no patient IDs, so
there is no patient-level separation between splits and some near-duplicate leakage remains (documented in that report).

Experiment scripts (results are stored under `runs/`):

- `audit_dataset.py` – dataset audit and split construction
- `run_phase_a.py` – baseline model
- `run_phase_b.py` – frozen-feature improvements (selects the bundle served by the app)
- `run_phase_d.py`, `run_phase_d_correction.py` – SVM Grad-CAM experiments (correction not yet complete; not used by the app)
- `evaluate.py` – validation evaluation, selection freezing and the guarded final test evaluation
- `analyze_confidence_threshold.py` – validation-only analysis behind the warning tiers

## Legacy (not used by the web app)

The original pipeline is kept for reference only: `train.py` trains a ResNet50 + SVM on the original
`dataset/Training` folder (not the audited split) and writes `artifacts/svm.joblib`, `artifacts/class_names.json` and
`artifacts/cnn_classifier.keras` (a CNN head used for the old Grad-CAM). It reads the original `dataset/Training` and
`dataset/Testing` folders directly, not the audited split. The current app does **not** load `artifacts/`.
`gradcam.py` is the old heatmap code (only the Phase D experiment scripts still import it, for comparison),
`ml_utils.py` holds small shared helpers used by `train.py` and `audit_dataset.py`, and `generate_report.py` builds the
original project report.
