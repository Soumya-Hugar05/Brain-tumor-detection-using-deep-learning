# Brain Tumor MRI Detection (ResNet + SVM + Grad-CAM + Flask)

This project uses:
- **ResNet50 (CNN)** for **feature extraction**
- **SVM** for **classification**
- **Grad-CAM** for **explainability**
- **Flask** for **web deployment**

## Dataset
Place the Kaggle dataset in:

```
dataset/
  Training/
    <class folders>/
  Testing/
    <class folders>/
```

Your workspace already contains `dataset/Training` and `dataset/Testing`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train (creates artifacts)

```bash
python train.py
```

Artifacts are saved to `artifacts/`:
- `svm.joblib` (scaler + SVM)
- `class_names.json`
- `cnn_classifier.keras` (for Grad-CAM only)

## Run web app

```bash
python app.py
```

Open `http://127.0.0.1:5000/`.

