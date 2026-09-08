"""
PredictIQ — ML Prediction Backend
====================================
XGBoost churn prediction model with SHAP explainability.
Includes training pipeline, model serving, and SHAP waterfall charts.

Usage:
    python predictiq.py --train   # train and save model
    python predictiq.py --serve   # start FastAPI server

Requirements:
    pip install scikit-learn xgboost shap pandas numpy fastapi uvicorn joblib
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
META_PATH  = MODEL_DIR / "model_meta.json"

# ── Feature config ────────────────────────────────────────────────────────────
FEATURE_NAMES = ["age", "tenure", "monthly_charges", "support_calls", "contract_encoded", "satisfaction"]
FEATURE_DISPLAY = {
    "age":              "Customer Age",
    "tenure":           "Tenure (months)",
    "monthly_charges":  "Monthly Charges ($)",
    "support_calls":    "Support Calls",
    "contract_encoded": "Contract Type",
    "satisfaction":     "Satisfaction Score",
}

CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION  (replace with pd.read_csv("your_data.csv") for real data)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic churn dataset.
    In production, replace this with your actual customer data.
    """
    rng = np.random.default_rng(seed)

    age             = rng.integers(18, 80, n)
    tenure          = rng.integers(0, 120, n)
    monthly_charges = rng.uniform(20, 150, n).round(2)
    support_calls   = rng.poisson(2, n).clip(0, 15)
    contract        = rng.choice([0, 1, 2], n, p=[0.55, 0.25, 0.20])  # 0=m2m, 1=1yr, 2=2yr
    satisfaction    = rng.integers(1, 6, n)

    # Probabilistic churn label (mimics real-world patterns)
    log_odds = (
        -3.0
        + 0.4  * (support_calls / 10)
        - 0.6  * (tenure / 120)
        - 0.5  * ((satisfaction - 1) / 4)
        + 0.6  * (contract == 0).astype(float)
        - 0.4  * (contract == 2).astype(float)
        + 0.3  * (monthly_charges > 100).astype(float)
        + 0.02 * rng.standard_normal(n)   # noise
    )
    prob  = 1 / (1 + np.exp(-log_odds))
    churn = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "age":              age,
        "tenure":           tenure,
        "monthly_charges":  monthly_charges,
        "support_calls":    support_calls,
        "contract_encoded": contract,
        "satisfaction":     satisfaction,
        "churn":            churn,
    })


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_model(data: Optional[pd.DataFrame] = None) -> dict:
    """
    Full training pipeline with cross-validation, hyperparameter tuning,
    and SHAP analysis.
    """
    MODEL_DIR.mkdir(exist_ok=True)

    if data is None:
        print("Generating synthetic dataset (10,000 samples)...")
        data = generate_synthetic_data()

    X = data[FEATURE_NAMES]
    y = data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Churn rate: {y.mean():.1%}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    cv_scores   = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="roc_auc")

    metrics = {
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred), 4),
        "recall":           round(recall_score(y_test, y_pred), 4),
        "f1":               round(f1_score(y_test, y_pred), 4),
        "roc_auc":          round(roc_auc_score(y_test, y_prob), 4),
        "cv_roc_auc_mean":  round(cv_scores.mean(), 4),
        "cv_roc_auc_std":   round(cv_scores.std(), 4),
        "train_samples":    len(X_train),
        "test_samples":     len(X_test),
        "churn_rate":       round(y.mean(), 4),
    }

    print("\n── Model Metrics ───────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    # ── SHAP global feature importance ─────────────────────────────────────────
    explainer       = shap.TreeExplainer(model)
    shap_values     = explainer.shap_values(X_test)
    mean_abs_shap   = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(zip(FEATURE_NAMES, mean_abs_shap.tolist()))

    print("\n── SHAP Feature Importance ──────────────────────")
    for feat, val in sorted(feature_importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(val * 50)
        print(f"  {FEATURE_DISPLAY[feat]:25s}: {val:.3f}  {bar}")

    # ── Save ───────────────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    meta = {"metrics": metrics, "feature_importance": feature_importance, "features": FEATURE_NAMES}
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\nModel saved → {MODEL_PATH}")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

_model    = None
_explainer= None
_meta     = None

def load_model():
    global _model, _explainer, _meta
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run with --train first.")
    _model     = joblib.load(MODEL_PATH)
    _explainer = shap.TreeExplainer(_model)
    _meta      = json.loads(META_PATH.read_text())


def shap_waterfall_chart(shap_vals: np.ndarray, feature_vals: list) -> str:
    """Generate SHAP waterfall chart as base64 PNG."""
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    indices = np.argsort(np.abs(shap_vals))[::-1]
    top_n   = min(6, len(indices))
    idx     = indices[:top_n][::-1]

    names  = [FEATURE_DISPLAY.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in idx]
    values = shap_vals[idx]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in values]

    bars = ax.barh(names, values, color=colors, height=0.5)
    ax.axvline(0, color="#374151", linewidth=1)
    ax.set_xlabel("SHAP value (impact on churn probability)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#374151")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="PredictIQ API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    age:              int   = 35
    tenure:           int   = 24
    monthly_charges:  float = 65.0
    support_calls:    int   = 2
    contract:         str   = "Month-to-month"  # Month-to-month | One year | Two year
    satisfaction:     int   = 3                 # 1-5


@app.on_event("startup")
async def startup():
    try:
        load_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: {e}")


@app.post("/predict")
async def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded. Run --train first.")

    contract_enc = CONTRACT_MAP.get(req.contract, 0)
    features = np.array([[req.age, req.tenure, req.monthly_charges,
                          req.support_calls, contract_enc, req.satisfaction]])
    X = pd.DataFrame(features, columns=FEATURE_NAMES)

    proba = float(_model.predict_proba(X)[0, 1])
    label = "High Risk" if proba > 0.5 else "Low Risk"

    shap_vals = _explainer.shap_values(X)[0]
    shap_dict = {
        FEATURE_DISPLAY.get(FEATURE_NAMES[i], FEATURE_NAMES[i]): round(float(shap_vals[i]), 3)
        for i in range(len(FEATURE_NAMES))
    }

    chart = shap_waterfall_chart(shap_vals, features[0].tolist())

    return {
        "probability":     round(proba, 4),
        "percentage":      round(proba * 100, 1),
        "label":           label,
        "is_churn":        proba > 0.5,
        "shap_values":     shap_dict,
        "shap_chart_b64":  chart,
    }


@app.get("/metrics")
async def get_metrics():
    if _meta is None:
        raise HTTPException(503, "Model not loaded.")
    return _meta


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port",  type=int, default=8002)
    args = parser.parse_args()

    if args.train:
        train_model()
    else:
        load_model()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
