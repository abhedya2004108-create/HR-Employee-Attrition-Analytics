# -*- coding: utf-8 -*-
"""
train_model.py
--------------
Trains an XGBoost classifier on the IBM HR Attrition dataset.
Saves the trained pipeline + feature metadata to ../models/
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "data", "HR-Employee-Attrition.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# ── Drop uninformative constant columns ───────────────────────────────────────
drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── Target ────────────────────────────────────────────────────────────────────
y = (df["Attrition"].str.strip().str.lower() == "yes").astype(int)
df.drop(columns=["Attrition"], inplace=True)

# ── Feature lists ─────────────────────────────────────────────────────────────
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
numerical_cols   = df.select_dtypes(include=["number"]).columns.tolist()
all_features     = numerical_cols + categorical_cols

print(f"Numerical features  : {numerical_cols}")
print(f"Categorical features: {categorical_cols}")

X = df[all_features]

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Preprocessor ─────────────────────────────────────────────────────────────
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# ── Classifier ────────────────────────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    eval_metric="logloss",
    random_state=42,
)

# ── Imbalanced pipeline (SMOTE only on train split) ───────────────────────────
pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", xgb),
    ]
)

# ── Cross-validation ──────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"\nCV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Fit on full training set ──────────────────────────────────────────────────
pipeline.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n-- Classification Report --")
print(classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))
print(f"Test ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")

# ── Confusion matrix plot ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Leave"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Test Set")
plt.tight_layout()
fig.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=120)
plt.close(fig)
print("Saved confusion_matrix.png")

# ── Feature importances ───────────────────────────────────────────────────────
ohe_feature_names = (
    pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_cols)
    .tolist()
)
feature_names = numerical_cols + ohe_feature_names

importances = pipeline.named_steps["classifier"].feature_importances_
feat_imp_df  = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(20)
)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.barh(feat_imp_df["feature"][::-1], feat_imp_df["importance"][::-1], color="#3b82d4")
ax2.set_xlabel("XGBoost Importance")
ax2.set_title("Top-20 Feature Importances")
plt.tight_layout()
fig2.savefig(os.path.join(MODEL_DIR, "feature_importances.png"), dpi=120)
plt.close(fig2)
print("Saved feature_importances.png")

# ── Persist model + metadata ──────────────────────────────────────────────────
joblib.dump(pipeline, os.path.join(MODEL_DIR, "attrition_model.pkl"))
print("Saved attrition_model.pkl")

metadata = {
    "numerical_cols": numerical_cols,
    "categorical_cols": categorical_cols,
    "all_features": all_features,
    "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
    "cv_roc_auc_std":  round(float(cv_scores.std()), 4),
    "test_roc_auc":    round(float(roc_auc_score(y_test, y_proba)), 4),
    "categorical_options": {col: sorted(df[col].dropna().unique().tolist()) for col in categorical_cols},
    "numerical_ranges": {
        col: {"min": float(df[col].min()), "max": float(df[col].max()), "mean": round(float(df[col].mean()), 2)}
        for col in numerical_cols
    },
}

with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved metadata.json")
print("\n✅  Training complete.")
