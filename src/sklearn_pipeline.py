"""
Comparative Scikit-Learn Pipeline for Baseline Benchmarking
Validates local vs. distributed cluster performance for Diabetes prediction.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

ZERO_AS_MISSING_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

LABEL_COL = "Outcome"


def load_and_preprocess_pandas(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads diabetes.csv via Pandas and cleans biologically invalid zeros.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_stats = {}
    for col in ZERO_AS_MISSING_COLS:
        zero_count = (df[col] == 0).sum()
        missing_stats[col] = {
            "missing_zeros": int(zero_count),
            "missing_percentage": round((zero_count / len(df)) * 100, 2)
        }

    # Replace 0 with NaN for imputation
    cleaned_df = df.copy()
    cleaned_df[ZERO_AS_MISSING_COLS] = cleaned_df[ZERO_AS_MISSING_COLS].replace(0, np.nan)

    positive_count = int((df[LABEL_COL] == 1).sum())
    negative_count = int((df[LABEL_COL] == 0).sum())

    metadata = {
        "total_records": len(df),
        "feature_count": len(FEATURE_COLS),
        "positive_cases": positive_count,
        "negative_cases": negative_count,
        "prevalence_rate": round((positive_count / len(df)) * 100, 2),
        "missing_stats": missing_stats
    }

    return cleaned_df, metadata


def build_and_train_sklearn_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    impute_strategy: str = "median",
    C: float = 1.0,
    max_iter: int = 1000,
    threshold: float = 0.5
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains an end-to-end Scikit-Learn Pipeline:
    1. Imputation (median or mean)
    2. Standardization (StandardScaler)
    3. Logistic Regression with L2 regularization
    """
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[LABEL_COL]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[LABEL_COL]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=C, max_iter=max_iter, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Predict probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = 0.5

    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    pr_precision, pr_recall, pr_thresh = precision_recall_curve(y_test, y_prob)

    eval_results = {
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
        },
        "accuracy": round(float(accuracy), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "roc_auc": round(float(roc_auc), 4),
        "total_test_samples": len(y_test),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": pr_precision.tolist(), "recall": pr_recall.tolist()},
        "y_true": y_test.tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist()
    }

    return pipeline, eval_results
