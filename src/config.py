"""
Production Configuration Module for Diabetes Prediction Platform
Cluster Computing Architecture
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


# Base Project Directories
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

# Ensure required directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)


def get_default_dataset_path() -> str:
    """Finds the dataset path, checking root and data directories."""
    candidates = [
        os.path.join(PROJECT_DIR, "diabetes.csv"),
        os.path.join(DATA_DIR, "diabetes.csv"),
        "diabetes.csv",
        "data/diabetes.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return os.path.join(PROJECT_DIR, "diabetes.csv")


# Core Features Schema
FEATURE_COLS: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Clinical features where 0 represents missing/unrecorded data
ZERO_AS_MISSING_COLS: List[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

LABEL_COL: str = "Outcome"

# Clinical Validation Ranges (Physiologically realistic limits)
CLINICAL_BOUNDS: Dict[str, Dict[str, Any]] = {
    "Pregnancies": {
        "min": 0, "max": 25, "default": 2, "unit": "count",
        "description": "Number of times pregnant"
    },
    "Glucose": {
        "min": 0, "max": 350, "default": 120, "unit": "mg/dL",
        "normal_range": (70, 99),
        "description": "Plasma glucose concentration at 2 hours in an OGTT"
    },
    "BloodPressure": {
        "min": 0, "max": 220, "default": 72, "unit": "mm Hg",
        "normal_range": (60, 80),
        "description": "Diastolic blood pressure"
    },
    "SkinThickness": {
        "min": 0, "max": 110, "default": 23, "unit": "mm",
        "normal_range": (10, 30),
        "description": "Triceps skin fold thickness"
    },
    "Insulin": {
        "min": 0, "max": 900, "default": 85, "unit": "mu U/ml",
        "normal_range": (16, 166),
        "description": "2-Hour serum insulin"
    },
    "BMI": {
        "min": 0.0, "max": 75.0, "default": 32.0, "unit": "kg/m²",
        "normal_range": (18.5, 24.9),
        "description": "Body mass index (weight in kg / (height in m)^2)"
    },
    "DiabetesPedigreeFunction": {
        "min": 0.05, "max": 3.0, "default": 0.45, "unit": "score",
        "normal_range": (0.05, 0.5),
        "description": "Diabetes pedigree function (genetic score)"
    },
    "Age": {
        "min": 18, "max": 120, "default": 33, "unit": "years",
        "description": "Age in years"
    }
}

# Risk Stratification Tiers
RISK_TIERS = {
    "LOW": {
        "label": "Low Risk",
        "min_prob": 0.0,
        "max_prob": 35.0,
        "color": "#10b981",
        "recommendation": "Maintain a healthy balanced lifestyle and continue routine annual check-ups."
    },
    "MODERATE": {
        "label": "Moderate Risk",
        "min_prob": 35.0,
        "max_prob": 65.0,
        "color": "#f59e0b",
        "recommendation": "Consult a primary healthcare provider for HbA1c screening and lifestyle intervention."
    },
    "HIGH": {
        "label": "High Risk",
        "min_prob": 65.0,
        "max_prob": 100.0,
        "color": "#ef4444",
        "recommendation": "Immediate clinical evaluation and comprehensive diagnostic blood panels recommended."
    }
}


@dataclass
class ProductionConfig:
    """Production runtime settings with environment variable support."""
    app_name: str = os.getenv("APP_NAME", "DiabetesDiseasePredictionPlatform")
    dataset_path: str = os.getenv("DATASET_PATH", get_default_dataset_path())
    model_save_path: str = os.getenv("MODEL_PATH", os.path.join(MODELS_DIR, "diabetes_pipeline_model.joblib"))
    metrics_save_path: str = os.getenv("METRICS_PATH", os.path.join(PROJECT_DIR, "evaluation_results.json"))
    spark_master: str = os.getenv("SPARK_MASTER", "local[*]")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    default_threshold: float = float(os.getenv("DECISION_THRESHOLD", "0.50"))
    default_test_size: float = float(os.getenv("TEST_SIZE", "0.20"))
    default_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    default_reg_param: float = float(os.getenv("REG_PARAM", "0.01"))
    default_elastic_net: float = float(os.getenv("ELASTIC_NET", "0.0"))
    default_max_iter: int = int(os.getenv("MAX_ITER", "100"))
    default_impute_strategy: str = os.getenv("IMPUTE_STRATEGY", "median")


config = ProductionConfig()

