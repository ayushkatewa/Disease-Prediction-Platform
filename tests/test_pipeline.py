"""
Comprehensive Production Test Suite for Diabetes Disease Prediction Platform
Tests Scikit-Learn Pipeline, Model Service, API Endpoints, and PySpark MLlib.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config import (
    FEATURE_COLS,
    ZERO_AS_MISSING_COLS,
    LABEL_COL,
    CLINICAL_BOUNDS,
    get_default_dataset_path
)
from src.sklearn_pipeline import (
    load_and_preprocess_pandas,
    build_and_train_sklearn_pipeline
)
from src.model_service import ModelService

CSV_PATH = get_default_dataset_path()


def check_spark_runtime_available() -> bool:
    """Verifies that PySpark can initialize and execute with the local JVM."""
    try:
        import pyspark
        from src.spark_pipeline import get_spark_session
        s = get_spark_session(app_name="TestInitCheck", master="local[1]", log_level="OFF")
        df = s.createDataFrame([(1, 2)], ["a", "b"])
        df.count()
        s.stop()
        return True
    except Exception:
        return False

HAS_SPARK_RUNTIME = check_spark_runtime_available()


# ---------------------------------------------------------
# Test Group 1: Dataset & Preprocessing Integrity
# ---------------------------------------------------------
def test_data_file_exists():
    """Validates that the dataset exists and is accessible."""
    assert os.path.exists(CSV_PATH), f"Dataset file does not exist at {CSV_PATH}"


def test_pandas_data_loading_and_cleaning():
    """Validates zero-as-missing replacement and summary metadata calculation."""
    cleaned_df, metadata = load_and_preprocess_pandas(CSV_PATH)

    assert metadata["total_records"] == 768
    assert metadata["feature_count"] == 8
    assert metadata["positive_cases"] > 0
    assert metadata["negative_cases"] > 0
    assert len(metadata["missing_stats"]) == len(ZERO_AS_MISSING_COLS)

    # Verify that in cleaned DataFrame, biologically invalid zeros have been replaced with NaN
    for col in ZERO_AS_MISSING_COLS:
        zero_count = (cleaned_df[col] == 0).sum()
        assert zero_count == 0, f"Column {col} still has raw zeros instead of NaN!"


# ---------------------------------------------------------
# Test Group 2: Scikit-Learn Production Pipeline
# ---------------------------------------------------------
def test_sklearn_baseline_pipeline():
    """Tests fitting and evaluation of Scikit-Learn Pipeline."""
    cleaned_df, _ = load_and_preprocess_pandas(CSV_PATH)
    train_df = cleaned_df.iloc[:600]
    test_df = cleaned_df.iloc[600:]

    pipeline, eval_results = build_and_train_sklearn_pipeline(
        train_df=train_df,
        test_df=test_df,
        impute_strategy="median",
        C=1.0,
        max_iter=200
    )

    assert pipeline is not None
    assert 0.5 <= eval_results["accuracy"] <= 1.0
    assert 0.0 <= eval_results["f1_score"] <= 1.0
    assert 0.5 <= eval_results["roc_auc"] <= 1.0

    cm = eval_results["confusion_matrix"]
    assert cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"] == eval_results["total_test_samples"]


# ---------------------------------------------------------
# Test Group 3: ModelService & Clinical Risk Stratification
# ---------------------------------------------------------
def test_model_service_training_and_inference():
    """Tests the ModelService singleton, validation, and single inference."""
    service = ModelService.get_instance()
    results = service.train_and_save(csv_path=CSV_PATH)

    assert service.model is not None
    assert results["eval_metrics"]["accuracy"] > 0.65

    # Test sample patient prediction
    sample_patient = {
        "Pregnancies": 3,
        "Glucose": 160.0,
        "BloodPressure": 80.0,
        "SkinThickness": 30.0,
        "Insulin": 150.0,
        "BMI": 34.5,
        "DiabetesPedigreeFunction": 0.65,
        "Age": 45
    }
    pred_res = service.predict_single(sample_patient)

    assert pred_res["prediction"] in [0, 1]
    assert 0.0 <= pred_res["probability_positive"] <= 100.0
    assert pred_res["risk_level"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert "status" in pred_res
    assert "recommendation" in pred_res
    assert len(pred_res["top_risk_factors"]) > 0


def test_model_service_input_validation():
    """Tests handling of missing and out-of-bounds inputs."""
    service = ModelService.get_instance()

    # Empty patient input (should be imputed with clinical defaults)
    empty_patient = {}
    cleaned, warnings = service.validate_patient_input(empty_patient)

    assert len(cleaned) == len(FEATURE_COLS)
    assert len(warnings) == len(FEATURE_COLS)

    # Out of bounds input
    extreme_patient = {
        "Pregnancies": 50,  # Max is 25
        "Glucose": 999.0    # Max is 350
    }
    _, ext_warnings = service.validate_patient_input(extreme_patient)
    assert len(ext_warnings) >= 2


def test_model_service_batch_prediction():
    """Tests batch prediction functionality."""
    service = ModelService.get_instance()
    records = [
        {"Pregnancies": 1, "Glucose": 90, "BloodPressure": 65, "SkinThickness": 20, "Insulin": 50, "BMI": 22.0, "DiabetesPedigreeFunction": 0.25, "Age": 24},
        {"Pregnancies": 6, "Glucose": 185, "BloodPressure": 88, "SkinThickness": 38, "Insulin": 220, "BMI": 39.5, "DiabetesPedigreeFunction": 0.95, "Age": 55}
    ]
    batch_results = service.predict_batch(records)
    assert len(batch_results) == 2
    assert batch_results[0]["probability_positive"] < batch_results[1]["probability_positive"]


# ---------------------------------------------------------
# Test Group 4: FastAPI Endpoints (Integration)
# ---------------------------------------------------------
def test_api_endpoints():
    """Tests the REST API routes using FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)

    # 1. Health check
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert "status" in resp_health.json()

    # 2. Metrics
    resp_metrics = client.get("/metrics")
    assert resp_metrics.status_code == 200
    assert "metrics" in resp_metrics.json()

    # 3. Features
    resp_features = client.get("/features")
    assert resp_features.status_code == 200
    assert "features" in resp_features.json()

    # 4. Predict
    patient_payload = {
        "Pregnancies": 2,
        "Glucose": 130.0,
        "BloodPressure": 75.0,
        "SkinThickness": 25.0,
        "Insulin": 100.0,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 38
    }
    resp_pred = client.post("/predict", json=patient_payload)
    assert resp_pred.status_code == 200
    data = resp_pred.json()
    assert "probability_positive" in data
    assert "risk_level" in data


# ---------------------------------------------------------
# Test Group 5: PySpark MLlib Pipeline (Conditional)
# ---------------------------------------------------------
@pytest.mark.skipif(not HAS_SPARK_RUNTIME, reason="Apache Spark runtime unavailable (Spark 4.x requires Java 17+; local Java is Java 8)")
class TestSparkPipeline:
    @classmethod
    @pytest.fixture(scope="class")
    def spark(cls):
        from src.spark_pipeline import (
            get_spark_session,
            load_and_preprocess_spark_data,
            train_spark_pipeline,
            evaluate_spark_predictions
        )
        spark_session = get_spark_session(
            app_name="TestDiabetesPipeline",
            master="local[2]",
            log_level="ERROR"
        )
        yield spark_session
        spark_session.stop()

    def test_spark_data_loading(self, spark):
        df, metadata = load_and_preprocess_spark_data(spark, CSV_PATH)
        assert metadata["total_records"] == 768
        assert metadata["feature_count"] == 8

    def test_spark_training(self, spark):
        cleaned_df, _ = load_and_preprocess_spark_data(spark, CSV_PATH)
        train_df, test_df = cleaned_df.randomSplit([0.8, 0.2], seed=42)

        pipeline_model, lr_model = train_spark_pipeline(
            train_df,
            impute_strategy="median",
            reg_param=0.01,
            max_iter=30
        )
        assert pipeline_model is not None
        assert len(lr_model.coefficients) == len(FEATURE_COLS)

        preds = pipeline_model.transform(test_df)
        eval_metrics = evaluate_spark_predictions(preds, threshold=0.5)
        assert 0.0 <= eval_metrics["accuracy"] <= 1.0
