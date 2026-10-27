"""
Production REST API for Health Data Disease Prediction (Diabetes)
Built with FastAPI, Pydantic, and Scikit-Learn / PySpark MLlib
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

# Ensure project root is in sys.path
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import (
    FEATURE_COLS,
    CLINICAL_BOUNDS,
    config
)
from src.logger import get_logger
from src.model_service import ModelService

logger = get_logger("API")

# Initialize FastAPI App
app = FastAPI(
    title="Health Data Disease Prediction Cluster API",
    description="""
    Production-grade distributed machine learning inference engine for Type-2 Diabetes prediction.
    Features automated missing value imputation, standardized clinical odds ratios, and 
    real-time patient risk stratification.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontends (e.g. Streamlit or React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_START_TIME = time.time()


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------
class PatientInput(BaseModel):
    """Clinical diagnostic measurements for an individual patient."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Pregnancies": 4,
                "Glucose": 155.0,
                "BloodPressure": 80.0,
                "SkinThickness": 32.0,
                "Insulin": 175.0,
                "BMI": 35.8,
                "DiabetesPedigreeFunction": 0.72,
                "Age": 48
            }
        }
    )

    Pregnancies: int = Field(default=2, ge=0, le=25, description="Number of times pregnant")
    Glucose: float = Field(default=120.0, ge=0.0, le=350.0, description="Plasma glucose concentration (mg/dL)")
    BloodPressure: float = Field(default=72.0, ge=0.0, le=220.0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(default=23.0, ge=0.0, le=110.0, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(default=85.0, ge=0.0, le=900.0, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(default=32.0, ge=0.0, le=75.0, description="Body mass index (kg/m²)")
    DiabetesPedigreeFunction: float = Field(default=0.45, ge=0.05, le=3.0, description="Genetic diabetes pedigree score")
    Age: int = Field(default=33, ge=18, le=120, description="Age in years")


class BatchPatientInput(BaseModel):
    """Batch list of patient records for bulk inference."""
    patients: List[PatientInput]
    threshold: Optional[float] = Field(default=0.50, ge=0.1, le=0.9, description="Probability cutoff threshold")


class TrainingRequest(BaseModel):
    """Hyperparameter payload for on-demand model retraining."""
    train_ratio: float = Field(default=0.80, ge=0.5, le=0.95)
    reg_param: float = Field(default=0.01, gt=0.0)
    max_iter: int = Field(default=500, ge=50, le=2000)
    impute_strategy: str = Field(default="median", pattern="^(median|mean)$")
    threshold: float = Field(default=0.50, ge=0.1, le=0.9)


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    """Service overview and quick documentation links."""
    return {
        "service": "Health Data Disease Prediction API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/predict/batch (POST)",
            "metrics": "/metrics",
            "features": "/features",
            "documentation": "/docs"
        }
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    model_service = ModelService.get_instance()
    uptime_sec = round(time.time() - APP_START_TIME, 2)
    has_model = model_service.model is not None

    # Check Spark status safely
    spark_available = False
    spark_version = "Not Loaded"
    try:
        import pyspark
        spark_version = pyspark.__version__
        spark_available = True
    except Exception:
        pass

    return {
        "status": "healthy" if has_model else "initializing",
        "uptime_seconds": uptime_sec,
        "engine": model_service.engine_name,
        "model_loaded": has_model,
        "spark_runtime": {
            "available": spark_available,
            "version": spark_version
        },
        "dataset_path": config.dataset_path
    }


@app.post("/predict", tags=["Inference"])
def predict_patient(
    patient: PatientInput,
    threshold: float = Query(default=0.50, ge=0.10, le=0.90, description="Decision probability threshold")
):
    """
    Performs real-time diabetes risk assessment for an individual patient.
    Computes calibrated probability, risk tier badge, and physiological risk drivers.
    """
    try:
        service = ModelService.get_instance()
        result = service.predict_single(patient.model_dump(), threshold=threshold)
        return result
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict/batch", tags=["Inference"])
def batch_predict(payload: BatchPatientInput):
    """
    Processes a list of patient profiles for bulk screening.
    """
    try:
        service = ModelService.get_instance()
        records = [p.model_dump() for p in payload.patients]
        results = service.predict_batch(records, threshold=payload.threshold)
        return {
            "total_records": len(results),
            "threshold_used": payload.threshold,
            "predictions": results
        }
    except Exception as e:
        logger.error(f"Batch inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


@app.get("/metrics", tags=["Model Analytics"])
def get_model_metrics():
    """
    Retrieves current production model evaluation metrics (Accuracy, F1, ROC-AUC, Confusion Matrix).
    """
    service = ModelService.get_instance()
    if service.metrics is None:
        service.train_and_save()

    return {
        "engine": service.engine_name,
        "metrics": service.metrics
    }


@app.get("/features", tags=["Model Analytics"])
def get_feature_importance():
    """
    Retrieves standardized logistic regression weights and calculated Odds Ratios ($e^{\beta}$).
    """
    service = ModelService.get_instance()
    if service.feature_importance is None:
        service.train_and_save()

    return {
        "intercept": service.intercept,
        "features": service.feature_importance.to_dict(orient="records")
    }


@app.post("/train", tags=["Model Lifecycle"])
def retrain_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Triggers an asynchronous model retraining job with the requested hyperparameters.
    """
    service = ModelService.get_instance()

    def run_training_task():
        try:
            logger.info("Executing background training job...")
            service.train_and_save(
                train_ratio=request.train_ratio,
                reg_val=request.reg_param,
                max_iter=request.max_iter,
                impute_strategy=request.impute_strategy,
                threshold=request.threshold
            )
            logger.info("Background training job completed successfully.")
        except Exception as err:
            logger.error(f"Background training job failed: {err}", exc_info=True)

    background_tasks.add_task(run_training_task)

    return {
        "status": "training_scheduled",
        "message": "Model retraining job submitted. Check /health and /metrics for updates.",
        "parameters": request.model_dump()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=config.api_host, port=config.api_port, reload=True)
