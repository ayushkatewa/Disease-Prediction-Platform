"""
PySpark MLlib Pipeline for Health Data Disease Prediction (Diabetes)
Cluster Computing Architecture Implementation
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Ensure Spark libraries are accessible
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

# Clinical feature columns in Pima Indians Diabetes Dataset
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

# Features where 0 indicates missing/invalid biological measurement
ZERO_AS_MISSING_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

LABEL_COL = "Outcome"


def get_spark_session(
    app_name: str = "DiabetesDiseasePredictionCluster",
    master: str = "local[*]",
    log_level: str = "WARN"
) -> SparkSession:
    """
    Initializes or retrieves an active SparkSession configured for local
    multi-core or distributed cluster execution.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    return spark


def load_and_preprocess_spark_data(
    spark: SparkSession,
    csv_path: str
) -> Tuple[pyspark.sql.DataFrame, Dict[str, Any]]:
    """
    Loads diabetes.csv into a Spark DataFrame and performs cluster-aware preprocessing:
    1. Identifies biologically invalid zero values in clinical columns (Glucose, BP, etc.)
    2. Replaces them with null/NaN for distributed imputation
    3. Casts all input columns to DoubleType for Spark MLlib VectorAssembler compatibility.

    Returns:
        spark_df: preprocessed Spark DataFrame
        metadata: dictionary containing dataset statistics and missingness audits
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at path: {csv_path}")

    # Read CSV via distributed Spark SQL DataFrameReader
    raw_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(csv_path)
    )

    total_rows = raw_df.count()

    # Calculate zero counts before imputation (missingness audit)
    missing_stats = {}
    for col_name in ZERO_AS_MISSING_COLS:
        zero_count = raw_df.filter(F.col(col_name) == 0).count()
        missing_stats[col_name] = {
            "missing_zeros": zero_count,
            "missing_percentage": round((zero_count / total_rows) * 100, 2)
        }

    # Replace biologically impossible 0 with null
    cleaned_df = raw_df
    for col_name in ZERO_AS_MISSING_COLS:
        cleaned_df = cleaned_df.withColumn(
            col_name,
            F.when(F.col(col_name) == 0, None).otherwise(F.col(col_name))
        )

    # Cast all feature columns to DoubleType for MLlib transformers
    for col_name in FEATURE_COLS:
        cleaned_df = cleaned_df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    cleaned_df = cleaned_df.withColumn(LABEL_COL, F.col(LABEL_COL).cast(DoubleType()))

    # Calculate target distribution
    positive_count = cleaned_df.filter(F.col(LABEL_COL) == 1.0).count()
    negative_count = cleaned_df.filter(F.col(LABEL_COL) == 0.0).count()

    metadata = {
        "total_records": total_rows,
        "feature_count": len(FEATURE_COLS),
        "positive_cases": positive_count,
        "negative_cases": negative_count,
        "prevalence_rate": round((positive_count / total_rows) * 100, 2),
        "missing_stats": missing_stats,
        "num_partitions": cleaned_df.rdd.getNumPartitions()
    }

    return cleaned_df, metadata


def build_spark_ml_pipeline(
    impute_strategy: str = "median",
    reg_param: float = 0.01,
    elastic_net_param: float = 0.0,
    max_iter: int = 100,
    threshold: float = 0.5
) -> Pipeline:
    """
    Constructs an end-to-end distributed Spark MLlib Pipeline:
    Stage 1: Imputer (handles missing/null clinical readings using median/mean)
    Stage 2: VectorAssembler (aggregates 8 features into dense vector)
    Stage 3: StandardScaler (standardizes to 0 mean, unit variance to prevent feature scale bias)
    Stage 4: LogisticRegression (distributed L-BFGS optimizer)
    """
    # Stage 1: Median/Mean Imputer
    imputer = Imputer(
        inputCols=ZERO_AS_MISSING_COLS,
        outputCols=ZERO_AS_MISSING_COLS,
        strategy=impute_strategy
    )

    # Stage 2: Vector Assembler
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="unscaled_features",
        handleInvalid="keep"
    )

    # Stage 3: Feature Normalization (StandardScaler)
    scaler = StandardScaler(
        inputCol="unscaled_features",
        outputCol="features",
        withStd=True,
        withMean=True
    )

    # Stage 4: Logistic Regression Classifier
    lr = LogisticRegression(
        featuresCol="features",
        labelCol=LABEL_COL,
        predictionCol="prediction",
        rawPredictionCol="rawPrediction",
        probabilityCol="probability",
        regParam=reg_param,
        elasticNetParam=elastic_net_param,
        maxIter=max_iter,
        threshold=threshold
    )

    pipeline = Pipeline(stages=[imputer, assembler, scaler, lr])
    return pipeline


def train_spark_pipeline(
    train_df: pyspark.sql.DataFrame,
    impute_strategy: str = "median",
    reg_param: float = 0.01,
    elastic_net_param: float = 0.0,
    max_iter: int = 100,
    threshold: float = 0.5
) -> Tuple[PipelineModel, LogisticRegressionModel]:
    """
    Fits the Spark ML pipeline on the training partition across the cluster.
    Returns the fitted PipelineModel and the underlying LogisticRegressionModel.
    """
    pipeline = build_spark_ml_pipeline(
        impute_strategy=impute_strategy,
        reg_param=reg_param,
        elastic_net_param=elastic_net_param,
        max_iter=max_iter,
        threshold=threshold
    )
    pipeline_model = pipeline.fit(train_df)
    lr_model: LogisticRegressionModel = pipeline_model.stages[-1]
    return pipeline_model, lr_model


def evaluate_spark_predictions(
    predictions_df: pyspark.sql.DataFrame,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates predictions from Spark ML pipeline on the test set.
    Calculates Accuracy, F1-Score, Precision, Recall, Specificity, ROC-AUC,
    and constructs the complete Confusion Matrix.
    """
    # Extract predictions and actual outcomes
    # Note: If threshold is customized after model fitting, adjust prediction column dynamically
    extract_prob_udf = F.udf(lambda v: float(v[1]), DoubleType())
    eval_df = predictions_df.withColumn("pos_prob", extract_prob_udf(F.col("probability")))
    if threshold != 0.5:
        eval_df = eval_df.withColumn(
            "custom_prediction",
            F.when(F.col("pos_prob") >= threshold, 1.0).otherwise(0.0)
        )
        pred_col = "custom_prediction"
    else:
        pred_col = "prediction"

    # Distributed grouping for Confusion Matrix
    cm_rows = (
        eval_df
        .groupBy(LABEL_COL, pred_col)
        .count()
        .collect()
    )

    tp, tn, fp, fn = 0, 0, 0, 0
    for row in cm_rows:
        actual = int(row[LABEL_COL])
        pred = int(row[pred_col])
        count = int(row["count"])

        if actual == 1 and pred == 1:
            tp = count
        elif actual == 0 and pred == 0:
            tn = count
        elif actual == 0 and pred == 1:
            fp = count
        elif actual == 1 and pred == 0:
            fn = count

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Calculate Spark Evaluators
    roc_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    pr_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    try:
        roc_auc = float(roc_eval.evaluate(predictions_df))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(pr_eval.evaluate(predictions_df))
    except Exception:
        pr_auc = 0.0

    # Extract probability curves for plotting
    pandas_eval = (
        eval_df
        .select(
            LABEL_COL,
            pred_col,
            "pos_prob"
        )
        .toPandas()
    )

    return {
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "matrix": [[tn, fp], [fn, tp]]
        },
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1_score, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "total_test_samples": total,
        "eval_df": pandas_eval
    }


def extract_feature_importance(
    lr_model: LogisticRegressionModel
) -> pd.DataFrame:
    """
    Extracts standardized coefficients and calculated Odds Ratios
    from the fitted PySpark LogisticRegressionModel.
    """
    coefficients = lr_model.coefficients.toArray()
    intercept = float(lr_model.intercept)

    odds_ratios = np.exp(coefficients)

    fi_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Coefficient": coefficients,
        "Odds_Ratio": odds_ratios,
        "Absolute_Weight": np.abs(coefficients)
    }).sort_values(by="Absolute_Weight", ascending=False).reset_index(drop=True)

    return fi_df, intercept


def predict_patient_risk(
    pipeline_model: PipelineModel,
    patient_dict: Dict[str, float],
    spark: SparkSession
) -> Dict[str, Any]:
    """
    Performs inference for an individual patient record using the trained distributed pipeline.
    """
    # Create single-row Spark DataFrame
    row_data = [(
        float(patient_dict.get("Pregnancies", 0)),
        float(patient_dict.get("Glucose", 120)) if float(patient_dict.get("Glucose", 120)) > 0 else None,
        float(patient_dict.get("BloodPressure", 70)) if float(patient_dict.get("BloodPressure", 70)) > 0 else None,
        float(patient_dict.get("SkinThickness", 20)) if float(patient_dict.get("SkinThickness", 20)) > 0 else None,
        float(patient_dict.get("Insulin", 80)) if float(patient_dict.get("Insulin", 80)) > 0 else None,
        float(patient_dict.get("BMI", 25.0)) if float(patient_dict.get("BMI", 25.0)) > 0 else None,
        float(patient_dict.get("DiabetesPedigreeFunction", 0.35)),
        float(patient_dict.get("Age", 30)),
        0.0  # Placeholder Outcome
    )]

    schema = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]

    single_df = spark.createDataFrame(row_data, schema)
    pred_res = pipeline_model.transform(single_df).collect()[0]

    prob_vec = pred_res["probability"]
    positive_prob = float(prob_vec[1])
    predicted_class = int(pred_res["prediction"])

    if positive_prob >= 0.65:
        risk_level = "High Risk"
        color = "#ef4444"
    elif positive_prob >= 0.35:
        risk_level = "Moderate Risk"
        color = "#f59e0b"
    else:
        risk_level = "Low Risk"
        color = "#10b981"

    return {
        "prediction": predicted_class,
        "probability_positive": round(positive_prob * 100, 2),
        "risk_level": risk_level,
        "color": color
    }
