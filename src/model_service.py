"""
Production Model Service for Diabetes Disease Prediction
Provides model persistence, loading, validation, single and batch inference,
and clinical risk explanation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from src.config import (
    FEATURE_COLS,
    ZERO_AS_MISSING_COLS,
    LABEL_COL,
    CLINICAL_BOUNDS,
    RISK_TIERS,
    config
)
from src.logger import get_logger
from src.sklearn_pipeline import (
    load_and_preprocess_pandas,
    build_and_train_sklearn_pipeline
)

logger = get_logger("ModelService")


class ModelService:
    """
    Central production service managing model lifecycle, training, serialization,
    and clinical inference.
    """

    _instance: Optional["ModelService"] = None

    def __init__(self):
        self.model = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.intercept: float = 0.0
        self.engine_name: str = "Scikit-Learn Baseline"
        self._load_persisted_model()

    @classmethod
    def get_instance(cls) -> "ModelService":
        """Singleton pattern for fast shared access across API and dashboard."""
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def _load_persisted_model(self) -> bool:
        """Attempts to load a pre-trained model and metrics from disk."""
        model_path = config.model_save_path
        metrics_path = config.metrics_save_path

        if os.path.exists(model_path):
            try:
                saved_payload = joblib.load(model_path)
                self.model = saved_payload.get("model")
                self.feature_importance = saved_payload.get("feature_importance")
                self.intercept = saved_payload.get("intercept", 0.0)
                self.metrics = saved_payload.get("metrics")
                self.engine_name = saved_payload.get("engine", "Scikit-Learn Pipeline")
                logger.info(f"Successfully loaded persisted model from: {model_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load persisted model: {e}")

        # If metrics exist in JSON, load them
        if os.path.exists(metrics_path) and self.metrics is None:
            try:
                with open(metrics_path, "r") as f:
                    self.metrics = json.load(f)
                logger.info(f"Loaded existing evaluation metrics from: {metrics_path}")
            except Exception as e:
                logger.warning(f"Failed to read metrics JSON: {e}")

        return False

    def train_and_save(
        self,
        csv_path: Optional[str] = None,
        train_ratio: float = 0.8,
        reg_val: float = 0.01,
        max_iter: int = 1000,
        impute_strategy: str = "median",
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Trains the production Scikit-Learn pipeline, evaluates it,
        and persists artifacts to disk.
        """
        dataset_path = csv_path or config.dataset_path
        logger.info(f"Training production model on dataset: {dataset_path}")

        cleaned_df, meta = load_and_preprocess_pandas(dataset_path)

        # Stratified train/test split to prevent sampling bias
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            cleaned_df,
            test_size=(1.0 - train_ratio),
            random_state=config.default_seed,
            stratify=cleaned_df[LABEL_COL]
        )

        c_inv = 1.0 / max(reg_val, 1e-5)
        pipeline, eval_results = build_and_train_sklearn_pipeline(
            train_df=train_df,
            test_df=test_df,
            impute_strategy=impute_strategy,
            C=c_inv,
            max_iter=max_iter,
            threshold=threshold
        )

        coefs = pipeline.named_steps["classifier"].coef_[0]
        intercept = float(pipeline.named_steps["classifier"].intercept_[0])
        fi_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Coefficient": coefs,
            "Odds_Ratio": np.exp(coefs),
            "Absolute_Weight": np.abs(coefs)
        }).sort_values(by="Absolute_Weight", ascending=False).reset_index(drop=True)

        self.model = pipeline
        self.feature_importance = fi_df
        self.intercept = intercept
        self.metrics = eval_results
        self.engine_name = "Scikit-Learn Production Pipeline"

        # Persist model
        payload = {
            "model": pipeline,
            "feature_importance": fi_df,
            "intercept": intercept,
            "metrics": eval_results,
            "engine": self.engine_name,
            "metadata": meta
        }
        joblib.dump(payload, config.model_save_path)
        logger.info(f"Model successfully saved to: {config.model_save_path}")

        # Update JSON metrics file
        metrics_dict = {
            "engine": self.engine_name,
            "hyperparameters": {
                "reg_param": reg_val,
                "max_iter": max_iter,
                "train_ratio": train_ratio,
                "threshold": threshold,
                "impute_strategy": impute_strategy
            },
            "metrics": {
                "accuracy": eval_results["accuracy"],
                "f1_score": eval_results["f1_score"],
                "precision": eval_results["precision"],
                "recall": eval_results["recall"],
                "specificity": eval_results["specificity"],
                "roc_auc": eval_results["roc_auc"]
            },
            "confusion_matrix": eval_results["confusion_matrix"]
        }
        with open(config.metrics_save_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Evaluation metrics updated in: {config.metrics_save_path}")

        return {
            "metadata": meta,
            "eval_metrics": eval_results,
            "feature_importance": fi_df,
            "intercept": intercept
        }

    def validate_patient_input(self, patient_dict: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
        """
        Validates clinical biomarker values against medical physiological ranges.
        Returns cleaned numeric dict and list of validation warnings/flags.
        """
        cleaned = {}
        warnings = []

        for feature in FEATURE_COLS:
            if feature not in patient_dict or patient_dict[feature] is None:
                default_val = CLINICAL_BOUNDS[feature]["default"]
                cleaned[feature] = float(default_val)
                warnings.append(f"Missing '{feature}'; defaulted to {default_val} {CLINICAL_BOUNDS[feature]['unit']}.")
                continue

            try:
                val = float(patient_dict[feature])
            except (ValueError, TypeError):
                default_val = CLINICAL_BOUNDS[feature]["default"]
                cleaned[feature] = float(default_val)
                warnings.append(f"Invalid format for '{feature}'; defaulted to {default_val}.")
                continue

            bounds = CLINICAL_BOUNDS[feature]
            if val < bounds["min"] or val > bounds["max"]:
                warnings.append(
                    f"'{feature}' value {val} is outside expected clinical bounds "
                    f"[{bounds['min']}, {bounds['max']} {bounds['unit']}]."
                )

            cleaned[feature] = val

        return cleaned, warnings

    def predict_single(
        self,
        patient_dict: Dict[str, Any],
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Executes single patient prediction with clinical risk stratification
        and actionable medical guidance.
        """
        # Ensure a trained model exists
        if self.model is None:
            logger.info("No active model found in memory. Triggering initial training.")
            self.train_and_save()

        cleaned_patient, warnings = self.validate_patient_input(patient_dict)

        # Prepare DataFrame with zero-to-NaN masking for missing physiological readings
        df = pd.DataFrame([cleaned_patient])
        for z_col in ZERO_AS_MISSING_COLS:
            if df[z_col].iloc[0] == 0:
                df[z_col] = np.nan

        # Predict probability
        prob_positive = float(self.model.predict_proba(df[FEATURE_COLS])[0][1])
        predicted_class = int(prob_positive >= threshold)
        prob_pct = round(prob_positive * 100, 2)

        # Risk Stratification
        if prob_pct >= RISK_TIERS["HIGH"]["min_prob"]:
            tier_info = RISK_TIERS["HIGH"]
        elif prob_pct >= RISK_TIERS["MODERATE"]["min_prob"]:
            tier_info = RISK_TIERS["MODERATE"]
        else:
            tier_info = RISK_TIERS["LOW"]

        # Risk Factor Breakdown
        top_risk_factors = []
        if self.feature_importance is not None:
            for _, row in self.feature_importance.head(4).iterrows():
                feat = row["Feature"]
                val = cleaned_patient[feat]
                normal = CLINICAL_BOUNDS[feat].get("normal_range")
                is_elevated = normal is not None and val > normal[1]
                top_risk_factors.append({
                    "feature": feat,
                    "patient_value": val,
                    "odds_ratio": round(row["Odds_Ratio"], 2),
                    "is_elevated": bool(is_elevated),
                    "normal_range": normal
                })

        return {
            "prediction": predicted_class,
            "status": "Diabetic (Positive)" if predicted_class == 1 else "Healthy (Negative)",
            "probability_positive": prob_pct,
            "risk_level": tier_info["label"],
            "color": tier_info["color"],
            "recommendation": tier_info["recommendation"],
            "decision_threshold_used": threshold,
            "validation_warnings": warnings,
            "top_risk_factors": top_risk_factors,
            "patient_data": cleaned_patient
        }

    def predict_batch(
        self,
        records: List[Dict[str, Any]],
        threshold: float = 0.50
    ) -> List[Dict[str, Any]]:
        """Executes batch predictions for a list of patient records."""
        results = []
        for record in records:
            res = self.predict_single(record, threshold=threshold)
            results.append(res)
        return results

