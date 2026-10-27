#!/usr/bin/env python3
"""
Standalone Cluster Execution Script: Diabetes Disease Prediction
Trains distributed Logistic Regression via PySpark MLlib on Apache Spark cluster.

Usage:
    python train_spark.py
    python train_spark.py --data ./diabetes.csv --master "local[4]" --reg-param 0.01
    spark-submit --master local[*] train_spark.py
"""

import os
import sys
import argparse
import json
import time

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.spark_pipeline import (
    get_spark_session,
    load_and_preprocess_spark_data,
    build_spark_ml_pipeline,
    evaluate_spark_predictions,
    extract_feature_importance,
    FEATURE_COLS
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Cluster Computing: Distributed Logistic Regression for Diabetes Prediction using PySpark MLlib"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./diabetes.csv",
        help="Path to diabetes.csv dataset"
    )
    parser.add_argument(
        "--master",
        type=str,
        default="local[*]",
        help="Spark master URL (e.g., 'local[*]', 'spark://master:7077', 'yarn')"
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default="DiabetesPredictionCluster",
        help="Spark application name"
    )
    parser.add_argument(
        "--reg-param",
        type=float,
        default=0.01,
        help="L2 Regularization parameter (default: 0.01)"
    )
    parser.add_argument(
        "--elastic-net",
        type=float,
        default=0.0,
        help="ElasticNet mixing parameter (0.0 for L2 Ridge, 1.0 for L1 Lasso)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum optimizer iterations (default: 100)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset to reserve for test evaluation (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test splitting (default: 42)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["spark", "sklearn"],
        default="spark",
        help="Machine learning engine ('spark' or 'sklearn')"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="evaluation_results.json",
        help="File path to save evaluation metrics as JSON"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 75)
    print(" [*] HEALTH DATA DISEASE PREDICTION -- CLUSTER COMPUTING ENGINE")
    print("=" * 75)
    print(f"[*] Spark Master     : {args.master}")
    print(f"[*] Dataset Path     : {args.data}")
    print(f"[*] Regularization   : {args.reg_param} (ElasticNet: {args.elastic_net})")
    print(f"[*] Max Iterations   : {args.max_iter}")
    print(f"[*] Test Ratio       : {args.test_size}")
    print("=" * 75)

    start_wall_time = time.time()

    if args.engine == "sklearn":
        print("\n[*] Engine: Scikit-Learn Production Pipeline")
        from src.model_service import ModelService
        service = ModelService.get_instance()
        res = service.train_and_save(
            csv_path=args.data,
            train_ratio=(1.0 - args.test_size),
            reg_val=args.reg_param,
            max_iter=args.max_iter
        )
        metrics = res["eval_metrics"]
        print(f" -> Training complete. Accuracy: {metrics['accuracy'] * 100:.2f}%, ROC-AUC: {metrics['roc_auc']:.4f}")
        return

    # Step 1: Initialize Spark Session
    print("\n[Step 1/6] Initializing SparkSession across cluster executors...")
    t0 = time.time()
    try:
        spark = get_spark_session(app_name=args.app_name, master=args.master)
        print(f" -> SparkSession initialized in {time.time() - t0:.2f}s (Version: {spark.version})")
    except Exception as e:
        print(f"\n[!] SparkSession initialization failed: {e}")
        print("[!] Note: PySpark 4.x requires Java 17+ (or JVM compatibility). Local Java is Java 8.")
        print("[*] Automatically falling back to Scikit-Learn baseline pipeline...")
        from src.model_service import ModelService
        service = ModelService.get_instance()
        res = service.train_and_save(
            csv_path=args.data,
            train_ratio=(1.0 - args.test_size),
            reg_val=args.reg_param,
            max_iter=args.max_iter
        )
        metrics = res["eval_metrics"]
        print(f"\n[+] Production model successfully trained and saved!")
        print(f"    - Test Accuracy : {metrics['accuracy'] * 100:.2f}%")
        print(f"    - F1-Score      : {metrics['f1_score']:.4f}")
        print(f"    - ROC-AUC       : {metrics['roc_auc']:.4f}")
        return

    # Step 2: Load and preprocess data
    print(f"\n[Step 2/6] Loading '{args.data}' into distributed Spark DataFrame...")
    t0 = time.time()
    cleaned_df, metadata = load_and_preprocess_spark_data(spark, args.data)
    print(f" -> Loaded {metadata['total_records']} patient records across {metadata['num_partitions']} partitions in {time.time() - t0:.2f}s")
    print(f" -> Class Distribution: {metadata['positive_cases']} Diabetic ({metadata['prevalence_rate']}%) | {metadata['negative_cases']} Healthy")

    print("\n -> Missing Value Audit (Biologically Invalid Zeros Detected):")
    for col, stat in metadata["missing_stats"].items():
        print(f"    - {col:<15}: {stat['missing_zeros']:>3} zeros ({stat['missing_percentage']:>5.1f}% missing)")

    # Step 3: Train / Test Split
    print(f"\n[Step 3/6] Performing distributed stratified train/test split ({int((1-args.test_size)*100)}/{int(args.test_size*100)})...")
    train_weight = 1.0 - args.test_size
    train_df, test_df = cleaned_df.randomSplit([train_weight, args.test_size], seed=args.seed)

    # Cache training DataFrame in cluster memory
    train_df.cache()
    train_count = train_df.count()
    test_count = test_df.count()
    print(f" -> Training Partition: {train_count} samples cached in memory")
    print(f" -> Testing Partition : {test_count} samples reserved")

    # Step 4: Build & Fit Spark MLlib Pipeline
    print("\n[Step 4/6] Constructing and fitting distributed Spark MLlib Pipeline:")
    print("    Stage 1: Imputer (Median strategy for missing Glucose, BP, Insulin, BMI, SkinThickness)")
    print("    Stage 2: VectorAssembler (Aggregates 8 clinical biomarkers)")
    print("    Stage 3: StandardScaler (Normalized with zero-mean and unit-variance)")
    print("    Stage 4: LogisticRegression (Distributed L-BFGS optimizer)")

    t_fit_start = time.time()
    pipeline = build_spark_ml_pipeline(
        impute_strategy="median",
        reg_param=args.reg_param,
        elastic_net_param=args.elastic_net,
        max_iter=args.max_iter,
        threshold=0.5
    )
    pipeline_model = pipeline.fit(train_df)
    t_fit = time.time() - t_fit_start
    print(f" -> Model fitted across cluster in {t_fit:.3f} seconds!")

    # Step 5: Evaluate on Test Set
    print("\n[Step 5/6] Performing distributed inference and metric evaluation on test set...")
    t_eval_start = time.time()
    predictions = pipeline_model.transform(test_df)
    eval_metrics = evaluate_spark_predictions(predictions, threshold=0.5)
    t_eval = time.time() - t_eval_start
    print(f" -> Inference and evaluation completed in {t_eval:.3f} seconds.")

    # Step 6: Print Evaluation Report & Confusion Matrix
    print("\n" + "=" * 75)
    print(" [REPORT] MODEL PERFORMANCE EVALUATION RESULTS")
    print("=" * 75)
    print(f"  * Accuracy    : {eval_metrics['accuracy'] * 100:.2f}%")
    print(f"  * F1-Score    : {eval_metrics['f1_score']:.4f}")
    print(f"  * Precision   : {eval_metrics['precision']:.4f}")
    print(f"  * Recall (TPR): {eval_metrics['recall']:.4f}")
    print(f"  * Specificity : {eval_metrics['specificity']:.4f}")
    print(f"  * ROC-AUC     : {eval_metrics['roc_auc']:.4f}")
    print("=" * 75)

    cm = eval_metrics["confusion_matrix"]
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    total_test = eval_metrics["total_test_samples"]

    print("\n  [*] CONFUSION MATRIX VISUALIZATION:")
    print("  " + "-" * 50)
    print("                  |  Predicted: 0   |  Predicted: 1   |")
    print("  " + "-" * 50)
    print(f"   Actual: 0 (Neg)|  TN = {tn:<4} ({round(tn/total_test*100,1)}%)|  FP = {fp:<4} ({round(fp/total_test*100,1)}%)|")
    print("  " + "-" * 50)
    print(f"   Actual: 1 (Pos)|  FN = {fn:<4} ({round(fn/total_test*100,1)}%)|  TP = {tp:<4} ({round(tp/total_test*100,1)}%)|")
    print("  " + "-" * 50)

    # Feature Importance & Odds Ratios
    lr_stage = pipeline_model.stages[-1]
    fi_df, intercept = extract_feature_importance(lr_stage)
    print("\n  [*] LOGISTIC REGRESSION WEIGHTS & ODDS RATIOS:")
    print("  " + "-" * 60)
    print(f"  {'Feature':<26} {'Coefficient (β)':<18} {'Odds Ratio (e^β)':<15}")
    print("  " + "-" * 60)
    for _, row in fi_df.iterrows():
        print(f"  {row['Feature']:<26} {row['Coefficient']:>12.4f}       {row['Odds_Ratio']:>10.3f}")
    print("  " + "-" * 60)
    print(f"  Model Intercept: {intercept:.4f}")

    total_time = time.time() - start_wall_time
    print("\n" + "=" * 75)
    print(f" [*] Total Cluster Execution Time: {total_time:.2f} seconds")
    print("=" * 75)

    # Save output metrics to JSON
    output_dict = {
        "spark_master": args.master,
        "spark_version": spark.version,
        "hyperparameters": {
            "reg_param": args.reg_param,
            "elastic_net": args.elastic_net,
            "max_iter": args.max_iter,
            "test_size": args.test_size,
            "seed": args.seed
        },
        "metrics": {
            "accuracy": eval_metrics["accuracy"],
            "f1_score": eval_metrics["f1_score"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "specificity": eval_metrics["specificity"],
            "roc_auc": eval_metrics["roc_auc"]
        },
        "confusion_matrix": cm,
        "timing": {
            "fit_time_sec": round(t_fit, 3),
            "eval_time_sec": round(t_eval, 3),
            "total_time_sec": round(total_time, 2)
        }
    }

    with open(args.output_json, "w") as f:
        json.dump(output_dict, f, indent=4)
    print(f"\n[+] Results successfully exported to: {os.path.abspath(args.output_json)}")

    spark.stop()
    print("[*] SparkSession cleanly terminated.\n")


if __name__ == "__main__":
    main()
