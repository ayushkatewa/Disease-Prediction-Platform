"""
Cluster Computing Benchmark Suite for PySpark
Measures distributed scaling, partition performance, and execution latency.
"""

import time
import os
from typing import Dict, Any, List
import pandas as pd
from pyspark.sql import SparkSession
from src.spark_pipeline import (
    load_and_preprocess_spark_data,
    build_spark_ml_pipeline,
    evaluate_spark_predictions
)


def run_partition_scaling_benchmark(
    spark: SparkSession,
    csv_path: str,
    partition_list: List[int] = [1, 2, 4, 8],
    replications: int = 1
) -> pd.DataFrame:
    """
    Executes the PySpark MLlib training pipeline across different partition counts
    to simulate and evaluate cluster parallelization, data distribution, and overhead.
    """
    results = []

    for num_parts in partition_list:
        run_times = []
        eval_times = []
        accuracies = []
        f1_scores = []

        for _ in range(replications):
            start_total = time.time()

            # Load and repartition DataFrame
            cleaned_df, _ = load_and_preprocess_spark_data(spark, csv_path)
            partitioned_df = cleaned_df.repartition(num_parts)

            # Split into train and test
            train_df, test_df = partitioned_df.randomSplit([0.8, 0.2], seed=42)

            # Build and train
            t_fit_start = time.time()
            pipeline = build_spark_ml_pipeline()
            model = pipeline.fit(train_df)
            t_fit_end = time.time()

            # Evaluate
            t_eval_start = time.time()
            preds = model.transform(test_df)
            eval_metrics = evaluate_spark_predictions(preds)
            t_eval_end = time.time()

            total_elapsed = time.time() - start_total
            run_times.append(t_fit_end - t_fit_start)
            eval_times.append(t_eval_end - t_eval_start)
            accuracies.append(eval_metrics["accuracy"])
            f1_scores.append(eval_metrics["f1_score"])

        avg_fit = sum(run_times) / len(run_times)
        avg_eval = sum(eval_times) / len(eval_times)
        avg_acc = sum(accuracies) / len(accuracies)
        avg_f1 = sum(f1_scores) / len(f1_scores)

        results.append({
            "Partitions": num_parts,
            "Fit_Time_Sec": round(avg_fit, 3),
            "Eval_Time_Sec": round(avg_eval, 3),
            "Total_Pipeline_Sec": round(avg_fit + avg_eval, 3),
            "Test_Accuracy": round(avg_acc, 4),
            "Test_F1_Score": round(avg_f1, 4)
        })

    bench_df = pd.DataFrame(results)

    # Compute relative speedup compared to single partition baseline
    base_time = bench_df.loc[bench_df["Partitions"] == 1, "Total_Pipeline_Sec"].values[0] if 1 in partition_list else bench_df["Total_Pipeline_Sec"].iloc[0]
    bench_df["Speedup_Factor"] = [round(base_time / t, 2) if t > 0 else 1.0 for t in bench_df["Total_Pipeline_Sec"]]

    return bench_df


def get_cluster_environment_info(spark: SparkSession) -> Dict[str, Any]:
    """
    Extracts runtime cluster configuration and resource metadata from the SparkContext.
    """
    sc = spark.sparkContext
    conf = sc.getConf()

    return {
        "App_Name": sc.appName,
        "Spark_Version": sc.version,
        "Master_URL": sc.master,
        "Default_Parallelism": sc.defaultParallelism,
        "Driver_Memory": conf.get("spark.driver.memory", "Not explicitly specified"),
        "Executor_Memory": conf.get("spark.executor.memory", "Local driver default"),
        "Spark_User": sc.sparkUser(),
        "Application_ID": sc.applicationId,
        "Python_Version": f"{sc.pythonVer}.x",
        "Web_UI_URL": sc.uiWebUrl if sc.uiWebUrl else "UI offline or embedded"
    }
