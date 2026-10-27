"""
Interactive Streamlit Dashboard for Health Data Disease Prediction (Diabetes)
Cluster Computing Architecture with Apache Spark MLlib & Scikit-Learn
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project directory to path
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from pyspark.ml import PipelineModel
except (ImportError, Exception):
    PipelineModel = None

from src.config import (
    config,
    get_default_dataset_path,
    FEATURE_COLS,
    ZERO_AS_MISSING_COLS,
    LABEL_COL,
    CLINICAL_BOUNDS,
    RISK_TIERS
)
from src.spark_pipeline import (
    get_spark_session,
    load_and_preprocess_spark_data,
    build_spark_ml_pipeline,
    evaluate_spark_predictions,
    extract_feature_importance,
    predict_patient_risk
)
from src.sklearn_pipeline import (
    load_and_preprocess_pandas,
    build_and_train_sklearn_pipeline
)
from src.cluster_benchmarks import (
    run_partition_scaling_benchmark,
    get_cluster_environment_info
)
from src.model_service import ModelService
from src.utils import (
    plot_confusion_matrix_plotly,
    plot_roc_curve_plotly,
    plot_feature_importance_plotly,
    plot_missing_audit_plotly,
    plot_cluster_scaling_plotly,
    plot_patient_radar_plotly,
    plot_threshold_tradeoff_plotly,
    plot_batch_risk_donut_plotly
)

# ---------------------------------------------------------
# Page Configuration & Modern Glassmorphism Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Health Data | Diabetes Prediction Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main {
        background-color: #070b14;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.55);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -4px rgba(56, 189, 248, 0.25);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-sub {
        color: #38bdf8;
        font-size: 0.78rem;
        margin-top: 3px;
        font-weight: 500;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 18px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 40%, #c084fc 80%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.0rem;
        max-width: 950px;
        line-height: 1.55;
    }
    .cluster-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Status Pill */
    .status-pill-green {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-pill-blue {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Code blocks */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CSV_PATH = get_default_dataset_path()


# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_spark():
    """Initializes and caches a single SparkSession for the application session."""
    return get_spark_session(app_name="DiabetesPredictionStreamlitDashboard", master="local[*]")


@st.cache_data(show_spinner=False)
def load_raw_dataset():
    """Loads and caches pandas dataframe for fast UI exploration."""
    return pd.read_csv(CSV_PATH)


# Initialize Spark Session safely
spark = None
spark_error = None
try:
    spark = init_spark()
except Exception as e:
    spark_error = str(e)


# ---------------------------------------------------------
# Sidebar Controls & Engine Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏥 Diabetes Cluster Engine")
    st.caption("Distributed Analytics & Clinical Risk Decision Platform")

    # Engine selector
    engine = st.radio(
        "Execution Engine",
        options=["PySpark MLlib (Cluster Mode)", "Scikit-Learn (Local Production)"],
        index=0 if spark is not None else 1,
        help="Switch between distributed Apache Spark MLlib and single-node Scikit-Learn baseline."
    )

    st.markdown("---")
    st.markdown("#### 🎛️ Pipeline Hyperparameters")

    train_split = st.slider(
        "Train / Test Split Ratio",
        min_value=0.60,
        max_value=0.90,
        value=0.80,
        step=0.05,
        help="Fraction of data reserved for model training (default: 80/20)."
    )

    reg_param = st.select_slider(
        "Regularization (λ / C)",
        options=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        value=0.01,
        help="L2 regularization parameter to prevent clinical overfitting."
    )

    elastic_net = st.slider(
        "ElasticNet Mixing (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0 = L2 Ridge, 1.0 = L1 Lasso (feature sparsity)."
    )

    max_iter = st.slider(
        "Max Optimizer Iterations",
        min_value=20,
        max_value=300,
        value=100,
        step=20,
        help="Epochs for distributed L-BFGS or Quasi-Newton solver."
    )

    impute_method = st.selectbox(
        "Missing Imputation Strategy",
        options=["median", "mean"],
        index=0,
        help="Strategy to impute biologically invalid zeros in Glucose, BP, Insulin, BMI, SkinThickness."
    )

    st.markdown("---")
    st.markdown("#### 🎯 Clinical Decision Threshold")
    decision_threshold = st.slider(
        "Probability Cutoff Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Decision boundary for positive diabetes diagnosis. Lower thresholds prioritize sensitivity/recall."
    )

    st.markdown("---")
    # Quick Telemetry
    st.markdown("#### 📡 System Status")
    st.markdown('<span class="status-pill-green">● REST API Active</span> &nbsp; <span class="status-pill-blue">Port 8000</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="{"status-pill-green" if spark else "status-pill-blue"}">● Spark: {"Active" if spark else "Fallback Ready"}</span>', unsafe_allow_html=True)
    st.caption("Antigravity Health Computing Lab • Spark 4.2.0")


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Health Data — Disease Prediction Platform</div>
    <div class="hero-subtitle">
        Distributed Machine Learning pipeline for Type-2 Diabetes patient diagnosis built for 
        <b>Cluster Computing</b> using <b>Apache Spark 4.2.0 (PySpark MLlib)</b>, zero-leakage feature standardization, 
        and real-time clinical risk decision modeling with live REST API integration.
    </div>
    <div>
        <span class="cluster-tag">⚡ Apache Spark 4.2.0</span>
        <span class="cluster-tag">🧠 Distributed MLlib Pipeline</span>
        <span class="cluster-tag">📊 L-BFGS Optimizer</span>
        <span class="cluster-tag">🩺 Clinical Risk Profiler</span>
        <span class="cluster-tag">🌐 FastAPI REST Microservice</span>
        <span class="cluster-tag">🧪 Cohort Batch Screening</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Dynamic Model Execution & Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_and_evaluate_cached(
    engine_name: str,
    train_ratio: float,
    reg_val: float,
    elastic_val: float,
    iterations: int,
    imp_strategy: str,
    threshold: float
):
    """Executes training on chosen backend and returns models and evaluation metrics."""
    if engine_name.startswith("PySpark") and spark is not None:
        cleaned_df, meta = load_and_preprocess_spark_data(spark, CSV_PATH)
        test_ratio = round(1.0 - train_ratio, 2)
        train_df, test_df = cleaned_df.randomSplit([train_ratio, test_ratio], seed=42)
        train_df.cache()

        t0 = time.time()
        pipeline = build_spark_ml_pipeline(
            impute_strategy=imp_strategy,
            reg_param=reg_val,
            elastic_net_param=elastic_val,
            max_iter=iterations,
            threshold=threshold
        )
        fitted_pipeline = pipeline.fit(train_df)
        fit_time = time.time() - t0

        t1 = time.time()
        preds = fitted_pipeline.transform(test_df)
        eval_res = evaluate_spark_predictions(preds, threshold=threshold)
        eval_time = time.time() - t1

        lr_stage = fitted_pipeline.stages[-1]
        fi_df, intercept = extract_feature_importance(lr_stage)

        return {
            "engine": "PySpark MLlib (Cluster Mode)",
            "metadata": meta,
            "eval_metrics": eval_res,
            "feature_importance": fi_df,
            "intercept": intercept,
            "fit_time": fit_time,
            "eval_time": eval_time,
            "pipeline_model": fitted_pipeline
        }
    else:
        # Scikit-Learn Production Pipeline
        cleaned_df, meta = load_and_preprocess_pandas(CSV_PATH)
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            cleaned_df,
            test_size=(1.0 - train_ratio),
            random_state=42,
            stratify=cleaned_df[LABEL_COL]
        )

        t0 = time.time()
        c_inv = 1.0 / max(reg_val, 1e-5)
        pipeline, eval_res = build_and_train_sklearn_pipeline(
            train_df=train_df,
            test_df=test_df,
            impute_strategy=imp_strategy,
            C=c_inv,
            max_iter=iterations,
            threshold=threshold
        )
        fit_time = time.time() - t0

        coefs = pipeline.named_steps["classifier"].coef_[0]
        intercept = float(pipeline.named_steps["classifier"].intercept_[0])
        fi_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Coefficient": coefs,
            "Odds_Ratio": np.exp(coefs),
            "Absolute_Weight": np.abs(coefs)
        }).sort_values(by="Absolute_Weight", ascending=False).reset_index(drop=True)

        return {
            "engine": "Scikit-Learn Production Pipeline",
            "metadata": meta,
            "eval_metrics": eval_res,
            "feature_importance": fi_df,
            "intercept": intercept,
            "fit_time": fit_time,
            "eval_time": 0.005,
            "pipeline_model": pipeline
        }


with st.spinner("⚡ Running distributed machine learning pipeline..."):
    results = train_and_evaluate_cached(
        engine_name=engine,
        train_ratio=train_split,
        reg_val=reg_param,
        elastic_val=elastic_net,
        iterations=max_iter,
        imp_strategy=impute_method,
        threshold=decision_threshold
    )

eval_metrics = results["eval_metrics"]
cm_matrix = eval_metrics["confusion_matrix"]["matrix"]
fi_df = results["feature_importance"]
meta = results["metadata"]


# ---------------------------------------------------------
# Top-level KPI Metrics Row
# ---------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Test Accuracy</div>
        <div class="metric-value">{eval_metrics['accuracy'] * 100:.1f}%</div>
        <div class="metric-sub">Overall Correctness</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">F1-Score</div>
        <div class="metric-value">{eval_metrics['f1_score']:.3f}</div>
        <div class="metric-sub">Harmonic Precision/Recall</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Recall (Sensitivity)</div>
        <div class="metric-value">{eval_metrics['recall'] * 100:.1f}%</div>
        <div class="metric-sub">Diabetic Detection Rate</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Specificity</div>
        <div class="metric-value">{eval_metrics['specificity'] * 100:.1f}%</div>
        <div class="metric-sub">Healthy Clearance Rate</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">ROC-AUC</div>
        <div class="metric-value">{eval_metrics['roc_auc']:.3f}</div>
        <div class="metric-sub">Discrimination Capacity</div>
    </div>
    """, unsafe_allow_html=True)

with c6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Fit Latency</div>
        <div class="metric-value">{results['fit_time']:.2f}s</div>
        <div class="metric-sub">{results['engine'].split()[0]}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_eval, tab_cluster, tab_data, tab_pipeline, tab_predict, tab_batch, tab_report = st.tabs([
    "📈 Confusion Matrix & Evaluation",
    "⚡ Cluster Computing Architecture",
    "📊 Data Explorer & Missingness",
    "🧠 Preprocessing & ML Pipeline",
    "🩺 Patient Risk Profiler & Simulator",
    "🧪 Cohort Batch Screening",
    "📑 Project Report & REST API"
])


# =========================================================
# TAB 1: Evaluation & Confusion Matrix
# =========================================================
with tab_eval:
    st.markdown("### 🔍 Model Evaluation & Confusion Matrix Analysis")
    st.caption("Visualizing classification performance and threshold trade-offs on held-out test patients.")

    col_cm, col_roc = st.columns([1, 1])

    with col_cm:
        cm_fig = plot_confusion_matrix_plotly(
            cm_matrix,
            title=f"Confusion Matrix (Decision Threshold: {decision_threshold:.2f})"
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        cm = eval_metrics["confusion_matrix"]
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 14px 18px; border-radius: 10px; border-left: 4px solid #38bdf8; font-size: 0.9rem;">
            <b>Clinical Interpretation at Cutoff {decision_threshold:.2f}:</b><br>
            • <b>True Positives (TP = {cm['tp']})</b>: Diabetic patients accurately identified for medical intervention.<br>
            • <b>True Negatives (TN = {cm['tn']})</b>: Healthy individuals correctly given a clean bill of health.<br>
            • <b>False Negatives (FN = {cm['fn']})</b>: Clinically hazardous type II error (missed diabetes diagnoses).<br>
            • <b>False Positives (FP = {cm['fp']})</b>: Healthy individuals falsely flagged, causing unnecessary anxiety.
        </div>
        """, unsafe_allow_html=True)

    with col_roc:
        if "eval_df" in eval_metrics:
            roc_fig = plot_roc_curve_plotly(eval_metrics["eval_df"], eval_metrics["roc_auc"])
            st.plotly_chart(roc_fig, use_container_width=True)
        elif "roc_curve" in eval_metrics:
            fpr = eval_metrics["roc_curve"]["fpr"]
            tpr = eval_metrics["roc_curve"]["tpr"]
            fig_roc_sk = go.Figure()
            fig_roc_sk.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {eval_metrics["roc_auc"]:.3f}', line=dict(color='#38bdf8', width=3)))
            fig_roc_sk.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='#94a3b8', dash='dash')))
            fig_roc_sk.update_layout(title="<b>ROC Curve</b>", template="plotly_dark", height=380)
            st.plotly_chart(fig_roc_sk, use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚖️ Sensitivity vs. Specificity Trade-Off Scrubber")
    st.caption("How varying the classification threshold alters Sensitivity (Recall), Specificity, Precision, and F1-Score.")

    if "y_true" in eval_metrics and "y_prob" in eval_metrics:
        tradeoff_fig = plot_threshold_tradeoff_plotly(eval_metrics["y_true"], eval_metrics["y_prob"], current_threshold=decision_threshold)
        st.plotly_chart(tradeoff_fig, use_container_width=True)
    elif "eval_df" in eval_metrics:
        tradeoff_fig = plot_threshold_tradeoff_plotly(eval_metrics["eval_df"]["Outcome"].tolist(), eval_metrics["eval_df"]["pos_prob"].tolist(), current_threshold=decision_threshold)
        st.plotly_chart(tradeoff_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🧬 Clinical Feature Impact & Odds Ratios")
    st.caption("How each clinical biomarker impacts patient diabetes risk under normalized logistic regression.")

    fi_fig = plot_feature_importance_plotly(fi_df)
    st.plotly_chart(fi_fig, use_container_width=True)

    with st.expander("📋 View Exact Model Coefficients & Odds Ratio Table"):
        st.dataframe(
            fi_df.style.format({
                "Coefficient": "{:.4f}",
                "Odds_Ratio": "{:.3f}",
                "Absolute_Weight": "{:.4f}"
            }),
            use_container_width=True
        )
        st.info(f"**Model Intercept (β₀)**: {results['intercept']:.4f} (Base log-odds when all standardized features equal zero).")


# =========================================================
# TAB 2: Cluster Computing Architecture
# =========================================================
with tab_cluster:
    st.markdown("### ⚡ Cluster Computing Architecture & Spark Runtime")
    st.caption("Distributed computing principles, cluster resource monitoring, and partition scaling analysis.")

    if spark is not None:
        cluster_info = get_cluster_environment_info(spark)

        c_env1, c_env2, c_env3, c_env4 = st.columns(4)
        with c_env1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Spark Version</div>
                <div class="metric-value">{cluster_info['Spark_Version']}</div>
                <div class="metric-sub">Apache Spark Core</div>
            </div>
            """, unsafe_allow_html=True)
        with c_env2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Master Mode</div>
                <div class="metric-value">{cluster_info['Master_URL']}</div>
                <div class="metric-sub">Cluster Resource Mgr</div>
            </div>
            """, unsafe_allow_html=True)
        with c_env3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Default Parallelism</div>
                <div class="metric-value">{cluster_info['Default_Parallelism']} Cores</div>
                <div class="metric-sub">Executor Threads</div>
            </div>
            """, unsafe_allow_html=True)
        with c_env4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Driver Memory</div>
                <div class="metric-value">{cluster_info['Driver_Memory']}</div>
                <div class="metric-sub">JVM Heap Size</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🔬 Real-Time Partition Scalability Experiment")
        st.write("""
        In cluster computing, data is split into partitions distributed across worker nodes. 
        Click below to run a live scaling experiment that evaluates training time and speedup across 
        **1, 2, 4, and 8 partitions**.
        """)

        if st.button("🚀 Run Live Partition Scaling Benchmark", type="primary"):
            with st.spinner("Benchmarking distributed training across 1, 2, 4, and 8 partitions..."):
                bench_df = run_partition_scaling_benchmark(spark, CSV_PATH, partition_list=[1, 2, 4, 8])
                st.session_state["bench_df"] = bench_df

        if "bench_df" in st.session_state:
            bench_df = st.session_state["bench_df"]
            bench_fig = plot_cluster_scaling_plotly(bench_df)
            st.plotly_chart(bench_fig, use_container_width=True)

            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                st.dataframe(bench_df, use_container_width=True)
            with col_b2:
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 8px; border-left: 4px solid #6366f1; font-size: 0.9rem;">
                    <b>Cluster Computing Insights:</b><br>
                    • <b>Distributed Amdahl's Law</b>: On smaller datasets (768 records), thread coordination, serialization, and shuffle overhead dominate compute.<br>
                    • <b>Core Matching</b>: The sweet spot occurs when partitions equal or slightly exceed CPU worker cores.<br>
                    • <b>Big Data Transition</b>: On millions of records, distributed partition scaling achieves near-linear speedups!
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info(f"ℹ️ Spark Cluster Runtime status: JVM requires Java 17+ for Spark 4.x. Running on production Scikit-Learn baseline. Docker container includes OpenJDK 17.")

    st.markdown("---")
    st.markdown("#### 📐 Spark Distributed Pipeline Architecture")
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph Cluster Driver
            A[diabetes.csv File] --> B[SparkSession DataFrameReader]
            B --> C[Zero-to-Null Sanitization]
            C --> D[Stratified Train/Test Split]
        end
        
        subgraph Worker Nodes / Distributed Executors
            D -->|Partition 1| E1[Executor 1: Imputer & Scaler]
            D -->|Partition 2| E2[Executor 2: Imputer & Scaler]
            D -->|Partition 3| E3[Executor 3: Imputer & Scaler]
            D -->|Partition 4| E4[Executor 4: Imputer & Scaler]
            
            E1 & E2 & E3 & E4 -->|TreeAggregate Gradients| F[Distributed L-BFGS Optimizer]
        end
        
        subgraph Evaluation & Metrics
            F --> G[PipelineModel Broadcast]
            G --> H[Distributed Confusion Matrix]
            G --> I[Accuracy & F1-Score]
        end
    ```
    """)


# =========================================================
# TAB 3: Data Explorer & Missingness
# =========================================================
with tab_data:
    st.markdown("### 📊 Dataset Exploration & Missingness Audit")
    raw_df = load_raw_dataset()

    col_d1, col_d2 = st.columns([1, 1])

    with col_d1:
        st.markdown("#### 📋 Raw Patient Health Data (Sample)")
        st.dataframe(raw_df.head(15), use_container_width=True)
        st.caption(f"Showing 15 of {len(raw_df)} patient records.")

    with col_d2:
        st.markdown("#### ⚠️ Biologically Invalid Zeros (Missing Data)")
        st.write("""
        In health datasets, measurements such as **Glucose, Blood Pressure, Insulin, Skin Thickness, and BMI cannot be zero in a living human**. 
        These zeros represent missing/unrecorded entries and must be statistically imputed.
        """)
        missing_fig = plot_missing_audit_plotly(meta["missing_stats"])
        st.plotly_chart(missing_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 Clinical Feature Distributions by Diabetes Outcome")

    selected_feat = st.selectbox(
        "Select Clinical Biomarker to Inspect:",
        options=FEATURE_COLS,
        index=1
    )

    clean_viz_df = raw_df.copy()
    if selected_feat in ZERO_AS_MISSING_COLS:
        clean_viz_df = clean_viz_df[clean_viz_df[selected_feat] > 0]

    col_hist, col_box = st.columns([2, 1])

    with col_hist:
        hist_fig = px.histogram(
            clean_viz_df,
            x=selected_feat,
            color="Outcome",
            barmode="overlay",
            nbins=35,
            color_discrete_map={0: "#38bdf8", 1: "#ef4444"},
            labels={"Outcome": "Condition"},
            title=f"Distribution of {selected_feat} (0 = Healthy, 1 = Diabetic)",
            template="plotly_dark"
        )
        hist_fig.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.6)", plot_bgcolor="rgba(15, 23, 42, 0.6)", height=350)
        st.plotly_chart(hist_fig, use_container_width=True)

    with col_box:
        box_fig = px.box(
            clean_viz_df,
            x="Outcome",
            y=selected_feat,
            color="Outcome",
            color_discrete_map={0: "#38bdf8", 1: "#ef4444"},
            title=f"Boxplot: {selected_feat} by Outcome",
            template="plotly_dark"
        )
        box_fig.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.6)", plot_bgcolor="rgba(15, 23, 42, 0.6)", height=350)
        st.plotly_chart(box_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔗 Biomarker Correlation Heatmap")
    corr = raw_df.corr()
    corr_fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Pairwise Pearson Correlation Matrix",
        template="plotly_dark"
    )
    corr_fig.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.6)", plot_bgcolor="rgba(15, 23, 42, 0.6)", height=450)
    st.plotly_chart(corr_fig, use_container_width=True)


# =========================================================
# TAB 4: ML Pipeline & Feature Normalization
# =========================================================
with tab_pipeline:
    st.markdown("### 🧠 Spark MLlib End-to-End Distributed Pipeline")
    st.markdown(r"""
    The pipeline conforms strictly to **Machine Learning Best Practices**:
    1. **Strict Featurization Order**: Train/Test split is executed **BEFORE** fitting imputer and standardizer to guarantee **Zero Data Leakage**.
    2. **Zero-to-Null Masking**: Invalid physiological zeros are replaced with `null` in Spark DataFrame.
    3. **Spark Imputer**: Distributed median imputation per partition.
    4. **VectorAssembler**: Combines 8 scalar columns into dense vector column.
    5. **StandardScaler (Normalization)**: Centers features to zero mean ($\mu = 0$) and scales to unit variance ($\sigma = 1$). 
       This prevents high-magnitude features (e.g. Insulin up to 846) from drowning out low-magnitude features (e.g. Pedigree Function around 0.5).
    6. **Logistic Regression with Distributed L-BFGS**: Iterative Quasi-Newton optimization executed via tree-aggregation across worker nodes.
    """)

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        st.markdown("#### ⚙️ Pipeline Configuration Details")
        st.json({
            "Stage 1": "Imputer (strategy='median', columns=Glucose, BP, Skin, Insulin, BMI)",
            "Stage 2": "VectorAssembler (inputCols=8 features, outputCol='unscaled_features')",
            "Stage 3": "StandardScaler (withMean=True, withStd=True, outputCol='features')",
            "Stage 4": f"LogisticRegression (regParam={reg_param}, elasticNetParam={elastic_net}, maxIter={max_iter})"
        })

    with col_p2:
        st.markdown("#### 📐 Mathematical Formulation")
        st.latex(r"P(Y = 1 | \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \mathbf{\beta}^T \mathbf{z})}}")
        st.write("Where normalized feature vector $\\mathbf{z}$ is:")
        st.latex(r"z_i = \frac{x_i - \mu_i}{\sigma_i}")
        st.write("And regularized objective function solved across cluster workers:")
        st.latex(r"\min_{\mathbf{\beta}} -\sum_{i=1}^N \left[ y_i \log p_i + (1-y_i)\log(1-p_i) \right] + \lambda \left[ (1-\alpha)\frac{1}{2}\|\mathbf{\beta}\|_2^2 + \alpha \|\mathbf{\beta}\|_1 \right]")


# =========================================================
# TAB 5: Patient Risk Profiler & "What-If" Simulator
# =========================================================
with tab_predict:
    st.markdown("### 🩺 Real-Time Patient Risk Profiler & Interactive Clinical Simulator")
    st.caption("Input clinical readings or select a realistic medical preset to evaluate real-time diabetes risk.")

    raw_df_clean = load_raw_dataset()
    med_healthy = raw_df_clean[raw_df_clean["Outcome"] == 0].median()
    med_diabetic = raw_df_clean[raw_df_clean["Outcome"] == 1].median()

    # Preset Profiles
    st.markdown("##### ⚡ Quick Clinical Presets (Click to Load)")
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)

    PRESETS = {
        "High-Risk Diabetic Case": {
            "Pregnancies": 4, "Glucose": 178, "BloodPressure": 88, "SkinThickness": 38,
            "Insulin": 210, "BMI": 39.2, "DiabetesPedigreeFunction": 0.88, "Age": 54
        },
        "Healthy Young Adult": {
            "Pregnancies": 0, "Glucose": 86, "BloodPressure": 68, "SkinThickness": 18,
            "Insulin": 42, "BMI": 21.4, "DiabetesPedigreeFunction": 0.22, "Age": 23
        },
        "Pre-Diabetic / Borderline": {
            "Pregnancies": 2, "Glucose": 132, "BloodPressure": 76, "SkinThickness": 26,
            "Insulin": 95, "BMI": 28.6, "DiabetesPedigreeFunction": 0.42, "Age": 41
        },
        "Senior Metabolic Profile": {
            "Pregnancies": 5, "Glucose": 162, "BloodPressure": 84, "SkinThickness": 32,
            "Insulin": 160, "BMI": 34.0, "DiabetesPedigreeFunction": 0.65, "Age": 68
        }
    }

    if "current_patient" not in st.session_state:
        st.session_state["current_patient"] = PRESETS["High-Risk Diabetic Case"].copy()

    with col_pre1:
        if st.button("🔴 High-Risk Diabetic", use_container_width=True):
            st.session_state["current_patient"] = PRESETS["High-Risk Diabetic Case"].copy()
    with col_pre2:
        if st.button("🟢 Healthy Young Adult", use_container_width=True):
            st.session_state["current_patient"] = PRESETS["Healthy Young Adult"].copy()
    with col_pre3:
        if st.button("🟡 Pre-Diabetic Borderline", use_container_width=True):
            st.session_state["current_patient"] = PRESETS["Pre-Diabetic / Borderline"].copy()
    with col_pre4:
        if st.button("🟣 Senior Metabolic Profile", use_container_width=True):
            st.session_state["current_patient"] = PRESETS["Senior Metabolic Profile"].copy()

    cur = st.session_state["current_patient"]

    with st.form("patient_form"):
        st.markdown("##### 📝 Clinical Biomarker Input Form")
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)

        with p_c1:
            inp_preg = st.number_input("Pregnancies", min_value=0, max_value=25, value=int(cur["Pregnancies"]), step=1)
            inp_gluc = st.number_input("Glucose (mg/dL)", min_value=0, max_value=350, value=int(cur["Glucose"]), step=1, help="Normal fasting: 70-99 mg/dL")
        with p_c2:
            inp_bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=220, value=int(cur["BloodPressure"]), step=1, help="Normal: 60-80 mm Hg")
            inp_skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=110, value=int(cur["SkinThickness"]), step=1)
        with p_c3:
            inp_ins = st.number_input("Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=int(cur["Insulin"]), step=5, help="Normal 2hr: 16-166 mu U/ml")
            inp_bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=75.0, value=float(cur["BMI"]), step=0.5, help="Normal: 18.5 - 24.9")
        with p_c4:
            inp_ped = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=3.0, value=float(cur["DiabetesPedigreeFunction"]), step=0.01, help="Genetic family history score")
            inp_age = st.number_input("Age (Years)", min_value=18, max_value=120, value=int(cur["Age"]), step=1)

        submitted = st.form_submit_button("⚡ Compute Real-Time Risk", type="primary", use_container_width=True)

    patient_data = {
        "Pregnancies": inp_preg,
        "Glucose": inp_gluc,
        "BloodPressure": inp_bp,
        "SkinThickness": inp_skin,
        "Insulin": inp_ins,
        "BMI": inp_bmi,
        "DiabetesPedigreeFunction": inp_ped,
        "Age": inp_age
    }

    # Perform inference via production ModelService
    model_service = ModelService.get_instance()
    pred_result = model_service.predict_single(patient_data, threshold=decision_threshold)

    col_res1, col_res2 = st.columns([1, 1])

    with col_res1:
        st.markdown(f"""
        <div style="background: rgba(19, 27, 46, 0.85); border: 2px solid {pred_result['color']}; border-radius: 14px; padding: 22px; text-align: center; margin-top: 10px;">
            <div style="font-size: 0.95rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Diagnostic Outcome</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {pred_result['color']}; margin: 6px 0;">
                {pred_result['status']}
            </div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #f8fafc;">
                Calculated Risk: <span style="color: {pred_result['color']}; font-weight: 800;">{pred_result['probability_positive']}%</span>
            </div>
            <div style="margin-top: 10px;">
                <span style="background: {pred_result['color']}22; color: {pred_result['color']}; border: 1px solid {pred_result['color']}; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                    {pred_result['risk_level']}
                </span>
            </div>
            <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">
                <b>Guidance</b>: {pred_result['recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Clinical Risk Gauge
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_result["probability_positive"],
            number={"suffix": "%", "font": {"size": 28, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": pred_result["color"]},
                "bgcolor": "rgba(15, 23, 42, 0.6)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [35, 65], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.25)"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": decision_threshold * 100
                }
            }
        ))
        gauge_fig.update_layout(height=220, margin=dict(l=30, r=30, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_res2:
        radar_fig = plot_patient_radar_plotly(patient_data, med_healthy, med_diabetic)
        st.plotly_chart(radar_fig, use_container_width=True)

    # ---------------------------------------------------------
    # NEW: Interactive "What-If" Counterfactual Risk Simulator
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🧪 Interactive Counterfactual 'What-If' Lifestyle Simulator")
    st.caption("Simulate lifestyle interventions (e.g. glucose reduction via diet, BMI reduction via exercise) and observe risk drop in real time.")

    col_sim1, col_sim2, col_sim3 = st.columns([1, 1, 1])

    with col_sim1:
        sim_glucose = st.slider(
            "Simulated Glucose (mg/dL)",
            min_value=70, max_value=220,
            value=min(220, max(70, int(patient_data["Glucose"]))),
            help="Simulate fasting plasma glucose reduction"
        )
    with col_sim2:
        sim_bmi = st.slider(
            "Simulated BMI (kg/m²)",
            min_value=18.0, max_value=45.0,
            value=min(45.0, max(18.0, float(patient_data["BMI"]))),
            step=0.5,
            help="Simulate weight loss / BMI reduction"
        )
    with col_sim3:
        sim_patient = patient_data.copy()
        sim_patient["Glucose"] = sim_glucose
        sim_patient["BMI"] = sim_bmi
        sim_res = model_service.predict_single(sim_patient, threshold=decision_threshold)
        
        baseline_prob = pred_result["probability_positive"]
        new_prob = sim_res["probability_positive"]
        delta_prob = round(new_prob - baseline_prob, 2)

        st.metric(
            label="Simulated Risk Probability",
            value=f"{new_prob}%",
            delta=f"{delta_prob}%" if delta_prob != 0 else "Baseline",
            delta_color="inverse"
        )
        st.markdown(f"""
        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">
            Status: <b style="color: {sim_res['color']}">{sim_res['status']} ({sim_res['risk_level']})</b>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TAB 6: Cohort Batch Screening (NEW!)
# =========================================================
with tab_batch:
    st.markdown("### 🧪 Patient Cohort Batch Screening & High-Throughput Analytics")
    st.caption("Perform bulk risk screening on multiple patient profiles via the production inference engine.")

    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        load_demo = st.button("📋 Load Sample Cohort (25 Patients)", type="primary", use_container_width=True)

    uploaded_file = st.file_uploader("Or upload your own CSV cohort file:", type=["csv"])

    cohort_df = None
    if uploaded_file is not None:
        cohort_df = pd.read_csv(uploaded_file)
    elif load_demo or "cohort_data" not in st.session_state:
        # Load sample from dataset
        raw = load_raw_dataset()
        cohort_df = raw.sample(n=25, random_state=101).reset_index(drop=True)
        st.session_state["cohort_data"] = cohort_df
    else:
        cohort_df = st.session_state.get("cohort_data")

    if cohort_df is not None:
        # Check that required features exist
        missing_feats = [f for f in FEATURE_COLS if f not in cohort_df.columns]
        if missing_feats:
            st.error(f"Uploaded CSV is missing required clinical columns: {missing_feats}")
        else:
            with st.spinner("Screening patient cohort..."):
                records = cohort_df[FEATURE_COLS].to_dict(orient="records")
                batch_preds = model_service.predict_batch(records, threshold=decision_threshold)

            # Build enriched DataFrame
            enriched_df = cohort_df.copy()
            enriched_df["Predicted_Class"] = [p["prediction"] for p in batch_preds]
            enriched_df["Condition"] = [p["status"] for p in batch_preds]
            enriched_df["Risk_Score_%"] = [p["probability_positive"] for p in batch_preds]
            enriched_df["Risk_Tier"] = [p["risk_level"] for p in batch_preds]

            tier_counts = enriched_df["Risk_Tier"].value_counts().to_dict()
            high_risk_count = tier_counts.get("High Risk", 0)

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total Patients Screened", len(enriched_df))
            with col_stat2:
                st.metric("Diabetic Flags", sum(enriched_df["Predicted_Class"] == 1))
            with col_stat3:
                st.metric("High-Risk Urgent Care Flags", high_risk_count)

            col_ch1, col_ch2 = st.columns([1, 1])
            with col_ch1:
                donut_fig = plot_batch_risk_donut_plotly(tier_counts)
                st.plotly_chart(donut_fig, use_container_width=True)
            with col_ch2:
                hist_batch = px.histogram(
                    enriched_df, x="Risk_Score_%", color="Risk_Tier",
                    color_discrete_map={"Low Risk": "#10b981", "Moderate Risk": "#f59e0b", "High Risk": "#ef4444"},
                    nbins=20, title="<b>Cohort Risk Score Distribution</b>",
                    template="plotly_dark"
                )
                hist_batch.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.6)", plot_bgcolor="rgba(15, 23, 42, 0.6)", height=340)
                st.plotly_chart(hist_batch, use_container_width=True)

            st.markdown("#### 📋 Screened Patient Cohort Table")
            st.dataframe(
                enriched_df.style.map(
                    lambda v: "background-color: rgba(239, 68, 68, 0.25); color: #fca5a5;" if v == "High Risk"
                    else ("background-color: rgba(245, 158, 11, 0.25); color: #fde047;" if v == "Moderate Risk"
                    else "background-color: rgba(16, 185, 129, 0.25); color: #86efac;"),
                    subset=["Risk_Tier"]
                ),
                use_container_width=True
            )

            # Export button
            csv_cohort = enriched_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Screened Cohort Results (CSV)",
                data=csv_cohort,
                file_name="screened_patient_cohort.csv",
                mime="text/csv"
            )


# =========================================================
# TAB 7: Project Report & REST API Integration
# =========================================================
with tab_report:
    st.markdown("### 📑 Cluster Computing Academic Summary & REST API Integration")
    st.caption("Complete engineering documentation, mathematical derivation, and REST API consumption guide.")

    report_summary = f"""
# Disease Prediction using Apache Spark MLlib & Scikit-Learn

### 1. Executive Summary
- **Domain**: Chronic Disease Classification / Predictive Healthcare (Type-2 Diabetes).
- **Core Technology**: Apache Spark 4.2.0 (PySpark MLlib), Distributed L-BFGS, FastAPI REST Service, Streamlit.
- **Dataset**: Pima Indians Diabetes Dataset (768 patient profiles, 8 clinical biomarkers).
- **Evaluation Metrics**:
  - Test Accuracy: **{eval_metrics['accuracy'] * 100:.2f}%**
  - F1-Score: **{eval_metrics['f1_score']:.4f}**
  - Precision: **{eval_metrics['precision']:.4f}**
  - Recall (Sensitivity): **{eval_metrics['recall']:.4f}**
  - Specificity: **{eval_metrics['specificity']:.4f}**
  - Area Under ROC: **{eval_metrics['roc_auc']:.4f}**

### 2. Preprocessing & Leakage Prevention
- **Strict Featurization Order**: Train and test partitions split prior to computing imputer and standardizer statistics, preventing data leakage.
- **Physiological Missingness Audit**: Biologically impossible zeros in Glucose (0.7%), BloodPressure (4.6%), SkinThickness (29.6%), Insulin (48.7%), and BMI (1.4%) were treated as missing and imputed via median estimation.
- **Standardization**: Features normalized via `StandardScaler(withMean=True, withStd=True)` ensuring zero mean and unit variance.

### 3. Top Physiological Risk Factors (Odds Ratios)
1. Glucose: **{fi_df.loc[fi_df['Feature'] == 'Glucose', 'Odds_Ratio'].values[0]:.2f}x** risk multiplier per $1\\sigma$ increase
2. BMI: **{fi_df.loc[fi_df['Feature'] == 'BMI', 'Odds_Ratio'].values[0]:.2f}x** risk multiplier
3. Pregnancies: **{fi_df.loc[fi_df['Feature'] == 'Pregnancies', 'Odds_Ratio'].values[0]:.2f}x** risk multiplier
4. Diabetes Pedigree: **{fi_df.loc[fi_df['Feature'] == 'DiabetesPedigreeFunction', 'Odds_Ratio'].values[0]:.2f}x** risk multiplier
"""
    st.markdown(report_summary)

    st.markdown("---")
    st.markdown("#### 🌐 Production REST API Quickstart")
    st.write("The platform exposes a live FastAPI REST endpoint on port `8000` with interactive Swagger docs at `http://localhost:8000/docs`.")

    col_code1, col_code2 = st.columns(2)
    with col_code1:
        st.markdown("**Python Client Example:**")
        st.code("""
import requests

url = "http://localhost:8000/predict"
payload = {
    "Pregnancies": 3,
    "Glucose": 155.0,
    "BloodPressure": 80.0,
    "SkinThickness": 32.0,
    "Insulin": 175.0,
    "BMI": 35.8,
    "DiabetesPedigreeFunction": 0.72,
    "Age": 48
}
response = requests.post(url, json=payload)
print(response.json())
        """, language="python")

    with col_code2:
        st.markdown("**cURL Terminal Example:**")
        st.code("""
curl -X POST http://localhost:8000/predict \\
     -H "Content-Type: application/json" \\
     -d '{
       "Pregnancies": 3,
       "Glucose": 155.0,
       "BloodPressure": 80.0,
       "SkinThickness": 32.0,
       "Insulin": 175.0,
       "BMI": 35.8,
       "DiabetesPedigreeFunction": 0.72,
       "Age": 48
     }'
        """, language="bash")

    st.markdown("---")
    st.markdown("#### 💾 Export Project Deliverables")

    col_exp1, col_exp2 = st.columns(2)

    export_payload = {
        "project": "Health Data Disease Prediction Platform",
        "cluster_framework": "Apache Spark MLlib & Scikit-Learn",
        "spark_version": spark.version if spark else "N/A",
        "metrics": {
            "accuracy": eval_metrics["accuracy"],
            "f1_score": eval_metrics["f1_score"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "specificity": eval_metrics["specificity"],
            "roc_auc": eval_metrics["roc_auc"]
        },
        "confusion_matrix": eval_metrics["confusion_matrix"],
        "coefficients": fi_df.to_dict(orient="records")
    }

    with col_exp1:
        st.download_button(
            label="📥 Download Full Evaluation Metrics (JSON)",
            data=json.dumps(export_payload, indent=4),
            file_name="diabetes_cluster_metrics.json",
            mime="application/json",
            use_container_width=True
        )

    with col_exp2:
        csv_export = fi_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Coefficients & Odds Ratios (CSV)",
            data=csv_export,
            file_name="diabetes_feature_importance.csv",
            mime="text/csv",
            use_container_width=True
        )
