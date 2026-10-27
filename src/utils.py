"""
Utility functions for plotting, metrics formatting, and visualizations.
Uses Plotly for interactive web dashboard components.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_confusion_matrix_plotly(
    cm_matrix: List[List[int]],
    title: str = "Confusion Matrix (Test Evaluation)"
) -> go.Figure:
    """
    Renders an interactive, publication-grade Confusion Matrix heatmap
    with explicit labels for TP, TN, FP, FN.
    """
    tn, fp = cm_matrix[0][0], cm_matrix[0][1]
    fn, tp = cm_matrix[1][0], cm_matrix[1][1]
    total = tn + fp + fn + tp

    z = [[tn, fp], [fn, tp]]

    text_annotations = [
        [
            f"<b>True Negative (TN)</b><br>{tn}<br>({round(tn/total*100, 1)}%)",
            f"<b>False Positive (FP)</b><br>{fp}<br>({round(fp/total*100, 1)}%)"
        ],
        [
            f"<b>False Negative (FN)</b><br>{fn}<br>({round(fn/total*100, 1)}%)",
            f"<b>True Positive (TP)</b><br>{tp}<br>({round(tp/total*100, 1)}%)"
        ]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted: Healthy (0)", "Predicted: Diabetic (1)"],
        y=["Actual: Healthy (0)", "Actual: Diabetic (1)"],
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 15, "color": "white", "family": "Inter, sans-serif"},
        colorscale=[
            [0.0, "#0f172a"],
            [0.5, "#2563eb"],
            [1.0, "#059669"]
        ],
        showscale=True,
        hoverongaps=False
    ))

    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 17, "color": "#f8fafc"}
        },
        xaxis={"title": "Predicted Diagnosis", "tickfont": {"size": 13, "color": "#cbd5e1"}},
        yaxis={"title": "Actual Clinical Ground Truth", "tickfont": {"size": 13, "color": "#cbd5e1"}, "autorange": "reversed"},
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=380
    )

    return fig


def plot_roc_curve_plotly(
    eval_df: pd.DataFrame,
    roc_auc: float
) -> go.Figure:
    """
    Renders an interactive ROC (Receiver Operating Characteristic) curve with AUC area.
    """
    from sklearn.metrics import roc_curve

    y_true = eval_df["Outcome"].values
    y_prob = eval_df["pos_prob"].values

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig = go.Figure()

    # ROC Line
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"Logistic Regression (AUC = {roc_auc:.3f})",
        line=dict(color="#38bdf8", width=3),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.15)"
    ))

    # Baseline diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Chance (AUC = 0.500)",
        line=dict(color="#94a3b8", width=2, dash="dash")
    ))

    fig.update_layout(
        title="<b>Receiver Operating Characteristic (ROC) Curve</b>",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=380,
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig


def plot_threshold_tradeoff_plotly(
    y_true: List[int],
    y_prob: List[float],
    current_threshold: float = 0.50
) -> go.Figure:
    """
    Plots Sensitivity, Specificity, Precision, and F1-Score curves across
    decision thresholds from 0.05 to 0.95 to help clinicians select optimal cutoffs.
    """
    from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

    thresholds = np.linspace(0.05, 0.95, 37)
    recalls, specificities, precisions, f1s = [], [], [], []

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)

    for th in thresholds:
        preds = (y_prob_arr >= th).astype(int)
        rec = recall_score(y_true_arr, preds, zero_division=0)
        prec = precision_score(y_true_arr, preds, zero_division=0)
        f1 = f1_score(y_true_arr, preds, zero_division=0)

        cm = confusion_matrix(y_true_arr, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        recalls.append(round(rec, 3))
        precisions.append(round(prec, 3))
        f1s.append(round(f1, 3))
        specificities.append(round(spec, 3))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=thresholds, y=recalls, mode="lines+markers",
        name="Sensitivity (Recall)", line=dict(color="#ef4444", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=specificities, mode="lines+markers",
        name="Specificity", line=dict(color="#10b981", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=precisions, mode="lines+markers",
        name="Precision", line=dict(color="#38bdf8", width=2.5, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=f1s, mode="lines+markers",
        name="F1-Score", line=dict(color="#a855f7", width=3)
    ))

    # Add vertical line for active threshold
    fig.add_vline(
        x=current_threshold, line_width=2, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Selected Cutoff ({current_threshold:.2f})",
        annotation_position="top right"
    )

    fig.update_layout(
        title="<b>Clinical Metric Trade-Off Curves Across Decision Thresholds</b>",
        xaxis_title="Classification Probability Threshold",
        yaxis_title="Metric Score (0.0 - 1.0)",
        yaxis_range=[0, 1.05],
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_feature_importance_plotly(fi_df: pd.DataFrame) -> go.Figure:
    """
    Renders a bar chart of Logistic Regression feature coefficients and Odds Ratios.
    """
    plot_df = fi_df.sort_values(by="Coefficient", ascending=True)

    colors = ["#ef4444" if val > 0 else "#38bdf8" for val in plot_df["Coefficient"]]

    fig = go.Figure(go.Bar(
        x=plot_df["Coefficient"],
        y=plot_df["Feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#ffffff", width=0.5)),
        text=[f"OR: {or_val:.2f}" for or_val in plot_df["Odds_Ratio"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.3f}<br>Odds Ratio: %{text}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Clinical Feature Impact (Standardized Coefficients & Odds Ratios)</b>",
        xaxis_title="Standardized Model Coefficient (β)",
        yaxis_title="Clinical Biomarker",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=380,
        margin=dict(l=40, r=60, t=50, b=40)
    )

    return fig


def plot_missing_audit_plotly(missing_stats: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Visualizes the count and percentage of biologically missing zeros in the raw data.
    """
    features = list(missing_stats.keys())
    counts = [missing_stats[f]["missing_zeros"] for f in features]
    percentages = [missing_stats[f]["missing_percentage"] for f in features]

    fig = go.Figure(go.Bar(
        x=features,
        y=counts,
        text=[f"{p}%" for p in percentages],
        textposition="outside",
        marker=dict(
            color=percentages,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Missing %")
        ),
        hovertemplate="<b>%{x}</b><br>Missing Zeros: %{y}<br>Missing %: %{text}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Missing Data Audit (Biologically Invalid Zero Counts)</b>",
        xaxis_title="Clinical Measurement",
        yaxis_title="Count of Missing Records (Zeros)",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig


def plot_cluster_scaling_plotly(bench_df: pd.DataFrame) -> go.Figure:
    """
    Visualizes cluster partition scaling with dual axes:
    Left Y: Execution time (seconds)
    Right Y: Parallel Speedup Factor
    """
    fig = go.Figure()

    # Fit Time
    fig.add_trace(go.Bar(
        x=bench_df["Partitions"].astype(str),
        y=bench_df["Fit_Time_Sec"],
        name="Fit Time (Sec)",
        marker_color="#6366f1"
    ))

    # Total Time Line
    fig.add_trace(go.Scatter(
        x=bench_df["Partitions"].astype(str),
        y=bench_df["Total_Pipeline_Sec"],
        name="Total Pipeline Time (Sec)",
        mode="lines+markers",
        line=dict(color="#f43f5e", width=3)
    ))

    # Speedup Line (Secondary Y)
    fig.add_trace(go.Scatter(
        x=bench_df["Partitions"].astype(str),
        y=bench_df["Speedup_Factor"],
        name="Relative Speedup",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#10b981", width=3, dash="dot")
    ))

    fig.update_layout(
        title="<b>Spark Cluster Scaling Analysis (Partitions vs. Latency & Speedup)</b>",
        xaxis_title="Number of Spark RDD Partitions",
        yaxis=dict(title="Execution Time (Seconds)"),
        yaxis2=dict(
            title="Relative Speedup Factor",
            overlaying="y",
            side="right"
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        legend=dict(x=0.05, y=0.95),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig


def plot_patient_radar_plotly(
    patient_dict: Dict[str, float],
    median_healthy: pd.Series,
    median_diabetic: pd.Series
) -> go.Figure:
    """
    Renders an interactive Radar / Spider chart comparing an individual patient's
    normalized clinical readings against the healthy and diabetic population medians.
    """
    features = [
        "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "Age"
    ]

    bounds = {
        "Glucose": 200.0,
        "BloodPressure": 120.0,
        "SkinThickness": 60.0,
        "Insulin": 300.0,
        "BMI": 50.0,
        "Age": 80.0
    }

    patient_vals = [min(100.0, (patient_dict.get(f, 0) / bounds[f]) * 100) for f in features]
    healthy_vals = [min(100.0, (median_healthy[f] / bounds[f]) * 100) for f in features]
    diabetic_vals = [min(100.0, (median_diabetic[f] / bounds[f]) * 100) for f in features]

    features_closed = features + [features[0]]
    patient_vals += [patient_vals[0]]
    healthy_vals += [healthy_vals[0]]
    diabetic_vals += [diabetic_vals[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=healthy_vals,
        theta=features_closed,
        fill="toself",
        name="Healthy Median",
        fillcolor="rgba(16, 185, 129, 0.15)",
        line=dict(color="#10b981", width=2)
    ))

    fig.add_trace(go.Scatterpolar(
        r=diabetic_vals,
        theta=features_closed,
        fill="toself",
        name="Diabetic Median",
        fillcolor="rgba(239, 68, 68, 0.15)",
        line=dict(color="#ef4444", width=2)
    ))

    fig.add_trace(go.Scatterpolar(
        r=patient_vals,
        theta=features_closed,
        fill="none",
        name="Current Patient Profile",
        line=dict(color="#38bdf8", width=3.5)
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color="#94a3b8")),
            angularaxis=dict(tickfont=dict(size=12, color="#f8fafc"))
        ),
        title="<b>Patient Biomarker Profile vs. Population Medians</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(x=0.8, y=1.1)
    )

    return fig


def plot_batch_risk_donut_plotly(risk_counts: Dict[str, int]) -> go.Figure:
    """
    Renders an interactive donut chart of risk stratification distribution for cohort screening.
    """
    labels = list(risk_counts.keys())
    values = list(risk_counts.values())
    colors_map = {
        "Low Risk": "#10b981",
        "Moderate Risk": "#f59e0b",
        "High Risk": "#ef4444"
    }
    colors = [colors_map.get(lbl, "#38bdf8") for lbl in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="label+percent",
        textfont=dict(size=13, color="white"),
        hoverinfo="label+value+percent"
    )])

    fig.update_layout(
        title="<b>Patient Cohort Risk Stratification</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=340,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        annotations=[dict(text=f"Total<br><b>{sum(values)}</b>", x=0.5, y=0.5, font_size=16, showarrow=False, font_color="#f8fafc")]
    )

    return fig
