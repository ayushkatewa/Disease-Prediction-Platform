<div align="center">

# 🏥 Health Data — Disease Prediction Platform
### Distributed Machine Learning & Clinical Decision Support on Apache Spark

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Tests Passing](https://img.shields.io/badge/Tests-Passing%20(pytest)-2ea44f?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

**An enterprise-grade, distributed machine learning platform designed for Cluster Computing.**  
Built with **Apache Spark (PySpark MLlib)**, **Scikit-Learn**, a **FastAPI REST API**, and a modern **Streamlit Web Dashboard** featuring real-time Confusion Matrix analytics, patient risk profiling, interactive "What-If" lifestyle simulation, and cohort batch screening.

[Explore Features](#-key-features) • [Quick Start](#-quick-start) • [REST API Docs](#-production-rest-api-reference) • [Cluster Architecture](#-cluster-computing-architecture) • [Docker Deployment](#-docker-deployment)

</div>

---

## 📌 Executive Summary & Clinical Context

- **Clinical Problem**: Type-2 Diabetes is a chronic metabolic condition affecting over 500 million adults globally. Early diagnosis allows proactive behavioral and pharmaceutical interventions that prevent microvascular and cardiovascular damage.
- **Dataset**: Pima Indians Diabetes Dataset (`diabetes.csv`, 768 patient clinical profiles, 8 physiological biomarkers).
- **Core Engineering Challenge**: Health data frequently exhibits biologically invalid zeros (e.g. Glucose = 0, Blood Pressure = 0), non-standardized feature scales (Insulin reaching 846 vs. Pedigree Function around 0.5), and data leakage risks.
- **The Solution**: An end-to-end distributed ML pipeline utilizing **distributed median imputation**, **StandardScaler normalization**, **L-BFGS Quasi-Newton optimization**, strict **zero-leakage train/test partitioning**, a **FastAPI inference microservice**, and an interactive **Streamlit dashboard**.

---

## ✨ Key Features

| Capability | Engineering Highlight | Clinical Value |
| :--- | :--- | :--- |
| **⚡ Distributed MLlib** | Apache Spark 4.2.0 RDD / DataFrame tree-aggregated gradients via L-BFGS | Scales horizontally across multi-node clusters |
| **🛡️ Zero-Leakage Featurization** | Imputer and StandardScaler statistics ($\mu, \sigma$) fitted strictly on train partition | Eliminates test-set contamination and optimistic evaluation bias |
| **🌐 Production REST API** | FastAPI with Pydantic v2 schemas, OpenAPI/Swagger docs, and CORS enabled | Instant integration with Electronic Health Record (EHR) systems |
| **🧪 "What-If" Simulator** | Interactive counterfactual slider simulating glucose and BMI reduction | Empowers clinicians to demonstrate risk reduction from lifestyle change |
| **📋 Cohort Batch Screening** | High-throughput batch inference with CSV upload and risk tier tags | Rapid community and hospital population health screening |
| **📈 Live Metric Scrubber** | Real-time Sensitivity vs. Specificity trade-off curves across decision thresholds | Allows customizing threshold based on clinical diagnostic priorities |
| **🐳 Containerized Deployment** | Multi-service `docker-compose.yml` with OpenJDK 17 and Python 3.11 | Reproducible, 1-command deployment across cloud or on-premise clusters |

---

## 🏗️ Cluster Computing Architecture

```
                                  ┌───────────────────────────────┐
                                  │   Production Entrypoints      │
                                  │   (FastAPI / Streamlit / CLI) │
                                  └───────────────┬───────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
   ┌───────────────────────────┐                                     ┌───────────────────────────┐
   │    PySpark MLlib Engine   │                                     │   Scikit-Learn Pipeline   │
   │  (Distributed Cluster)    │                                     │    (Local / Production)   │
   ├───────────────────────────┤                                     ├───────────────────────────┤
   │ • Partitioned DataFrames  │                                     │ • Stratified Splitting    │
   │ • Distributed Imputation  │                                     │ • Median Imputation       │
   │ • VectorAssembler         │                                     │ • StandardScaler          │
   │ • StandardScaler (Norm)   │                                     │ • Logistic Regression     │
   │ • Distributed L-BFGS      │                                     │ • Model Serialization     │
   └─────────────┬─────────────┘                                     └─────────────┬─────────────┘
                 │                                                                 │
                 └────────────────────────────────┬────────────────────────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │    Model Service & Serving    │
                                  ├───────────────────────────────┤
                                  │ • Pydantic Input Validation   │
                                  │ • Clinical Range Checking     │
                                  │ • Risk Tier Stratification    │
                                  │ • Feature Odds Ratios (e^β)   │
                                  │ • Model Persistence (.joblib) │
                                  └───────────────┬───────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
   ┌───────────────────────────┐                                     ┌───────────────────────────┐
   │   FastAPI REST Service    │                                     │    Streamlit Dashboard    │
   │   (Port 8000 / OpenAPI)   │                                     │   (Port 8501 / Plotly)    │
   │   • GET  /health          │                                     │   • Live Confusion Matrix │
   │   • POST /predict         │                                     │   • Cluster Architecture  │
   │   • POST /predict/batch   │                                     │   • Data Explorer & Audit │
   │   • GET  /metrics         │                                     │   • Patient Radar Profile │
   │   • GET  /features        │                                     │   • What-If Risk Simulator│
   │   • POST /train           │                                     │   • Cohort Batch Screen   │
   └───────────────────────────┘                                     └───────────────────────────┘
```

### Key Distributed Computing Concepts Demonstrated
1. **Partitioning & Resilient Distributed Datasets (RDDs)**: Data is split across executor cores to parallelize transformation DAGs.
2. **Distributed L-BFGS (Quasi-Newton)**: Each worker computes partial gradients on its local partition. Partial gradients are tree-aggregated (`treeAggregate`) in $O(\log k)$ communication depth, avoiding driver bandwidth saturation.
3. **Catalyst Optimizer**: Spark constructs an optimized logical plan, pushing projections and filters to storage prior to execution.
4. **Fault Tolerance & Resilience**: Pipeline automatically falls back to Scikit-Learn if Spark cluster nodes or JVM configurations are unavailable.

---

## 📊 Preprocessing & Physiological Missingness

In health datasets, measurements such as **Glucose, Blood Pressure, Insulin, Skin Thickness, and BMI cannot physically be 0 in a living human**. In the raw dataset, 0 indicates an unrecorded or missing observation:

| Biomarker | Raw Zero Count | Missing % | Clinical Imputation Strategy |
| :--- | :---: | :---: | :--- |
| **Glucose** | 5 | 0.7% | Distributed Median Imputation |
| **Blood Pressure** | 35 | 4.6% | Distributed Median Imputation |
| **Skin Thickness** | 227 | 29.6% | Distributed Median Imputation |
| **Insulin** | 374 | 48.7% | Distributed Median Imputation |
| **BMI** | 11 | 1.4% | Distributed Median Imputation |

### Feature Standardization
All 8 features are transformed via `StandardScaler(withMean=True, withStd=True)`:
$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$
This prevents large-scale attributes (e.g. Insulin up to $846\,\mu\text{U/ml}$) from distorting smaller-scale attributes (e.g. Pedigree Function $\sim 0.5$).

---

## 📈 Evaluation Results & Confusion Matrix

Evaluated on the held-out test partition:

| Metric | Score | Clinical Meaning |
| :--- | :---: | :--- |
| **Accuracy** | **79.67%** | Overall correct diagnoses |
| **F1-Score** | **0.6667** | Harmonic mean of precision and recall |
| **Precision** | **0.7812** | Likelihood that a flagged patient is truly diabetic |
| **Recall (Sensitivity)** | **0.5814** | Fraction of diabetic patients detected (minimizing false negatives) |
| **Specificity** | **0.9125** | Fraction of healthy patients correctly cleared |
| **ROC-AUC** | **0.8640** | High discriminatory ability across all decision boundaries |

### Confusion Matrix (Test Partition)
```
                       Predicted: Healthy (0)     Predicted: Diabetic (1)
 Actual: Healthy (0)       TN = 73 (59.3%)             FP = 7  (5.7%)
 Actual: Diabetic (1)      FN = 18 (14.6%)             TP = 25 (20.3%)
```

### Physiological Risk Drivers (Odds Ratios $e^\beta$)
1. **Glucose ($\text{OR} = 2.69$)**: Every $1\sigma$ increase in fasting glucose multiplies diabetes odds by **2.69x**.
2. **BMI ($\text{OR} = 1.83$)**: Elevated Body Mass Index increases odds by **1.83x**.
3. **Pregnancies ($\text{OR} = 1.44$)**: History of gestational metabolic load increases risk by **1.44x**.
4. **Diabetes Pedigree Function ($\text{OR} = 1.36$)**: Hereditary genetic predisposition increases risk by **1.36x**.

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/diabetes-cluster-computing.git
cd diabetes-cluster-computing

# Install Python requirements
pip install -r requirements.txt
```

### 2. Verify System & Environment Diagnostics
```bash
python run.py check
```

### 3. Train the Model & Persist Artifacts
```bash
python run.py train
```

### 4. Launch the Interactive Streamlit Web Dashboard
```bash
python run.py web
# Access in browser at: http://localhost:8501
```

### 5. Launch the FastAPI Production REST Server
```bash
python run.py api
# Interactive Swagger Documentation: http://localhost:8000/docs
# Interactive ReDoc:                 http://localhost:8000/redoc
```

### 6. Run the Test Suite
```bash
python run.py test
# Or directly:
pytest tests/test_pipeline.py -v
```

---

## 🌐 Production REST API Reference

The FastAPI service provides high-throughput, validated endpoints for health systems.

### `GET /health`
Returns system status, active engine, model persistence state, and Spark status.
```bash
curl -X GET http://localhost:8000/health
```

### `POST /predict`
Evaluates an individual patient and returns calibrated probability, risk tier, and key risk drivers.
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
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
```
**Response:**
```json
{
  "prediction": 1,
  "status": "Diabetic (Positive)",
  "probability_positive": 73.29,
  "risk_level": "High Risk",
  "color": "#ef4444",
  "recommendation": "Immediate clinical evaluation and comprehensive diagnostic blood panels recommended.",
  "decision_threshold_used": 0.5,
  "top_risk_factors": [
    {
      "feature": "Glucose",
      "patient_value": 155.0,
      "odds_ratio": 3.34,
      "is_elevated": true,
      "normal_range": [70, 99]
    },
    {
      "feature": "BMI",
      "patient_value": 35.8,
      "odds_ratio": 2.02,
      "is_elevated": true,
      "normal_range": [18.5, 24.9]
    }
  ]
}
```

### `POST /predict/batch`
Processes multiple patient records simultaneously for population screening.

### `GET /metrics`
Returns current model accuracy, F1-score, precision, recall, specificity, ROC-AUC, and confusion matrix.

### `GET /features`
Returns standardized logistic regression coefficients and calculated Odds Ratios.

---

## 🐳 Docker Deployment

The project includes multi-stage containerization with **OpenJDK 17** and **Python 3.11**:

```bash
# Build and launch both Dashboard (8501) and REST API (8000)
docker compose up --build -d

# Check running containers
docker compose ps

# View live container logs
docker compose logs -f
```

---

## 📁 Repository Structure

```
diabetes-cluster-computing/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Automated GitHub Actions CI workflow
├── data/
│   └── diabetes.csv                   # Pima Indians Diabetes clinical dataset
├── models/
│   └── diabetes_pipeline_model.joblib # Persisted production model artifact
├── src/
│   ├── __init__.py
│   ├── config.py                      # Centralized configuration & physiological bounds
│   ├── logger.py                      # Rotating file and console logging
│   ├── model_service.py               # Model lifecycle, validation, inference singleton
│   ├── spark_pipeline.py              # PySpark MLlib distributed pipeline
│   ├── sklearn_pipeline.py            # Comparative Scikit-Learn baseline
│   ├── cluster_benchmarks.py          # Partition scalability benchmark suite
│   └── utils.py                       # Plotly visualization components
├── tests/
│   └── test_pipeline.py               # Automated pytest suite (7 unit & integration tests)
├── .dockerignore                      # Docker ignore rules
├── .gitignore                         # Git ignore rules for Python, Spark, and Docker
├── api.py                             # FastAPI REST API with Swagger docs
├── app.py                             # Interactive Streamlit Web Dashboard (7 Tabs)
├── docker-compose.yml                 # Multi-service container orchestration
├── Dockerfile                         # Production container definition (Java 17 + Python)
├── evaluation_results.json            # Model evaluation metrics & confusion matrix
├── LICENSE                            # MIT License
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Production Python dependencies
├── run.py                             # Unified CLI runner (check, train, api, web, test)
└── train_spark.py                     # CLI cluster PySpark training script
```

---

## 🧪 Automated Testing

Run the test suite to validate data integrity, Scikit-Learn pipeline, ModelService validation, API routes, and Spark pipeline:

```bash
pytest tests/test_pipeline.py -v
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <b>Built for Cluster Computing & Clinical AI Research</b> • Apache Spark 4.2.0 • FastAPI • Streamlit
</div>
