# Production Multi-Stage Dockerfile for Health Data Disease Prediction Platform
# Supports Apache Spark 4.x / PySpark MLlib and Scikit-Learn

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PATH="${JAVA_HOME}/bin:${PATH}" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    API_PORT=8000 \
    API_HOST=0.0.0.0

WORKDIR /app

# Install system dependencies & OpenJDK 17 for Spark
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code
COPY . .

# Run initial model training to persist artifacts
RUN python run.py train || true

# Expose ports: 8501 (Streamlit Dashboard), 8000 (FastAPI REST API)
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default entrypoint: launch Streamlit Dashboard
CMD ["python", "run.py", "web"]

