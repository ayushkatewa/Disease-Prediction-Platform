"""
Unified Production CLI Runner for Diabetes Disease Prediction Platform
Provides command shortcuts for training, testing, API serving, and web dashboard.

Usage:
    python run.py check      # Verify system diagnostics and dependencies
    python run.py train      # Train and persist production ML pipeline
    python run.py api        # Launch FastAPI REST inference server
    python run.py web        # Launch Streamlit interactive dashboard
    python run.py test       # Execute test suite
"""

import os
import sys
import argparse
import subprocess

# Ensure UTF-8 output on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config import config, get_default_dataset_path
from src.logger import get_logger

logger = get_logger("ProductionRunner")


def cmd_check():
    """Validates runtime environment, dependencies, and datasets."""
    print("=" * 70)
    print(" [HEALTH] SYSTEM & CLUSTER COMPUTING DIAGNOSTIC CHECK")
    print("=" * 70)

    # 1. Python Environment
    print(f"[*] Python Version  : {sys.version.split()[0]} ({sys.executable})")

    # 2. Dataset Verification
    data_path = get_default_dataset_path()
    if os.path.exists(data_path):
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"[+] Dataset Found   : {data_path} ({len(df)} rows, {len(df.columns)} columns)")
    else:
        print(f"[-] Dataset Missing : {data_path}")

    # 3. Model Artifact Check
    model_path = config.model_save_path
    if os.path.exists(model_path):
        print(f"[+] Persisted Model : {model_path} (Ready for instant serving)")
    else:
        print(f"[!] Persisted Model : Not yet trained. Run 'python run.py train'")

    # 4. Dependency Checks
    packages = ["pyspark", "streamlit", "plotly", "fastapi", "uvicorn", "sklearn", "pandas", "numpy", "pytest"]
    print("\n[*] Checking Installed Packages:")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "Available")
            print(f"    - {pkg:<15}: {version}")
        except ImportError:
            print(f"    - {pkg:<15}: NOT INSTALLED")

    # 5. Java / Spark Check
    print("\n[*] Java / Spark Runtime Status:")
    try:
        res = subprocess.run(["java", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        java_out = res.stderr or res.stdout
        print(f"    Java detected: {java_out.splitlines()[0] if java_out else 'Unknown'}")
    except Exception as e:
        print(f"    Java check failed: {e}")

    print("=" * 70)


def cmd_train(args):
    """Executes model training and artifact persistence."""
    from src.model_service import ModelService
    logger.info("Starting model training pipeline...")
    service = ModelService.get_instance()
    results = service.train_and_save(
        csv_path=args.data,
        train_ratio=args.train_ratio,
        reg_val=args.reg_param,
        max_iter=args.max_iter,
        threshold=args.threshold
    )
    metrics = results["eval_metrics"]
    print("\n" + "=" * 60)
    print(" [REPORT] TRAINING & EVALUATION REPORT")
    print("=" * 60)
    print(f"  • Test Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"  • F1-Score      : {metrics['f1_score']:.4f}")
    print(f"  • Precision     : {metrics['precision']:.4f}")
    print(f"  • Recall        : {metrics['recall']:.4f}")
    print(f"  • Specificity   : {metrics['specificity']:.4f}")
    print(f"  • ROC-AUC       : {metrics['roc_auc']:.4f}")
    print("=" * 60)
    print(f"[+] Model saved to: {config.model_save_path}\n")


def cmd_api(args):
    """Starts the FastAPI inference server."""
    logger.info(f"Starting FastAPI server on {args.host}:{args.port}...")
    import uvicorn
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)


def cmd_web(args):
    """Starts the Streamlit web dashboard."""
    logger.info(f"Launching Streamlit dashboard on port {args.port}...")
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        os.path.join(PROJECT_DIR, "app.py"),
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    subprocess.run(cmd)


def cmd_test():
    """Runs automated unit and integration tests."""
    logger.info("Executing pytest test suite...")
    cmd = [sys.executable, "-m", "pytest", "tests/test_pipeline.py", "-v"]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Production Runner: Health Data Disease Prediction Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: check
    subparsers.add_parser("check", help="Run environment diagnostic check")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train and save production model")
    train_parser.add_argument("--data", type=str, default=None, help="Dataset CSV path")
    train_parser.add_argument("--train-ratio", type=float, default=0.80, help="Train/test ratio")
    train_parser.add_argument("--reg-param", type=float, default=0.01, help="Regularization parameter")
    train_parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations")
    train_parser.add_argument("--threshold", type=float, default=0.50, help="Classification threshold")

    # Command: api
    api_parser = subparsers.add_parser("api", help="Launch FastAPI REST server")
    api_parser.add_argument("--host", type=str, default=config.api_host, help="Host address")
    api_parser.add_argument("--port", type=int, default=config.api_port, help="Port number")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Command: web
    web_parser = subparsers.add_parser("web", help="Launch Streamlit dashboard")
    web_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    web_parser.add_argument("--port", type=int, default=config.streamlit_port, help="Port number")

    # Command: test
    subparsers.add_parser("test", help="Run automated test suite")

    args = parser.parse_args()

    if not args.command or args.command == "check":
        cmd_check()
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "api":
        cmd_api(args)
    elif args.command == "web":
        cmd_web(args)
    elif args.command == "test":
        cmd_test()


if __name__ == "__main__":
    main()
