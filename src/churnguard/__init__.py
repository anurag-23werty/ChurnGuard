"""ChurnGuard: churn prediction and retention platform."""

import os

# Silence MLflow's agent hint. Set here because `python -m churnguard.<module>` loads this
# package before any module that imports mlflow.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

__version__ = "0.1.0"
