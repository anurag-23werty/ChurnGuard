"""MLflow tracking, model registry and the champion/challenger promotion gate.

Every trained model is registered and aliased `challenger`. It becomes `champion` (the
alias serving loads) only if `promotion_decision` says so: it must beat the baseline and
the current champion on held-out data and be well calibrated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from churnguard.config import PROJECT_ROOT, Settings  # noqa: E402

CHAMPION, CHALLENGER = "champion", "challenger"
MIN_PR_AUC_GAIN = 0.01
MAX_ECE = 0.01


def tracking_uri(settings: Settings) -> str:
    uri = settings.mlflow.tracking_uri
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        return f"{prefix}{PROJECT_ROOT / uri[len(prefix):]}"
    return uri


def setup(settings: Settings, artifact_root: Path | None = None) -> str:
    mlflow.set_tracking_uri(tracking_uri(settings))
    exp = mlflow.get_experiment_by_name(settings.mlflow.experiment)
    if exp is None:
        artifacts = (artifact_root or PROJECT_ROOT / "mlartifacts").as_uri()
        mlflow.create_experiment(settings.mlflow.experiment, artifact_location=artifacts)
    mlflow.set_experiment(settings.mlflow.experiment)
    return mlflow.get_tracking_uri()


class ChurnPyfunc(mlflow.pyfunc.PythonModel):
    """Registry wrapper: `mlflow.pyfunc.load_model("models:/<name>@champion")` returns
    calibrated churn probabilities for a frame of registry features."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

    # No type hints: MLflow would infer an `Any` schema from them; the signature comes from
    # the registry-feature input example instead.
    def predict(self, context, model_input, params=None):
        return pd.DataFrame({"churn_probability": self.model.predict_proba(model_input)},
                            index=model_input.index)


@dataclass(frozen=True)
class Decision:
    promote: bool
    reasons: list[str]


def promotion_decision(candidate: dict, baseline: dict, champion: dict | None,
                       permutation_passed: bool) -> Decision:
    """All inputs are valid-split metric dicts from modeling.metrics.evaluate."""
    reasons = []
    if not permutation_passed:
        reasons.append("baseline permutation control did not pass")
    if candidate["pr_auc"] < baseline["pr_auc"] + MIN_PR_AUC_GAIN:
        reasons.append(f"PR-AUC {candidate['pr_auc']:.4f} < baseline "
                       f"{baseline['pr_auc']:.4f} + {MIN_PR_AUC_GAIN}")
    if candidate["log_loss"] >= baseline["log_loss"]:
        reasons.append(f"log loss {candidate['log_loss']:.4f} >= baseline "
                       f"{baseline['log_loss']:.4f}")
    if candidate["ece"] > MAX_ECE:
        reasons.append(f"ECE {candidate['ece']:.4f} > {MAX_ECE}")
    if champion is not None and candidate["pr_auc"] <= champion["pr_auc"]:
        reasons.append(f"PR-AUC {candidate['pr_auc']:.4f} <= champion {champion['pr_auc']:.4f}")
    return Decision(promote=not reasons, reasons=reasons or ["all gates passed"])


def log_and_register(settings: Settings, model_path: Path, run_id: str,
                     input_example: pd.DataFrame) -> str:
    """Log the pyfunc under `run_id`, register it, alias it challenger. Returns version."""
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ChurnPyfunc(),
            artifacts={"model": str(model_path)},
            input_example=input_example,
            code_paths=[str(PROJECT_ROOT / "src" / "churnguard")],
        )
    mv = mlflow.register_model(info.model_uri, settings.mlflow.registered_model)
    MlflowClient().set_registered_model_alias(settings.mlflow.registered_model, CHALLENGER,
                                              mv.version)
    return mv.version


def champion_metrics(settings: Settings) -> dict | None:
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(settings.mlflow.registered_model, CHAMPION)
    except mlflow.exceptions.MlflowException:
        return None
    m = client.get_run(mv.run_id).data.metrics
    return {k.removeprefix("valid_"): v for k, v in m.items() if k.startswith("valid_")}


def promote(settings: Settings, version: str) -> None:
    MlflowClient().set_registered_model_alias(settings.mlflow.registered_model, CHAMPION,
                                              version)


def load_champion(settings: Settings, alias: str = CHAMPION):
    """The ChurnModel behind a registry alias, loaded exactly as serving loads it."""
    mlflow.set_tracking_uri(tracking_uri(settings))
    uri = f"models:/{settings.mlflow.registered_model}@{alias}"
    return mlflow.pyfunc.load_model(uri).unwrap_python_model().model
