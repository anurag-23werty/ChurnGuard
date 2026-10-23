"""MLflow tracking, model registry and the champion/challenger promotion gate.

Registry CLI:
  python -m churnguard.tracking list
  python -m churnguard.tracking demote --model churnguard-gbm --alias champion

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
MIN_MEAN_PR_AUC_GAIN = 0.005
MAX_WORST_AUC_DROP = 0.01
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


def promotion_decision(candidate: dict, baseline: dict, latest_ece: float,
                       champion: dict | None, permutation_passed: bool) -> Decision:
    """`candidate`, `baseline`, `champion` are rows of evaluation.summarise() (metrics over
    every out-of-time month). One good month is not enough: the worst month counts too."""
    reasons = []
    if not permutation_passed:
        reasons.append("baseline permutation control did not pass")
    if candidate["mean_pr_auc"] < baseline["mean_pr_auc"] + MIN_MEAN_PR_AUC_GAIN:
        reasons.append(f"mean PR-AUC {candidate['mean_pr_auc']:.4f} < baseline "
                       f"{baseline['mean_pr_auc']:.4f} + {MIN_MEAN_PR_AUC_GAIN}")
    if candidate["mean_log_loss"] > baseline["mean_log_loss"]:
        reasons.append(f"mean log loss {candidate['mean_log_loss']:.4f} > baseline "
                       f"{baseline['mean_log_loss']:.4f}")
    if candidate["worst_roc_auc"] < baseline["worst_roc_auc"] - MAX_WORST_AUC_DROP:
        reasons.append(f"worst-month AUC {candidate['worst_roc_auc']:.4f} < baseline "
                       f"{baseline['worst_roc_auc']:.4f} - {MAX_WORST_AUC_DROP}")
    if latest_ece > MAX_ECE:
        reasons.append(f"latest-month ECE {latest_ece:.4f} > {MAX_ECE}")
    if champion is not None and candidate["mean_pr_auc"] <= champion.get("mean_pr_auc", 0):
        reasons.append(f"mean PR-AUC {candidate['mean_pr_auc']:.4f} <= champion "
                       f"{champion['mean_pr_auc']:.4f}")
    return Decision(promote=not reasons, reasons=reasons or ["all gates passed"])


def log_and_register(settings: Settings, model_path: Path, run_id: str,
                     input_example: pd.DataFrame, model_type: str = "lgbm") -> str:
    """Log the pyfunc under `run_id`, register it, alias it challenger. Returns version.

    Any object with `predict_proba(X) -> 1-D array` can be served this way."""
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ChurnPyfunc(),
            artifacts={"model": str(model_path)},
            input_example=input_example,
            code_paths=[str(PROJECT_ROOT / "src" / "churnguard")],
        )
    mv = mlflow.register_model(info.model_uri, settings.mlflow.registered_model,
                               tags={"model_type": model_type})
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
    return {k.removeprefix("oot_"): v for k, v in m.items() if k.startswith("oot_")}


def promote(settings: Settings, version: str) -> None:
    MlflowClient().set_registered_model_alias(settings.mlflow.registered_model, CHAMPION,
                                              version)


def champion_info(settings: Settings, alias: str = CHAMPION) -> dict:
    mlflow.set_tracking_uri(tracking_uri(settings))
    mv = MlflowClient().get_model_version_by_alias(settings.mlflow.registered_model, alias)
    return {"name": settings.mlflow.registered_model, "version": str(mv.version),
            "model_type": mv.tags.get("model_type", "unknown"), "run_id": mv.run_id,
            "alias": alias}


def load_champion(settings: Settings, alias: str = CHAMPION):
    """The ChurnModel behind a registry alias, loaded exactly as serving loads it."""
    mlflow.set_tracking_uri(tracking_uri(settings))
    uri = f"models:/{settings.mlflow.registered_model}@{alias}"
    return mlflow.pyfunc.load_model(uri).unwrap_python_model().model


def demote(model: str, alias: str = CHAMPION) -> bool:
    """Remove `alias` from registered model `model`. Returns False if it was not set."""
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(model, alias)
    except mlflow.exceptions.MlflowException:
        return False
    client.delete_registered_model_alias(model, alias)
    return True


def main(argv: list[str] | None = None) -> int:
    import argparse

    from churnguard.config import get_settings

    parser = argparse.ArgumentParser(description="Model registry operations")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    d = sub.add_parser("demote")
    d.add_argument("--model", required=True)
    d.add_argument("--alias", default=CHAMPION)
    args = parser.parse_args(argv)

    setup(get_settings())
    client = MlflowClient()
    if args.cmd == "list":
        for rm in client.search_registered_models():
            for mv in client.search_model_versions(f"name='{rm.name}'"):
                # search results omit aliases; the per-version lookup includes them
                aliases = list(client.get_model_version(rm.name, mv.version).aliases)
                print(f"{rm.name:16} v{mv.version:<3} type={mv.tags.get('model_type', '?'):10} "
                      f"aliases={aliases} run={mv.run_id[:8]}")
    else:
        print("demoted" if demote(args.model, args.alias)
              else f"{args.model} has no alias {args.alias}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
