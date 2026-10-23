"""FastAPI service for on-demand scoring.

  GET  /health                          live model name/version/type
  GET  /members/{member_id}/score       features built from silver as of a date, scored
  POST /score                           score caller-supplied registry features

Both scoring routes go through serving.scoring, the same code as the batch job, so a
member scored here and in the batch file gets the same number (tested).

Run:
  uvicorn churnguard.serving.api:app --port 8000      (or `make serve`)
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, create_model

from churnguard import tracking
from churnguard.config import Settings, get_settings
from churnguard.db import connect
from churnguard.features.registry import registry
from churnguard.serving.scoring import (
    N_REASONS,
    features_for,
    score_frame,
    single_member_population,
)

_PY_TYPES = {"int": int, "float": float, "str": str}
AsOf = Annotated[date, Query(description="decision date, e.g. 2017-04-01")]

# Request schema generated from the registry: a missing or mistyped feature is a 422.
FeatureRow = create_model(
    "FeatureRow",
    **{f.name: (_PY_TYPES[f.dtype] | None, Field(default=..., description=f.description))
       for f in registry().values()},
)


class Reason(BaseModel):
    feature: str
    logit_contribution: float


class ScoreResponse(BaseModel):
    member_id: int | None
    as_of_date: date | None
    p_churn: float
    monthly_revenue: float
    clv: float
    expected_value: float
    contact: bool
    reasons: list[Reason]
    model: dict[str, Any]


class ScoreRequest(BaseModel):
    features: FeatureRow  # type: ignore[valid-type]
    member_id: int | None = None


def _to_response(row: pd.Series, info: dict) -> ScoreResponse:
    reasons = [Reason(feature=row[f"reason_{i}"], logit_contribution=row[f"reason_{i}_logit"])
               for i in range(1, N_REASONS + 1) if isinstance(row[f"reason_{i}"], str)]
    return ScoreResponse(
        member_id=None if pd.isna(row.get("member_id")) else int(row["member_id"]),
        as_of_date=None if pd.isna(row.get("as_of_date")) else pd.Timestamp(
            row["as_of_date"]).date(),
        p_churn=float(row["p_churn"]), monthly_revenue=float(row["monthly_revenue"]),
        clv=float(row["clv"]), expected_value=float(row["ev"]), contact=bool(row["contact"]),
        reasons=reasons, model=info)


def create_app(settings: Settings | None = None, model=None,
               model_info: dict | None = None) -> FastAPI:
    """Model and settings are injectable so tests run without a registry."""
    settings = settings or get_settings()
    app = FastAPI(title="ChurnGuard", version="0.1.0")

    @lru_cache(maxsize=1)
    def _model():
        if model is not None:
            return model, model_info or {"name": "injected", "version": "0"}
        return tracking.load_champion(settings), tracking.champion_info(settings)

    @app.get("/health")
    def health() -> dict:
        _, info = _model()
        return {"status": "ok", "model": info}

    @app.get("/members/{member_id}/score", response_model=ScoreResponse)
    def score_member(member_id: int, as_of: AsOf):
        m, info = _model()
        con = connect(settings)
        pop = single_member_population(con, settings, member_id, as_of)
        raw = features_for(con, settings, pop)
        if raw.empty:
            raise HTTPException(404, f"member {member_id} has no transactions before {as_of}")
        return _to_response(score_frame(raw, m, settings.retention).iloc[0], info)

    @app.post("/score", response_model=ScoreResponse)
    def score_features(req: ScoreRequest):
        m, info = _model()
        raw = pd.DataFrame([req.features.model_dump()])
        raw["member_id"], raw["as_of_date"], raw["last_expire"] = req.member_id, None, None
        return _to_response(score_frame(raw, m, settings.retention).iloc[0], info)

    return app


def __getattr__(name: str):
    # `uvicorn churnguard.serving.api:app` builds the app lazily (no registry at import).
    if name == "app":
        return create_app()
    raise AttributeError(name)
