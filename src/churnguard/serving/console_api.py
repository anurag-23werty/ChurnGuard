"""Read-only JSON endpoints for the web console (web/).

Everything here reads artifacts the pipeline already wrote; nothing is recomputed or
modelled, so the console shows exactly what the batch, selection, backtest and
monitoring jobs produced. A missing artifact is a 404 naming the make target that
creates it.

  GET /api/months                       scored months, newest first
  GET /api/overview?month=              KPIs, risk tiers, reason mix, score histogram
  GET /api/members?month=&q=&tier=...   paginated, filtered, sorted score rows
  GET /api/members/{member_id}?month=   one member's score row + model input snapshot
  GET /api/history                      served months + monitoring replay
  GET /api/policy                       retention economics in force
  GET /api/retention/plan?budget=       contact list re-selected under a campaign budget
  GET /api/export/contacts.csv?budget=  that contact list as a CRM CSV
  GET /api/registry                     registered model versions and aliases
  GET /api/reports/{name}               selection | baselines | backtest | replay
  GET /api/monitoring                   months with a monitoring report
  GET /api/monitoring/{month}           that month's monitoring report
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from churnguard.config import Settings
from churnguard.console.data import budget_selection, crm_export, load_scores
from churnguard.serving.scoring import N_REASONS

# Display bands only: who gets contacted is decided by expected value, not by tier.
TIERS = (("critical", 0.5, 1.0), ("high", 0.2, 0.5), ("medium", 0.05, 0.2), ("low", 0.0, 0.05))
TIER_SQL = ("CASE WHEN p_churn >= 0.5 THEN 'critical' WHEN p_churn >= 0.2 THEN 'high' "
            "WHEN p_churn >= 0.05 THEN 'medium' ELSE 'low' END")
HIST_BINS = 20
SORTS = {"ev_rank": "ev_rank", "p_churn": "p_churn", "ev": "ev",
         "monthly_revenue": "monthly_revenue", "clv": "clv"}
REPORTS = {
    "selection": ("selection/selection.json", "make select"),
    "baselines": ("baselines/metrics.json", "make baselines"),
    "backtest": ("decisioning/backtest_valid.json", "make backtest"),
    "replay": ("monitoring/replay.json", "make monitor-replay"),
}


def _clean(v):
    """JSON-safe scalar: NaN/NaT -> None, numpy/duckdb scalars -> Python."""
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _rows(cur: duckdb.DuckDBPyConnection) -> list[dict]:
    cols = [d[0] for d in cur.description]
    return [{c: _clean(v) for c, v in zip(cols, r, strict=True)} for r in cur.fetchall()]


def _read_json(path: Path, hint: str):
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found; run `{hint}`")
    return json.loads(path.read_text())


def _default_registry(settings: Settings) -> list[dict]:
    import mlflow
    from mlflow.tracking import MlflowClient

    from churnguard.tracking import tracking_uri

    mlflow.set_tracking_uri(tracking_uri(settings))
    client = MlflowClient()
    out = []
    for rm in client.search_registered_models():
        for mv in client.search_model_versions(f"name='{rm.name}'"):
            full = client.get_model_version(rm.name, mv.version)  # search omits aliases
            out.append({"name": rm.name, "version": str(mv.version),
                        "model_type": mv.tags.get("model_type"), "aliases": list(full.aliases),
                        "run_id": mv.run_id, "created_at": mv.creation_timestamp})
    return out


def build_router(settings: Settings,
                 registry_fn: Callable[[Settings], list[dict]] | None = None) -> APIRouter:
    """`registry_fn` is injectable so tests do not touch the real MLflow store."""
    router = APIRouter(prefix="/api", tags=["console"])
    scores_root = settings.paths.gold / "scores"
    reports = settings.paths.reports

    def months() -> list[str]:
        return sorted((p.name.removeprefix("score_month=")
                       for p in scores_root.glob("score_month=*")
                       if (p / "data.parquet").exists()), reverse=True)

    def month_dir(month: str | None) -> tuple[str, Path]:
        available = months()
        if not available:
            raise HTTPException(404, "no scored months; run `make score`")
        month = month or available[0]
        if month not in available:
            raise HTTPException(404, f"month {month} not scored; available: {available}")
        return month, scores_root / f"score_month={month}"

    def con() -> duckdb.DuckDBPyConnection:
        c = duckdb.connect()
        c.execute(f"SET threads = {settings.duckdb.threads}")
        return c

    @router.get("/months")
    def list_months() -> dict:
        return {"months": months()}

    @router.get("/overview")
    def overview(month: str | None = None) -> dict:
        month, d = month_dir(month)
        src = f"read_parquet('{d / 'data.parquet'}')"
        c = con()
        head = _rows(c.execute(f"""
            SELECT count(*) AS members, avg(p_churn) AS mean_p, median(p_churn) AS median_p,
                   count(*) FILTER (WHERE contact) AS contacted,
                   coalesce(sum(ev) FILTER (WHERE contact), 0) AS expected_net_value,
                   coalesce(sum(cost) FILTER (WHERE contact), 0) AS campaign_cost,
                   sum(p_churn * monthly_revenue) AS revenue_at_risk,
                   any_value(model_name) AS model_name, any_value(model_version) AS model_version,
                   any_value(model_type) AS model_type, max(scored_at) AS scored_at
            FROM {src}"""))[0]
        tiers = {r["tier"]: r for r in _rows(c.execute(f"""
            SELECT {TIER_SQL} AS tier, count(*) AS members,
                   sum(p_churn * monthly_revenue) AS revenue_at_risk,
                   count(*) FILTER (WHERE contact) AS contacted
            FROM {src} GROUP BY 1"""))}
        n = head["members"] or 1
        tier_rows = []
        for name, lo, hi in TIERS:
            t = tiers.get(name, {"members": 0, "revenue_at_risk": 0.0, "contacted": 0})
            tier_rows.append({"tier": name, "min_p": lo, "max_p": hi, "members": t["members"],
                              "share": t["members"] / n,
                              "revenue_at_risk": t["revenue_at_risk"] or 0.0,
                              "contacted": t["contacted"]})
        reasons = _rows(c.execute(f"""
            SELECT coalesce(reason_1, '(none)') AS feature, count(*) AS members,
                   count(*) / sum(count(*)) OVER () AS share
            FROM {src} WHERE contact GROUP BY 1 ORDER BY 2 DESC"""))
        hist = {r["bin"]: r["members"] for r in _rows(c.execute(f"""
            SELECT least(floor(p_churn * {HIST_BINS}), {HIST_BINS - 1})::INT AS bin,
                   count(*) AS members FROM {src} GROUP BY 1"""))}
        return {
            "month": month,
            "model": {"name": head.pop("model_name"), "version": head.pop("model_version"),
                      "model_type": head.pop("model_type")},
            **head,
            "contact_share": head["contacted"] / n,
            "tiers": tier_rows,
            "reasons": reasons,
            "histogram": [{"lo": i / HIST_BINS, "hi": (i + 1) / HIST_BINS,
                           "members": hist.get(i, 0)} for i in range(HIST_BINS)],
        }

    @router.get("/members")
    def members(month: str | None = None,
                q: str | None = Query(None, max_length=64,
                                      description="member_id prefix or exact msno"),
                tier: Literal["critical", "high", "medium", "low"] | None = None,
                contact: bool | None = None,
                sort: Literal["ev_rank", "p_churn", "ev", "monthly_revenue", "clv"] = "ev_rank",
                order: Literal["asc", "desc"] | None = None,
                page: int = Query(1, ge=1),
                page_size: int = Query(25, ge=1, le=200)) -> dict:
        month, d = month_dir(month)
        where, params = [], []
        if q and q.strip():
            q = q.strip()
            if q.isdigit():
                where.append("CAST(member_id AS VARCHAR) LIKE ?")
                params.append(f"{q}%")
            else:
                where.append("msno = ?")
                params.append(q)
        if tier:
            where.append(f"{TIER_SQL} = ?")
            params.append(tier)
        if contact is not None:
            where.append("contact = ?")
            params.append(contact)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        direction = (order or ("asc" if sort == "ev_rank" else "desc")).upper()
        reason_cols = ", ".join(f"reason_{i}, reason_{i}_logit" for i in range(1, N_REASONS + 1))
        c = con()
        c.execute(f"CREATE VIEW s AS SELECT * FROM read_parquet('{d / 'data.parquet'}')")
        total = c.execute(f"SELECT count(*) FROM s {clause}", params).fetchone()[0]
        rows = _rows(c.execute(f"""
            SELECT member_id, msno, ev_rank, p_churn, {TIER_SQL} AS tier, monthly_revenue, clv,
                   cost, ev, contact, {reason_cols}
            FROM s {clause}
            ORDER BY {SORTS[sort]} {direction}, member_id
            LIMIT {page_size} OFFSET {(page - 1) * page_size}""", params))
        for r in rows:
            r["reasons"] = [{"feature": r.pop(f"reason_{i}"), "logit": r.pop(f"reason_{i}_logit")}
                            for i in range(1, N_REASONS + 1)]
            r["reasons"] = [x for x in r["reasons"] if x["feature"]]
        return {"month": month, "total": total, "page": page, "page_size": page_size,
                "rows": rows}

    @router.get("/members/{member_id}")
    def member(member_id: int, month: str | None = None) -> dict:
        month, d = month_dir(month)
        c = con()
        hit = _rows(c.execute(f"""SELECT *, {TIER_SQL} AS tier,
                (SELECT count(*) FROM read_parquet('{d / 'data.parquet'}') WHERE contact)
                    AS contacted_total
            FROM read_parquet('{d / 'data.parquet'}') WHERE member_id = ?""", [member_id]))
        if not hit:
            raise HTTPException(404, f"member {member_id} was not scored in {month}")
        score = hit[0]
        score["reasons"] = [{"feature": score.pop(f"reason_{i}"),
                             "logit": score.pop(f"reason_{i}_logit")}
                            for i in range(1, N_REASONS + 1)]
        score["reasons"] = [x for x in score["reasons"] if x["feature"]]
        features = None
        fpath = d / "features.parquet"
        if fpath.exists():
            f = _rows(c.execute(f"SELECT * FROM read_parquet('{fpath}') WHERE member_id = ?",
                                [member_id]))
            features = f[0] if f else None
        return {"month": month, "score": score, "features": features,
                "policy": asdict(settings.retention)}

    @router.get("/history")
    def history() -> dict:
        served = []
        c = con()
        for m in months():
            r = _rows(c.execute(f"""
                SELECT count(*) AS members, avg(p_churn) AS mean_p,
                       count(*) FILTER (WHERE contact) AS contacted, max(scored_at) AS scored_at,
                       any_value(model_name) AS model_name,
                       any_value(model_version) AS model_version
                FROM read_parquet('{scores_root / f'score_month={m}' / 'data.parquet'}')"""))[0]
            served.append({"month": m, **r})
        replay_path = reports / REPORTS["replay"][0]
        replay = json.loads(replay_path.read_text()) if replay_path.exists() else None
        return {"served": served, "replay": replay}

    @router.get("/policy")
    def policy() -> dict:
        return asdict(settings.retention)

    def planned(month: str | None, budget: float | None) -> tuple[str, pd.DataFrame]:
        month, _ = month_dir(month)
        return month, budget_selection(load_scores(settings, month), budget)

    @router.get("/retention/plan")
    def retention_plan(month: str | None = None,
                       budget: float | None = Query(None, ge=0, le=1e9)) -> dict:
        """Same greedy EV/cost selection the batch job uses, under a different budget."""
        month, s = planned(month, budget)
        c = s[s.contact]
        return {"month": month, "budget": budget, "members": len(s), "contacted": len(c),
                "contact_share": len(c) / len(s) if len(s) else 0.0,
                "campaign_cost": float(c.cost.sum()),
                "expected_net_value": float(c.ev.sum()),
                "expected_benefit": float((c.ev + c.cost).sum()),
                "min_p_contacted": float(c.p_churn.min()) if len(c) else None,
                "policy": asdict(settings.retention)}

    @router.get("/export/contacts.csv")
    def export_contacts(month: str | None = None,
                        budget: float | None = Query(None, ge=0, le=1e9)) -> Response:
        month, s = planned(month, budget)
        name = f"churnguard_contacts_{month}" + ("" if budget is None else f"_budget{budget:.0f}")
        return Response(crm_export(s), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})

    @router.get("/registry")
    def registry() -> dict:
        try:
            versions = (registry_fn or _default_registry)(settings)
        except Exception as e:  # noqa: BLE001 - any registry failure is "unavailable"
            raise HTTPException(503, f"model registry unavailable: {type(e).__name__}") from e
        return {"versions": versions}

    @router.get("/reports/{name}")
    def report(name: Literal["selection", "baselines", "backtest", "replay"]):
        rel, hint = REPORTS[name]
        return _read_json(reports / rel, hint)

    @router.get("/monitoring")
    def monitoring_months() -> dict:
        found = sorted((p.stem.removeprefix("month=")
                        for p in (reports / "monitoring").glob("month=*.json")), reverse=True)
        return {"months": found}

    @router.get("/monitoring/{month}")
    def monitoring(month: str):
        if not month.replace("-", "").isdigit():
            raise HTTPException(422, "month must look like 2017-04")
        return _read_json(reports / "monitoring" / f"month={month}.json",
                          f"make monitor MONTH={month}")

    return router
