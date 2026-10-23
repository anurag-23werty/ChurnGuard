"""Read-only console endpoints: they must report exactly what the batch and reports hold."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from churnguard.serving.api import create_app
from tests.test_serving import INFO, batch, model, silver  # noqa: F401 - shared fixtures

VERSIONS = [{"name": "churnguard", "version": "1", "model_type": "logreg",
             "aliases": ["champion"], "run_id": "abc", "created_at": 0}]


@pytest.fixture
def client(batch, silver, model):  # noqa: F811
    return TestClient(create_app(silver, model, INFO, registry_fn=lambda s: VERSIONS))


def test_overview_matches_the_batch(client, batch):  # noqa: F811
    s = batch["scores"]
    assert client.get("/api/months").json() == {"months": ["2017-04"]}
    o = client.get("/api/overview").json()
    assert o["month"] == "2017-04" and o["members"] == len(s)
    assert o["contacted"] == int(s.contact.sum())
    assert o["mean_p"] == pytest.approx(s.p_churn.mean())
    assert o["revenue_at_risk"] == pytest.approx((s.p_churn * s.monthly_revenue).sum())
    assert o["expected_net_value"] == pytest.approx(s.loc[s.contact, "ev"].sum())
    assert sum(t["members"] for t in o["tiers"]) == len(s)
    assert [t["tier"] for t in o["tiers"]] == ["critical", "high", "medium", "low"]
    assert sum(h["members"] for h in o["histogram"]) == len(s)
    assert len(o["histogram"]) == 20
    assert o["model"] == {"name": "test", "version": "7", "model_type": "logreg"}
    assert client.get("/api/overview", params={"month": "2016-01"}).status_code == 404


def test_members_filter_sort_and_page(client, batch):  # noqa: F811
    s = batch["scores"]
    first = client.get("/api/members", params={"page_size": 5}).json()
    assert first["total"] == len(s) and len(first["rows"]) == 5
    assert [r["ev_rank"] for r in first["rows"]] == [1, 2, 3, 4, 5]
    second = client.get("/api/members", params={"page_size": 5, "page": 2}).json()
    assert [r["ev_rank"] for r in second["rows"]] == [6, 7, 8, 9, 10]

    by_p = client.get("/api/members", params={"sort": "p_churn", "page_size": 200}).json()
    ps = [r["p_churn"] for r in by_p["rows"]]
    assert ps == sorted(ps, reverse=True)

    contact = client.get("/api/members", params={"contact": True, "page_size": 200}).json()
    assert contact["total"] == int(s.contact.sum())
    assert all(r["contact"] for r in contact["rows"])

    low = client.get("/api/members", params={"tier": "low", "page_size": 200}).json()
    assert all(r["tier"] == "low" and r["p_churn"] < 0.05 for r in low["rows"])

    one = client.get("/api/members", params={"q": "msno_3"}).json()
    assert [r["member_id"] for r in one["rows"]] == [3]
    prefix = client.get("/api/members", params={"q": "1", "page_size": 200}).json()
    assert prefix["rows"] and all(str(r["member_id"]).startswith("1") for r in prefix["rows"])

    assert client.get("/api/members", params={"sort": "msno"}).status_code == 422
    assert client.get("/api/members", params={"page_size": 500}).status_code == 422


def test_member_detail(client, batch):  # noqa: F811
    row = batch["scores"].set_index("member_id").loc[3]
    body = client.get("/api/members/3").json()
    assert body["score"]["p_churn"] == pytest.approx(row.p_churn, abs=1e-12)
    assert body["score"]["ev"] == pytest.approx(row.ev, abs=1e-9)
    assert [r["feature"] for r in body["score"]["reasons"]] == [
        row[f"reason_{i}"] for i in (1, 2, 3) if isinstance(row[f"reason_{i}"], str)]
    assert body["features"]["member_id"] == 3
    assert body["policy"]["save_rate"] == pytest.approx(0.2)
    assert client.get("/api/members/999").status_code == 404


def test_reports_and_monitoring(client, silver):  # noqa: F811
    missing = client.get("/api/reports/selection")
    assert missing.status_code == 404 and "make select" in missing.json()["detail"]
    assert client.get("/api/reports/other").status_code == 422

    rep = silver.paths.reports
    (rep / "selection").mkdir(parents=True)
    (rep / "selection" / "selection.json").write_text(json.dumps({"selected": "logreg"}))
    assert client.get("/api/reports/selection").json() == {"selected": "logreg"}

    (rep / "monitoring").mkdir()
    (rep / "monitoring" / "month=2017-04.json").write_text(json.dumps({"month": "2017-04"}))
    assert client.get("/api/monitoring").json() == {"months": ["2017-04"]}
    assert client.get("/api/monitoring/2017-04").json() == {"month": "2017-04"}
    assert client.get("/api/monitoring/2017-05").status_code == 404
    assert client.get("/api/monitoring/..%2Fx").status_code in (404, 422)

    h = client.get("/api/history").json()
    assert [m["month"] for m in h["served"]] == ["2017-04"] and h["replay"] is None
    assert client.get("/api/policy").json()["contact_cost"] == pytest.approx(2.0)


def test_registry_injected_and_unavailable(batch, silver, model):  # noqa: F811
    ok = TestClient(create_app(silver, model, INFO, registry_fn=lambda s: VERSIONS))
    assert ok.get("/api/registry").json() == {"versions": VERSIONS}

    def broken(settings):
        raise RuntimeError("no db")

    down = TestClient(create_app(silver, model, INFO, registry_fn=broken))
    r = down.get("/api/registry")
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]


def test_retention_plan_and_export(client, batch):  # noqa: F811
    s = batch["scores"]
    full = client.get("/api/retention/plan").json()
    assert full["contacted"] == int(s.contact.sum())
    assert full["expected_net_value"] == pytest.approx(s.loc[s.contact, "ev"].sum())

    c = s[s.contact]
    budget = float(c.loc[(c.ev / c.cost).idxmax(), "cost"])  # room for the first pick only
    one = client.get("/api/retention/plan", params={"budget": budget}).json()
    assert one["contacted"] == 1 and one["campaign_cost"] <= budget
    assert client.get("/api/retention/plan", params={"budget": -1}).status_code == 422

    csv = client.get("/api/export/contacts.csv", params={"budget": budget})
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    assert "attachment" in csv.headers["content-disposition"]
    lines = csv.text.strip().splitlines()
    assert lines[0].startswith("ev_rank,msno,p_churn") and len(lines) == 2
