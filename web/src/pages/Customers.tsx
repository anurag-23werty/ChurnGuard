import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { useMembers, useOverview } from "../api/hooks";
import type { SortKey, Tier } from "../api/types";
import { IconArrowDown, IconArrowUp, IconChevronLeft, IconChevronRight, IconSearch } from "../components/icons";
import { ExportButton } from "../components/ExportButton";
import { PageHead, useMonthHref } from "../components/Shell";
import { Card, EmptyState, ErrorState, Segmented, Skeleton, TierBadge, Track } from "../components/ui";
import { featureLabel } from "../lib/features";
import { int, monthLabel, ntd, pct, TIER_COLOR } from "../lib/format";
import { useMonth } from "../lib/month";

const TIERS: (Tier | "all")[] = ["all", "critical", "high", "medium", "low"];
const SIZES = [25, 50, 100];
const COLUMNS: { key: SortKey | null; label: string; align?: "r" }[] = [
  { key: "ev_rank", label: "EV rank", align: "r" },
  { key: null, label: "Member" },
  { key: "p_churn", label: "Churn probability" },
  { key: null, label: "Tier" },
  { key: "monthly_revenue", label: "Monthly rev.", align: "r" },
  { key: "clv", label: "CLV", align: "r" },
  { key: "ev", label: "Expected value", align: "r" },
  { key: null, label: "Action" },
  { key: null, label: "Top driver" },
];

export function CustomersPage() {
  const { month } = useMonth();
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const href = useMonthHref();
  const overview = useOverview(month);

  const q = params.get("q") ?? "";
  const tier = (params.get("tier") as Tier | null) ?? undefined;
  const contact = params.get("contact") === "1" ? true : undefined;
  const sort = (params.get("sort") as SortKey | null) ?? "ev_rank";
  const order = (params.get("order") as "asc" | "desc" | null) ?? (sort === "ev_rank" ? "asc" : "desc");
  const page = Math.max(1, Number(params.get("page")) || 1);
  const pageSize = SIZES.includes(Number(params.get("size"))) ? Number(params.get("size")) : 25;

  const update = (patch: Record<string, string | undefined>, resetPage = true) =>
    setParams(
      (p) => {
        const n = new URLSearchParams(p);
        for (const [k, v] of Object.entries(patch)) (v === undefined || v === "" ? n.delete(k) : n.set(k, v));
        if (resetPage) n.delete("page");
        return n;
      },
      { replace: true },
    );

  // Debounced search box bound to ?q=.
  const [draft, setDraft] = useState(q);
  useEffect(() => setDraft(q), [q]);
  useEffect(() => {
    if (draft === q) return;
    const t = window.setTimeout(() => update({ q: draft.trim() || undefined }), 300);
    return () => window.clearTimeout(t);
  }, [draft]);

  const members = useMembers({ month, q: q || undefined, tier, contact, sort, order, page, page_size: pageSize });
  const data = members.data;
  const pages = data ? Math.max(1, Math.ceil(data.total / pageSize)) : 1;
  const tierCounts = new Map(overview.data?.tiers.map((t) => [t.tier, t.members]) ?? []);

  const toggleSort = (key: SortKey) => {
    if (sort === key) update({ order: order === "asc" ? "desc" : "asc" });
    else update({ sort: key, order: undefined });
  };

  return (
    <>
      <PageHead
        title="Customers"
        actions={<ExportButton month={month} contacted={overview.data?.contacted} />}
      >
        {month ? `${monthLabel(month)} scores` : "Scores"}
        {overview.data ? ` · ${int(overview.data.members)} members` : ""} · search, filter and open any member's risk profile
      </PageHead>

      <div className="card toolbar" style={{ padding: 12 }}>
        <label className="search">
          <span className="sr-only">Search by member ID or msno</span>
          <IconSearch />
          <input
            className="input"
            type="search"
            placeholder="Member ID prefix or exact msno"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
        </label>
        <Segmented
          label="Risk tier"
          value={tier ?? "all"}
          onChange={(v) => update({ tier: v === "all" ? undefined : v })}
          options={TIERS.map((t) => ({
            value: t,
            label: (
              <>
                {t === "all" ? "All" : t[0].toUpperCase() + t.slice(1)}
                {t !== "all" && tierCounts.has(t) && <span className="num small faint">{compact(tierCounts.get(t)!)}</span>}
              </>
            ),
          }))}
        />
        <label className="check">
          <input type="checkbox" checked={!!contact} onChange={(e) => update({ contact: e.target.checked ? "1" : undefined })} />
          Contact list only
        </label>
        {(q || tier || contact) && (
          <button type="button" className="btn ghost" onClick={() => { setDraft(""); update({ q: undefined, tier: undefined, contact: undefined }); }}>
            Clear filters
          </button>
        )}
      </div>

      <Card flush>
        {members.isError ? (
          <ErrorState error={members.error} what="Members" retry={() => members.refetch()} />
        ) : (
          <>
            <div className="table-wrap" aria-busy={members.isFetching}>
              <table className="data" style={{ minWidth: 1080, opacity: members.isPlaceholderData ? 0.6 : 1, transition: "opacity .15s" }}>
                <caption className="sr-only">Scored members, sorted by {sort} {order}</caption>
                <thead>
                  <tr>
                    {COLUMNS.map((c) => (
                      <th key={c.label} scope="col" className={c.align} aria-sort={c.key && c.key === sort ? (order === "asc" ? "ascending" : "descending") : undefined}>
                        {c.key ? (
                          <button type="button" className="sort" aria-pressed={c.key === sort} onClick={() => toggleSort(c.key!)}>
                            {c.label}
                            {c.key === sort && (order === "asc" ? <IconArrowUp /> : <IconArrowDown />)}
                          </button>
                        ) : (
                          c.label
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {!data
                    ? Array.from({ length: 8 }, (_, i) => (
                        <tr key={i}>
                          {COLUMNS.map((c) => (
                            <td key={c.label}>
                              <Skeleton h={12} w={c.label === "Top driver" ? 180 : 64} />
                            </td>
                          ))}
                        </tr>
                      ))
                    : data.rows.map((r) => (
                        <tr key={r.member_id}>
                          <td className="r num muted">{int(r.ev_rank)}</td>
                          <td>
                            <button
                              type="button"
                              className="linkish"
                              onClick={() => navigate(href(`/customers/${r.member_id}`))}
                              aria-label={`Open member ${r.member_id}`}
                            >
                              {r.member_id}
                            </button>
                          </td>
                          <td>
                            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                              <div style={{ width: 72 }}>
                                <Track value={r.p_churn} color={TIER_COLOR[r.tier]} height={6} />
                              </div>
                              <span className="num">{pct(r.p_churn)}</span>
                            </div>
                          </td>
                          <td>
                            <TierBadge tier={r.tier} />
                          </td>
                          <td className="r num">{ntd(r.monthly_revenue)}</td>
                          <td className="r num">{ntd(r.clv)}</td>
                          <td className={`r num ${r.ev > 0 ? "pos" : "muted"}`}>{ntd(r.ev, { signed: true })}</td>
                          <td>
                            {r.contact ? (
                              <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                                <span className="dot" style={{ color: "var(--violet)" }} />
                                Contact
                              </span>
                            ) : (
                              <span className="small muted">Hold</span>
                            )}
                          </td>
                          <td>
                            <span className="badge">{featureLabel(r.reasons[0]?.feature)}</span>
                          </td>
                        </tr>
                      ))}
                </tbody>
              </table>
            </div>
            {data && data.total === 0 && (
              <EmptyState
                title="No members match these filters"
                action={
                  <button type="button" className="btn" onClick={() => { setDraft(""); update({ q: undefined, tier: undefined, contact: undefined }); }}>
                    Clear filters
                  </button>
                }
              >
                Member search matches an ID prefix or an exact msno.
              </EmptyState>
            )}
            <div className="pager">
              <span className="num small muted">
                {data && data.total > 0
                  ? `${int((page - 1) * pageSize + 1)}–${int(Math.min(page * pageSize, data.total))} of ${int(data.total)}`
                  : data
                    ? "0 results"
                    : "Loading…"}
              </span>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <label className="small muted" style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                  Rows
                  <select className="select num" value={pageSize} onChange={(e) => update({ size: e.target.value })}>
                    {SIZES.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </label>
                <button type="button" className="btn icon" aria-label="Previous page" disabled={page <= 1} onClick={() => update({ page: String(page - 1) }, false)}>
                  <IconChevronLeft />
                </button>
                <span className="num small muted" style={{ padding: "0 4px" }}>
                  {int(page)} / {int(pages)}
                </span>
                <button type="button" className="btn icon" aria-label="Next page" disabled={!data || page >= pages} onClick={() => update({ page: String(page + 1) }, false)}>
                  <IconChevronRight />
                </button>
              </div>
            </div>
          </>
        )}
      </Card>
    </>
  );
}

function compact(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n);
}
