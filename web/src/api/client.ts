import type {
  Backtest,
  Baselines,
  Health,
  History,
  MemberDetail,
  MemberPage,
  MemberQuery,
  MonitoringReport,
  Overview,
  Policy,
  RegistryVersion,
  RetentionPlan,
  ScoreResponse,
  Selection,
} from "./types";

/** An HTTP error from the API. `status === 0` means the service could not be reached. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }

  /** The endpoint answered but the artifact behind it does not exist (yet). */
  get unavailable(): boolean {
    return this.status === 404 || this.status === 503;
  }
}

type Params = Record<string, string | number | boolean | undefined | null>;

export function buildUrl(path: string, params: Params = {}): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `${path}?${s}` : path;
}

async function detailOf(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* not JSON */
  }
  return res.statusText || `HTTP ${res.status}`;
}

export async function getJson<T>(path: string, params?: Params, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(buildUrl(path, params), { signal, headers: { Accept: "application/json" } });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "Cannot reach the ChurnGuard API. Is `make serve` running?");
  }
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  return (await res.json()) as T;
}

export const api = {
  health: (s?: AbortSignal) => getJson<Health>("/health", {}, s),
  months: (s?: AbortSignal) => getJson<{ months: string[] }>("/api/months", {}, s),
  overview: (month?: string, s?: AbortSignal) => getJson<Overview>("/api/overview", { month }, s),
  members: (q: MemberQuery, s?: AbortSignal) => getJson<MemberPage>("/api/members", { ...q }, s),
  member: (id: number, month?: string, s?: AbortSignal) =>
    getJson<MemberDetail>(`/api/members/${id}`, { month }, s),
  history: (s?: AbortSignal) => getJson<History>("/api/history", {}, s),
  policy: (s?: AbortSignal) => getJson<Policy>("/api/policy", {}, s),
  registry: (s?: AbortSignal) => getJson<{ versions: RegistryVersion[] }>("/api/registry", {}, s),
  plan: (month?: string, budget?: number | null, s?: AbortSignal) =>
    getJson<RetentionPlan>("/api/retention/plan", { month, budget }, s),
  selection: (s?: AbortSignal) => getJson<Selection>("/api/reports/selection", {}, s),
  baselines: (s?: AbortSignal) => getJson<Baselines>("/api/reports/baselines", {}, s),
  backtest: (s?: AbortSignal) => getJson<Backtest>("/api/reports/backtest", {}, s),
  monitoringMonths: (s?: AbortSignal) => getJson<{ months: string[] }>("/api/monitoring", {}, s),
  monitoring: (month: string, s?: AbortSignal) =>
    getJson<MonitoringReport>(`/api/monitoring/${month}`, {}, s),
  /** Live scoring through the model service (same code path as the batch job). */
  scoreMember: (id: number, asOf: string) =>
    getJson<ScoreResponse>(`/members/${id}/score`, { as_of: asOf }),
  exportUrl: (month?: string, budget?: number | null) =>
    buildUrl("/api/export/contacts.csv", { month, budget }),
};
