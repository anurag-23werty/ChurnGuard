// Response shapes of the FastAPI service (src/churnguard/serving/api.py, console_api.py).
// Report payloads mirror the JSON files the pipeline writes under reports/.

export type Tier = "critical" | "high" | "medium" | "low";

export interface ModelRef {
  name: string;
  version: string;
  model_type: string;
}

export interface Health {
  status: string;
  model: ModelRef & { run_id?: string; alias?: string };
}

export interface TierRow {
  tier: Tier;
  min_p: number;
  max_p: number;
  members: number;
  share: number;
  revenue_at_risk: number;
  contacted: number;
}

export interface Overview {
  month: string;
  model: ModelRef;
  members: number;
  mean_p: number;
  median_p: number;
  contacted: number;
  contact_share: number;
  expected_net_value: number;
  campaign_cost: number;
  revenue_at_risk: number;
  scored_at: string;
  tiers: TierRow[];
  reasons: { feature: string; members: number; share: number }[];
  histogram: { lo: number; hi: number; members: number }[];
}

export interface Reason {
  feature: string;
  logit: number | null;
}

export interface MemberRow {
  member_id: number;
  msno: string;
  ev_rank: number;
  p_churn: number;
  tier: Tier;
  monthly_revenue: number;
  clv: number;
  cost: number;
  ev: number;
  contact: boolean;
  reasons: Reason[];
}

export interface MemberPage {
  month: string;
  total: number;
  page: number;
  page_size: number;
  rows: MemberRow[];
}

export type SortKey = "ev_rank" | "p_churn" | "ev" | "monthly_revenue" | "clv";

export interface MemberQuery {
  month?: string;
  q?: string;
  tier?: Tier;
  contact?: boolean;
  sort?: SortKey;
  order?: "asc" | "desc";
  page?: number;
  page_size?: number;
}

export interface Policy {
  save_rate: number;
  horizon_months: number;
  baseline_monthly_churn: number;
  contact_cost: number;
  incentive_months: number;
  budget: number | null;
}

export interface MemberDetail {
  month: string;
  score: MemberRow & {
    as_of_date: string;
    last_expire: string;
    expected_benefit: number;
    score_month: string;
    model_name: string;
    model_version: string;
    model_type: string;
    scored_at: string;
    contacted_total: number;
  };
  features: Record<string, string | number | null> | null;
  policy: Policy;
}

export interface ScoreResponse {
  member_id: number | null;
  as_of_date: string | null;
  p_churn: number;
  monthly_revenue: number;
  clv: number;
  expected_value: number;
  contact: boolean;
  reasons: { feature: string; logit_contribution: number }[];
  model: Record<string, string>;
}

export interface History {
  served: {
    month: string;
    members: number;
    mean_p: number;
    contacted: number;
    scored_at: string;
    model_name: string;
    model_version: string;
  }[];
  replay: ReplayRow[] | null;
}

export interface ReplayRow {
  month: string;
  reference: string;
  score_psi: number;
  mean_p: number;
  major: number;
  moderate: number;
  top_drift: string;
  decision: string;
}

export interface RegistryVersion {
  name: string;
  version: string;
  model_type: string | null;
  aliases: string[];
  run_id: string;
  created_at: number;
}

export interface RetentionPlan {
  month: string;
  budget: number | null;
  members: number;
  contacted: number;
  contact_share: number;
  campaign_cost: number;
  expected_net_value: number;
  expected_benefit: number;
  min_p_contacted: number | null;
  policy: Policy;
}

export interface Metrics {
  n?: number;
  base_rate?: number;
  roc_auc: number;
  pr_auc: number;
  log_loss: number;
  ece: number;
  brier?: number;
  precision_top10?: number;
  recall_top10?: number;
  lift_top10?: number;
  precision_top5?: number;
}

export interface Selection {
  candidates: string[];
  scored_months: string[];
  per_month: (Metrics & { candidate: string; month: string; lift_top10: number })[];
  summary: {
    candidate: string;
    mean_roc_auc: number;
    worst_roc_auc: number;
    mean_pr_auc: number;
    worst_pr_auc: number;
    mean_log_loss: number;
    mean_ece: number;
    mean_lift_top10: number;
  }[];
  gate: Record<string, { promote: boolean; reasons: string[] }>;
  selected: string;
  calibration: {
    applied: boolean;
    holdout_month: string;
    raw_log_loss: number;
    raw_ece: number;
    cal_log_loss: number;
    cal_ece: number;
  };
  final_training_rows: number;
  test: Record<string, Record<string, Metrics>>;
}

export interface BaselineModel {
  fit_seconds: number;
  valid: Metrics;
  test: Metrics;
}

export interface Baselines {
  prior: BaselineModel;
  rule: BaselineModel;
  logreg: BaselineModel;
  _permutation_control: { k: number; sample: number; aucs: number[]; mean: number; sd: number };
}

export interface Backtest {
  split: string;
  members: number;
  churn_rate: number;
  policy: Policy;
  sensitivity: {
    save_rate: number;
    contacted: number;
    contacted_share: number;
    break_even_p_at_149: number;
    realised_net_value: number;
    realised_value_contact_everyone: number;
  }[];
  targeting_curve: {
    contact_fraction: number;
    contacted: number;
    by_ev: number;
    by_risk: number;
    random: number;
  }[];
  best_fixed_fraction_by_ev: { contact_fraction: number; realised_net_value: number };
}

export interface FeatureDrift {
  feature: string;
  kind: string;
  psi: number;
  ref_null: number;
  cur_null: number;
  severity: "stable" | "moderate" | "major";
  expected_drift: boolean;
}

export interface MonitoringReport {
  month: string;
  reference_months: string[];
  data_quality: {
    rows: number;
    volume_ratio: number;
    volume_ok: boolean;
    max_null_jump: { feature: string; delta: number } | null;
    p_in_range: boolean;
  };
  score_drift: {
    psi: number;
    severity: string;
    mean_p_reference: number;
    mean_p_current: number;
  };
  feature_drift: FeatureDrift[];
  performance: {
    month: string;
    status: "pending" | "partial" | "complete";
    scores: string;
    members: number;
    decided: number;
    decided_share: number;
    in_sample: boolean;
    metrics?: Metrics & { base_rate: number; mean_pred: number };
  }[];
  decision: { status: "ok" | "warn" | "retrain"; retrain_reasons: string[]; warnings: string[] };
}
