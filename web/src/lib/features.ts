// Plain-language names for the 52 registry features (src/churnguard/features/registry.yaml).

export type Family = "member" | "billing" | "engagement";

interface FeatureInfo {
  label: string;
  family: Family;
  unit?: "days" | "secs" | "ntd" | "share" | "flag" | "code" | "count";
}

export const FEATURES: Record<string, FeatureInfo> = {
  city: { label: "City", family: "member", unit: "code" },
  age: { label: "Age", family: "member" },
  age_missing: { label: "Age missing", family: "member", unit: "flag" },
  gender: { label: "Gender", family: "member" },
  registered_via: { label: "Registration channel", family: "member", unit: "code" },
  tenure_days: { label: "Tenure", family: "member", unit: "days" },
  has_member_record: { label: "Has member record", family: "member", unit: "flag" },
  days_to_expiry: { label: "Days until plan expiry", family: "billing", unit: "days" },
  last_plan_days: { label: "Plan length", family: "billing", unit: "days" },
  last_list_price: { label: "Plan list price", family: "billing", unit: "ntd" },
  last_amount_paid: { label: "Amount last paid", family: "billing", unit: "ntd" },
  last_discount: { label: "Discount on last payment", family: "billing", unit: "ntd" },
  last_is_auto_renew: { label: "Auto-renew on last transaction", family: "billing", unit: "flag" },
  last_is_cancel: { label: "Last transaction was a cancellation", family: "billing", unit: "flag" },
  last_payment_method: { label: "Payment method", family: "billing", unit: "code" },
  days_since_last_txn: { label: "Days since last transaction", family: "billing", unit: "days" },
  days_since_first_txn: { label: "Days since first transaction", family: "billing", unit: "days" },
  n_txn_30d: { label: "Transactions, 30 days", family: "billing", unit: "count" },
  n_txn_90d: { label: "Transactions, 90 days", family: "billing", unit: "count" },
  n_txn_180d: { label: "Transactions, 180 days", family: "billing", unit: "count" },
  n_txn_365d: { label: "Transactions, 365 days", family: "billing", unit: "count" },
  n_cancel_30d: { label: "Cancellations, 30 days", family: "billing", unit: "count" },
  n_cancel_90d: { label: "Cancellations, 90 days", family: "billing", unit: "count" },
  n_cancel_180d: { label: "Cancellations, 180 days", family: "billing", unit: "count" },
  n_cancel_365d: { label: "Cancellations, 365 days", family: "billing", unit: "count" },
  amount_paid_90d: { label: "Paid, 90 days", family: "billing", unit: "ntd" },
  amount_paid_365d: { label: "Paid, 365 days", family: "billing", unit: "ntd" },
  n_free_txn_365d: { label: "Free transactions, 365 days", family: "billing", unit: "count" },
  auto_renew_share_180d: { label: "Auto-renew share, 180 days", family: "billing", unit: "share" },
  n_cancel_all: { label: "Cancellations, all time", family: "billing", unit: "count" },
  n_payment_methods: { label: "Payment methods used", family: "billing", unit: "count" },
  active_days_7d: { label: "Active days, 7 days", family: "engagement", unit: "count" },
  active_days_14d: { label: "Active days, 14 days", family: "engagement", unit: "count" },
  active_days_30d: { label: "Active days, 30 days", family: "engagement", unit: "count" },
  active_days_90d: { label: "Active days, 90 days", family: "engagement", unit: "count" },
  total_secs_7d: { label: "Listening time, 7 days", family: "engagement", unit: "secs" },
  total_secs_14d: { label: "Listening time, 14 days", family: "engagement", unit: "secs" },
  total_secs_30d: { label: "Listening time, 30 days", family: "engagement", unit: "secs" },
  total_secs_90d: { label: "Listening time, 90 days", family: "engagement", unit: "secs" },
  num_unq_7d: { label: "Unique songs, 7 days", family: "engagement", unit: "count" },
  num_unq_30d: { label: "Unique songs, 30 days", family: "engagement", unit: "count" },
  num_unq_90d: { label: "Unique songs, 90 days", family: "engagement", unit: "count" },
  num_plays_7d: { label: "Plays, 7 days", family: "engagement", unit: "count" },
  num_plays_30d: { label: "Plays, 30 days", family: "engagement", unit: "count" },
  num_plays_90d: { label: "Plays, 90 days", family: "engagement", unit: "count" },
  completion_rate_30d: { label: "Completion rate, 30 days", family: "engagement", unit: "share" },
  completion_rate_90d: { label: "Completion rate, 90 days", family: "engagement", unit: "share" },
  skip_rate_30d: { label: "Skip rate, 30 days", family: "engagement", unit: "share" },
  secs_per_active_day_30d: { label: "Listening per active day", family: "engagement", unit: "secs" },
  days_since_last_listen: { label: "Days since last listen", family: "engagement", unit: "days" },
  secs_trend_7_30: { label: "Listening trend, 7 vs 30 days", family: "engagement" },
  active_trend_7_30: { label: "Activity trend, 7 vs 30 days", family: "engagement" },
};

export function featureLabel(name: string | null | undefined): string {
  if (!name) return "—";
  return FEATURES[name]?.label ?? name;
}
