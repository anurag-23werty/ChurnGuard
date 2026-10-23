import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { api } from "./client";
import type { MemberQuery } from "./types";

// Reports change only when the pipeline reruns, so they are cached for the session.
const STATIC = { staleTime: Infinity } as const;

export const useHealth = () =>
  useQuery({ queryKey: ["health"], queryFn: ({ signal }) => api.health(signal), refetchInterval: 30_000 });
export const useMonths = () =>
  useQuery({ queryKey: ["months"], queryFn: ({ signal }) => api.months(signal), ...STATIC });
export const useOverview = (month?: string) =>
  useQuery({
    queryKey: ["overview", month],
    queryFn: ({ signal }) => api.overview(month, signal),
    enabled: !!month,
    ...STATIC,
  });
export const useMembers = (q: MemberQuery) =>
  useQuery({
    queryKey: ["members", q],
    queryFn: ({ signal }) => api.members(q, signal),
    enabled: !!q.month,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });
export const useMember = (id: number, month?: string) =>
  useQuery({
    queryKey: ["member", id, month],
    queryFn: ({ signal }) => api.member(id, month, signal),
    enabled: Number.isFinite(id) && !!month,
    ...STATIC,
  });
export const useHistory = () =>
  useQuery({ queryKey: ["history"], queryFn: ({ signal }) => api.history(signal), ...STATIC });
export const usePolicy = () =>
  useQuery({ queryKey: ["policy"], queryFn: ({ signal }) => api.policy(signal), ...STATIC });
export const useRegistry = () =>
  useQuery({ queryKey: ["registry"], queryFn: ({ signal }) => api.registry(signal), staleTime: 60_000 });
export const usePlan = (month?: string, budget?: number | null) =>
  useQuery({
    queryKey: ["plan", month, budget],
    queryFn: ({ signal }) => api.plan(month, budget, signal),
    enabled: !!month,
    placeholderData: keepPreviousData,
    ...STATIC,
  });
export const useSelection = () =>
  useQuery({ queryKey: ["selection"], queryFn: ({ signal }) => api.selection(signal), ...STATIC });
export const useBaselines = () =>
  useQuery({ queryKey: ["baselines"], queryFn: ({ signal }) => api.baselines(signal), ...STATIC });
export const useBacktest = () =>
  useQuery({ queryKey: ["backtest"], queryFn: ({ signal }) => api.backtest(signal), ...STATIC });
export const useMonitoringMonths = () =>
  useQuery({ queryKey: ["monitoring-months"], queryFn: ({ signal }) => api.monitoringMonths(signal), ...STATIC });
export const useMonitoring = (month?: string) =>
  useQuery({
    queryKey: ["monitoring", month],
    queryFn: ({ signal }) => api.monitoring(month!, signal),
    enabled: !!month,
    ...STATIC,
  });
