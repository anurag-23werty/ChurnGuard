import { createContext, useContext, useMemo, type ReactNode } from "react";
import { useSearchParams } from "react-router-dom";

import { useMonths } from "../api/hooks";

interface MonthState {
  month: string | undefined;
  months: string[];
  setMonth: (m: string) => void;
  isLoading: boolean;
  error: unknown;
}

const Ctx = createContext<MonthState | null>(null);

/** The selected score month lives in the URL (?month=), so views are shareable. */
export function MonthProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams();
  const { data, isLoading, error } = useMonths();
  const months = useMemo(() => data?.months ?? [], [data]);
  const wanted = params.get("month");
  const month = wanted && months.includes(wanted) ? wanted : months[0];
  const value = useMemo<MonthState>(
    () => ({
      month,
      months,
      isLoading,
      error,
      setMonth: (m) =>
        setParams((p) => {
          const next = new URLSearchParams(p);
          next.set("month", m);
          return next;
        }),
    }),
    [month, months, isLoading, error, setParams],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useMonth(): MonthState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useMonth outside MonthProvider");
  return v;
}
