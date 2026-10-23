import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";

type Kind = "success" | "error" | "info";
interface Toast {
  id: number;
  kind: Kind;
  title: string;
  body?: string;
}

const Ctx = createContext<((t: Omit<Toast, "id">) => void) | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const next = useRef(1);
  const push = useCallback((t: Omit<Toast, "id">) => {
    const id = next.current++;
    setToasts((xs) => [...xs.slice(-3), { ...t, id }]);
    window.setTimeout(() => setToasts((xs) => xs.filter((x) => x.id !== id)), 5000);
  }, []);
  const value = useMemo(() => push, [push]);
  return (
    <Ctx.Provider value={value}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`}>
            <span className="dot" />
            <div className="stack" style={{ ["--gap" as string]: "2px" }}>
              <strong style={{ fontWeight: 500 }}>{t.title}</strong>
              {t.body && <span className="muted small">{t.body}</span>}
            </div>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}

export function useToast() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useToast outside ToastProvider");
  return v;
}
