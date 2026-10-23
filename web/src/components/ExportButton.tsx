import { useState } from "react";

import { api, ApiError } from "../api/client";
import { int, monthLabel, ntd } from "../lib/format";
import { IconDownload } from "./icons";
import { useToast } from "./toast";
import { ConfirmDialog } from "./ui";

/** Download the contact list as a CRM CSV, after a confirmation that states its size. */
export function ExportButton({
  month,
  contacted,
  budget,
  label = "Export contact list",
}: {
  month?: string;
  contacted?: number;
  budget?: number | null;
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  const run = async () => {
    setOpen(false);
    setBusy(true);
    try {
      const res = await fetch(api.exportUrl(month, budget));
      if (!res.ok) throw new ApiError(res.status, (await res.json().catch(() => null))?.detail ?? res.statusText);
      const blob = await res.blob();
      const name = /filename="([^"]+)"/.exec(res.headers.get("content-disposition") ?? "")?.[1] ?? "contacts.csv";
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      URL.revokeObjectURL(url);
      const rows = (await blob.text()).trim().split("\n").length - 1;
      toast({ kind: "success", title: `Exported ${int(rows)} members`, body: name });
    } catch (e) {
      const detail = e instanceof ApiError ? e.detail : (e as Error).message;
      toast({ kind: "error", title: "Export failed", body: detail });
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <button type="button" className="btn" onClick={() => setOpen(true)} disabled={!month || busy}>
        <IconDownload />
        {busy ? "Exporting…" : label}
      </button>
      <ConfirmDialog
        open={open}
        title="Export contact list?"
        confirmLabel="Download CSV"
        onConfirm={run}
        onCancel={() => setOpen(false)}
      >
        {contacted !== undefined ? `${int(contacted)} members` : "The positive-EV members"} from the {month ? monthLabel(month) : ""} scores
        {budget ? `, selected within a ${ntd(budget)} budget,` : ""} will be downloaded with their msno, churn probability, expected value and
        top three reasons. The file contains member identifiers: share it only with the CRM team.
      </ConfirmDialog>
    </>
  );
}
