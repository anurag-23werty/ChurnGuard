import type { SVGProps } from "react";

const base: SVGProps<SVGSVGElement> = {
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
};

type P = SVGProps<SVGSVGElement>;

export const IconOverview = (p: P) => (
  <svg {...base} {...p}>
    <rect x="3" y="3" width="7" height="9" rx="1.5" />
    <rect x="14" y="3" width="7" height="5" rx="1.5" />
    <rect x="14" y="12" width="7" height="9" rx="1.5" />
    <rect x="3" y="16" width="7" height="5" rx="1.5" />
  </svg>
);
export const IconCustomers = (p: P) => (
  <svg {...base} {...p}>
    <circle cx="9" cy="8" r="3.5" />
    <path d="M2.5 20c.8-3.6 3.4-5.5 6.5-5.5s5.7 1.9 6.5 5.5M16 4.6a3.5 3.5 0 0 1 0 6.8M18 14.8c1.8.7 3 2.4 3.5 5.2" />
  </svg>
);
export const IconPredictions = (p: P) => (
  <svg {...base} {...p}>
    <path d="M3 12h4l3-8 4 16 3-8h4" />
  </svg>
);
export const IconRetention = (p: P) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="12" r="9" />
    <circle cx="12" cy="12" r="5" />
    <circle cx="12" cy="12" r="1.5" />
  </svg>
);
export const IconModel = (p: P) => (
  <svg {...base} {...p}>
    <path d="M3 21h18M6 17v-6M11 17V6M16 17v-4M21 17V9" />
  </svg>
);
export const IconMonitoring = (p: P) => (
  <svg {...base} {...p}>
    <path d="M12 3l8 3v6c0 4.5-3.4 8.2-8 9-4.6-.8-8-4.5-8-9V6z" />
    <path d="M8.5 12.5l2.5 2.5 4.5-5" />
  </svg>
);
export const IconSearch = (p: P) => (
  <svg {...base} strokeWidth={2} {...p}>
    <circle cx="11" cy="11" r="7" />
    <path d="M20 20l-3.5-3.5" />
  </svg>
);
export const IconDownload = (p: P) => (
  <svg {...base} {...p}>
    <path d="M12 4v11M7 10l5 5 5-5M5 20h14" />
  </svg>
);
export const IconChevronLeft = (p: P) => (
  <svg {...base} strokeWidth={2} {...p}>
    <path d="M15 6l-6 6 6 6" />
  </svg>
);
export const IconChevronRight = (p: P) => (
  <svg {...base} strokeWidth={2} {...p}>
    <path d="M9 6l6 6-6 6" />
  </svg>
);
export const IconArrowDown = (p: P) => (
  <svg {...base} width={12} height={12} strokeWidth={2} {...p}>
    <path d="M12 5v14M6 13l6 6 6-6" />
  </svg>
);
export const IconArrowUp = (p: P) => (
  <svg {...base} width={12} height={12} strokeWidth={2} {...p}>
    <path d="M12 19V5M6 11l6-6 6 6" />
  </svg>
);
export const IconCheck = (p: P) => (
  <svg {...base} strokeWidth={2.2} {...p}>
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </svg>
);
export const IconAlert = (p: P) => (
  <svg {...base} {...p}>
    <path d="M12 3l9.5 17h-19z" />
    <path d="M12 10v4M12 17.5v.01" />
  </svg>
);
export const IconEmpty = (p: P) => (
  <svg {...base} width={28} height={28} strokeWidth={1.6} {...p}>
    <circle cx="11" cy="11" r="7" />
    <path d="M20 20l-3.5-3.5M8 11h6" />
  </svg>
);
export const IconCopy = (p: P) => (
  <svg {...base} {...p}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V5a1 1 0 0 1 1-1h10" />
  </svg>
);
export const BrandMark = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--violet)" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2" />
  </svg>
);
