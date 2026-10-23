import { describe, expect, it } from "vitest";

import { int, monthLabel, ntd, pct, signedPt, tierOf } from "./format";

describe("format", () => {
  it("formats money compactly with a real minus sign", () => {
    expect(ntd(6_415_237)).toBe("NT$6.42M");
    expect(ntd(720_290, { signed: true })).toBe("+NT$720k");
    expect(ntd(-2_365_092)).toBe("−NT$2.37M");
    expect(ntd(68.74)).toBe("NT$68.7");
    expect(ntd(1572)).toBe("NT$1.6k");
  });

  it("never prints NaN", () => {
    expect(pct(Number.NaN)).toBe("—");
    expect(int(undefined)).toBe("—");
    expect(ntd(null)).toBe("—");
  });

  it("matches the backend tier bands", () => {
    expect([0.6, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01].map(tierOf)).toEqual([
      "critical",
      "critical",
      "high",
      "high",
      "medium",
      "medium",
      "low",
    ]);
  });

  it("labels months and point changes", () => {
    expect(monthLabel("2017-04")).toBe("Apr 2017");
    expect(signedPt(0.0472 - 0.0497)).toBe("−0.25 pt");
  });
});
