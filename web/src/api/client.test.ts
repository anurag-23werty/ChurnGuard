import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, buildUrl, getJson } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("drops empty params", () => {
    expect(buildUrl("/api/members", { month: "2017-04", q: "", tier: undefined, page: 2, contact: false })).toBe(
      "/api/members?month=2017-04&page=2&contact=false",
    );
  });

  it("surfaces the backend detail and marks missing artifacts unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ detail: "selection.json not found; run `make select`" }), { status: 404 })),
    );
    const err = (await getJson("/api/reports/selection").catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(404);
    expect(err.unavailable).toBe(true);
    expect(err.detail).toContain("make select");
  });

  it("reports an unreachable service as status 0", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    const err = (await getJson("/health").catch((e: unknown) => e)) as ApiError;
    expect(err.status).toBe(0);
    expect(err.detail).toMatch(/make serve/);
  });
});
