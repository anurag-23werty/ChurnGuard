import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ApiError } from "../api/client";
import { ErrorState } from "./ui";

describe("ErrorState", () => {
  it("shows a missing artifact as unavailable, with the backend's hint and no retry", () => {
    render(<ErrorState error={new ApiError(404, "backtest_valid.json not found; run `make backtest`")} what="Backtest" retry={() => {}} />);
    expect(screen.getByText("Backtest unavailable")).toBeTruthy();
    expect(screen.getByText(/make backtest/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
  });

  it("offers a retry for real failures", () => {
    render(<ErrorState error={new ApiError(500, "boom")} what="Members" retry={() => {}} />);
    expect(screen.getByRole("alert")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
  });
});
