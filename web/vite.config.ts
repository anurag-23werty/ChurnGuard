/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The FastAPI service (make serve) owns /api, /health and /members/{id}/score. In dev and
// preview Vite proxies them, so the browser sees one origin and the backend needs no CORS.
const API = process.env.CHURNGUARD_API ?? "http://127.0.0.1:8000";
const proxy = {
  "/api": API,
  "/health": API,
  "/members": API,
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  test: { environment: "jsdom" },
});
