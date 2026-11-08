import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the page talks to the Node gateway; in production the gateway serves this build itself.
const gateway = process.env.GATEWAY_URL ?? "http://localhost:3000";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/query": gateway, "/api": gateway, "/health": gateway } },
});
