import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildGateway } from "./gateway.ts";

const here = fileURLToPath(new URL(".", import.meta.url));

const app = buildGateway({
  upstreamUrl: process.env.UPSTREAM_URL ?? "http://localhost:8080",
  webDir: process.env.WEB_DIR ?? resolve(here, "../../web/dist"),
  questionsPerMinute: Number(process.env.QUESTIONS_PER_MINUTE ?? 20),
  // One hop: the address Render's proxy saw. Never `true`, which would trust an address the visitor wrote.
  trustProxy: process.env.TRUST_PROXY === "1" ? 1 : false,
  gatewaySecret: process.env.GATEWAY_SECRET || undefined,
  logger: true,
});

const port = Number(process.env.PORT ?? 3000);
app.listen({ port, host: "0.0.0.0" }).catch((error) => {
  app.log.error(error);
  process.exit(1);
});
