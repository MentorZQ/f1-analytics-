import { z } from "zod";

/**
 * Who paid for an answer. "router" is a one-fact question answered from the data with no model call;
 * "none" is the fixed reply to a prompt-injection-only question.
 */
export const PaidBy = z.enum(["saved", "visitor", "free", "none", "router"]);

export const SourceSchema = z.object({
  chunk_type: z.string(),
  section: z.string().nullable(),
  drivers: z.array(z.string()),
  text: z.string(),
});

export const SectionSchema = z.object({
  section: z.string(),
  type: z.string(),
  /** Seconds gained by driver A over driver B in this section. Positive means A was faster. */
  gain_s: z.number(),
  start_m: z.number().nullable(),
  end_m: z.number().nullable(),
  /** True when the section was part of the context the model saw. */
  retrieved: z.boolean(),
});

export const DeltasSchema = z.object({
  driver_a: z.string(),
  driver_b: z.string(),
  lap_gap_s: z.number(),
  /** Exact lap-time gap from timing, when the server has it. Preferred over lap_gap_s. */
  lap_gap_exact_s: z.number().nullable().optional(),
  lap_session: z.string().nullable().optional(),
  sections: z.array(SectionSchema),
});

export const MetricsSchema = z
  .object({
    retrieve_s: z.number().optional(),
    llm_s: z.number().optional(),
    total_s: z.number().optional(),
    input_tokens: z.number().optional(),
    output_tokens: z.number().optional(),
    /** null when no model ran: a routed answer or a fixed refusal. */
    model: z.string().nullable().optional(),
  })
  .passthrough();

export const QueryResponseSchema = z.object({
  answer: z.string(),
  session: z.string().nullable(),
  sources: z.array(SourceSchema),
  deltas: DeltasSchema.nullable(),
  paid_by: PaidBy,
  free_remaining: z.number().int().nonnegative().optional(),
  metrics: MetricsSchema.optional(),
});

export const MAX_QUESTION_CHARS = 500;

export const QueryRequestSchema = z.object({
  question: z.string().trim().min(1, "Please enter a question.").max(MAX_QUESTION_CHARS),
  session: z.string().nullable().optional(),
});

export const SessionsResponseSchema = z.object({
  sessions: z.array(z.object({ key: z.string(), label: z.string() })),
  examples: z.array(z.object({ session: z.string(), question: z.string() })),
  max_question_chars: z.number().int().positive(),
});

/** Codes the Python server and this gateway can return. The UI branches on these, not on message text. */
export const ErrorCode = z.enum([
  "empty_question",
  "unknown_session",
  "invalid_key",
  "too_many_requests",
  "free_limit_reached",
  "daily_budget_reached",
  "no_server_key",
  "busy",
  "empty_answer",
  "upstream_error",
  // raised by the gateway itself
  "invalid_request",
  "upstream_unreachable",
  "bad_upstream_response",
  "gateway_rate_limited",
]);

export const ApiErrorSchema = z.object({
  detail: z.object({ code: z.string(), message: z.string() }),
});

export type QueryResponse = z.infer<typeof QueryResponseSchema>;
export type QueryRequest = z.infer<typeof QueryRequestSchema>;
export type SessionsResponse = z.infer<typeof SessionsResponseSchema>;
export type Deltas = z.infer<typeof DeltasSchema>;
export type Section = z.infer<typeof SectionSchema>;
export type Source = z.infer<typeof SourceSchema>;
export type ApiError = z.infer<typeof ApiErrorSchema>;
export type KnownErrorCode = z.infer<typeof ErrorCode>;

/** Errors where the right next step is for the visitor to supply a key. */
export const KEY_ERRORS: ReadonlySet<string> = new Set([
  "free_limit_reached",
  "daily_budget_reached",
  "no_server_key",
  "invalid_key",
]);
