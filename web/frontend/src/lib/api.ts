export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown) {
    super(
      typeof detail === "string"
        ? detail
        : ((detail as any)?.message ?? `Request failed (${status})`),
    );
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    ...init,
  });
  const text = await res.text();
  const body = text ? safeParse(text) : null;
  if (!res.ok) throw new ApiError(res.status, (body as any)?.detail ?? text);
  return body as T;
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

const get = <T>(p: string) => request<T>(p);
const post = <T>(p: string, body?: unknown) =>
  request<T>(p, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export type Health = {
  ok: boolean;
  instance_ready: boolean;
  busy: boolean;
  queue_depth: number;
};

export type SecretsView = {
  api_key_masked: string;
  has_api_key: boolean;
  base_url: string;
  model: string;
  answer_model: string;
};

export type PingResult = {
  ok: boolean;
  model: string;
  reply?: string;
  error?: string;
  models_error?: string;
  models_visible?: number;
  models_sample?: string[];
  model_listed?: boolean;
  latency_ms?: number;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

export type Detected = {
  compressor: { path: string | null; found: boolean };
  embedder: { path: string | null; found: boolean; dims: number | null };
  device: string;
  gpus: { index: number; name: string; free_mb: number; total_mb: number }[];
  ollama: { running: boolean; host: string; models: string[] };
  tiktoken_ok: boolean;
  searched: string[];
  candidates: string[];
};

export type MemoryConfig = Record<string, any>;

export type UiField = {
  path: string;
  label: string;
  label_zh?: string;
  kind: "bool" | "text" | "path" | "int" | "number" | "select" | "model";
  options?: string[];
  default?: unknown;
  help?: string;
  help_zh?: string;
  showIf?: string;
  danger?: boolean;
  min?: number;
  max?: number;
  step?: number;
};

export type UiSection = {
  id: string;
  title: string;
  title_zh?: string;
  subtitle: string;
  subtitle_zh?: string;
  fields: UiField[];
};

export type Preset = {
  id: string;
  name: string;
  name_zh?: string;
  description: string;
  description_zh?: string;
  config: MemoryConfig;
};

export type ConfigSchema = {
  schema: Record<string, any>;
  ui: UiSection[];
  presets: Preset[];
};

export type ValidationIssue = {
  path: string;
  message: string;
  message_zh?: string;
  level?: "danger" | "warn";
};
export type Validation = {
  ok: boolean;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
};

export type Component = { name: string; impl: string; note: string };

export type InstanceStatus = {
  ready: boolean;
  config: MemoryConfig | null;
  components: Component[];
  created_at: number | null;
  init_seconds: number | null;
  collection?: string | null;
  points?: number | null;
  retrieve_strategy?: string | null;
  last_error: string | null;
  gpu?: { index: number; name: string; total_mb: number; used_mb: number }[];
};

export type JobSummary = {
  id: string;
  type: string;
  status: "pending" | "running" | "succeeded" | "failed" | "cancelled";
  stage: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  duration: number | null;
  error: string | null;
};

export type JobEvent = {
  seq: number;
  kind: "log" | "stage" | "status" | "done";
  level?: string;
  logger?: string;
  message?: string;
  ts?: number;
  stage?: string;
  detail?: Record<string, unknown>;
  status?: string;
  result?: unknown;
  error?: string | null;
};

export type JobDetail = JobSummary & {
  params: Record<string, unknown>;
  result: any;
  traceback: string | null;
  events: JobEvent[];
};

export type ChatMessage = {
  role: string;
  content: string;
  time_stamp?: string;
};

export type MemoryHit = {
  id: string;
  score: number;
  time_stamp: string;
  weekday: string;
  memory: string;
  original_memory: string;
  compressed_memory: string;
  topic_id: number | null;
  topic_summary: string;
  category: string;
  subcategory: string;
  memory_class: string;
  speaker_id: string;
  speaker_name: string;
  consolidated: boolean;
  bam_tags: string[];
};

export type RetrieveResult = {
  query: string;
  count: number;
  latency_ms: number;
  results: MemoryHit[];
};

export type MemoryRow = Record<string, any> & { id: string };

export type ChatResult = {
  answer: string;
  model: string;
  messages: { role: string; content: string }[];
  latency_ms: number;
  usage: {
    prompt_tokens: number | null;
    completion_tokens: number | null;
    total_tokens: number | null;
  } | null;
  retrieved: MemoryHit[];
  used_count: number;
};

export const api = {
  health: () => get<Health>("/health"),

  getSecrets: () => get<SecretsView>("/secrets"),
  setSecrets: (v: Partial<SecretsView & { api_key: string }>) =>
    post<SecretsView>("/secrets", v),
  ping: (v: { api_key?: string; base_url?: string; model?: string }) =>
    post<PingResult>("/secrets/ping", v),

  models: (body: {
    api_key?: string;
    base_url?: string;
    backend?: string;
    host?: string;
  }) => post<{ models: string[]; error: string | null }>("/models", body),
  autodetect: () => get<Detected>("/autodetect"),
  configSchema: () => get<ConfigSchema>("/config/schema"),
  validate: (config: MemoryConfig) =>
    post<Validation>("/config/validate", { config }),

  instance: () => get<InstanceStatus>("/instance"),
  initInstance: (config: MemoryConfig) =>
    post<{ job_id: string; warnings: ValidationIssue[] }>("/instance/init", {
      config,
    }),

  addMemory: (body: {
    messages: ChatMessage[];
    force_segment?: boolean;
    force_extract?: boolean;
    metadata_prompt?: unknown;
  }) => post<{ job_id: string }>("/memory/add", body),

  offlineUpdate: (body: {
    top_k: number;
    keep_top_n: number;
    score_threshold: number;
  }) => post<{ job_id: string }>("/update/offline", body),

  retrieve: (body: {
    query: string;
    limit?: number;
    filters?: Record<string, unknown> | null;
  }) => post<RetrieveResult>("/retrieve", body),

  memories: (params: { limit?: number; offset?: string }) => {
    const qs = new URLSearchParams();
    if (params.limit) qs.set("limit", String(params.limit));
    if (params.offset) qs.set("offset", params.offset);
    return get<{
      count: number;
      next_offset: string | null;
      rows: MemoryRow[];
    }>(`/memories?${qs}`);
  },

  chat: (body: {
    question: string;
    limit?: number;
    filters?: Record<string, unknown> | null;
    model?: string;
    system_prompt?: string;
    temperature?: number;
    selected_ids?: string[] | null;
    use_memory?: boolean;
  }) => post<ChatResult>("/chat", body),

  job: (id: string) => get<JobDetail>(`/jobs/${id}`),
};
