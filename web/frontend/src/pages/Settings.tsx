import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Check, X } from "lucide-react";
import {
  api,
  type MemoryConfig,
  type PingResult,
  type UiField,
  type ValidationIssue,
} from "@/lib/api";
import {
  coerce,
  deletePath,
  getPath,
  prune,
  setPath,
  shouldShow,
} from "@/lib/configPath";
import { setDraft, useDraft } from "@/lib/configStore";
import { pickLocalized, useT, type Lang } from "@/lib/i18n";
import { cn, formatNumber } from "@/lib/utils";
import { useJobStream } from "@/lib/useJobStream";
import { PageHeader, Step } from "@/components/Layout";
import JobConsole from "@/components/JobConsole";
import {
  Button,
  Field,
  Input,
  Note,
  Select,
  Spinner,
  Suggest,
  Switch,
} from "@/components/ui";
import { setPrefs, usePrefs } from "@/lib/prefs";

function Control({
  field,
  config,
  onChange,
  issue,
  lang,
  models = [],
}: {
  field: UiField;
  config: MemoryConfig;
  onChange: (path: string, value: unknown) => void;
  issue?: string;
  lang: Lang;
  models?: string[];
}) {
  const raw = getPath(config, field.path);
  const id = `cfg-${field.path}`;
  const label = pickLocalized(field, "label", lang);
  const help = pickLocalized(field, "help", lang) || undefined;

  if (field.kind === "bool") {
    return (
      <Field
        label={label}
        hint={issue ?? help}
        htmlFor={id}
        inline
        invalid={Boolean(issue)}
      >
        <Switch
          checked={Boolean(raw)}
          onChange={(next) => onChange(field.path, next)}
          label={label}
        />
      </Field>
    );
  }

  return (
    <Field
      label={label}
      hint={issue ?? help}
      htmlFor={id}
      invalid={Boolean(issue)}
    >
      {field.kind === "model" ? (
        <Suggest
          id={id}
          value={raw ?? ""}
          options={models}
          onChange={(next) => onChange(field.path, next)}
          placeholder={
            field.default !== undefined ? String(field.default) : undefined
          }
          className="font-mono text-tiny"
        />
      ) : field.kind === "select" ? (
        <Select
          id={id}
          value={raw ?? ""}
          options={(field.options ?? []).map((o) => ({ value: o, label: o }))}
          onChange={(e) => onChange(field.path, coerce(field, e.target.value))}
        />
      ) : (
        <Input
          id={id}
          type={
            field.kind === "int" || field.kind === "number" ? "number" : "text"
          }
          step={field.step}
          value={raw ?? ""}
          spellCheck={false}
          className={field.kind === "path" ? "font-mono text-tiny" : undefined}
          placeholder={
            field.default !== undefined ? String(field.default) : undefined
          }
          onChange={(e) => onChange(field.path, coerce(field, e.target.value))}
        />
      )}
    </Field>
  );
}

export default function Settings() {
  const queryClient = useQueryClient();
  const draft = useDraft();
  const { t, lang } = useT();
  const prefs = usePrefs();

  const schema = useQuery({
    queryKey: ["configSchema"],
    queryFn: api.configSchema,
  });
  const secrets = useQuery({ queryKey: ["secrets"], queryFn: api.getSecrets });
  const instance = useQuery({
    queryKey: ["instance"],
    queryFn: api.instance,
    refetchInterval: 8000,
  });

  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [ping, setPing] = useState<PingResult | null>(null);

  const stream = useJobStream(jobId);
  const config = draft ?? {};
  const backendForModels =
    getPath(config, "memory_manager.model_name") ?? "openai";
  const ollamaHost =
    getPath(config, "memory_manager.configs.host") ?? undefined;

  useEffect(() => {
    if (!draft && schema.data?.presets.length)
      setDraft(schema.data.presets[0].config);
  }, [draft, schema.data]);

  useEffect(() => {
    if (secrets.data) setBaseUrl(secrets.data.base_url);
  }, [secrets.data]);

  const validation = useQuery({
    queryKey: ["validate", draft, secrets.data?.has_api_key],
    queryFn: () => api.validate(prune(draft ?? {})),
    enabled: Boolean(draft),
  });

  const autofill = useMutation({
    mutationFn: async () => {
      const d = await api.autodetect();
      let next = draft ?? {};
      const applied: string[] = [];
      const missing: string[] = [];

      if (d.compressor.found && d.compressor.path) {
        next = setPath(
          next,
          "pre_compressor.configs.llmlingua_config.model_name",
          d.compressor.path,
        );
        applied.push(`${t("set.autoCompressor")} ${d.compressor.path}`);
      } else {
        missing.push(t("set.autoNoCompressor"));
      }

      if (d.embedder.found && d.embedder.path) {
        next = setPath(next, "text_embedder.configs.model", d.embedder.path);
        applied.push(`${t("set.autoEmbedder")} ${d.embedder.path}`);
        if (d.embedder.dims) {
          next = setPath(
            next,
            "text_embedder.configs.embedding_dims",
            d.embedder.dims,
          );
          next = setPath(
            next,
            "embedding_retriever.configs.embedding_model_dims",
            d.embedder.dims,
          );
          applied.push(t("set.autoDims", { n: d.embedder.dims }));
        }
      } else {
        missing.push(t("set.autoNoEmbedder"));
      }

      next = setPath(
        next,
        "text_embedder.configs.model_kwargs.device",
        d.device,
      );
      next = setPath(
        next,
        "pre_compressor.configs.llmlingua_config.device_map",
        d.device,
      );
      const gpu = d.gpus.find((g) => `cuda:${g.index}` === d.device);
      applied.push(
        gpu
          ? t("set.autoDeviceGpu", {
              device: d.device,
              free: Math.round(gpu.free_mb / 1024),
            })
          : t("set.autoDeviceCpu"),
      );

      if (d.ollama.running) {
        applied.push(
          t("set.autoOllama", {
            n: d.ollama.models.length,
            host: d.ollama.host,
          }),
        );
      }
      if (!d.tiktoken_ok) missing.push(t("set.autoTiktoken"));

      setDraft(next);
      return { applied, missing };
    },
  });

  const models = useQuery({
    queryKey: [
      "models",
      secrets.data?.has_api_key,
      baseUrl,
      backendForModels,
      ollamaHost,
    ],
    queryFn: () =>
      api.models({
        base_url: baseUrl || undefined,
        backend: backendForModels,
        host: ollamaHost,
      }),
    enabled:
      backendForModels === "ollama" || Boolean(secrets.data?.has_api_key),
    retry: false,
    staleTime: 60_000,
  });

  const saveKey = useMutation({
    mutationFn: () =>
      api.setSecrets({
        ...(apiKey ? { api_key: apiKey } : {}),
        base_url: baseUrl,
      }),
    onSuccess: () => {
      setApiKey("");
      queryClient.invalidateQueries({ queryKey: ["secrets"] });
      queryClient.invalidateQueries({ queryKey: ["validate"] });
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });

  const test = useMutation({
    mutationFn: () =>
      api.ping({
        api_key: apiKey || undefined,
        base_url: baseUrl || undefined,
        model: getPath(config, "memory_manager.configs.model") ?? undefined,
      }),
    onSuccess: (result) => {
      setPing(result);
      if (result.ok) saveKey.mutate();
    },
  });

  const start = useMutation({
    mutationFn: async () => {
      await api.setSecrets({
        ...(apiKey ? { api_key: apiKey } : {}),
        base_url: baseUrl,
        model: getPath(config, "memory_manager.configs.model") ?? undefined,
      });
      setApiKey("");
      await queryClient.invalidateQueries({ queryKey: ["secrets"] });
      return api.initInstance(prune(draft ?? {}));
    },
    onSuccess: (data) => setJobId(data.job_id),
  });

  useEffect(() => {
    if (stream.status === "succeeded" || stream.status === "failed") {
      queryClient.invalidateQueries({ queryKey: ["instance"] });
    }
  }, [stream.status, queryClient]);

  if (schema.isLoading) {
    return (
      <p className="flex items-center gap-2 py-24 text-base text-ink3">
        <Spinner /> {t("cfg.loading")}
      </p>
    );
  }
  if (!schema.data) return <Note tone="bad" title={t("cfg.loadFailed")} />;

  const update = (path: string, value: unknown) =>
    setDraft(
      value === undefined
        ? deletePath(config, path)
        : setPath(config, path, value),
    );

  const issues = new Map<string, string>();
  validation.data?.errors.forEach((e) =>
    issues.set(e.path, pickLocalized(e, "message", lang)),
  );

  const blocking = (validation.data?.errors ?? []).filter(
    (e) => e.path !== "memory_manager.configs.api_key",
  );
  const warnings = validation.data?.warnings ?? [];
  const activePreset = schema.data.presets.find(
    (p) => JSON.stringify(p.config) === JSON.stringify(config),
  );

  const s = instance.data;
  const ready = s?.ready ?? false;
  const backend = getPath(config, "memory_manager.model_name");
  const needsKey = backend === "openai" || backend === "deepseek";
  const looksLikeUrl = /^https?:\/\//i.test(apiKey.trim());
  const hasKey = Boolean(secrets.data?.has_api_key || apiKey);
  const canStart = (!needsKey || hasKey) && blocking.length === 0;

  return (
    <>
      <PageHeader title={t("set.title")} description={t("set.intro")} />

      <div className="space-y-10">
        <Step n={1} title={t("cfg.stepPreset")} hint={t("cfg.stepPresetHint")}>
          <div className="mb-6 rounded-lg border border-rule bg-raised px-4 py-3.5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="min-w-0">
                <p className="text-base font-medium text-ink">
                  {t("set.autoTitle")}
                </p>
                <p className="mt-0.5 max-w-prose text-tiny leading-relaxed text-ink3">
                  {t("set.autoHint")}
                </p>
              </div>
              <Button
                onClick={() => autofill.mutate()}
                loading={autofill.isPending}
              >
                {t("set.autoRun")}
              </Button>
            </div>

            {autofill.data && (
              <ul className="mt-4 space-y-1 border-t border-ruleSoft pt-3 text-tiny text-ink2">
                {autofill.data.applied.map((line, i) => (
                  <li key={i} className="flex gap-2">
                    <Check className="mt-[3px] h-3 w-3 shrink-0 text-goodInk" />
                    <span>{line}</span>
                  </li>
                ))}
                {autofill.data.missing.map((line, i) => (
                  <li key={`m${i}`} className="flex gap-2 text-warnInk">
                    <AlertTriangle className="mt-[3px] h-3 w-3 shrink-0" />
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            {schema.data.presets.map((preset) => {
              const active = activePreset?.id === preset.id;
              return (
                <button
                  key={preset.id}
                  onClick={() => setDraft(preset.config)}
                  className={cn(
                    "rounded-lg border p-4 text-left transition-colors",
                    active
                      ? "border-ink bg-sunken/60"
                      : "border-rule hover:border-ink3",
                  )}
                >
                  <span className="flex items-center gap-2">
                    <span
                      className={cn(
                        "h-3 w-3 shrink-0 rounded-full border-[3px]",
                        active ? "border-ink" : "border-rule",
                      )}
                    />
                    <span className="text-base font-medium text-ink">
                      {pickLocalized(preset, "name", lang)}
                    </span>
                  </span>
                  <span className="mt-1.5 block text-tiny leading-relaxed text-ink3">
                    {pickLocalized(preset, "description", lang)}
                  </span>
                </button>
              );
            })}
          </div>
        </Step>

        <Step
          n={2}
          title={t("cfg.stepKey")}
          hint={needsKey ? t("cfg.stepKeyHint") : t("cfg.stepKeyLocal")}
        >
          <div className="grid max-w-3xl gap-x-10 gap-y-5 md:grid-cols-2">
            <Field
              label={t("env.apiKey")}
              hint={
                secrets.data?.has_api_key
                  ? t("env.apiKeyStored", {
                      masked: secrets.data.api_key_masked,
                    })
                  : t("env.apiKeyNone")
              }
              invalid={needsKey && !hasKey}
            >
              <Input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={
                  secrets.data?.has_api_key
                    ? t("env.apiKeyPlaceholderStored")
                    : "sk-…"
                }
                autoComplete="off"
                spellCheck={false}
                className="font-mono text-tiny"
              />
            </Field>
            <Field label={t("env.baseUrl")} hint={t("env.baseUrlHint")}>
              <Input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.deepseek.com/v1"
                spellCheck={false}
                className="font-mono text-tiny"
              />
            </Field>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-4">
            <Button
              variant="primary"
              onClick={() => saveKey.mutate()}
              loading={saveKey.isPending}
            >
              {t("common.save")}
            </Button>
            <Button onClick={() => test.mutate()} loading={test.isPending}>
              {t("env.testConnection")}
            </Button>
            {saveKey.isError && (
              <span className="text-sm text-badInk">
                {String((saveKey.error as Error).message)}
              </span>
            )}
            {saveKey.isSuccess && !saveKey.isPending && (
              <span className="text-sm text-goodInk">{t("env.saved")}</span>
            )}
            {ping && (
              <span
                className={cn(
                  "text-sm",
                  ping.ok ? "text-goodInk" : "text-badInk",
                )}
              >
                {ping.ok
                  ? t("env.pingOk", {
                      ms: ping.latency_ms ?? 0,
                      reply: ping.reply ?? "",
                    })
                  : t("env.pingBad")}
              </span>
            )}
          </div>

          {ping && !ping.ok && (
            <div className="mt-3">
              <Note tone="bad" title={t("env.pingBad")}>
                <p className="font-mono text-tiny">
                  {ping.error ?? ping.models_error}
                </p>
                {looksLikeUrl && (
                  <p className="mt-2">{t("env.keyLooksLikeUrl")}</p>
                )}
                {baseUrl && !/\/v\d$/.test(baseUrl.replace(/\/$/, "")) && (
                  <p className="mt-2">{t("env.baseUrlNoVersion")}</p>
                )}
              </Note>
            </div>
          )}
        </Step>

        <Step
          n={3}
          title={t("cfg.stepDetails")}
          hint={t("cfg.stepDetailsHint")}
        >
          <div className="space-y-8">
            {schema.data.ui.map((section) => {
              const visible = section.fields.filter((f) =>
                shouldShow(f, config),
              );
              if (!visible.length) return null;
              return (
                <div key={section.id}>
                  <p className="eyebrow mb-4">
                    {pickLocalized(section, "title", lang)}
                  </p>
                  <div className="grid max-w-3xl gap-x-10 gap-y-5 md:grid-cols-2">
                    {visible.map((field) => (
                      <Control
                        key={field.path}
                        field={field}
                        config={config}
                        onChange={update}
                        issue={issues.get(field.path)}
                        lang={lang}
                        models={models.data?.models ?? []}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </Step>

        <Step
          n={4}
          title={t("set.consolidation")}
          hint={t("set.consolidationHint")}
        >
          <div className="grid max-w-3xl gap-x-10 gap-y-5 md:grid-cols-3">
            <Field label="top_k" hint={t("con.topKHint")}>
              <Input
                type="number"
                min={1}
                max={200}
                value={prefs.topK}
                onChange={(e) => setPrefs({ topK: Number(e.target.value) })}
              />
            </Field>
            <Field label="keep_top_n" hint={t("con.keepTopNHint")}>
              <Input
                type="number"
                min={1}
                max={100}
                value={prefs.keepTopN}
                onChange={(e) => setPrefs({ keepTopN: Number(e.target.value) })}
              />
            </Field>
            <Field label="score_threshold" hint={t("con.thresholdHint")}>
              <Input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={prefs.scoreThreshold}
                onChange={(e) =>
                  setPrefs({ scoreThreshold: Number(e.target.value) })
                }
              />
            </Field>
          </div>
        </Step>

        <Step n={5} title={t("cfg.stepStart")} hint={t("cfg.stepStartHint")}>
          {(blocking.length > 0 || warnings.length > 0) && (
            <div className="mb-6 space-y-4">
              {blocking.map((item, i) => (
                <Note
                  key={`e${i}`}
                  tone="bad"
                  icon={<X className="h-4 w-4" />}
                  title={item.path}
                >
                  {pickLocalized(item, "message", lang)}
                </Note>
              ))}
              {warnings.map((item, i) => (
                <Note
                  key={`w${i}`}
                  tone="warn"
                  icon={<AlertTriangle className="h-4 w-4" />}
                  title={item.path}
                >
                  {pickLocalized(item, "message", lang)}
                </Note>
              ))}
            </div>
          )}

          <div className="flex flex-wrap items-center gap-4">
            <Button
              variant="primary"
              onClick={() => start.mutate()}
              loading={start.isPending || stream.running}
              disabled={!canStart}
            >
              {ready ? t("cfg.restart") : t("cfg.start")}
            </Button>
            <span className="text-sm text-ink3">
              {needsKey && !hasKey
                ? t("cfg.needKey")
                : ready && s
                  ? t("cfg.running", {
                      collection: s.collection ?? "",
                      n: formatNumber(s.points ?? 0),
                      seconds: (s.init_seconds ?? 0).toFixed(1),
                    })
                  : t("cfg.notRunning")}
            </span>
          </div>

          {jobId && (
            <div className="mt-6">
              <JobConsole
                stream={stream}
                title="LightMemory.from_config"
                height="14rem"
              />
            </div>
          )}

          {start.isError && (
            <div className="mt-6">
              <Note tone="bad" title={t("inst.startFailed")}>
                {(() => {
                  const detail = (start.error as any)?.detail;
                  if (Array.isArray(detail?.errors)) {
                    return detail.errors
                      .map((e: ValidationIssue) =>
                        pickLocalized(e, "message", lang),
                      )
                      .join(" ");
                  }
                  return String((start.error as Error)?.message ?? start.error);
                })()}
              </Note>
            </div>
          )}
        </Step>
      </div>
    </>
  );
}
