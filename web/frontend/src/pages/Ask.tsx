import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowRight, Check, Search, SlidersHorizontal } from "lucide-react";
import { api, type MemoryHit, type MemoryRow } from "@/lib/api";
import { useT, type Translate } from "@/lib/i18n";
import { setPrefs, usePrefs } from "@/lib/prefs";
import { cn, formatNumber, shortStamp } from "@/lib/utils";
import {
  Button,
  CodeBlock,
  Empty,
  Field,
  Input,
  Note,
  Select,
  Spinner,
  Tag,
} from "@/components/ui";

const SAMPLE_QUERIES: Record<string, string[]> = {
  zh: ["我有什么忌口", "我住在哪里", "我在做什么工作"],
  en: ["what should I avoid eating", "where do I live", "what am I working on"],
};

function MetaLine({
  parts,
  tag,
}: {
  parts: (string | null)[];
  tag?: React.ReactNode;
}) {
  const items = parts.filter(Boolean) as string[];
  return (
    <p className="mt-1.5 flex flex-wrap items-center gap-x-2 text-micro tracking-normal text-ink4">
      {items.map((part, i) => (
        <span key={i} className="flex items-center gap-2">
          {i > 0 && <span className="text-rule">·</span>}
          {part}
        </span>
      ))}
      {tag}
    </p>
  );
}

function HitRow({
  hit,
  index,
  selected,
  onToggle,
  t,
  topScore,
}: {
  hit: MemoryHit;
  index: number;
  selected: boolean;
  onToggle: () => void;
  t: Translate;
  topScore: number;
}) {
  return (
    <li>
      <button
        onClick={onToggle}
        aria-pressed={selected}
        className={cn(
          "group flex w-full gap-4 rounded-xl border px-3 py-3.5 text-left transition-[background-color,border-color]",
          selected
            ? "border-accent/20 bg-accent-wash/35"
            : "border-transparent hover:border-ruleSoft hover:bg-sunken/60",
        )}
      >
        <span
          className={cn(
            "mt-1 flex h-4 w-4 shrink-0 items-center justify-center rounded border transition-colors",
            selected
              ? "border-accent bg-accent text-white"
              : "border-rule group-hover:border-ink3",
          )}
          aria-hidden
        >
          {selected && <Check className="h-3 w-3" strokeWidth={3.5} />}
        </span>

        <span className="min-w-0 flex-1">
          <span className="mb-1 flex items-center gap-2">
            <span className="font-mono text-micro tabular-nums tracking-normal text-ink3">
              {hit.score.toFixed(3)}
            </span>
            <span className="h-[3px] w-12 overflow-hidden rounded-full bg-accent/12">
              <span
                className="block h-full rounded-full bg-accent"
                style={{
                  width: `${Math.max(4, Math.min(100, (hit.score / topScore) * 100))}%`,
                }}
              />
            </span>
            <span className="ml-auto text-micro tracking-normal text-ink4">
              {index + 1}
            </span>
          </span>

          <span className="block max-w-readable text-lead leading-relaxed text-ink">
            {hit.memory}
          </span>

          <MetaLine
            parts={[shortStamp(hit.time_stamp), hit.category, hit.memory_class]}
            tag={
              hit.consolidated ? (
                <Tag>{t("exp.consolidatedTag")}</Tag>
              ) : undefined
            }
          />
        </span>
      </button>
    </li>
  );
}

function DetailPanel({ hit, t }: { hit?: MemoryHit; t: Translate }) {
  if (!hit) {
    return (
      <div className="flex min-h-[13rem] items-center justify-center p-6 text-center text-sm text-ink3">
        {t("ask.detailEmpty")}
      </div>
    );
  }

  const details: [string, string][] = [
    [t("ask.detailScore"), hit.score.toFixed(3)],
    [t("ask.detailTime"), shortStamp(hit.time_stamp) || "—"],
    [t("exp.fCategory"), hit.category || "—"],
    [
      t("exp.fSpeaker"),
      hit.speaker_name && hit.speaker_name !== "None" ? hit.speaker_name : "—",
    ],
    [
      t("exp.fTopic"),
      hit.topic_id != null ? t("pipe.topicN", { n: hit.topic_id }) : "—",
    ],
  ];

  return (
    <div className="space-y-5 p-5">
      <p className="text-base leading-7 text-ink">{hit.memory}</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-4 border-t border-ruleSoft pt-4">
        {details.map(([label, value]) => (
          <div key={label} className="min-w-0">
            <p className="text-micro text-ink4">{label}</p>
            <p
              className={cn(
                "mt-1 truncate text-sm text-ink2",
                label === t("ask.detailScore") && "font-mono tabular-nums",
              )}
            >
              {value}
            </p>
          </div>
        ))}
      </div>
      {hit.consolidated && <Tag>{t("exp.consolidatedTag")}</Tag>}
    </div>
  );
}

function StoredRow({ row, t }: { row: MemoryRow; t: Translate }) {
  return (
    <li className="px-3 py-3.5">
      <p className="max-w-readable text-lead leading-relaxed text-ink">
        {row.memory}
      </p>
      <MetaLine
        parts={[shortStamp(row.time_stamp), row.category, row.memory_class]}
        tag={
          row.consolidated ? <Tag>{t("exp.consolidatedTag")}</Tag> : undefined
        }
      />
    </li>
  );
}

export default function Ask() {
  const { t, lang } = useT();
  const instance = useQuery({ queryKey: ["instance"], queryFn: api.instance });
  const secrets = useQuery({ queryKey: ["secrets"], queryFn: api.getSecrets });

  const prefs = usePrefs();
  const [query, setQuery] = useState("");
  const limit = prefs.retrieveLimit;
  const [showFilters, setShowFilters] = useState(false);
  const [fCategory, setFCategory] = useState("");
  const [fSpeaker, setFSpeaker] = useState("");
  const [fTopic, setFTopic] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeHitId, setActiveHitId] = useState<string | null>(null);
  const [pages, setPages] = useState<string[]>([]);

  const filters = useMemo(() => {
    const f: Record<string, unknown> = {};
    if (fCategory.trim()) f.category = fCategory.trim();
    if (fSpeaker.trim()) f.speaker_name = fSpeaker.trim();
    if (fTopic.trim() && !Number.isNaN(Number(fTopic)))
      f.topic_id = Number(fTopic);
    return Object.keys(f).length ? f : null;
  }, [fCategory, fSpeaker, fTopic]);
  const activeFilters = filters ? Object.keys(filters).length : 0;

  const retrieve = useMutation({
    mutationFn: () => api.retrieve({ query, limit, filters }),
    onSuccess: (data) => {
      setSelected(new Set(data.results.map((r) => String(r.id))));
      setActiveHitId(data.results.length ? String(data.results[0].id) : null);
    },
  });

  const ask = useMutation({
    mutationFn: () =>
      api.chat({
        question: query,
        limit,
        filters,
        selected_ids: Array.from(selected),
        use_memory: true,
      }),
  });

  useEffect(() => {
    ask.reset();
  }, [query]);

  const hits = retrieve.data?.results ?? [];
  const topScore = hits.length ? Math.max(...hits.map((h) => h.score)) || 1 : 1;
  const chosen = useMemo(
    () => hits.filter((h) => selected.has(String(h.id))),
    [hits, selected],
  );
  const activeHit =
    hits.find((hit) => String(hit.id) === activeHitId) ?? hits[0];

  const contextPreview = useMemo(() => {
    if (!chosen.length) return t("ask.noMemories");
    return chosen
      .map(
        (h, i) =>
          `[${i + 1}] ${[h.time_stamp, h.weekday].filter(Boolean).join(" ")} ${h.memory}`,
      )
      .join("\n");
  }, [chosen, t]);

  const ready = instance.data?.ready ?? false;
  const submit = () => query.trim() && retrieve.mutate();

  const browsing = !retrieve.data && !retrieve.isPending;
  const offset = pages[pages.length - 1];
  const stored = useQuery({
    queryKey: ["memories", offset],
    queryFn: () => api.memories({ limit: 25, offset }),
    enabled: ready && browsing,
  });
  const rows = stored.data?.rows ?? [];

  return (
    <>
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-h1 font-semibold tracking-[-0.03em] text-ink sm:text-[2rem]">
            {t("ask.title")}
          </h1>
          <p className="mt-1.5 text-sm text-ink2">{t("ask.intro")}</p>
        </div>
        <div className="hidden items-center gap-2 text-tiny text-ink3 sm:flex">
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              ready ? "bg-good" : "bg-ink4",
            )}
          />
          {ready
            ? t("shell.entries", {
                n: formatNumber(instance.data?.points ?? 0),
              })
            : t("shell.notReady")}
        </div>
      </header>

      <div className="workspace-composer searchbar mt-8 min-h-16 text-left">
        <Search className="h-[18px] w-[18px] shrink-0 text-ink4" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && submit()}
          placeholder={t("ask.placeholder")}
          className="field-flat min-w-0 flex-1 text-h3 placeholder:text-ink4"
          disabled={!ready}
        />
        <Select
          flat
          value={String(limit)}
          options={["5", "10", "20", "50"].map((v) => ({
            value: v,
            label: t("ask.top", { n: v }),
          }))}
          onChange={(e) => setPrefs({ retrieveLimit: Number(e.target.value) })}
          className="hidden w-[4.5rem] shrink-0 text-tiny text-ink3 hover:text-ink sm:block"
        />
        <Button
          variant="primary"
          onClick={submit}
          loading={retrieve.isPending}
          disabled={!ready}
        >
          {t("ask.retrieve")}
        </Button>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-tiny">
        <button
          className={cn(
            "inline-flex items-center gap-1.5 transition-colors",
            activeFilters ? "text-accent" : "text-ink3 hover:text-ink",
          )}
          onClick={() => setShowFilters((v) => !v)}
        >
          <SlidersHorizontal className="h-3 w-3" />
          {t("ask.filters")}
          {activeFilters > 0 && <span>{activeFilters}</span>}
        </button>

        {ready && browsing && (
          <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-ink4">
            <span>{t("ask.tryThese")}</span>
            {(SAMPLE_QUERIES[lang] ?? SAMPLE_QUERIES.en).map((q) => (
              <button
                key={q}
                className="text-ink3 underline decoration-rule underline-offset-2 transition-colors hover:text-accent hover:decoration-accent/40"
                onClick={() => {
                  setQuery(q);
                  setTimeout(() => retrieve.mutate(), 0);
                }}
              >
                {q}
              </button>
            ))}
          </span>
        )}
      </div>

      {showFilters && (
        <div className="mt-5 grid gap-3 rounded-xl border border-ruleSoft bg-sunken/40 p-4 text-left sm:grid-cols-3">
          <Field label={t("exp.fCategory")} hint={t("ask.filterExact")}>
            <Input
              value={fCategory}
              onChange={(e) => setFCategory(e.target.value)}
              placeholder="work"
            />
          </Field>
          <Field label={t("exp.fSpeaker")} hint={t("ask.filterExact")}>
            <Input
              value={fSpeaker}
              onChange={(e) => setFSpeaker(e.target.value)}
              placeholder="speaker"
            />
          </Field>
          <Field label={t("exp.fTopic")} hint={t("ask.filterExact")}>
            <Input
              type="number"
              value={fTopic}
              onChange={(e) => setFTopic(e.target.value)}
              placeholder="0"
            />
          </Field>
        </div>
      )}

      {!ready && (
        <div className="mx-auto mt-10 max-w-3xl">
          <Note tone="warn" title={t("ask.notConfigured")}>
            <p>{t("ask.notConfiguredBody")}</p>
            <Link
              to="/settings"
              className="mt-2 inline-flex items-center gap-1.5 font-medium text-accent hover:underline"
            >
              {t("ask.goSettings")}
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </Note>
        </div>
      )}

      {retrieve.isError && (
        <div className="mx-auto mt-10 max-w-3xl">
          <Note tone="bad" title={t("ask.retrieveFailed")}>
            {String((retrieve.error as Error).message)}
          </Note>
        </div>
      )}

      {ready && browsing && (
        <div className="workspace-panel mx-auto mt-10 max-w-5xl overflow-hidden">
          <div className="flex items-baseline justify-between gap-4 border-b border-ruleSoft bg-sunken/45 px-6 py-4">
            <p className="text-sm font-medium text-ink">
              {t("ask.stored")}
              <span className="ml-2 font-normal text-ink3">
                {t("ask.storedCount", { n: instance.data?.points ?? 0 })}
              </span>
            </p>
            {(pages.length > 0 || stored.data?.next_offset) && (
              <span className="flex items-center gap-1">
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={pages.length === 0}
                  onClick={() => setPages((p) => p.slice(0, -1))}
                >
                  {t("common.previous")}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={!stored.data?.next_offset}
                  onClick={() =>
                    setPages((p) => [...p, stored.data!.next_offset!])
                  }
                >
                  {t("common.next")}
                </Button>
              </span>
            )}
          </div>

          {stored.isLoading && (
            <p className="flex items-center gap-2 py-12 text-base text-ink3">
              <Spinner /> {t("ask.loadingStored")}
            </p>
          )}
          {!stored.isLoading && rows.length === 0 && (
            <Empty title={t("ask.storeEmpty")}>{t("ask.storeEmptyHint")}</Empty>
          )}
          <ul className="stagger divide-y divide-ruleSoft px-3 sm:px-4">
            {rows.map((row) => (
              <StoredRow key={row.id} row={row} t={t} />
            ))}
          </ul>
        </div>
      )}

      {!browsing && (
        <div className="mt-10 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,.72fr)]">
          <div className="workspace-panel overflow-hidden">
            <div className="flex items-baseline justify-between gap-4 border-b border-ruleSoft bg-sunken/45 px-5 py-4">
              <p className="text-sm font-medium text-ink">
                {t("ask.retrieved")}
              </p>
              <p className="text-tiny text-ink3">
                {retrieve.data &&
                  t("ask.retrievedMeta", {
                    n: retrieve.data.count,
                    ms: retrieve.data.latency_ms,
                    sel: chosen.length,
                  })}
              </p>
            </div>

            {retrieve.isPending && (
              <p className="flex items-center gap-2 py-12 text-base text-ink3">
                <Spinner /> {t("ask.searching")}
              </p>
            )}
            {!retrieve.isPending && hits.length === 0 && (
              <Empty title={t("ask.noResults")}>
                {t("ask.noResultsQueried")}
              </Empty>
            )}

            <ul className="stagger px-2 py-2">
              {hits.map((hit, i) => (
                <HitRow
                  key={hit.id}
                  hit={hit}
                  index={i}
                  t={t}
                  topScore={topScore}
                  selected={selected.has(String(hit.id))}
                  onToggle={() => {
                    setActiveHitId(String(hit.id));
                    setSelected((prev) => {
                      const next = new Set(prev);
                      const id = String(hit.id);
                      if (next.has(id)) next.delete(id);
                      else next.add(id);
                      return next;
                    });
                  }}
                />
              ))}
            </ul>
          </div>

          <div className="space-y-6 lg:sticky lg:top-8 lg:self-start">
            <div className="workspace-panel overflow-hidden">
              <div className="flex items-baseline justify-between gap-4 border-b border-ruleSoft bg-sunken/45 px-5 py-4">
                <p className="text-sm font-medium text-ink">
                  {t("ask.context")}
                  <span className="ml-2 font-normal text-ink3">
                    {t("ask.contextMeta", {
                      n: chosen.length,
                      c: contextPreview.length,
                    })}
                  </span>
                </p>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => ask.mutate()}
                  loading={ask.isPending}
                  disabled={
                    !ready || !query.trim() || !secrets.data?.has_api_key
                  }
                >
                  {t("ask.ask")}
                </Button>
              </div>
              {!secrets.data?.has_api_key && (
                <div className="mb-3">
                  <Note tone="warn" title={t("ask.noKeyTitle")}>
                    {t("ask.noKeyBody")}{" "}
                    <Link to="/settings" className="link">
                      {t("nav.settings")}
                    </Link>
                  </Note>
                </div>
              )}
              <div className="p-4">
                <CodeBlock code={contextPreview} maxHeight="13rem" wrap />
              </div>
            </div>

            <div className="workspace-panel overflow-hidden">
              <div className="border-b border-ruleSoft bg-sunken/45 px-5 py-4">
                <p className="text-sm font-medium text-ink">
                  {t("ask.details")}
                </p>
              </div>
              <DetailPanel hit={activeHit} t={t} />
            </div>

            {ask.isError && (
              <Note tone="bad" title={t("ask.callFailed")}>
                {String((ask.error as Error).message)}
              </Note>
            )}

            {ask.data && (
              <div className="animate-fade-up rounded-2xl border border-accent/20 bg-accent-wash/40 p-6">
                <p className="mb-3 text-sm font-medium text-ink">
                  {t("ask.answer")}
                </p>
                <p className="max-w-readable whitespace-pre-wrap text-lead leading-[1.75] text-ink">
                  {ask.data.answer}
                </p>
                <p className="mt-4 border-t border-ruleSoft pt-3 text-micro tracking-normal text-ink4">
                  {ask.data.usage?.total_tokens
                    ? t("ask.answerMetaTokens", {
                        model: ask.data.model,
                        ms: ask.data.latency_ms,
                        tokens: formatNumber(ask.data.usage.total_tokens),
                        used: ask.data.used_count,
                      })
                    : t("ask.answerMeta", {
                        model: ask.data.model,
                        ms: ask.data.latency_ms,
                        used: ask.data.used_count,
                      })}
                </p>
                <details className="mt-4">
                  <summary className="cursor-pointer text-tiny text-ink3 transition-colors hover:text-ink">
                    {t("ask.showPrompt")}
                  </summary>
                  <div className="mt-3 space-y-3">
                    {ask.data.messages.map((m, i) => (
                      <div key={i}>
                        <p className="mb-1 text-micro tracking-normal text-ink4">
                          {m.role}
                        </p>
                        <CodeBlock code={m.content} maxHeight="13rem" wrap />
                      </div>
                    ))}
                  </div>
                </details>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
