import { useState } from "react";
import type { PipelineState, StageState } from "@/lib/pipeline";
import { diffTokens } from "@/lib/pipeline";
import { useT, type Translate } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import { Empty, Segmented } from "@/components/ui";

const STAGE_RAMP = [
  "#86B6EF",
  "#5598E7",
  "#2A78D6",
  "#1C5CAB",
  "#104281",
] as const;
function StageNode({
  stage,
  index,
  last,
  t,
}: {
  stage: StageState;
  index: number;
  last: boolean;
  t: Translate;
}) {
  const color = STAGE_RAMP[Math.min(index, STAGE_RAMP.length - 1)];
  const done = stage.status === "done";
  const active = stage.status === "active";
  const skipped = stage.status === "skipped";
  const lit = done || active;

  return (
    <>
      <div className="min-w-0 flex-1">
        <div className="flex h-4 items-center gap-1.5">
          <span
            className={cn(
              "h-1.5 w-1.5 shrink-0 rounded-full",
              active && "animate-breathe",
            )}
            style={{ background: lit ? color : "#D8D6D0" }}
          />
          <span
            className={cn(
              "truncate text-[0.625rem] tracking-[-0.02em]",
              lit ? "text-ink" : "text-ink3",
            )}
          >
            {t(stage.labelKey)}
          </span>
        </div>
        <p
          className={cn(
            "mt-2 text-h2 font-medium tabular-nums",
            lit ? "text-ink" : "text-ink4",
          )}
        >
          {stage.metric ?? (skipped ? t("pipe.skipped") : "—")}
        </p>
        <p className="mt-1 h-4 truncate text-micro tracking-normal text-ink3">
          {stage.detailKey ? t(stage.detailKey, stage.detailVars) : ""}
        </p>
      </div>
      {!last && (
        <div className="mx-1 mt-2 h-px w-2 shrink-0 self-start bg-rule lg:mx-2 lg:w-3" />
      )}
    </>
  );
}

function CompressionDiff({ state, t }: { state: PipelineState; t: Translate }) {
  const before = state.normalized ?? state.raw;
  const after = state.compressed;

  if (state.compressionSkipped) {
    return (
      <Empty title={t("pipe.compressionOff")}>
        {t("pipe.compressionOffHint")}
      </Empty>
    );
  }
  if (!before || !after) return <Empty title={t("pipe.noCompression")} />;

  return (
    <div className="space-y-5">
      {before.map((msg, i) => {
        const compressedMsg = after[i];
        if (!compressedMsg) return null;
        const tokens = diffTokens(
          msg.content ?? "",
          compressedMsg.content ?? "",
        );
        const dropped = tokens.filter((x) => !x.space && !x.kept).length;
        const total = tokens.filter((x) => !x.space).length;
        return (
          <div key={i}>
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              <span className="text-micro tracking-normal text-ink3">
                {msg.role}
              </span>
              <span className="font-mono text-micro tabular-nums tracking-normal text-ink3">
                {t("pipe.kept", { kept: total - dropped, total })}
              </span>
            </div>
            <p className="max-w-readable text-sm leading-relaxed">
              {tokens.map((token, j) =>
                token.space ? (
                  <span key={j}>{token.text}</span>
                ) : (
                  <span
                    key={j}
                    className={
                      token.kept ? "text-ink" : "text-ink4 line-through"
                    }
                  >
                    {token.text}
                  </span>
                ),
              )}
            </p>
          </div>
        );
      })}
    </div>
  );
}

function Segments({ state, t }: { state: PipelineState; t: Translate }) {
  if (state.segmentationSkipped) {
    return (
      <Empty title={t("pipe.segmentationOff")}>
        {t("pipe.segmentationOffHint")}
      </Empty>
    );
  }
  if (!state.segments?.length) return <Empty title={t("pipe.noSegments")} />;

  return (
    <div className="space-y-6">
      {state.segments.map((segment, i) => (
        <div key={i}>
          <div className="mb-2 flex items-baseline gap-3">
            <span className="text-sm font-medium text-ink">
              {t("pipe.segmentN", { i })}
            </span>
            <span className="text-tiny text-ink3">
              {t("pipe.segmentMessages", { n: segment.length })}
            </span>
          </div>
          <div className="space-y-1.5 border-l border-rule pl-4">
            {segment.map((msg, j) => (
              <p
                key={j}
                className="max-w-readable text-sm leading-relaxed text-ink2"
              >
                <span className="mr-2 text-micro tracking-normal text-ink4">
                  {msg.role}
                </span>
                {msg.content}
              </p>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function Entries({ state, t }: { state: PipelineState; t: Translate }) {
  if (state.extractionSkipped) {
    return (
      <Empty title={t("pipe.extractionSkipped")}>
        {t("pipe.noEntriesHint")}
      </Empty>
    );
  }
  if (!state.entries.length) return <Empty title={t("pipe.noEntries")} />;

  return (
    <ul className="divide-y divide-ruleSoft">
      {state.entries.map((entry) => (
        <li key={entry.index} className="py-3 first:pt-0 last:pb-0">
          <p className="max-w-readable text-base text-ink">{entry.memory}</p>
          <p className="mt-1 flex flex-wrap items-center gap-x-3 text-micro tracking-normal text-ink3">
            <span>{entry.time}</span>
            <span>{entry.weekday}</span>
            <span>{t("pipe.topicN", { n: entry.topicId })}</span>
            {entry.speakerName && entry.speakerName !== "None" && (
              <span>{entry.speakerName}</span>
            )}
          </p>
        </li>
      ))}
    </ul>
  );
}

export default function PipelineView({ state }: { state: PipelineState }) {
  const { t } = useT();
  const [tab, setTab] = useState<"entries" | "segments" | "compress">(
    "entries",
  );

  return (
    <div>
      <div className="flex items-start">
        {state.stages.map((stage, i) => (
          <StageNode
            key={stage.key}
            stage={stage}
            index={i}
            last={i === state.stages.length - 1}
            t={t}
          />
        ))}
      </div>

      <div className="mt-3 flex items-center gap-4 text-micro tracking-normal text-ink3">
        <span className="flex items-center gap-1.5">
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ background: STAGE_RAMP[0] }}
          />
          {t("pipe.sensory")}
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ background: STAGE_RAMP[3] }}
          />
          {t("pipe.shortterm")}
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ background: STAGE_RAMP[4] }}
          />
          {t("pipe.longterm")}
        </span>
      </div>

      <div className="mt-8 border-b border-rule">
        <Segmented
          value={tab}
          onChange={setTab}
          items={[
            {
              value: "entries",
              label: t("pipe.tabEntries"),
              count: state.entries.length,
            },
            {
              value: "segments",
              label: t("pipe.tabSegments"),
              count: state.segments?.length ?? 0,
            },
            { value: "compress", label: t("pipe.tabCompression") },
          ]}
        />
      </div>

      <div className="max-h-[32rem] overflow-y-auto pt-6">
        {tab === "entries" && <Entries state={state} t={t} />}
        {tab === "segments" && <Segments state={state} t={t} />}
        {tab === "compress" && <CompressionDiff state={state} t={t} />}
      </div>
    </div>
  );
}
