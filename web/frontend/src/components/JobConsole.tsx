import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { JobStream } from "@/lib/useJobStream";
import { useT } from "@/lib/i18n";
import { cn, formatTime } from "@/lib/utils";
import {
  Button,
  Segmented,
  Spinner,
  Surface,
  SurfaceHead,
  Tag,
} from "@/components/ui";

const LEVEL_STYLE: Record<string, string> = {
  DEBUG: "text-ink4",
  INFO: "text-ink2",
  WARNING: "text-warnInk",
  ERROR: "text-badInk",
  CRITICAL: "text-badInk",
};

type Filter = "info" | "all" | "problems";

export default function JobConsole({
  stream,
  title,
  meta,
  emptyHint,
  className,
  height = "22rem",
}: {
  stream: JobStream;
  title?: string;
  meta?: string;
  emptyHint?: string;
  className?: string;
  height?: string;
}) {
  const { t } = useT();
  const [filter, setFilter] = useState<Filter>("info");
  const [pinned, setPinned] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  const logs = useMemo(() => {
    if (filter === "all") return stream.logs;
    if (filter === "problems")
      return stream.logs.filter(
        (l) => l.level === "WARNING" || l.level === "ERROR",
      );
    return stream.logs.filter((l) => l.level !== "DEBUG");
  }, [stream.logs, filter]);

  useEffect(() => {
    if (!pinned || !scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [logs.length, pinned]);

  const tone: "good" | "bad" | "neutral" =
    stream.status === "succeeded"
      ? "good"
      : stream.status === "failed"
        ? "bad"
        : "neutral";

  return (
    <Surface className={cn("overflow-hidden", className)}>
      <SurfaceHead
        title={title ?? t("job.runLog")}
        meta={
          meta ??
          (stream.stage ? t("job.stage", { s: stream.stage }) : undefined)
        }
        actions={
          <>
            {stream.running ? (
              <Spinner className="h-3.5 w-3.5" />
            ) : (
              <Tag tone={tone}>{t(`job.${stream.status}`)}</Tag>
            )}
            <div className="ml-2">
              <Segmented
                value={filter}
                onChange={setFilter}
                items={[
                  { value: "info", label: t("job.filterInfo") },
                  { value: "all", label: t("job.filterAll") },
                  { value: "problems", label: t("job.filterProblems") },
                ]}
              />
            </div>
          </>
        }
      />

      <div className="relative">
        <div
          ref={scrollRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            setPinned(el.scrollHeight - el.scrollTop - el.clientHeight < 40);
          }}
          className="overflow-y-auto px-4 py-3 font-mono text-micro leading-[1.55] tracking-normal"
          style={{ height }}
        >
          {logs.length === 0 ? (
            <p className="py-10 text-center text-tiny font-sans text-ink3">
              {emptyHint ?? t("job.empty")}
            </p>
          ) : (
            logs.map((log, i) => (
              <div key={`${log.seq}-${i}`} className="flex gap-3 py-px">
                <span className="shrink-0 text-ink4">{formatTime(log.ts)}</span>
                <span
                  className={cn(
                    "min-w-0 flex-1 whitespace-pre-wrap break-words",
                    LEVEL_STYLE[log.level ?? "INFO"],
                  )}
                >
                  {log.message}
                </span>
              </div>
            ))
          )}
        </div>

        {!pinned && (
          <Button
            size="sm"
            variant="secondary"
            className="absolute bottom-3 right-4 shadow-raise"
            onClick={() => {
              setPinned(true);
              if (scrollRef.current)
                scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
            }}
          >
            <ChevronDown className="h-3.5 w-3.5" />
            {t("common.follow")}
          </Button>
        )}
      </div>

      {stream.error && (
        <div className="border-t border-ruleSoft px-4 py-3">
          <p className="max-w-readable text-sm leading-relaxed text-badInk">
            {stream.error}
          </p>
        </div>
      )}
    </Surface>
  );
}
