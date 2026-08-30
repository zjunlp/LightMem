import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Trash2 } from "lucide-react";
import { api, type ChatMessage } from "@/lib/api";
import { parsePipeline } from "@/lib/pipeline";
import { useT } from "@/lib/i18n";
import { useJobStream } from "@/lib/useJobStream";
import { isoStamp } from "@/lib/utils";
import { PageHeader } from "@/components/Layout";
import { usePrefs } from "@/lib/prefs";
import JobConsole from "@/components/JobConsole";
import PipelineView from "@/components/PipelineView";
import {
  Button,
  Field,
  Note,
  Section,
  Select,
  Switch,
  Textarea,
} from "@/components/ui";

type Msg = ChatMessage & { key: string };

let counter = 0;
const withKeys = (messages: ChatMessage[]): Msg[] =>
  messages.map((m) => ({
    ...m,
    key: `m${counter++}`,
    time_stamp: m.time_stamp ?? isoStamp(new Date(Date.now() + counter * 1000)),
  }));

function MessageRow({
  msg,
  onChange,
  onRemove,
  placeholder,
}: {
  msg: Msg;
  onChange: (next: Partial<Msg>) => void;
  onRemove: () => void;
  placeholder: string;
}) {
  return (
    <div className="group py-4 first:pt-0">
      <div className="mb-1.5 flex items-center gap-2">
        <Select
          flat
          value={msg.role}
          options={["user", "assistant", "system"]}
          onChange={(e) => onChange({ role: e.target.value })}
          className="w-20 text-micro tracking-normal text-ink3 hover:text-ink"
        />
        <span className="flex-1" />
        <button
          onClick={onRemove}
          className="shrink-0 p-1 text-ink4 opacity-0 transition hover:text-badInk group-hover:opacity-100"
          aria-label="remove"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
      <Textarea
        value={msg.content}
        onChange={(e) => onChange({ content: e.target.value })}
        rows={3}
        className="resize-none border-0 bg-transparent px-0 py-0 text-sm focus:ring-0"
        placeholder={placeholder}
      />
    </div>
  );
}

export default function Ingest() {
  const queryClient = useQueryClient();
  const { t } = useT();
  const prefs = usePrefs();

  const instance = useQuery({ queryKey: ["instance"], queryFn: api.instance });
  const ready = instance.data?.ready ?? false;

  const [messages, setMessages] = useState<Msg[]>(() =>
    withKeys([
      { role: "user", content: "" },
      { role: "assistant", content: "" },
    ]),
  );
  const [forceSegment, setForceSegment] = useState(true);
  const [forceExtract, setForceExtract] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const hasContent = messages.some((m) => m.content.trim().length > 0);
  const stream = useJobStream(jobId);
  const pipeline = useMemo(() => parsePipeline(stream.events), [stream.events]);

  const submit = useCallback(async () => {
    setSubmitError(null);
    try {
      const { job_id } = await api.addMemory({
        messages: messages
          .filter((m) => m.content.trim())
          .map(({ role, content, time_stamp }) => ({
            role,
            content,
            time_stamp,
          })),
        force_segment: forceSegment,
        force_extract: forceExtract,
      });
      setJobId(job_id);
    } catch (err) {
      setSubmitError(String((err as Error).message ?? err));
    }
  }, [messages, forceSegment, forceExtract]);

  const consolidate = useMutation({
    mutationFn: () =>
      api.offlineUpdate({
        top_k: prefs.topK,
        keep_top_n: prefs.keepTopN,
        score_threshold: prefs.scoreThreshold,
      }),
    onSuccess: (data) => setJobId(data.job_id),
  });

  useEffect(() => {
    if (stream.status === "succeeded" || stream.status === "failed") {
      queryClient.invalidateQueries({ queryKey: ["instance"] });
    }
  }, [stream.status, queryClient]);

  return (
    <>
      <PageHeader
        title={t("ing.title")}
        description={t("ing.intro")}
        actions={
          <>
            <Button
              variant="ghost"
              onClick={() => consolidate.mutate()}
              loading={consolidate.isPending}
              disabled={!ready || (instance.data?.points ?? 0) === 0}
              title={t("ing.consolidateHint")}
            >
              {t("ing.consolidate")}
            </Button>
            <Button
              variant="primary"
              onClick={submit}
              loading={stream.running}
              disabled={!hasContent || !ready}
            >
              {t("ing.addMemory")}
            </Button>
          </>
        }
      />

      {!ready && (
        <div className="mb-10">
          <Note tone="warn" title={t("common.noInstanceTitle")}>
            {t("common.buildFirst")}{" "}
            <Link to="/settings" className="link">
              {t("nav.settings")}
            </Link>
          </Note>
        </div>
      )}

      {submitError && (
        <div className="mb-10">
          <Note tone="bad" title={t("ing.submitFailed")}>
            {submitError}
          </Note>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,0.78fr)_minmax(0,1.22fr)]">
        <div className="space-y-8 lg:sticky lg:top-8 lg:self-start">
          <div className="rounded-2xl border border-rule bg-raised p-5 shadow-raise sm:p-6">
            <Section
              title={t("ing.turns")}
              description={t("ing.turnsCount", { m: messages.length })}
              divided={false}
            >
              <div className="max-h-[26rem] divide-y divide-ruleSoft overflow-y-auto">
                {messages.map((msg, i) => (
                  <MessageRow
                    key={msg.key}
                    msg={msg}
                    placeholder={t("ing.contentPlaceholder")}
                    onChange={(next) =>
                      setMessages((prev) =>
                        prev.map((m, j) => (j === i ? { ...m, ...next } : m)),
                      )
                    }
                    onRemove={() =>
                      setMessages((prev) => prev.filter((_, j) => j !== i))
                    }
                  />
                ))}
                {!messages.length && (
                  <p className="py-6 text-tiny text-ink3">{t("ing.noTurns")}</p>
                )}
              </div>

              <div className="mt-4 flex items-center gap-4 text-tiny">
                <button
                  className="flex items-center gap-1 text-accent hover:underline"
                  onClick={() =>
                    setMessages((prev) => [
                      ...prev,
                      ...withKeys([
                        { role: "user", content: "" },
                        { role: "assistant", content: "" },
                      ]),
                    ])
                  }
                >
                  <Plus className="h-3 w-3" /> {t("ing.addTurn")}
                </button>
                <button
                  className="text-ink3 hover:text-ink"
                  onClick={() => setMessages([])}
                >
                  {t("common.clear")}
                </button>
              </div>
            </Section>
          </div>

          <div className="rounded-2xl border border-rule bg-raised p-5 shadow-raise sm:p-6">
            <Section title={t("ing.options")} divided={false}>
              <div className="divide-y divide-ruleSoft">
                <Field
                  label={t("ing.forceSegment")}
                  hint={t("ing.forceSegmentHint")}
                  inline
                >
                  <Switch checked={forceSegment} onChange={setForceSegment} />
                </Field>
                <Field
                  label={t("ing.forceExtract")}
                  hint={t("ing.forceExtractHint")}
                  inline
                >
                  <Switch checked={forceExtract} onChange={setForceExtract} />
                </Field>
              </div>
            </Section>
          </div>
        </div>

        <div className="space-y-8">
          <div className="rounded-2xl border border-rule bg-raised p-5 shadow-raise sm:p-6">
            <Section
              title={t("ing.pipeline")}
              description={t("ing.pipelineDesc")}
              divided={false}
            >
              <PipelineView state={pipeline} />
            </Section>
          </div>

          {jobId ? (
            <div className="rounded-2xl border border-rule bg-raised p-5 shadow-raise sm:p-6">
              <Section title={t("ing.log")} divided={false}>
                <JobConsole stream={stream} height="16rem" />
              </Section>
            </div>
          ) : (
            <p className="border-t border-rule pt-8 text-sm text-ink2">
              {t("ing.idle")}
            </p>
          )}
        </div>
      </div>
    </>
  );
}
