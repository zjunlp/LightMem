import { useEffect, useRef, useState } from "react";
import { api, type JobDetail, type JobEvent } from "./api";

export type JobStream = {
  events: JobEvent[];
  logs: JobEvent[];
  stage: string | null;
  status: JobDetail["status"] | "idle";
  error: string | null;
  running: boolean;
};

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

export function useJobStream(jobId: string | null): JobStream {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [stage, setStage] = useState<string | null>(null);
  const [status, setStatus] = useState<JobDetail["status"] | "idle">("idle");
  const [error, setError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!jobId) return;

    setEvents([]);
    setStage(null);
    setStatus("pending");
    setError(null);

    const source = new EventSource(`/api/jobs/${jobId}/events`);
    sourceRef.current = source;

    source.onmessage = (raw) => {
      let event: JobEvent;
      try {
        event = JSON.parse(raw.data);
      } catch {
        return;
      }

      setEvents((prev) => [...prev, event]);

      if (event.kind === "stage" && event.stage) setStage(event.stage);
      if (event.kind === "status" && event.status) {
        setStatus(event.status as JobDetail["status"]);
        if (event.error) setError(event.error);
      }
      if (event.kind === "done") {
        if (event.status) setStatus(event.status as JobDetail["status"]);
        source.close();
        sourceRef.current = null;
        api
          .job(jobId)
          .then((detail) => {
            setError(detail.error);
            setStatus(detail.status);
          })
          .catch(() => undefined);
      }
    };

    source.onerror = () => {
      if (sourceRef.current) {
        sourceRef.current.close();
        sourceRef.current = null;
        api
          .job(jobId)
          .then((detail) => {
            setStatus(detail.status);
            setError(detail.error);
            if (!TERMINAL.has(detail.status)) {
              setError("Lost the event stream. Reload to reattach.");
            }
          })
          .catch(() => setError("Lost the event stream."));
      }
    };

    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, [jobId]);

  return {
    events,
    logs: events.filter((e) => e.kind === "log"),
    stage,
    status,
    error,
    running: status === "pending" || status === "running",
  };
}
