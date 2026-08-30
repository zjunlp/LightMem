import type { JobEvent } from "./api";

export type StageKey =
  | "normalize"
  | "compress"
  | "segment"
  | "extract"
  | "store";

export type StageState = {
  key: StageKey;
  labelKey: string;
  status: "idle" | "active" | "done" | "skipped";
  detailKey?: string;
  detailVars?: Record<string, string | number>;
  metric?: string;
};

export type ParsedMessage = {
  role: string;
  content: string;
  time_stamp?: string;
};

export type ParsedEntry = {
  index: number;
  time: string;
  weekday: string;
  speakerId: string;
  speakerName: string;
  topicId: string;
  memory: string;
};

export type PipelineState = {
  stages: StageState[];
  raw: ParsedMessage[] | null;
  normalized: ParsedMessage[] | null;
  compressed: ParsedMessage[] | null;
  segments: ParsedMessage[][] | null;
  entries: ParsedEntry[];
  compressionRate: string | null;
  compressionSkipped: boolean;
  segmentationSkipped: boolean;
  segmentCount: number | null;
  extractTriggers: number | null;
  extractionSkipped: boolean;
  apiCalls: number | null;
  entriesCreated: number | null;
  inserted: number | null;
  topicsAssigned: number | null;
  savedToFile: boolean;
};

const EMPTY: PipelineState = {
  stages: [],
  raw: null,
  normalized: null,
  compressed: null,
  segments: null,
  entries: [],
  compressionRate: null,
  compressionSkipped: false,
  segmentationSkipped: false,
  segmentCount: null,
  extractTriggers: null,
  extractionSkipped: false,
  apiCalls: null,
  entriesCreated: null,
  inserted: null,
  topicsAssigned: null,
  savedToFile: false,
};

function tryJson<T>(text: string): T | null {
  try {
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

function stripCallId(message: string): string {
  return message.replace(/^\[[a-z_]+_\d{8}_\d{6}_\d+\]\s*/, "");
}

const ENTRY_RE =
  /^MemoryEntry\[(\d+)\]: time=(.*?), weekday=(.*?), speaker_id=(.*?), speaker_name=(.*?), topic_id=(.*?), memory=([\s\S]*)$/;

export function parsePipeline(events: JobEvent[]): PipelineState {
  const state: PipelineState = { ...EMPTY, entries: [] };

  for (const event of events) {
    if (event.kind !== "log" || !event.message) continue;
    const line = stripCallId(event.message);

    let m: RegExpMatchArray | null;

    if ((m = line.match(/^Raw input sample: ([\s\S]+)$/))) {
      state.raw = tryJson<ParsedMessage[]>(m[1]);
    } else if ((m = line.match(/^Normalized messages sample: ([\s\S]+)$/))) {
      state.normalized = tryJson<ParsedMessage[]>(m[1]);
    } else if ((m = line.match(/^Compressed messages sample: ([\s\S]+)$/))) {
      state.compressed = tryJson<ParsedMessage[]>(m[1]);
    } else if ((m = line.match(/^Target compression rate: (.+)$/))) {
      state.compressionRate = m[1].trim();
    } else if (/^Pre-compression disabled/.test(line)) {
      state.compressionSkipped = true;
    } else if (/^Topic segmentation disabled/.test(line)) {
      state.segmentationSkipped = true;
    } else if ((m = line.match(/^Generated (\d+) segments/))) {
      state.segmentCount = Number(m[1]);
    } else if ((m = line.match(/^Segments sample: ([\s\S]+)$/))) {
      state.segments = tryJson<ParsedMessage[][]>(m[1]);
    } else if ((m = line.match(/^Extraction triggered (\d+) times/))) {
      state.extractTriggers = Number(m[1]);
    } else if (/^Extraction not triggered/.test(line)) {
      state.extractionSkipped = true;
    } else if ((m = line.match(/^Assigned global topic IDs: total=(\d+)/))) {
      state.topicsAssigned = Number(m[1]);
    } else if (
      (m = line.match(/^Metadata generation completed with (\d+) API calls/))
    ) {
      state.apiCalls = Number(m[1]);
    } else if ((m = line.match(/^Created (\d+) MemoryEntry objects/))) {
      state.entriesCreated = Number(m[1]);
    } else if (
      (m = line.match(
        /^Successfully inserted (\d+) entries to vector database/,
      ))
    ) {
      state.inserted = Number(m[1]);
    } else if (/^Saving memory entries to file/.test(line)) {
      state.savedToFile = true;
    } else if ((m = line.match(ENTRY_RE))) {
      state.entries.push({
        index: Number(m[1]),
        time: m[2],
        weekday: m[3],
        speakerId: m[4],
        speakerName: m[5],
        topicId: m[6],
        memory: m[7],
      });
    }
  }

  state.stages = buildStages(state);
  return state;
}

function buildStages(s: PipelineState): StageState[] {
  const normalizeDone = Boolean(s.normalized ?? s.raw);

  const compress: StageState = s.compressionSkipped
    ? {
        key: "compress",
        labelKey: "pipe.compress",
        status: "skipped",
        detailKey: "pipe.precompressOff",
      }
    : {
        key: "compress",
        labelKey: "pipe.compress",
        status: s.compressed ? "done" : normalizeDone ? "active" : "idle",
        detailKey: s.compressionRate ? "pipe.targetRate" : undefined,
        detailVars: s.compressionRate ? { r: s.compressionRate } : undefined,
        metric: compressionMetric(s),
      };

  const segment: StageState = s.segmentationSkipped
    ? {
        key: "segment",
        labelKey: "pipe.segment",
        status: "skipped",
        detailKey: "pipe.oneSegment",
      }
    : {
        key: "segment",
        labelKey: "pipe.segment",
        status:
          s.segmentCount != null
            ? "done"
            : compress.status === "done"
              ? "active"
              : "idle",
        metric: s.segmentCount != null ? `${s.segmentCount}` : undefined,
        detailKey: s.topicsAssigned != null ? "pipe.topics" : undefined,
        detailVars:
          s.topicsAssigned != null ? { n: s.topicsAssigned } : undefined,
      };

  const extract: StageState = s.extractionSkipped
    ? {
        key: "extract",
        labelKey: "pipe.extract",
        status: "skipped",
        detailKey: "pipe.thresholdNotMet",
      }
    : {
        key: "extract",
        labelKey: "pipe.extract",
        status:
          s.entriesCreated != null
            ? "done"
            : s.extractTriggers != null
              ? "active"
              : "idle",
        metric: s.entriesCreated != null ? `${s.entriesCreated}` : undefined,
        detailKey: s.apiCalls != null ? "pipe.apiCalls" : undefined,
        detailVars: s.apiCalls != null ? { n: s.apiCalls } : undefined,
      };

  const store: StageState = {
    key: "store",
    labelKey: "pipe.store",
    status:
      s.inserted != null
        ? "done"
        : extract.status === "done"
          ? "active"
          : "idle",
    metric: s.inserted != null ? `${s.inserted}` : undefined,
  };

  return [
    {
      key: "normalize",
      labelKey: "pipe.normalize",
      status: normalizeDone ? "done" : "idle",
      metric:
        (s.normalized ?? s.raw)
          ? `${(s.normalized ?? s.raw)!.length}`
          : undefined,
      detailKey: "pipe.messages",
    },
    compress,
    segment,
    extract,
    store,
  ];
}

function compressionMetric(s: PipelineState): string | undefined {
  if (!s.compressed || !s.normalized) return undefined;
  const before = s.normalized.reduce((n, m) => n + (m.content?.length ?? 0), 0);
  const after = s.compressed.reduce((n, m) => n + (m.content?.length ?? 0), 0);
  if (!before) return undefined;
  return `${Math.round((after / before) * 100)}%`;
}

export function diffTokens(original: string, compressed: string) {
  const kept = new Set(
    compressed
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => w.replace(/^[^\w]+|[^\w]+$/g, "")),
  );
  return original.split(/(\s+)/).map((token) => {
    if (/^\s*$/.test(token)) return { text: token, kept: true, space: true };
    const normalized = token.toLowerCase().replace(/^[^\w]+|[^\w]+$/g, "");
    return { text: token, kept: kept.has(normalized), space: false };
  });
}
