import { useSyncExternalStore } from "react";

export type Lang = "zh" | "en";

const KEY = "lightmem.console.lang";

function initialLang(): Lang {
  try {
    const stored = localStorage.getItem(KEY);
    if (stored === "zh" || stored === "en") return stored;
  } catch {
    /* storage disabled */
  }
  return "zh";
}

let lang: Lang = initialLang();
const listeners = new Set<() => void>();

export function setLang(next: Lang) {
  lang = next;
  try {
    localStorage.setItem(KEY, next);
  } catch {
    /* ignore */
  }
  document.documentElement.lang = next === "zh" ? "zh-CN" : "en";
  listeners.forEach((fn) => fn());
}

function subscribe(fn: () => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

const getLang = () => lang;

export function useLang(): Lang {
  return useSyncExternalStore(subscribe, getLang, getLang);
}

/** Pick the localized variant of a server-provided field. */
export function pickLocalized<T extends Record<string, any>>(
  obj: T | undefined,
  field: string,
  current: Lang,
): string {
  if (!obj) return "";
  if (current === "zh") {
    const zh = obj[`${field}_zh`];
    if (typeof zh === "string" && zh) return zh;
  }
  return typeof obj[field] === "string" ? obj[field] : "";
}

type Dict = Record<string, string>;

const en: Dict = {
  "nav.settings": "Settings",
  "set.autoTitle": "One-click setup",
  "set.autoHint":
    "Finds the local models, the freest GPU, and the right vector dimension.",
  "set.autoRun": "Detect and fill",
  "set.autoCompressor": "Compressor:",
  "set.autoEmbedder": "Embedder:",
  "set.autoDims":
    "Vector dimension set to {n} on both the embedder and the store",
  "set.autoDeviceGpu": "Device: {device} ({free} GB free)",
  "set.autoDeviceCpu": "Device: cpu (no CUDA available)",
  "set.autoOllama": "Local Ollama is up at {host} with {n} model(s)",
  "set.autoNoCompressor":
    "No compressor model found — fill the path in by hand.",
  "set.autoNoEmbedder": "No embedding model found — fill the path in by hand.",
  "set.autoTiktoken":
    "tiktoken cache is cold; run web/scripts/prime_tiktoken_cache.sh or startup will hang.",
  "set.title": "Settings",
  "set.intro": "Restart to apply changes.",
  "set.consolidation": "Consolidation parameters",
  "set.consolidationHint": "",
  "ask.notConfigured": "Nothing is running yet",
  "ask.notConfiguredBody": "Start an instance in Settings to begin.",
  "ask.goSettings": "Open Settings",
  "ask.filters": "Filters",
  "ask.tryThese": "Try:",
  "ask.noKeyTitle": "No API key saved",
  "ask.noKeyBody": "Add one in Settings to enable answering.",
  "ask.storedCount": "{n} entries",
  "exp.fCategory": "Category",
  "exp.fSpeaker": "Speaker",
  "exp.fTopic": "Topic",
  "common.previous": "Previous",
  "common.next": "Next",
  "ask.filterExact": "Exact match. Blank means no filter.",
  "con.topKHint": "Neighbours pulled per entry",
  "con.keepTopNHint": "How many stay queued",
  "con.thresholdHint": "Merge similarity floor",
  "cfg.stepPreset": "Configuration",
  "cfg.stepPresetHint":
    "Pick a ready-made set; everything below stays editable.",
  "cfg.stepKey": "Enter your LLM credentials",
  "cfg.stepKeyLocal": "Local backends need no key.",
  "cfg.stepKeyHint": "Stored server-side, never sent back.",
  "cfg.stepDetails": "Check the details",
  "cfg.stepDetailsHint": "",
  "cfg.stepStart": "Start it",
  "cfg.stepStartHint": "",
  "cfg.needKey": "Enter an API key first",
  "ing.intro": "Turn a conversation into searchable memory.",
  "ask.intro": "Search your memory, then ask with the context you choose.",
  "nav.ingest": "Ingest",
  "nav.ask": "Retrieve & Ask",
  "shell.notReady": "No instance",
  "shell.entries": "{n} entries",
  "lang.toggle": "Switch language",

  "common.copy": "Copy",
  "common.copied": "Copied",
  "common.follow": "Follow",
  "common.clear": "Clear",
  "common.save": "Save",
  "common.noInstanceTitle": "No instance loaded",
  "common.buildFirst": "Start it on the Settings page first.",
  "env.apiKey": "API key",
  "env.apiKeyStored": "Stored {masked} · blank keeps it",
  "env.apiKeyNone": "Not set",
  "env.apiKeyPlaceholderStored": "•••••••• unchanged",
  "env.baseUrl": "Base URL",
  "env.baseUrlHint": "Blank = api.openai.com",
  "env.testConnection": "Test connection",
  "env.saved": "Saved",
  "env.pingOk": "Connected in {ms} ms — replied “{reply}”",
  "env.pingBad": "Could not reach the model",
  "env.keyLooksLikeUrl":
    "The API key field contains a URL. The key goes in the key field, the address in Base URL.",
  "env.baseUrlNoVersion": "Most gateways need a version suffix, e.g. …/v1",
  "cfg.loading": "Loading the config schema…",
  "cfg.loadFailed": "Could not load the config schema",
  "cfg.start": "Start",
  "cfg.restart": "Restart",
  "cfg.running": "Running · {collection} · {n} entries · loaded in {seconds}s",
  "cfg.notRunning": "Not running.",
  "inst.startFailed": "Could not start initialization",
  "ing.title": "Ingest",
  "ing.addMemory": "Add memory",
  "ing.consolidate": "Clean up",
  "ing.consolidateHint": "Merge similar memories and remove duplicates.",
  "ing.submitFailed": "Could not submit",
  "ing.turns": "Turns",
  "ing.turnsCount": "{m} messages",
  "ing.noTurns": "No turns yet.",
  "ing.addTurn": "Add turn",
  "ing.contentPlaceholder": "Message content…",
  "ing.options": "Options",
  "ing.forceSegment": "force_segment",
  "ing.forceSegmentHint": "Flush the buffer now.",
  "ing.forceExtract": "force_extract",
  "ing.forceExtractHint": "Extract even below threshold.",
  "ing.pipeline": "Pipeline",
  "ing.pipelineDesc": "Live view of the memory pipeline.",
  "ing.log": "add_memory log",
  "ing.idle": "Press “Add memory” to start.",
  "pipe.normalize": "Normalize",
  "pipe.compress": "Compress",
  "pipe.segment": "Segment",
  "pipe.extract": "Extract",
  "pipe.store": "Store",
  "pipe.sensory": "sensory",
  "pipe.shortterm": "short-term",
  "pipe.longterm": "long-term",
  "pipe.skipped": "skipped",
  "pipe.messages": "messages",
  "pipe.targetRate": "target {r}",
  "pipe.topics": "{n} topics",
  "pipe.apiCalls": "{n} API calls",
  "pipe.oneSegment": "one segment",
  "pipe.precompressOff": "pre_compress off",
  "pipe.thresholdNotMet": "threshold not met",
  "pipe.tabEntries": "Memory entries",
  "pipe.tabSegments": "Segments",
  "pipe.tabCompression": "Compression",
  "pipe.noEntries": "No memory entries yet",
  "pipe.noEntriesHint":
    "The short-term buffer had not filled enough. Turn on force_extract to push it through anyway.",
  "pipe.extractionSkipped": "Extraction was not triggered",
  "pipe.noSegments": "No segments yet",
  "pipe.segmentationOff": "Topic segmentation is off",
  "pipe.segmentationOffHint":
    "All messages were treated as one segment. Enable topic_segment to split the stream by topic.",
  "pipe.compressionOff": "Pre-compression is off",
  "pipe.compressionOffHint":
    "Enable pre_compress to watch LLMLingua-2 drop low-information tokens here.",
  "pipe.noCompression": "No compression data yet",
  "pipe.segmentN": "Segment {i}",
  "pipe.segmentMessages": "{n} messages",
  "pipe.kept": "{kept}/{total} kept",
  "pipe.topicN": "topic {n}",

  "ask.title": "Retrieve & Ask",
  "ask.placeholder": "Ask something the memories should answer…",
  "ask.top": "top {n}",
  "ask.retrieve": "Retrieve",
  "ask.stored": "Stored memories",
  "ask.loadingStored": "Loading entries…",
  "ask.storeEmpty": "The store is empty",
  "ask.storeEmptyHint": "Add a conversation on step 2 first.",
  "ask.retrieveFailed": "Retrieval failed",
  "ask.retrieved": "Retrieved memories",
  "ask.retrievedMeta": "{n} hits in {ms} ms · {sel} selected",
  "ask.searching": "Searching…",
  "ask.noResults": "No results yet",
  "ask.noResultsQueried": "The store returned nothing for this query.",
  "ask.context": "Context to inject",
  "ask.contextMeta": "{n} memories · {c} characters",
  "ask.details": "Memory details",
  "ask.detailEmpty": "Select a result to inspect it.",
  "ask.detailScore": "Score",
  "ask.detailTime": "Timestamp",
  "ask.ask": "Ask",
  "ask.callFailed": "The model call failed",
  "ask.answer": "Answer",
  "ask.answerMeta": "{model} · {ms} ms · {used} memories used",
  "ask.answerMetaTokens":
    "{model} · {ms} ms · {tokens} tokens · {used} memories used",
  "ask.showPrompt": "Show prompt",
  "ask.noMemories": "(no memories injected)",
  "exp.consolidatedTag": "consolidated",
  "job.runLog": "Run log",
  "job.stage": "stage · {s}",
  "job.filterInfo": "Info",
  "job.filterAll": "All",
  "job.filterProblems": "Problems",
  "job.empty": "Log lines stream here while the job runs.",
  "job.pending": "pending",
  "job.running": "running",
  "job.succeeded": "succeeded",
  "job.failed": "failed",
  "job.cancelled": "cancelled",
  "job.idle": "idle",
};

const zh: Dict = {
  "nav.settings": "设置",
  "set.autoTitle": "一键配置",
  "set.autoHint": "自动找到本地模型、最空闲的显卡和正确的向量维度。",
  "set.autoRun": "检测并填写",
  "set.autoCompressor": "压缩器：",
  "set.autoEmbedder": "嵌入模型：",
  "set.autoDims": "向量维度按 {n} 同时写入嵌入模型和向量库",
  "set.autoDeviceGpu": "设备：{device}（空闲 {free} GB）",
  "set.autoDeviceCpu": "设备：cpu（无可用 CUDA）",
  "set.autoOllama": "检测到本地 Ollama 在 {host}，有 {n} 个模型",
  "set.autoNoCompressor": "没找到压缩器模型，需要手动填路径。",
  "set.autoNoEmbedder": "没找到嵌入模型，需要手动填路径。",
  "set.autoTiktoken":
    "tiktoken 缓存未预热，先跑 web/scripts/prime_tiktoken_cache.sh，否则启动会卡死。",
  "set.title": "设置",
  "set.intro": "重启后生效。",
  "set.consolidation": "整合参数",
  "set.consolidationHint": "",
  "ask.notConfigured": "还没有跑起来",
  "ask.notConfiguredBody": "到设置启动实例。",
  "ask.goSettings": "打开设置",
  "ask.filters": "筛选",
  "ask.tryThese": "试试：",
  "ask.noKeyTitle": "尚未保存 API Key",
  "ask.noKeyBody": "到设置里填一个即可使用问答。",
  "ask.storedCount": "共 {n} 条",
  "exp.fCategory": "类别",
  "exp.fSpeaker": "说话人",
  "exp.fTopic": "主题",
  "common.previous": "上一页",
  "common.next": "下一页",
  "ask.filterExact": "精确匹配。留空表示不筛选。",
  "con.topKHint": "每条召回的近邻数",
  "con.keepTopNHint": "保留进队列的数量",
  "con.thresholdHint": "合并相似度下限",
  "cfg.stepPreset": "配置",
  "cfg.stepPresetHint": "选一套配好的参数，之后可单独改。",
  "cfg.stepKey": "填入 LLM 凭据",
  "cfg.stepKeyLocal": "本地后端不需要 key。",
  "cfg.stepKeyHint": "只存服务端，不回传浏览器。",
  "cfg.stepDetails": "确认细节",
  "cfg.stepDetailsHint": "",
  "cfg.stepStart": "启动",
  "cfg.stepStartHint": "",
  "cfg.needKey": "先填 API Key",
  "ing.intro": "把一段对话变成可检索的记忆。",
  "ask.intro": "检索记忆，选择上下文，再向模型提问。",
  "nav.ingest": "写入记忆",
  "nav.ask": "检索与问答",
  "shell.notReady": "无实例",
  "shell.entries": "{n} 条记忆",
  "lang.toggle": "切换语言",

  "common.copy": "复制",
  "common.copied": "已复制",
  "common.follow": "跟随最新",
  "common.clear": "清空",
  "common.save": "保存",
  "common.noInstanceTitle": "尚未加载实例",
  "common.buildFirst": "请先到「设置」页启动。",
  "env.apiKey": "API Key",
  "env.apiKeyStored": "已保存 {masked} · 留空不改",
  "env.apiKeyNone": "尚未填写",
  "env.apiKeyPlaceholderStored": "•••••••• 保持不变",
  "env.baseUrl": "Base URL",
  "env.baseUrlHint": "留空 = api.openai.com",
  "env.testConnection": "测试连接",
  "env.saved": "已保存",
  "env.pingOk": "{ms} ms 连通 — 模型回复「{reply}」",
  "env.pingBad": "无法访问该模型",
  "env.keyLooksLikeUrl":
    "API Key 里填的是一个网址。key 填 key，地址填到 Base URL。",
  "env.baseUrlNoVersion": "多数网关需要版本后缀，例如 …/v1",
  "cfg.loading": "正在加载配置 schema…",
  "cfg.loadFailed": "配置 schema 加载失败",
  "cfg.start": "启动实例",
  "cfg.restart": "重启实例",
  "cfg.running": "运行中 · {collection} · {n} 条记忆 · 载入耗时 {seconds}s",
  "cfg.notRunning": "尚未启动。",
  "inst.startFailed": "无法启动初始化",
  "ing.title": "写入记忆",
  "ing.addMemory": "写入记忆",
  "ing.consolidate": "清理重复",
  "ing.consolidateHint": "合并相似记忆，清理重复条目。",
  "ing.submitFailed": "提交失败",
  "ing.turns": "对话轮次",
  "ing.turnsCount": "{m} 条消息",
  "ing.noTurns": "还没有对话轮次。",
  "ing.addTurn": "新增一轮",
  "ing.contentPlaceholder": "消息内容…",
  "ing.options": "选项",
  "ing.forceSegment": "force_segment",
  "ing.forceSegmentHint": "立即清空缓冲区。",
  "ing.forceExtract": "force_extract",
  "ing.forceExtractHint": "低于阈值也执行抽取。",
  "ing.pipeline": "流水线",
  "ing.pipelineDesc": "实时查看记忆流水线。",
  "ing.log": "add_memory 日志",
  "ing.idle": "点「写入记忆」开始。",
  "pipe.normalize": "规范化",
  "pipe.compress": "压缩",
  "pipe.segment": "主题切分",
  "pipe.extract": "抽取",
  "pipe.store": "入库",
  "pipe.sensory": "感觉记忆",
  "pipe.shortterm": "短期记忆",
  "pipe.longterm": "长期记忆",
  "pipe.skipped": "已跳过",
  "pipe.messages": "消息数",
  "pipe.targetRate": "目标 {r}",
  "pipe.topics": "{n} 个主题",
  "pipe.apiCalls": "{n} 次 API 调用",
  "pipe.oneSegment": "合为一段",
  "pipe.precompressOff": "pre_compress 未开启",
  "pipe.thresholdNotMet": "未达阈值",
  "pipe.tabEntries": "记忆条目",
  "pipe.tabSegments": "主题段",
  "pipe.tabCompression": "压缩对比",
  "pipe.noEntries": "还没有记忆条目",
  "pipe.noEntriesHint": "短期缓冲区还没填够。打开 force_extract 可以强制推进。",
  "pipe.extractionSkipped": "未触发抽取",
  "pipe.noSegments": "还没有主题段",
  "pipe.segmentationOff": "主题切分未开启",
  "pipe.segmentationOffHint":
    "所有消息被当作一段处理。开启 topic_segment 可按主题切分。",
  "pipe.compressionOff": "预压缩未开启",
  "pipe.compressionOffHint":
    "开启 pre_compress 后，可以在这里看到 LLMLingua-2 丢弃了哪些低信息量 token。",
  "pipe.noCompression": "还没有压缩数据",
  "pipe.segmentN": "第 {i} 段",
  "pipe.segmentMessages": "{n} 条消息",
  "pipe.kept": "保留 {kept}/{total}",
  "pipe.topicN": "主题 {n}",

  "ask.title": "检索与问答",
  "ask.placeholder": "问一个记忆库应该能回答的问题…",
  "ask.top": "前 {n} 条",
  "ask.retrieve": "检索",
  "ask.stored": "库中的记忆",
  "ask.loadingStored": "正在读取…",
  "ask.storeEmpty": "记忆库是空的",
  "ask.storeEmptyHint": "先到第 2 步写入一段对话。",
  "ask.retrieveFailed": "检索失败",
  "ask.retrieved": "检索结果",
  "ask.retrievedMeta": "{n} 条命中，耗时 {ms} ms · 已选 {sel} 条",
  "ask.searching": "检索中…",
  "ask.noResults": "还没有结果",
  "ask.noResultsQueried": "该查询在记忆库中没有命中。",
  "ask.context": "待注入上下文",
  "ask.contextMeta": "{n} 条记忆 · {c} 字符",
  "ask.details": "记忆详情",
  "ask.detailEmpty": "选择一条结果查看详情。",
  "ask.detailScore": "相关度",
  "ask.detailTime": "时间",
  "ask.ask": "提问",
  "ask.callFailed": "模型调用失败",
  "ask.answer": "回答",
  "ask.answerMeta": "{model} · {ms} ms · 使用 {used} 条记忆",
  "ask.answerMetaTokens":
    "{model} · {ms} ms · {tokens} tokens · 使用 {used} 条记忆",
  "ask.showPrompt": "查看 prompt",
  "ask.noMemories": "（未注入任何记忆）",
  "exp.consolidatedTag": "已整合",
  "job.runLog": "运行日志",
  "job.stage": "阶段 · {s}",
  "job.filterInfo": "关键",
  "job.filterAll": "全部",
  "job.filterProblems": "问题",
  "job.empty": "任务运行时，日志会实时输出到这里。",
  "job.pending": "排队中",
  "job.running": "运行中",
  "job.succeeded": "成功",
  "job.failed": "失败",
  "job.cancelled": "已取消",
  "job.idle": "空闲",
};

const dicts: Record<Lang, Dict> = { zh, en };

export type Translate = (
  key: string,
  vars?: Record<string, string | number>,
) => string;

function translate(
  current: Lang,
  key: string,
  vars?: Record<string, string | number>,
): string {
  const text = dicts[current][key] ?? en[key] ?? key;
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (match, name) =>
    Object.prototype.hasOwnProperty.call(vars, name)
      ? String(vars[name])
      : match,
  );
}

export function useT(): { t: Translate; lang: Lang } {
  const current = useLang();
  return {
    lang: current,
    t: (key, vars) => translate(current, key, vars),
  };
}
