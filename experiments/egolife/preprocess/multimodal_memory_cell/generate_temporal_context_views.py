from __future__ import annotations

import argparse
import json
import os
import sys
import logging
import re
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from em2mem.llm import LLMModel

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )


# =========================================================
# Prompts
# =========================================================

TEXT_SUMMARY_SYSTEM_PROMPTS = {
    "3min": """As an Event Summary Documentation Specialist, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

#### Specific Requirements

1. Structure the Events Clearly
- Merge related events into major events and arrange them in chronological order to ensure a smooth logical flow.
- Logical segmentation can be based on location, task, or theme.
- Each event should have a clear progression without fragmentation.

2. Retain Key Information
- The primary character's ("I") actions and decisions must be clearly presented.
- Important interactions with other people that affect the main event flow must be included.
- The purpose and method of key actions should be preserved when possible.

3. Use Aggregated Text Metadata as Supporting Constraints
- You will receive:
  - ordered_event_units: chronological event descriptions
  - aggregated_text_metadata: canonicalized actions, objects, topics, and speakers
- Use ordered_event_units as the main chronological backbone.
- Use aggregated_text_metadata only to avoid missing important event information.
- Do NOT mechanically copy metadata lists into the summary.
- You may also receive critical_speech lines extracted from transcript evidence.
- If critical_speech contains an explicit proposal, plan, commitment, instruction, source-transfer relation, or exact named item, preserve that information faithfully.
- Preserve exact nouns when they are important to the event meaning, instead of replacing them with broader categories.
- If a lower-level unit explicitly states where an object came from / was placed / was before, keep that source relation in the summary.
- If a lower-level unit explicitly states a proposal or intention, keep the proposer/agent and the proposed content together.

4. Concise Expression, Remove Redundancies
- Keep the facts clear, avoiding atmosphere, emotions, or abstract content.
- Remove trivial or repetitive details.
- Do not include local micro-actions unless they are important to the event flow.

5. Strictly Adhere to Facts
- Do not make assumptions or add interpretations.
- Maintain the correct chronological order.

#### Output Format
Return valid JSON only:
{
  "summary_text": "string"
}
""",
    "10min": """As an Event Summary Documentation Specialist, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

#### Specific Requirements

1. Structure the Events Clearly
- Merge related lower-level events into a coherent 10-minute event summary.
- Arrange the content chronologically.
- Group information by ongoing task, discussion, or theme when appropriate.

2. Retain Key Information
- Preserve the main actions and decisions of "I".
- Preserve important interactions with other participants.
- Preserve recurring objects, discussion topics, and speakers only when they matter to the main event flow.

3. Use Aggregated Text Metadata as Supporting Constraints
- You will receive:
  - ordered_event_units: chronological child summaries
  - aggregated_text_metadata: filtered canonicalized recurring actions, objects, topics, speakers
- Use ordered_event_units as the main chronological backbone.
- Use aggregated_text_metadata to preserve important recurring details.
- Do NOT turn metadata lists into direct prose unless clearly necessary.
- You may also receive critical_speech lines extracted from transcript evidence.
- If critical_speech contains an explicit proposal, plan, commitment, instruction, source-transfer relation, or exact named item, preserve that information faithfully.
- Preserve exact nouns when they are important to the event meaning, instead of replacing them with broader categories.
- If a lower-level unit explicitly states where an object came from / was placed / was before, keep that source relation in the summary.
- If a lower-level unit explicitly states a proposal or intention, keep the proposer/agent and the proposed content together.

4. Concise Expression, Remove Redundancies
- Focus on major developments, task flow, and conclusions.
- Remove repeated details and overly local observations.
- Summarize into event stages instead of listing many individual actions.
- Do NOT include micro gestures, local posture changes, or low-level repetitive motions.

5. Strictly Adhere to Facts
- Do not speculate.
- Keep the summary factual and chronological.

#### Output Format
Return valid JSON only:
{
  "summary_text": "string"
}
""",
    "1h": """As an Event Summary Documentation Specialist, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

#### Specific Requirements

1. Structure the Events Clearly
- Merge related lower-level events into a coherent 1-hour event summary.
- Organize the summary chronologically around major activities, transitions, and outcomes.
- Group long-running activities together when appropriate.

2. Retain Key Information
- Preserve the major actions and decisions of "I".
- Preserve important recurring interactions with other people.
- Preserve persistent objects and discussion topics only when they matter to the major activity flow.

3. Use Aggregated Text Metadata as Supporting Constraints
- You will receive:
  - ordered_event_units: chronological child summaries
  - aggregated_text_metadata: filtered canonicalized recurring actions, objects, topics, speakers
- Use ordered_event_units as the main event flow.
- Use aggregated_text_metadata only to preserve major recurring information.
- Do NOT copy metadata lists into the summary.
- You may also receive critical_speech lines extracted from transcript evidence.
- If critical_speech contains an explicit proposal, plan, commitment, instruction, source-transfer relation, or exact named item, preserve that information faithfully.
- Preserve exact nouns when they are important to the event meaning, instead of replacing them with broader categories.
- If a lower-level unit explicitly states where an object came from / was placed / was before, keep that source relation in the summary.
- If a lower-level unit explicitly states a proposal or intention, keep the proposer/agent and the proposed content together.

4. Concise Expression, Remove Redundancies
- Focus on major activities, major transitions, and outcomes.
- Omit overly local, repetitive, or low-importance details.
- Do NOT include posture, facial gestures, or micro-actions.
- Prefer high-level activity phases over detailed action chains.

5. Strictly Adhere to Facts
- Do not make assumptions.
- Keep the summary factual and chronologically correct.

#### Output Format
Return valid JSON only:
{
  "summary_text": "string"
}
""",
}

VISUAL_SUMMARY_SYSTEM_PROMPTS = {
    "3min": """You are a Visual Event Summary Documentation Specialist for egocentric videos.

Your job is to summarize the visual content of a short time period based only on visual evidence.

#### Specific Requirements

1. Structure the Visual Content Clearly
- Use ordered visual descriptions as the main chronological backbone.
- Summarize the visible scene and the most important visible objects.
- Focus on scene continuity and visible changes.

2. Retain Key Visual Information
- Preserve dominant scene context.
- Preserve recurring visible objects that are visually salient.
- Preserve visual transitions only if clearly supported.

3. Use Aggregated Visual Metadata as Supporting Constraints
- You will receive:
  - ordered_visual_units: chronological visual descriptions
  - aggregated_visual_metadata: filtered canonicalized scene statistics and visual objects
- Use ordered_visual_units as the primary backbone.
- Use aggregated_visual_metadata only to avoid missing important visible context.
- Do NOT mechanically copy metadata lists into prose.

4. Concise Expression, Remove Redundancies
- Avoid detailed narration of every small movement.
- Do not infer invisible actions, intentions, or dialogue.
- Keep the summary focused on what is visually present.

5. Strictly Adhere to Facts
- Only describe what is visually supported.
- Do not speculate.

#### Output Format
Return valid JSON only:
{
  "visual_summary": "string"
}
""",
    "10min": """You are a Visual Event Summary Documentation Specialist for egocentric videos.

Your job is to summarize the visual content of a medium time period based only on visual evidence.

#### Specific Requirements

1. Structure the Visual Content Clearly
- Use ordered visual descriptions as the main chronological backbone.
- Organize the visual summary around dominant scene context, recurring visible objects, and major visual transitions.

2. Retain Key Visual Information
- Preserve dominant scene context.
- Preserve stable and repeated visible objects.
- Preserve major visual changes only if clearly supported.

3. Use Aggregated Visual Metadata as Supporting Constraints
- You will receive:
  - ordered_visual_units: chronological child visual descriptions
  - aggregated_visual_metadata: filtered canonicalized scene statistics and visual objects
- Use ordered_visual_units as the primary visual backbone.
- Use aggregated_visual_metadata to preserve recurring visual context.
- Do NOT turn metadata lists into direct prose.

4. Concise Expression, Remove Redundancies
- Focus on stable visual context rather than local transient details.
- Avoid repeating similar visual observations.
- Do NOT mention low-information background objects unless they are central to the visual scene.

5. Strictly Adhere to Facts
- Only describe what is visually supported.
- Do not speculate.

#### Output Format
Return valid JSON only:
{
  "visual_summary": "string"
}
""",
    "1h": """You are a Visual Event Summary Documentation Specialist for egocentric videos.

Your job is to summarize the visual content of a long time period based only on visual evidence.

#### Specific Requirements

1. Structure the Visual Content Clearly
- Use ordered visual descriptions as the main backbone.
- Summarize major scene contexts, dominant visible environments, and persistent visible objects over time.

2. Retain Key Visual Information
- Preserve dominant scene context.
- Preserve persistent visible objects only when they are important to the major visual flow.
- Preserve major scene transitions only if clearly supported.

3. Use Aggregated Visual Metadata as Supporting Constraints
- You will receive:
  - ordered_visual_units: chronological child visual descriptions
  - aggregated_visual_metadata: filtered canonicalized scene statistics and visual objects
- Use ordered_visual_units as the primary backbone.
- Use aggregated_visual_metadata only to preserve major recurring visual context.
- Do NOT copy metadata lists into the summary.

4. Concise Expression, Remove Redundancies
- Focus on broad visual continuity and major changes.
- Omit local repetitive details and low-importance objects.

5. Strictly Adhere to Facts
- Only describe what is visually supported.
- Do not speculate.

#### Output Format
Return valid JSON only:
{
  "visual_summary": "string"
}
""",
}


# =========================================================
# Basic helpers
# =========================================================

STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "at", "for", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "do", "did", "does",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "this", "that", "these", "those", "it", "its", "can", "will", "would",
    "should", "could", "there", "their", "some", "any", "just", "then", "than",
    "into", "onto", "from", "up", "down", "out", "over", "under", "again"
}


model = LLMModel(model_name="gpt-5-mini")

def call_gpt(
    prompt: str,
    system_message: str = "You are an effective assistant.",
    max_tokens=2200,
    temperature=0.9,
    top_p=0.95,
) -> Optional[str]:
    try: 
        openai_key = os.getenv("OPENAI_API_KEY")
        response = model.generate(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_output_tokens=max_tokens*2
        )
        return response
    except Exception as e:
        logger.warning("[call_gpt] failed: %s", e)
        warnings.warn(f"[call_gpt] failed: {e}")
        return None


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    return json.loads(strip_code_fences(text))


def normalize_day(date_val: Any) -> Optional[int]:
    if date_val is None:
        return None
    m = re.search(r"(\d+)", str(date_val))
    return int(m.group(1)) if m else None


def normalize_day_str(date_val: Any) -> str:
    d = normalize_day(date_val)
    return f"DAY{d}" if d is not None else str(date_val)


def normalize_time_str(t: Any) -> str:
    return str(t).zfill(8)


def time_to_seconds(time_val):
    time_val = int(time_val)
    hours = time_val // 1000000
    minutes = (time_val % 1000000) // 10000
    seconds = (time_val % 10000) // 100
    return hours * 3600 + minutes * 60 + seconds


def canonicalize_text(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def ensure_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def title_case_name(x: str) -> str:
    x = str(x).strip()
    if not x:
        return ""
    return " ".join(part.capitalize() for part in re.split(r"\s+", x))


def resolve_max_workers(max_workers: Optional[int] = None) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))

    for env_name in ["GEN_EVENT_MAX_WORKERS", "MAX_WORKERS_LEVEL", "MAX_WORKERS_FILES"]:
        env_val = os.getenv(env_name)
        if env_val:
            try:
                return max(1, int(env_val))
            except Exception:
                pass
    return min(8, max(1, (os.cpu_count() or 4)))


# =========================================================
# Critical speech / protected mentions
# =========================================================

CRITICAL_SPEECH_PATTERNS = [
    r"\bwe can\b",
    r"\blet'?s\b",
    r"\bi was thinking\b",
    r"\bhow about\b",
    r"\bi can buy\b",
    r"\bi'll\b|\bi will\b",
    r"\bwe'll\b|\bwe will\b",
    r"\btake\b.*\bfrom\b",
    r"\bbring\b",
    r"\bplace\b.*\bon\b",
    r"\bput\b.*\bon\b",
    r"\bupstairs\b|\bdownstairs\b|\bup there\b|\bdown there\b",
    r"\bkeep this\b|\bkeep\b",
    r"\bthrow away\b|\bdiscard\b",
    r"\bthis is\b",
    r"\bthe purpose of\b",
    r"\bused for\b",
    r"\bidentify\b|\bconfirm\b|\bwhat is this\b",
]


GENERIC_LOW_INFO_NOUNS = {
    "thing", "things", "stuff", "item", "items", "object", "objects",
    "people", "person", "guy", "guys", "someone", "something",
    "table", "chair", "room", "area", "place", "side", "part"
}

PROTECTED_OBJECT_ALLOWLIST = {
    "phone", "hard drive", "tripod", "cable", "container", "whiteboard",
    "laptop", "tablet", "charger", "bag", "cart", "refrigerator",
    "marker", "pen"
}

PROTECTED_VERBISH_PATTERNS = [
    r"\b(going|doing|taking|bringing|putting|placing|keeping|throwing|installing|using|checking|making|having|get|got)\b",
    r"\b(can|will|would|should|could|just|take|bring|put|place|keep|throw|install)\b",
]

PROTECTED_BAD_SUBSTRINGS = {
    "take one from", "take these away", "can install", "just going", "you can throw",
    "we can", "i will", "i'll", "used for", "purpose of"
}


CRITICAL_PREDICATE_PATTERNS = {
    "proposal": [
        r"\bwe can\b", r"\blet'?s\b", r"\bhow about\b", r"\bi was thinking\b",
        r"\bsuggest\b", r"\bproposal\b", r"\bidea\b"
    ],
    "plan": [
        r"\bi will\b", r"\bi'll\b", r"\bgoing to\b", r"\bplan to\b",
        r"\bi can buy\b", r"\bwe will\b", r"\bwe'll\b"
    ],
    "source": [
        r"\btake\b", r"\bbring\b", r"\bcarry\b", r"\bplace\b", r"\bput\b",
        r"\bfrom\b", r"\bupstairs\b", r"\bdownstairs\b", r"\bup there\b", r"\bdown there\b"
    ],
    "discard": [
        r"\bkeep\b", r"\bthrow away\b", r"\bdiscard\b"
    ],
    "attribute": [
        r"\bthis is\b", r"\bit is\b", r"\bused for\b", r"\bpurpose\b"
    ],
    "confirm": [
        r"\bwhat is\b", r"\bidentify\b", r"\bconfirm\b"
    ],
}


def split_transcript_lines(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?<=\")\s+|(?<=\')\s+", str(text).strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = re.split(r"(?<=:)\s*|(?<=\.)\s*", str(text).strip())
        parts = [p.strip() for p in parts if p.strip()]
    return parts


def extract_critical_speech_lines(transcript_text: str, max_lines: int = 6) -> List[str]:
    lines = split_transcript_lines(transcript_text)
    kept = []
    for line in lines:
        s = canonicalize_text(line)
        if any(re.search(p, s) for p in CRITICAL_SPEECH_PATTERNS):
            kept.append(line.strip())

    seen = set()
    deduped = []
    for line in kept:
        norm = canonicalize_text(line)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(line)
    return deduped[:max_lines]


def simple_candidate_noun_phrases(text: str) -> List[str]:
    text = canonicalize_text(text)
    phrases = []
    for pat in [
        r"\b[a-z][a-z/-]{2,}(?:\s+[a-z][a-z/-]{2,}){0,2}\b",
    ]:
        for m in re.findall(pat, text):
            phrases.append(m.strip())

    seen = set()
    out = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def is_specific_mention(x: str) -> bool:
    x = canonicalize_text(x)
    if not x:
        return False

    toks = x.split()
    if len(toks) == 0 or len(toks) > 4:
        return False

    if x in PROTECTED_BAD_SUBSTRINGS:
        return False

    if all(tok in GENERIC_LOW_INFO_NOUNS or tok in STOPWORDS for tok in toks):
        return False

    informative = [t for t in toks if len(t) >= 3 and t not in GENERIC_LOW_INFO_NOUNS and t not in STOPWORDS]
    return len(informative) > 0


def normalize_protected_object_candidate(cand: str) -> str:
    cand = canonicalize_text(cand)
    if not cand or not is_specific_mention(cand):
        return ""

    if any(re.search(p, cand) for p in PROTECTED_VERBISH_PATTERNS):
        return ""

    if any(substr in cand for substr in PROTECTED_BAD_SUBSTRINGS):
        return ""

    obj = canonicalize_object(cand)
    if not obj or obj in LOW_INFO_TEXT_OBJECTS_HIGH or obj in {"paper", "people"}:
        return ""

    if obj not in PROTECTED_OBJECT_ALLOWLIST:
        return ""

    return obj


def extract_protected_mentions(transcript_text: str, max_mentions: int = 6) -> List[str]:
    text = canonicalize_text(transcript_text)
    if not text:
        return []

    protected = []
    chunks = re.split(r"(?<=[.!?])\s+|(?<=,)\s+", text)
    chunks = [c.strip() for c in chunks if c.strip()]

    for chunk in chunks:
        matched = False
        for pats in CRITICAL_PREDICATE_PATTERNS.values():
            if any(re.search(p, chunk) for p in pats):
                matched = True
                break
        if not matched:
            continue

        for cand in simple_candidate_noun_phrases(chunk):
            obj = normalize_protected_object_candidate(cand)
            if obj:
                protected.append(obj)

    seen = set()
    out = []
    for x in protected:
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out[:max_mentions]


def aggregate_critical_speech(items: List[Dict[str, Any]], max_lines: int = 10) -> List[str]:
    lines = []
    seen = set()
    for item in items:
        for line in ensure_list(item.get("critical_speech_lines", [])):
            norm = canonicalize_text(line)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            lines.append(str(line).strip())
            if len(lines) >= max_lines:
                return lines
    return lines


def aggregate_protected_mentions(items: List[Dict[str, Any]], max_mentions: int = 10) -> List[str]:
    mentions = []
    seen = set()
    for item in items:
        for mention in ensure_list(item.get("protected_mentions", [])):
            norm = canonicalize_text(mention)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            mentions.append(str(mention).strip())
            if len(mentions) >= max_mentions:
                return mentions
    return mentions


# =========================================================
# Canonicalization rules
# =========================================================

SCENE_CANONICAL_MAP: List[Tuple[str, str]] = [
    (r"\bindoor dining area\b", "dining area"),
    (r"\bdining room\b", "dining area"),
    (r"\bdining area\b", "dining area"),
    (r"\bindoor workspace\b", "office/workspace"),
    (r"\bworkspace\b", "office/workspace"),
    (r"\boffice/workspace\b", "office/workspace"),
    (r"\boffice\b", "office/workspace"),
    (r"\bkitchen\b", "kitchen"),
    (r"\bliving room\b", "living room"),
    (r"\bbedroom\b", "bedroom"),
    (r"\bbathroom\b", "bathroom"),
    (r"\bhallway\b", "hallway"),
    (r"\boutdoor|outside|street|parking\b", "outdoors"),
    (r"\bstore|shop|supermarket|market\b", "store/shop"),
    (r"\brestaurant|cafe|cafeteria\b", "restaurant/cafe"),
    (r"\bcar|vehicle interior\b", "car/interior"),
    (r"\bindoor\b", "other"),
]


def canonicalize_scene(scene: str) -> str:
    s = canonicalize_text(scene)
    if not s:
        return ""
    for pattern, target in SCENE_CANONICAL_MAP:
        if re.search(pattern, s):
            return target
    return s


OBJECT_CANONICAL_MAP: List[Tuple[str, str]] = [
    (r"\bcell ?phone|smartphone|mobile phone\b", "phone"),
    (r"\bphone\b", "phone"),
    (r"\bhard drives?\b|\bexternal drives?\b|\bdrive\b", "hard drive"),
    (r"\btripods?\b", "tripod"),
    (r"\bcables?\b|\bdata cable\b|\busb cable\b|\bcharging cable\b", "cable"),
    (r"\bpapers?\b|\bdocuments?\b|\bnotes?\b", "paper"),
    (r"\bcontainers?\b|\bboxes?\b|\bbin\b", "container"),
    (r"\bwhiteboard\b|\bboard\b", "whiteboard"),
    (r"\blaptop\b|\bcomputer\b", "laptop"),
    (r"\btablet\b|\bipad\b", "tablet"),
    (r"\bcharger\b", "charger"),
    (r"\bbag\b|\bbackpack\b", "bag"),
    (r"\bcart\b|\bshopping cart\b", "cart"),
    (r"\bfridge\b|\brefrigerator\b", "refrigerator"),
    (r"\bmarker\b", "marker"),
    (r"\bpen\b", "pen"),
    (r"\btable\b|\bdesk\b", "table"),
    (r"\bchair\b|\bchairs\b", "chair"),
    (r"\bstool\b|\bstools\b", "stool"),
    (r"\bpeople\b|\beveryone\b|\bgroup\b", "people"),
]


LOW_INFO_VISUAL_OBJECTS = {
    "people",
    "table",
    "chair",
    "stool",
    "paper",
}

LOW_INFO_TEXT_OBJECTS_HIGH = {
    "people",
    "table",
    "chair",
    "stool",
}


def singularize_basic(x: str) -> str:
    if x.endswith("ies") and len(x) > 4:
        return x[:-3] + "y"
    if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
        return x[:-1]
    return x


def canonicalize_object(obj: str) -> str:
    s = canonicalize_text(obj)
    if not s:
        return ""
    for pattern, target in OBJECT_CANONICAL_MAP:
        if re.search(pattern, s):
            return target
    s = singularize_basic(s)
    s = re.sub(r"\bsmall\b|\blarge\b|\bblack\b|\bwhite\b|\bred\b|\bblue\b|\bgreen\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


ACTION_CANONICAL_MAP: List[Tuple[str, str]] = [
    # explicit speech-act / QA-critical predicates
    (r"\bsuggest\b|\bproposal\b|\bpropose\b|\bi was thinking\b|\bwe can\b|\blet'?s\b|\bhow about\b", "proposal/suggestion"),
    (r"\bi can buy\b|\bi'll buy\b|\bi will buy\b|\bi'll go\b|\bi will go\b|\bplan to\b|\bgoing to\b", "plan/commitment"),
    (r"\bexplain\b|\bintroduce\b|\bdescribe\b|\bshow\b", "explain/introduce"),
    (r"\bask\b.*\bwhat\b|\bidentify\b|\bconfirm\b|\bwhat is this\b", "confirm/identify"),
    (r"\bkeep\b|\bthrow away\b|\bdiscard\b|\bclean up\b", "keep/discard"),
    (r"\btake\b.*\bfrom\b|\bbring\b|\bcarry\b.*\bto\b|\bplace\b.*\bon\b|\bput\b.*\bon\b|\bmove\b.*\bupstairs\b|\bmove\b.*\bdownstairs\b", "source transfer"),
    (r"\bwrite\b|\bmark\b|\blabel\b|\btimestamp\b", "write/mark"),

    # broader buckets
    (r"\bscroll\b|\bturn off phone\b|\bput away phone\b|\bhold phone\b|\bput phone\b|\buse phone\b|\bcheck phone\b", "phone handling"),
    (r"\bhand\b.*\bphone\b|\bpass\b.*\bphone\b|\bgive\b.*\bphone\b", "phone handoff"),
    (r"\bhold meeting\b|\bcontinue meeting\b|\bmeeting\b|\bdiscuss\b", "meeting/discussion"),
    (r"\bwalk\b|\bmove\b|\bgo to\b|\bapproach\b", "move between places"),
    (r"\bpoint\b|\blook at\b|\bcheck\b|\binspect\b|\bexamine\b", "inspect item"),
    (r"\borganize\b|\bsort\b|\barrange\b|\bclean\b|\bthrow away\b|\bpack\b", "organize items"),
    (r"\bcarry\b|\bpick up\b|\bgrab\b|\bhold\b|\bplace\b|\bput\b", "handle item"),
    (r"\beat\b|\bdrink\b|\bserve\b", "eat/drink"),
    (r"\bshop\b|\bpush cart\b|\bcheckout\b|\bbuy\b", "shopping"),
    (r"\bcook\b|\bprepare food\b|\bheat\b|\bmicrowave\b", "food preparation"),
    (r"\bnod\b|\blaugh\b|\bsmile\b|\btilt head\b|\bturn head\b|\blean back\b|\blean forward\b|\bsway body\b|\bcover face\b|\blower head\b|\bbow head\b|\bshake head\b", "micro gesture"),
]


def canonicalize_action(action: str) -> str:
    s = canonicalize_text(action)
    if not s:
        return ""
    for pattern, target in ACTION_CANONICAL_MAP:
        if re.search(pattern, s):
            return target
    s = re.sub(r"\bslightly\b|\bcarefully\b|\bawkwardly\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


TOPIC_CANONICAL_MAP: List[Tuple[str, str]] = [
    (r"\blast day\b|\bplan\b|\bplanning\b", "last day planning"),
    (r"\bsuggest\b|\bproposal\b|\bidea\b|\boutdoor activity\b", "activity suggestion"),
    (r"\bbuy\b|\bpurchase\b|\bacquire\b", "purchase or acquisition plan"),
    (r"\bfrom\b|\bupstairs\b|\bdownstairs\b|\bwhere it was before\b", "source location trace"),
    (r"\bkeep\b|\bthrow away\b|\bdiscard\b", "discard or keep decision"),
    (r"\bhard drive\b|\bdrive\b", "hard drive identification"),
    (r"\bcable\b|\bdata cable\b", "cable organization"),
    (r"\bmeeting\b|\bdiscussion\b", "meeting discussion"),
    (r"\bphone\b|\bmarking\b|\btimestamp\b", "phone coordination"),
    (r"\bcleanup\b|\bthrow away\b|\borganize\b", "cleanup and organization"),
]


def canonicalize_topic(topic: str) -> str:
    s = canonicalize_text(topic)
    if not s:
        return ""
    for pattern, target in TOPIC_CANONICAL_MAP:
        if re.search(pattern, s):
            return target
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize_speaker_name(name: str) -> str:
    s = canonicalize_text(name)
    if not s:
        return ""
    return title_case_name(s)


# =========================================================
# Level-aware filtering / promotion
# =========================================================

FIELD_LABEL_KEYS = {
    "action_threads": "action",
    "object_threads": "object",
    "topic_threads": "topic",
    "visual_object_threads": "object",
}

FIELD_STOPLISTS = {
    "action_threads": {"micro gesture"},
    "object_threads": set(),
    "topic_threads": set(),
    "visual_object_threads": LOW_INFO_VISUAL_OBJECTS,
}

FIELD_PRIORITY = {
    "action_threads": {
        "proposal/suggestion",
        "plan/commitment",
        "source transfer",
        "write/mark",
        "keep/discard",
        "confirm/identify",
        "explain/introduce",
        "phone handling",
        "phone handoff",
        "meeting/discussion",
        "move between places",
        "inspect item",
        "organize items",
        "handle item",
        "shopping",
        "food preparation",
        "eat/drink",
    },
    "object_threads": {
        "phone",
        "hard drive",
        "tripod",
        "cable",
        "whiteboard",
        "laptop",
        "charger",
        "bag",
        "cart",
        "refrigerator",
        "container",
        "marker",
        "pen",
    },
    "topic_threads": {
        "activity suggestion",
        "purchase or acquisition plan",
        "source location trace",
        "discard or keep decision",
        "phone coordination",
        "last day planning",
        "hard drive identification",
        "meeting discussion",
        "cable organization",
        "cleanup and organization",
    },
    "visual_object_threads": {
        "phone",
        "hard drive",
        "tripod",
        "cable",
        "whiteboard",
        "laptop",
        "charger",
        "bag",
        "container",
        "cart",
    },
}

LEVEL_FIELD_CONFIG = {
    "3min": {
        "action_threads": {"min_support": 1, "min_coverage": 0.0, "top_k": 10},
        "object_threads": {"min_support": 1, "min_coverage": 0.0, "top_k": 12},
        "topic_threads": {"min_support": 1, "min_coverage": 0.0, "top_k": 10},
        "visual_object_threads": {"min_support": 1, "min_coverage": 0.0, "top_k": 8},
    },
    "10min": {
        "action_threads": {"min_support": 2, "min_coverage": 0.34, "top_k": 8},
        "object_threads": {"min_support": 2, "min_coverage": 0.34, "top_k": 10},
        "topic_threads": {"min_support": 1, "min_coverage": 0.25, "top_k": 8},
        "visual_object_threads": {"min_support": 2, "min_coverage": 0.34, "top_k": 6},
    },
    "1h": {
        "action_threads": {"min_support": 3, "min_coverage": 0.25, "top_k": 6},
        "object_threads": {"min_support": 3, "min_coverage": 0.25, "top_k": 8},
        "topic_threads": {"min_support": 2, "min_coverage": 0.20, "top_k": 6},
        "visual_object_threads": {"min_support": 3, "min_coverage": 0.25, "top_k": 5},
    },
}


def filter_threads_by_level(
    threads: List[Dict[str, Any]],
    field_name: str,
    level_name: str,
    num_children: int,
) -> List[Dict[str, Any]]:
    label_key = FIELD_LABEL_KEYS[field_name]
    stoplist = FIELD_STOPLISTS[field_name]
    priority = FIELD_PRIORITY[field_name]
    cfg = LEVEL_FIELD_CONFIG[level_name][field_name]

    results = []
    for th in threads:
        label = str(th.get(label_key, "")).strip()
        if not label:
            continue

        support = int(th.get("support_units", 0))
        child_ids = ensure_list(th.get("child_ids", []))
        coverage = len(child_ids) / max(1, num_children)

        if label in stoplist:
            continue
        if field_name == "object_threads" and level_name in {"10min", "1h"} and label in LOW_INFO_TEXT_OBJECTS_HIGH:
            continue

        keep = False
        if label in priority:
            keep = (support >= 1) and (coverage >= min(0.2, cfg["min_coverage"]))
        else:
            keep = (support >= cfg["min_support"]) or (coverage >= cfg["min_coverage"])

        if keep:
            th = dict(th)
            th["child_coverage"] = round(coverage, 3)
            results.append(th)

    results.sort(
        key=lambda x: (
            -x.get("support_units", 0),
            -x.get("child_coverage", 0),
            -min(1, int(x.get("protected", False))),
            x.get(label_key, "")
        )
    )
    return results[: cfg["top_k"]]


def filter_scene_summary_by_level(scene_summary: Dict[str, Any], level_name: str) -> Dict[str, Any]:
    if not scene_summary:
        return {"dominant_scene": "", "scene_distribution": {}}

    dist = scene_summary.get("scene_distribution", {})
    if not dist:
        return {"dominant_scene": scene_summary.get("dominant_scene", ""), "scene_distribution": {}}

    if level_name == "3min":
        min_prob = 0.08
        top_k = 4
    elif level_name == "10min":
        min_prob = 0.10
        top_k = 4
    else:
        min_prob = 0.12
        top_k = 3

    filtered = [(k, v) for k, v in dist.items() if k != "other" and v >= min_prob]
    if not filtered:
        filtered = [(scene_summary.get("dominant_scene", ""), 1.0)] if scene_summary.get("dominant_scene") else []

    filtered = filtered[:top_k]
    total = sum(v for _, v in filtered) or 1.0
    new_dist = {k: round(v / total, 3) for k, v in filtered}
    dominant = filtered[0][0] if filtered else ""

    return {
        "dominant_scene": dominant,
        "scene_distribution": new_dist,
    }


# =========================================================
# Input normalization
# =========================================================

def normalize_doc(raw: Dict[str, Any]) -> Dict[str, Any]:
    metadata = raw.get("Metadata", {}) if isinstance(raw.get("Metadata"), dict) else {}

    date = raw.get("date", metadata.get("date"))
    start_time = raw.get("start_time", metadata.get("start_time"))
    end_time = raw.get("end_time", metadata.get("end_time"))

    date = normalize_day_str(date)
    start_time = normalize_time_str(start_time)
    end_time = normalize_time_str(end_time)

    doc_id = raw.get("doc_id", f"{date}_{start_time}_{end_time}")

    fine_caption = raw.get("fine_caption", "")
    if not fine_caption:
        fine_caption = raw.get("text", "")
    if not fine_caption:
        fine_caption = raw.get("Content", "")

    summary_text = raw.get("text", fine_caption)

    visual_backbone_text = raw.get("visual_summary", "")
    if not visual_backbone_text:
        visual_backbone_text = raw.get("keyframe_caption", "")

    transcript_text = (
        raw.get("transcript_text")
        or raw.get("transcript")
        or metadata.get("transcript_text")
        or ""
    )
    critical_speech_lines = extract_critical_speech_lines(transcript_text)
    protected_mentions = extract_protected_mentions(transcript_text)

    return {
        "doc_id": doc_id,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,

        "fine_caption": fine_caption,
        "text": summary_text,
        "transcript_text": transcript_text,
        "critical_speech_lines": critical_speech_lines,
        "protected_mentions": protected_mentions,

        "main_actions": ensure_list(raw.get("main_actions", [])),
        "salient_objects": ensure_list(raw.get("salient_objects", [])),
        "conversation_focus": ensure_list(raw.get("conversation_focus", [])),
        "speakers": ensure_list(raw.get("speakers", [])),

        "scene": raw.get("scene", ""),
        "visual_objects": ensure_list(raw.get("visual_objects", [])),
        "keyframe_caption": raw.get("keyframe_caption", ""),

        "visual_summary": raw.get("visual_summary", ""),
        "visual_backbone_text": visual_backbone_text,

        "action_threads": ensure_list(raw.get("action_threads", [])),
        "object_threads": ensure_list(raw.get("object_threads", [])),
        "topic_threads": ensure_list(raw.get("topic_threads", [])),
        "speaker_stats": ensure_list(raw.get("speaker_stats", [])),

        "scene_summary": raw.get("scene_summary", {}),
        "visual_object_threads": ensure_list(raw.get("visual_object_threads", [])),

        "child_ids": ensure_list(raw.get("child_ids", [])),
        "source_doc_ids": ensure_list(raw.get("source_doc_ids", [doc_id])),
    }


def sort_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        docs,
        key=lambda x: (normalize_day(x["date"]) or 0, time_to_seconds(x["start_time"]))
    )


# =========================================================
# Canonicalized aggregation
# =========================================================

def aggregate_canonical_threads(
    items: List[Dict[str, Any]],
    raw_field: str,
    thread_field: str,
    label_key: str,
    out_key: str,
    canonicalizer,
    top_k: int = 16,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}

    for item in items:
        item_id = item["doc_id"]
        start_time = item["start_time"]
        end_time = item["end_time"]

        if item.get(thread_field):
            for th in item[thread_field]:
                if not isinstance(th, dict):
                    continue
                raw_label = th.get("canonical_label") or th.get(label_key) or th.get("value") or th.get("label")
                if not raw_label:
                    continue

                canonical_label = canonicalizer(raw_label)
                if not canonical_label:
                    continue

                if canonical_label not in buckets:
                    buckets[canonical_label] = {
                        out_key: canonical_label,
                        "canonical_label": canonical_label,
                        "aliases": set(),
                        "support_units": 0,
                        "first_seen": start_time,
                        "last_seen": end_time,
                        "child_ids": set(),
                    }

                alias_source = ensure_list(th.get("aliases", []))
                if not alias_source:
                    alias_source = [raw_label]
                for alias in alias_source:
                    alias = str(alias).strip()
                    if alias:
                        buckets[canonical_label]["aliases"].add(alias)

                buckets[canonical_label]["support_units"] += int(th.get("support_units", 1))
                buckets[canonical_label]["first_seen"] = min(buckets[canonical_label]["first_seen"], start_time)
                buckets[canonical_label]["last_seen"] = max(buckets[canonical_label]["last_seen"], end_time)
                buckets[canonical_label]["child_ids"].add(item_id)
            continue

        seen_this_item = set()
        for raw_val in ensure_list(item.get(raw_field, [])):
            raw_val = str(raw_val).strip()
            if not raw_val:
                continue

            canonical_label = canonicalizer(raw_val)
            if not canonical_label:
                continue
            if canonical_label in seen_this_item:
                continue
            seen_this_item.add(canonical_label)

            if canonical_label not in buckets:
                buckets[canonical_label] = {
                    out_key: canonical_label,
                    "canonical_label": canonical_label,
                    "aliases": set(),
                    "support_units": 0,
                    "first_seen": start_time,
                    "last_seen": end_time,
                    "child_ids": set(),
                }

            buckets[canonical_label]["aliases"].add(raw_val)
            buckets[canonical_label]["support_units"] += 1
            buckets[canonical_label]["first_seen"] = min(buckets[canonical_label]["first_seen"], start_time)
            buckets[canonical_label]["last_seen"] = max(buckets[canonical_label]["last_seen"], end_time)
            buckets[canonical_label]["child_ids"].add(item_id)

    results = []
    for _, v in buckets.items():
        results.append(
            {
                out_key: v[out_key],
                "canonical_label": v["canonical_label"],
                "aliases": sorted(v["aliases"]),
                "support_units": v["support_units"],
                "first_seen": v["first_seen"],
                "last_seen": v["last_seen"],
                "child_ids": sorted(v["child_ids"]),
            }
        )

    results.sort(key=lambda x: (-x["support_units"], x["first_seen"], x[out_key]))
    return results[:top_k]


def aggregate_speakers(items: List[Dict[str, Any]], top_k: int = 12) -> List[Dict[str, Any]]:
    counter = Counter()

    for item in items:
        if item.get("speaker_stats"):
            for s in item["speaker_stats"]:
                if not isinstance(s, dict):
                    continue
                speaker = canonicalize_speaker_name(s.get("speaker", ""))
                if speaker:
                    counter[speaker] += int(s.get("support_units", s.get("count_units", 1)))
        else:
            for speaker in set(ensure_list(item.get("speakers", []))):
                speaker = canonicalize_speaker_name(speaker)
                if speaker:
                    counter[speaker] += 1

    return [{"speaker": k, "support_units": v} for k, v in counter.most_common(top_k)]


def aggregate_scene(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    counter = Counter()

    for item in items:
        if item.get("scene_summary") and item["scene_summary"].get("scene_distribution"):
            for scene, score in item["scene_summary"]["scene_distribution"].items():
                canonical_scene = canonicalize_scene(scene)
                if not canonical_scene:
                    continue
                try:
                    counter[canonical_scene] += float(score)
                except Exception:
                    pass
        else:
            scene = canonicalize_scene(str(item.get("scene", "")))
            if scene:
                counter[scene] += 1.0

    if not counter:
        return {"dominant_scene": "", "scene_distribution": {}}

    total = sum(counter.values())
    distribution = {k: round(v / total, 3) for k, v in counter.most_common()}
    dominant_scene = counter.most_common(1)[0][0]
    return {
        "dominant_scene": dominant_scene,
        "scene_distribution": distribution,
    }


def merge_protected_mentions(
    threads: List[Dict[str, Any]],
    protected_mentions: List[str],
    out_key: str,
    max_additional: int = 3,
) -> List[Dict[str, Any]]:
    if out_key != "object":
        return threads

    existing = {canonicalize_text(x.get(out_key, "")) for x in threads}
    added = 0

    for m in protected_mentions:
        m_norm = canonicalize_text(m)
        if not m_norm or m_norm in existing:
            continue
        if m_norm not in PROTECTED_OBJECT_ALLOWLIST:
            continue

        threads.append({
            out_key: m_norm,
            "canonical_label": m_norm,
            "aliases": [m_norm],
            "support_units": 1,
            "first_seen": "99999999",
            "last_seen": "99999999",
            "child_ids": [],
            "protected": True,
            "child_coverage": 0.0,
        })
        existing.add(m_norm)
        added += 1
        if added >= max_additional:
            break

    return threads


def aggregate_text_metadata(items: List[Dict[str, Any]], level_name: str) -> Dict[str, Any]:
    num_children = len(items)

    action_threads = aggregate_canonical_threads(
        items,
        raw_field="main_actions",
        thread_field="action_threads",
        label_key="action",
        out_key="action",
        canonicalizer=canonicalize_action,
    )
    object_threads = aggregate_canonical_threads(
        items,
        raw_field="salient_objects",
        thread_field="object_threads",
        label_key="object",
        out_key="object",
        canonicalizer=canonicalize_object,
    )
    topic_threads = aggregate_canonical_threads(
        items,
        raw_field="conversation_focus",
        thread_field="topic_threads",
        label_key="topic",
        out_key="topic",
        canonicalizer=canonicalize_topic,
    )
    speaker_stats = aggregate_speakers(items)

    protected_mentions = aggregate_protected_mentions(items, max_mentions=10)
    object_threads = merge_protected_mentions(object_threads, protected_mentions, "object", max_additional=3)

    action_threads = filter_threads_by_level(action_threads, "action_threads", level_name, num_children)
    object_threads = filter_threads_by_level(object_threads, "object_threads", level_name, num_children)
    topic_threads = filter_threads_by_level(topic_threads, "topic_threads", level_name, num_children)

    if level_name == "3min":
        speaker_stats = speaker_stats[:8]
    elif level_name == "10min":
        speaker_stats = [s for s in speaker_stats if s["support_units"] >= 1][:6]
    else:
        speaker_stats = [s for s in speaker_stats if s["support_units"] >= 2][:5]

    return {
        "action_threads": action_threads,
        "object_threads": object_threads,
        "topic_threads": topic_threads,
        "speaker_stats": speaker_stats,
        "protected_mentions": protected_mentions[:6],
    }


def aggregate_visual_metadata(items: List[Dict[str, Any]], level_name: str) -> Dict[str, Any]:
    num_children = len(items)

    scene_summary = aggregate_scene(items)
    scene_summary = filter_scene_summary_by_level(scene_summary, level_name)

    visual_object_threads = aggregate_canonical_threads(
        items,
        raw_field="visual_objects",
        thread_field="visual_object_threads",
        label_key="object",
        out_key="object",
        canonicalizer=canonicalize_object,
    )
    visual_object_threads = filter_threads_by_level(
        visual_object_threads, "visual_object_threads", level_name, num_children
    )

    return {
        "scene_summary": scene_summary,
        "visual_object_threads": visual_object_threads,
    }


# =========================================================
# Prompt payloads
# =========================================================

def build_text_prompt_payload(items: List[Dict[str, Any]], aggregated_text: Dict[str, Any]) -> Dict[str, Any]:
    ordered_event_units = []
    for item in items:
        event_text = (item.get("fine_caption") or item.get("text") or "").strip()
        critical_speech = ensure_list(item.get("critical_speech_lines", []))
        if event_text or critical_speech:
            ordered_event_units.append(
                {
                    "time": f"{item['date']}_{item['start_time']}_{item['end_time']}",
                    "event": event_text,
                    "critical_speech": critical_speech,
                }
            )

    return {
        "ordered_event_units": ordered_event_units,
        "aggregated_text_metadata": {
            "action_threads": aggregated_text["action_threads"],
            "object_threads": aggregated_text["object_threads"],
            "topic_threads": aggregated_text["topic_threads"],
            "speaker_stats": aggregated_text["speaker_stats"],
            "protected_mentions": aggregated_text.get("protected_mentions", []),
        },
    }


def build_visual_prompt_payload(items: List[Dict[str, Any]], aggregated_visual: Dict[str, Any]) -> Dict[str, Any]:
    ordered_visual_units = []
    for item in items:
        visual_text = (item.get("visual_backbone_text") or "").strip()
        if visual_text:
            ordered_visual_units.append(
                {
                    "time": f"{item['date']}_{item['start_time']}_{item['end_time']}",
                    "visual_evidence": visual_text,
                }
            )

    return {
        "ordered_visual_units": ordered_visual_units,
        "aggregated_visual_metadata": {
            "scene_summary": aggregated_visual["scene_summary"],
            "visual_object_threads": aggregated_visual["visual_object_threads"],
        },
    }


# =========================================================
# Summarization
# =========================================================

def fallback_text_summary(items: List[Dict[str, Any]]) -> Dict[str, str]:
    texts = []
    for item in items:
        t = (item.get("fine_caption") or item.get("text") or "").strip()
        if t:
            texts.append(t)
        for line in ensure_list(item.get("critical_speech_lines", []))[:2]:
            if line:
                texts.append(str(line).strip())
    return {"summary_text": " ".join(texts[:8]).strip()}


def fallback_visual_summary(items: List[Dict[str, Any]], aggregated_visual: Dict[str, Any]) -> Dict[str, str]:
    visual_parts = []
    for item in items:
        v = (item.get("visual_backbone_text") or "").strip()
        if v:
            visual_parts.append(v)

    if visual_parts:
        return {"visual_summary": " ".join(visual_parts[:4]).strip()}

    dominant_scene = aggregated_visual.get("scene_summary", {}).get("dominant_scene", "")
    top_objects = [
        x["object"] for x in aggregated_visual.get("visual_object_threads", [])[:4]
        if x.get("object")
    ]

    if dominant_scene and top_objects:
        return {
            "visual_summary": f"The visual context is mainly in {dominant_scene}, with recurring visible objects such as {', '.join(top_objects)}."
        }
    if dominant_scene:
        return {"visual_summary": f"The visual context is mainly in {dominant_scene}."}
    return {"visual_summary": ""}


def summarize_text_batch(
    items: List[Dict[str, Any]],
    aggregated_text: Dict[str, Any],
    level_name: str,
) -> Dict[str, str]:
    payload = build_text_prompt_payload(items, aggregated_text)
    resp = call_gpt(
        prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        system_message=TEXT_SUMMARY_SYSTEM_PROMPTS[level_name],
        max_tokens=2048,
        temperature=0.3,
        top_p=0.95,
    )
    if not resp:
        return fallback_text_summary(items)

    try:
        data = safe_json_loads(resp)
        summary_text = str(data.get("summary_text", "")).strip()
        if not summary_text:
            return fallback_text_summary(items)
        return {"summary_text": summary_text}
    except Exception:
        return fallback_text_summary(items)


def summarize_visual_batch(
    items: List[Dict[str, Any]],
    aggregated_visual: Dict[str, Any],
    level_name: str,
) -> Dict[str, str]:
    payload = build_visual_prompt_payload(items, aggregated_visual)
    resp = call_gpt(
        prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        system_message=VISUAL_SUMMARY_SYSTEM_PROMPTS[level_name],
        max_tokens=1024,
        temperature=0.2,
        top_p=0.95,
    )
    if not resp:
        return fallback_visual_summary(items, aggregated_visual)

    try:
        data = safe_json_loads(resp)
        visual_summary = str(data.get("visual_summary", "")).strip()
        if not visual_summary:
            return fallback_visual_summary(items, aggregated_visual)
        return {"visual_summary": visual_summary}
    except Exception:
        return fallback_visual_summary(items, aggregated_visual)


# =========================================================
# Aggregate node
# =========================================================

def build_aggregate_node(items: List[Dict[str, Any]], level_name: str) -> Dict[str, Any]:
    items = sort_docs(items)

    aggregated_text = aggregate_text_metadata(items, level_name)
    aggregated_visual = aggregate_visual_metadata(items, level_name)

    text_summary = summarize_text_batch(items, aggregated_text, level_name)
    visual_summary = summarize_visual_batch(items, aggregated_visual, level_name)

    source_doc_ids = sorted(
        {
            sid
            for item in items
            for sid in ensure_list(item.get("source_doc_ids", [item["doc_id"]]))
        }
    )

    node = {
        "doc_id": f"{items[0]['date']}_{items[0]['start_time']}_{items[-1]['end_time']}_{level_name}",
        "level": level_name,
        "date": items[0]["date"],
        "start_time": items[0]["start_time"],
        "end_time": items[-1]["end_time"],

        "text": text_summary["summary_text"],
        "visual_summary": visual_summary["visual_summary"],

        "action_threads": aggregated_text["action_threads"],
        "object_threads": aggregated_text["object_threads"],
        "topic_threads": aggregated_text["topic_threads"],
        "speaker_stats": aggregated_text["speaker_stats"],

        "critical_speech_lines": aggregate_critical_speech(items, max_lines=10 if level_name == "3min" else 8),
        "protected_mentions": aggregate_protected_mentions(items, max_mentions=10),

        "scene_summary": aggregated_visual["scene_summary"],
        "visual_object_threads": aggregated_visual["visual_object_threads"],

        "child_ids": [item["doc_id"] for item in items],
        "source_doc_ids": source_doc_ids,
    }
    return node


def build_aggregate_node_safe(items: List[Dict[str, Any]], level_name: str) -> Dict[str, Any]:
    try:
        return build_aggregate_node(items, level_name)
    except Exception as e:
        logger.exception("Failed to build aggregate node for level=%s batch=%s-%s: %s", level_name, items[0].get("start_time"), items[-1].get("end_time"), e)
        items = sort_docs(items)
        aggregated_text = aggregate_text_metadata(items, level_name)
        aggregated_visual = aggregate_visual_metadata(items, level_name)
        text_summary = fallback_text_summary(items)
        visual_summary = fallback_visual_summary(items, aggregated_visual)
        source_doc_ids = sorted(
            {
                sid
                for item in items
                for sid in ensure_list(item.get("source_doc_ids", [item["doc_id"]]))
            }
        )
        return {
            "doc_id": f"{items[0]['date']}_{items[0]['start_time']}_{items[-1]['end_time']}_{level_name}",
            "level": level_name,
            "date": items[0]["date"],
            "start_time": items[0]["start_time"],
            "end_time": items[-1]["end_time"],
            "text": text_summary["summary_text"],
            "visual_summary": visual_summary["visual_summary"],
            "action_threads": aggregated_text["action_threads"],
            "object_threads": aggregated_text["object_threads"],
            "topic_threads": aggregated_text["topic_threads"],
            "speaker_stats": aggregated_text["speaker_stats"],
            "critical_speech_lines": aggregate_critical_speech(items, max_lines=10 if level_name == "3min" else 8),
            "protected_mentions": aggregate_protected_mentions(items, max_mentions=10),
            "scene_summary": aggregated_visual["scene_summary"],
            "visual_object_threads": aggregated_visual["visual_object_threads"],
            "child_ids": [item["doc_id"] for item in items],
            "source_doc_ids": source_doc_ids,
        }


# =========================================================
# Windowing / generation
# =========================================================

def bucket_by_window(items: List[Dict[str, Any]], window_seconds: int):
    buckets = defaultdict(list)

    for item in items:
        day = normalize_day(item["date"])
        if day is None:
            continue
        start_sec = time_to_seconds(item["start_time"])
        window_start = (start_sec // window_seconds) * window_seconds
        buckets[(day, window_start)].append(item)

    ordered_keys = sorted(buckets.keys(), key=lambda x: (x[0], x[1]))
    return ordered_keys, buckets


def generate_level(
    input_items: List[Dict[str, Any]],
    window_seconds: int,
    level_name: str,
    merged_output_path: str,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ordered_keys, buckets = bucket_by_window(input_items, window_seconds)
    worker_count = resolve_max_workers(max_workers)

    logger.info(
        "[gen_event] start level=%s input_items=%d windows=%d max_workers=%d",
        level_name, len(input_items), len(ordered_keys), worker_count
    )
    print(
        f"[gen_event] start level={level_name} input_items={len(input_items)} windows={len(ordered_keys)} max_workers={worker_count}",
        flush=True,
    )

    tasks = []
    expected_per_day = Counter()
    for idx, (day, window_start) in enumerate(ordered_keys):
        batch = sort_docs(buckets[(day, window_start)])
        if not batch:
            continue
        expected_per_day[day] += 1
        tasks.append((idx, day, window_start, batch))

    results_by_index: Dict[int, Tuple[int, Dict[str, Any]]] = {}
    completed_per_day = Counter()
    collected_by_day: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"genevt-{level_name}") as executor:
        future_to_meta = {
            executor.submit(build_aggregate_node_safe, batch, level_name): (idx, day, window_start)
            for idx, day, window_start, batch in tasks
        }

        pbar = tqdm(total=len(tasks), desc=f"gen_event:{level_name}", unit="window")
        try:
            for future in as_completed(future_to_meta):
                idx, day, window_start = future_to_meta[future]
                node = future.result()
                results_by_index[idx] = (day, node)
                completed_per_day[day] += 1
                collected_by_day[day].append((idx, node))
                pbar.update(1)
                pbar.set_postfix(day=day, finished=f"{completed_per_day[day]}/{expected_per_day[day]}")
        finally:
            pbar.close()

    merged = []
    for day in sorted(expected_per_day.keys()):
        nodes = [node for _, node in sorted(collected_by_day[day], key=lambda x: x[0])]
        merged.extend(sort_docs(nodes))
    save_json(merged_output_path, merged)

    return merged


# =========================================================
# Public entry
# =========================================================

def load_source_docs(input_json: str) -> List[Dict[str, Any]]:
    raw = load_json(input_json)
    if not isinstance(raw, list):
        raise ValueError("input_json must be a list of docs")
    docs = [normalize_doc(x) for x in raw]
    return sort_docs(docs)


def gen_temporal_context_views(person_name, save_path, input_json, max_workers: Optional[int] = None):
    base_docs = load_source_docs(input_json)

    os.makedirs(save_path, exist_ok=True)

    worker_count = resolve_max_workers(max_workers)
    logger.info("[gen_event] resolved max_workers=%d", worker_count)
    print(f"[gen_event] resolved max_workers={worker_count}", flush=True)

    level1 = generate_level(
        input_items=base_docs,
        window_seconds=180,
        level_name="3min",
        merged_output_path=os.path.join(save_path, f"temporal_context_views_3min.json"),
        max_workers=worker_count,
    )

    level2 = generate_level(
        input_items=level1,
        window_seconds=600,
        level_name="10min",
        merged_output_path=os.path.join(save_path, f"temporal_context_views_10min.json"),
        max_workers=worker_count,
    )

    level3 = generate_level(
        input_items=level2,
        window_seconds=3600,
        level_name="1h",
        merged_output_path=os.path.join(save_path, f"temporal_context_views_1h.json"),
        max_workers=worker_count,
    )

    print(f"[Done] 3min: {len(level1)}, 10min: {len(level2)}, 1h: {len(level3)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate temporal context views.")
    parser.add_argument("--person", default="A1_JAKE")
    parser.add_argument("--json-path", default="output/metadata/multimodal_memory_cell/A1_JAKE/A1_JAKE_record.json")
    parser.add_argument("--save-path", default="output/metadata/multimodal_memory_cell/A1_JAKE/temporal_context_views")
    args = parser.parse_args()

    gen_temporal_context_views(
        person_name=args.person,
        input_json=args.json_path,
        save_path=args.save_path,
    )