from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

from em2mem.llm import LLMModel

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENIE_MAX_WORKERS = int(os.getenv("OPENIE_MAX_WORKERS", os.getenv("MAX_WORKERS", "4")))
OPENIE_LOG_EVERY = int(os.getenv("OPENIE_LOG_EVERY", "50"))


# =========================================================
# Prompts
# =========================================================

DIRECT_30SEC_TRIPLET_SYSTEM_PROMPT = """# Role and Objective

You are an expert episodic triplet extractor for egocentric video memory.

Your task is to convert a short egocentric video segment into factual episodic triplets.
The input contains a first-person segment caption together with its corresponding transcript.

You must extract only grounded event facts that are explicitly supported by the input.

# Input

You will receive a JSON object with:
- `wearer_name`: the camera wearer name
- `fine_caption`: a concise first-person caption for the segment
- `caption_text`: caption-derived description of visible actions and objects
- `transcript_text`: transcript text for the segment
- `speakers`: speaker names mentioned in the transcript
- `main_actions`: short action phrases
- `salient_objects`: key objects or tools
- `conversation_focus`: main discussion topics

# Extraction Goal

Transform the segment into a small set of factual triplets in the form:

[head_entity, relation, tail_entity]

The triplets should capture:
1. the wearer's main actions;
2. object interactions;
3. key dialogue acts;
4. important interactions between people;
5. simple scene grounding when clearly stated.

# Guidelines

1. Use `wearer_name` instead of first-person pronouns such as I / me / my.
2. Keep triplets factual, concise, and directly grounded in the input.
3. Focus on the segment's main event skeleton rather than peripheral observations.
4. Preserve important object manipulations and interpersonal actions.
5. For speech, extract the communicative act and its topic, instead of copying long quoted sentences.
6. Do not create triplets for vague atmosphere, emotion, or speculation.
7. Do not use long quoted text as an entity.
8. Do not rewrite the event from an observer perspective.
9. Do not invent entities or facts not explicitly supported by the input.
10. Return valid JSON only.

# Preferred Relation Style

Use short, factual relations close to the original event phrasing, such as:
- hold
- use
- inspect
- hand_to
- place_on
- take_from
- discuss
- ask_about
- confirm
- say_about
- move
- organize
- occurs_in

When in doubt, prefer a simple action relation over an abstract relation.

# Output JSON Format

Return exactly:
{
  "triplets": [
    ["head_entity", "relation", "tail_entity"]
  ]
}
"""


HIGH_LEVEL_TRIPLET_SYSTEM_PROMPT = """# Role and Objective

You are an expert episodic triplet extractor for egocentric long-video memory.

Your task is to convert a structured high-level memory unit (3min / 10min / 1h) into a compact set of grounded episodic triplets.

The input contains:
- summary_text: chronological event summary
- critical_speech_lines: the most important quoted lines
- action_threads / object_threads / topic_threads: canonicalized metadata threads
- speaker_stats
- scene_summary
- visual_summary

You must extract only factual triplets that are explicitly supported by these inputs.

# Extraction Goal

Return:
1. `entities`: important normalized entities / topics mentioned in the unit
2. `triplets`: grounded event triplets in the form [head, relation, tail]

The triplets should preserve:
- main actions performed by the wearer
- key interactions between people
- important object manipulation / transfer / installation / writing / organization steps
- explicit plans, decisions, invitations, confirmations, and source/placement facts
- scene grounding when it is clearly relevant

# Critical Constraints

1. Use `wearer_name` instead of I / me / my.
2. Prefer explicit predicate-aligned facts over broad narrative paraphrases.
3. Keep relation phrases short and factual.
4. Do not copy long sentence fragments as entities.
5. Do not output time spans, stage headings, or discourse markers as entities.
6. Do not invent facts that are not supported by the input.
7. Use critical_speech_lines and metadata as grounding anchors when available.
8. Return valid JSON only.

# Preferred Relations

Prefer short relations such as:
- hold
- use
- inspect
- hand_to
- place_on
- take_from
- organize
- move
- discuss
- ask_about
- confirm
- say_about
- explain
- invite
- write_on
- assemble
- fit_into
- offer
- occurs_in

# Good Output Style

Good:
["Jake", "ask_about", "guest_count"]
["Shure", "offer", "write_on_blackboard"]
["Jake", "take_from", "hard drive"]
["Jake", "occurs_in", "dining area"]

Bad:
["11:24-11:27", "contains", "discussion"]
["Jake", "said that later they maybe could if possible", "some plan"]
["I pointed to the thing and maybe someone used it", "related_to", "object"]

# Output JSON Format

Return exactly:
{
  "entities": ["entity_or_topic", "..."],
  "triplets": [
    ["head_entity", "relation", "tail_entity"]
  ]
}
"""


class DirectTripletRawOutput(BaseModel):
    triplets: List[List[str]] = Field(default_factory=list)


class HighLevelTripletRawOutput(BaseModel):
    entities: List[str] = Field(default_factory=list)
    triplets: List[List[str]] = Field(default_factory=list)


# =========================================================
# Basic helpers
# =========================================================

PRONOUNS_TO_SKIP = {
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "there", "here", "something", "someone", "somebody", "anything", "everything",
}
FIRST_PERSON = {"i", "me", "my", "myself"}
GROUP_ALIASES = {
    "everyone": "group",
    "everybody": "group",
    "all": "group",
    "all of us": "group",
    "all people": "group",
}
RELATION_STOPLIST = {"be", "is", "are", "was", "were", "have", "has", "had", "do", "does", "did"}
BAD_RELATIONS = {
    "at", "with",
    "observe", "observes", "observed",
    "hear", "heard",
    "witness", "witnesses", "witnessed",
    "notice", "noticed",
    "refer_to", "refers_to",
    "face", "faces", "facing",
    "stand_by", "sit_by", "look_toward", "turn_to", "wave_to",
    "point_at", "place_hand_on", "rub", "contain", "contains", "involve", "involves",
}
ALLOWED_RELATIONS = {
    "hand_to",
    "hold",
    "use",
    "inspect",
    "ask_about",
    "confirm",
    "say_about",
    "discuss",
    "organize",
    "move",
    "place_on",
    "take_from",
    "occurs_in",
    "explain",
    "invite",
    "write_on",
    "assemble",
    "fit_into",
    "offer",
}
INFERRED_NAME_STOPWORDS = {
    "I", "We", "He", "She", "They", "It",
    "Okay", "Yes", "No", "The", "A", "An",
    "Then", "Later", "After", "Before", "Meanwhile",
    "Stage", "Primary", "Finally", "Overall",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}
LOW_VALUE_TOPICS_EXACT = {
    "awkwardness", "left", "right", "up", "down", "here", "there",
    "place_visibility", "open_item_first", "continue_left", "move_left", "move_right",
    "everyone", "group", "plastic_bag", "lining", "this_thing", "question", "questions",
}
LOW_VALUE_TOPIC_PATTERNS = [
    r"\bawkward", r"\bvisibility\b", r"\bleft\b", r"\bright\b", r"\bthis thing\b",
    r"\bplastic bag\b", r"\blining\b", r"\bstage \d+\b",
]
LOW_VALUE_ENTITY_EXACT = {"left", "right", "up", "down", "here", "there"}
TIME_SPAN_PATTERNS = [
    r"^\d{1,2}:\d{2}(?:[–\-]\d{1,2}:\d{2})?$",
    r"^\d{6,8}$",
    r"^\d{1,2}\s*[:.]\s*\d{2}\s*(?:[–\-]\s*\d{1,2}\s*[:.]\s*\d{2})?$",
    r"^\d{1,2}\s+\d{2}\s+\d{1,2}\s+\d{2}$",
]
LOW_VALUE_HEAD_PATTERNS = [
    r"^stage\s*\d+", r"^primary view", r"^overall flow", r"^planning and invite methods",
    r"^major visual transitions", r"^scene context",
]
LONG_RELATION_BAD_PATTERNS = [
    r"\bin presence of\b", r"\bwhile\b", r"\bduring\b", r"\bso that\b", r"\bwhich\b",
    r"\bthat\b", r"\bbecause\b",
]

RELATION_MAP: List[Tuple[str, str]] = [
    (r"hold[_ ]meeting[_ ]with|meeting[_ ]with|meet[_ ]with|met[_ ]with|talk[_ ]with|discuss[_ ]with", "discuss"),
    (r"hand.*to|pass.*to|give.*to|return.*to", "hand_to"),
    (r"hold|holding|held|grab|pick up|picked up|carry|carrying", "hold"),
    (r"scroll.*on|scrolling.*on|turn off|turned off|use|using|used|operate|operating", "use"),
    (r"check|inspect|examine|look at|looking at|looks at|point out", "inspect"),
    (r"ask about|asks about|ask|asks|question|questions", "ask_about"),
    (r"confirm|confirms|reply|replies|reply to|replies to|answer|answers", "confirm"),
    (r"say about|says about|say|says|tell|tells|mention|mentions", "say_about"),
    (r"discuss|discusses|talk about|talks about|speak about|speaks about|plan|plans", "discuss"),
    (r"explain|explains|introduce|introduces|describe|describes", "explain"),
    (r"invite|invites|inviting", "invite"),
    (r"write on|write onto|writes on|mark on|label on", "write_on"),
    (r"assemble|install|mount|set up|put together", "assemble"),
    (r"fit into|insert into|align into|push into", "fit_into"),
    (r"offer|offers|volunteer|volunteers", "offer"),
    (r"organize|sort|pack|clean|throw away|put away|put back|store|tidy", "organize"),
    (r"move|walk|walk to|walk into|go to|approach|enter|leave|return", "move"),
    (r"place on|put on|set on|put in front of|set in front of", "place_on"),
    (r"take from|remove from|get from|pull from|pull open", "take_from"),
    (r"is in|are in|inside|in", "occurs_in"),
]

OBJECT_CANONICAL_MAP: List[Tuple[str, str]] = [
    (r"\bcell ?phone|smartphone|mobile phone\b", "phone"),
    (r"\bphone\b", "phone"),
    (r"\bhard drives?\b|\bexternal drives?\b|\bdrive\b", "hard drive"),
    (r"\btripods?\b", "tripod"),
    (r"\bcables?\b|\bdata cable\b|\busb cable\b|\bcharging cable\b", "cable"),
    (r"\bpapers?\b|\bdocuments?\b|\bnotes?\b", "paper"),
    (r"\bcontainers?\b|\bboxes?\b|\bbin\b", "container"),
    (r"\bwhiteboard\b", "whiteboard"),
    (r"\bboard\b", "board"),
    (r"\blaptop\b|\bcomputer\b", "laptop"),
    (r"\btablet\b|\bipad\b", "tablet"),
    (r"\bcharger\b", "charger"),
    (r"\bbag\b|\bbackpack\b|\bplastic bag\b", "bag"),
    (r"\bcart\b|\bshopping cart\b", "cart"),
    (r"\bfridge\b|\brefrigerator\b", "refrigerator"),
    (r"\btable\b|\bdesk\b", "table"),
    (r"\bchair\b|\bchairs\b", "chair"),
    (r"\bstool\b|\bstools\b", "stool"),
    (r"\bclapperboard\b|\bslate\b", "clapperboard"),
    (r"\bmarker\b|\bpen\b", "marker"),
    (r"\bpower bank\b", "power bank"),
    (r"\bscrew hole\b", "screw hole"),
    (r"\bscrews?\b", "screw"),
    (r"\bstopwatch\b", "stopwatch"),
    (r"\bglasses\b|\bglass(es)? pair\b", "glasses"),
    (r"\bwhite laptop\b", "laptop"),
]

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


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_time_str(t: Any) -> str:
    return str(t).zfill(8)


def normalize_day_str(date_val: Any) -> str:
    if date_val is None:
        return ""
    s = str(date_val)
    m = re.search(r"DAY(\d+)", s)
    if m:
        return f"DAY{m.group(1)}"
    m = re.search(r"(\d+)", s)
    if m:
        return f"DAY{m.group(1)}"
    return s


def time_to_seconds(time_val: Any) -> int:
    time_val = int(time_val)
    hours = time_val // 1000000
    minutes = (time_val % 1000000) // 10000
    seconds = (time_val % 10000) // 100
    return hours * 3600 + minutes * 60 + seconds


def canonicalize_text(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def title_case_name(x: str) -> str:
    x = canonicalize_text(x)
    if not x:
        return ""
    return " ".join(p.capitalize() for p in x.split())


def infer_scale(input_path: str, units: List[Dict[str, Any]]) -> str:
    if units and isinstance(units[0], dict):
        if units[0].get("level"):
            return str(units[0]["level"])
        if units[0].get("fine_caption") and not units[0].get("level"):
            return "30sec"

    name = os.path.basename(input_path).lower()
    if "1h" in name:
        return "1h"
    if "10min" in name:
        return "10min"
    if "3min" in name:
        return "3min"
    return "30sec"


def get_doc_id(unit: Dict[str, Any]) -> str:
    if unit.get("doc_id"):
        return str(unit["doc_id"])
    date = normalize_day_str(unit.get("date", ""))
    start_time = normalize_time_str(unit.get("start_time", ""))
    end_time = normalize_time_str(unit.get("end_time", ""))
    level = str(unit.get("level", "")).strip()
    if level:
        return f"{date}_{start_time}_{end_time}_{level}"
    return f"{date}_{start_time}_{end_time}"


def get_unit_text(unit: Dict[str, Any]) -> str:
    text = str(unit.get("text", "")).strip()
    if text:
        return text
    fine_caption = str(unit.get("fine_caption", "")).strip()
    if fine_caption:
        return fine_caption
    content = str(unit.get("Content", "")).strip()
    return content


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        s = str(x).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def sort_units(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        units,
        key=lambda x: (
            normalize_day_str(x.get("date", "")),
            time_to_seconds(normalize_time_str(x.get("start_time", "0"))),
        ),
    )


def simple_tokenize(text: str) -> Set[str]:
    toks = re.findall(r"[a-zA-Z0-9_/-]+", canonicalize_text(text))
    return {t for t in toks if len(t) > 1}


def ensure_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def thread_canonical_values(thread_list: List[Dict[str, Any]], label_keys: List[str]) -> List[str]:
    values: List[str] = []
    for item in ensure_list(thread_list):
        if not isinstance(item, dict):
            continue
        val = ""
        for key in ["canonical_label", *label_keys]:
            if item.get(key):
                val = str(item.get(key)).strip()
                break
        if val:
            values.append(val)
    return unique_keep_order(values)


def speaker_names_from_stats(stats: List[Dict[str, Any]]) -> List[str]:
    names = []
    for item in ensure_list(stats):
        if isinstance(item, dict) and item.get("speaker"):
            names.append(title_case_name(item["speaker"]))
        elif isinstance(item, str):
            names.append(title_case_name(item))
    return unique_keep_order(names)


def infer_worker_count(requested: int) -> int:
    requested = max(1, int(requested))
    cpu_cap = max(1, (os.cpu_count() or 4))
    return min(requested, max(4, cpu_cap * 2))


def extract_candidate_person_names(unit: Dict[str, Any]) -> Set[str]:
    texts = [
        str(unit.get("fine_caption", "")).strip(),
        str(unit.get("caption_text", "")).strip(),
        str(unit.get("transcript_text", "")).strip(),
        str(unit.get("text", "")).strip(),
        " ".join(ensure_list(unit.get("critical_speech_lines", []))),
    ]
    names: Set[str] = set()
    for text in texts:
        for m in re.finditer(r"\b[A-Z][a-z]{1,24}\b", text):
            token = m.group(0)
            if token in INFERRED_NAME_STOPWORDS:
                continue
            names.add(token)
    return names


def is_low_value_topic(topic: str) -> bool:
    t = canonicalize_text(topic)
    if not t:
        return True
    if t in LOW_VALUE_TOPICS_EXACT:
        return True
    for pattern in LOW_VALUE_TOPIC_PATTERNS:
        if re.search(pattern, t):
            return True
    return False


def is_low_value_entity_text(entity: str) -> bool:
    e = canonicalize_text(entity)
    return e in LOW_VALUE_ENTITY_EXACT


def looks_like_time_span(text: str) -> bool:
    s = canonicalize_text(text)
    if not s:
        return False
    for pat in TIME_SPAN_PATTERNS:
        if re.fullmatch(pat, s):
            return True
    return False


def looks_like_bad_head(text: str) -> bool:
    s = canonicalize_text(text)
    if not s:
        return False
    if looks_like_time_span(s):
        return True
    for pat in LOW_VALUE_HEAD_PATTERNS:
        if re.search(pat, s):
            return True
    return False


def canonicalize_scene(scene: str) -> str:
    s = canonicalize_text(scene)
    if not s:
        return ""
    for pattern, target in SCENE_CANONICAL_MAP:
        if re.search(pattern, s):
            return target
    return s


def singularize_basic(x: str) -> str:
    irregular = {
        "glasses": "glasses",
        "scissors": "scissors",
        "pants": "pants",
        "shorts": "shorts",
    }
    if x in irregular:
        return irregular[x]
    if x.endswith("ies") and len(x) > 4:
        return x[:-3] + "y"
    if x.endswith("sses"):
        return x
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
    s = re.sub(r"\bsmall\b|\blarge\b|\bblack\b|\bwhite\b|\bred\b|\bblue\b|\bgreen\b|\bbig\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize_topic(topic: str) -> str:
    s = canonicalize_text(topic)
    if not s:
        return ""
    s = s.replace('"', "").replace("'", "")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if "stopwatch" in s and "need" in s:
        s = "stopwatch_need"
    elif "timestamp" in s and ("mark" in s or "marking" in s):
        s = "timestamp_marking"
    elif "plan" in s and "last day" in s:
        s = "plans_for_last_day"
    elif "hard drive" in s and ("name" in s or "identify" in s):
        s = "hard_drive_identification"
    elif "guest" in s and ("count" in s or re.search(r"\b4\s*[–-]\s*6\b", s)):
        s = "guest_count"
    elif "write" in s and "blackboard" in s:
        s = "write_on_blackboard"
    else:
        s = s.replace(" ", "_")

    if is_low_value_topic(s):
        return ""
    return s


def best_topic_match(text: str, topic_set: Set[str]) -> str:
    text_toks = simple_tokenize(text)
    if not text_toks or not topic_set:
        return ""
    best_topic = ""
    best_score = 0
    for topic in topic_set:
        topic_toks = simple_tokenize(topic)
        overlap = len(text_toks & topic_toks)
        if overlap > best_score:
            best_score = overlap
            best_topic = topic
    return best_topic if best_score >= 1 else ""


def looks_like_utterance_span(x: str) -> bool:
    s = str(x).strip()
    if not s:
        return False
    if '"' in s or "'" in s:
        return True
    if len(s.split()) >= 8:
        return True
    return False


def normalize_relation(rel: str) -> str:
    r = canonicalize_text(rel)
    if not r:
        return ""
    for pattern, target in RELATION_MAP:
        if re.search(pattern, r):
            return target
    r = re.sub(r"[^a-z0-9\s_]", "", r)
    r = re.sub(r"\s+", "_", r).strip("_")
    return r


def relation_is_too_long(raw_rel: str, norm_rel: str) -> bool:
    if not norm_rel:
        return True
    if norm_rel in ALLOWED_RELATIONS:
        return False
    if len(norm_rel.split("_")) > 3:
        return True
    raw = canonicalize_text(raw_rel)
    for pat in LONG_RELATION_BAD_PATTERNS:
        if re.search(pat, raw):
            return True
    return False


def get_scene_summary(unit: Dict[str, Any]) -> Dict[str, Any]:
    raw = unit.get("scene_summary", {})
    if isinstance(raw, dict):
        dominant_scene = canonicalize_scene(raw.get("dominant_scene", ""))
        raw_dist = raw.get("scene_distribution", {})
        dist: Dict[str, float] = {}
        if isinstance(raw_dist, dict):
            for k, v in raw_dist.items():
                ck = canonicalize_scene(k)
                if not ck:
                    continue
                try:
                    dist[ck] = float(v)
                except Exception:
                    pass
        if dominant_scene or dist:
            if not dominant_scene and dist:
                dominant_scene = max(dist.items(), key=lambda x: x[1])[0]
            if dominant_scene and not dist:
                dist = {dominant_scene: 1.0}
            return {"dominant_scene": dominant_scene, "scene_distribution": dist}

    scene = canonicalize_scene(unit.get("scene", ""))
    if scene:
        return {"dominant_scene": scene, "scene_distribution": {scene: 1.0}}
    return {}


def get_visual_summary(unit: Dict[str, Any]) -> str:
    visual_summary = str(unit.get("visual_summary", "")).strip()
    if visual_summary:
        return visual_summary
    scene_summary = get_scene_summary(unit)
    dominant_scene = str(scene_summary.get("dominant_scene", "")).strip()
    keyframe_caption = str(unit.get("keyframe_caption", "")).strip()
    visual_objects = unique_keep_order(unit.get("visual_objects", []) or [])

    parts: List[str] = []
    if dominant_scene:
        parts.append(f"Scene: {dominant_scene}.")
    if keyframe_caption:
        kc = keyframe_caption.rstrip(". ")
        if kc:
            parts.append(kc + ".")
    if visual_objects:
        parts.append("Visible objects: " + ", ".join(visual_objects[:8]) + ".")
    return " ".join(parts).strip()


def collect_metadata_index(unit: Dict[str, Any], person_name: str) -> Dict[str, Any]:
    person_name = title_case_name(person_name)

    person_alias: Dict[str, str] = {}
    topic_alias: Dict[str, str] = {}
    object_alias: Dict[str, str] = {}
    place_alias: Dict[str, str] = {}

    person_set: Set[str] = set()
    topic_set: Set[str] = set()
    object_set: Set[str] = set()
    place_set: Set[str] = set()

    def add_alias(mapping: Dict[str, str], alias: str, canonical: str):
        alias = canonicalize_text(alias)
        canonical = canonical.strip()
        if alias and canonical:
            mapping[alias] = canonical

    person_set.add(person_name)
    add_alias(person_alias, person_name, person_name)

    for spk in ensure_list(unit.get("speakers", [])):
        spk_c = title_case_name(spk)
        if spk_c:
            person_set.add(spk_c)
            add_alias(person_alias, spk, spk_c)

    for spk_name in speaker_names_from_stats(unit.get("speaker_stats", [])):
        if spk_name:
            person_set.add(spk_name)
            add_alias(person_alias, spk_name, spk_name)

    for name in extract_candidate_person_names(unit):
        name_c = title_case_name(name)
        if name_c:
            person_set.add(name_c)
            add_alias(person_alias, name, name_c)

    for obj in ensure_list(unit.get("salient_objects", [])):
        c = canonicalize_object(obj)
        if c:
            object_set.add(c)
            add_alias(object_alias, obj, c)

    for obj in ensure_list(unit.get("visual_objects", [])):
        c = canonicalize_object(obj)
        if c:
            object_set.add(c)
            add_alias(object_alias, obj, c)

    for th in ensure_list(unit.get("object_threads", [])):
        if not isinstance(th, dict):
            continue
        c = canonicalize_object(th.get("canonical_label") or th.get("object", ""))
        if c:
            object_set.add(c)
            add_alias(object_alias, c, c)
        for alias in ensure_list(th.get("aliases", [])):
            add_alias(object_alias, alias, c)

    for th in ensure_list(unit.get("visual_object_threads", [])):
        if not isinstance(th, dict):
            continue
        c = canonicalize_object(th.get("canonical_label") or th.get("object", ""))
        if c:
            object_set.add(c)
            add_alias(object_alias, c, c)
        for alias in ensure_list(th.get("aliases", [])):
            add_alias(object_alias, alias, c)

    for topic in ensure_list(unit.get("conversation_focus", [])):
        t = canonicalize_topic(topic)
        if t:
            topic_set.add(t)
            add_alias(topic_alias, topic, t)

    for th in ensure_list(unit.get("topic_threads", [])):
        if not isinstance(th, dict):
            continue
        t = canonicalize_topic(th.get("canonical_label") or th.get("topic", ""))
        if t:
            topic_set.add(t)
            add_alias(topic_alias, t, t)
        for alias in ensure_list(th.get("aliases", [])):
            add_alias(topic_alias, alias, t)

    scene = canonicalize_scene(unit.get("scene", ""))
    if scene:
        place_set.add(scene)
        add_alias(place_alias, scene, scene)

    scene_summary = get_scene_summary(unit)
    if isinstance(scene_summary, dict):
        ds = canonicalize_scene(scene_summary.get("dominant_scene", ""))
        if ds:
            place_set.add(ds)
            add_alias(place_alias, ds, ds)
        for place in scene_summary.get("scene_distribution", {}).keys():
            p = canonicalize_scene(place)
            if p:
                place_set.add(p)
                add_alias(place_alias, place, p)

    return {
        "person_alias": person_alias,
        "topic_alias": topic_alias,
        "object_alias": object_alias,
        "place_alias": place_alias,
        "person_set": person_set,
        "topic_set": topic_set,
        "object_set": object_set,
        "place_set": place_set,
        "person_name": person_name,
    }


def normalize_entity(raw: str, meta_idx: Dict[str, Any]) -> str:
    raw_str = str(raw).strip()
    s = canonicalize_text(raw_str)
    if not s:
        return ""

    if looks_like_time_span(s):
        return ""
    if s in FIRST_PERSON:
        return meta_idx["person_name"]
    if s in GROUP_ALIASES:
        return GROUP_ALIASES[s]
    if s in PRONOUNS_TO_SKIP:
        return ""

    if re.fullmatch(r"[A-Z][a-z]{1,24}", raw_str):
        return title_case_name(raw_str)

    if looks_like_bad_head(s):
        return ""

    if looks_like_utterance_span(s):
        matched_topic = best_topic_match(s, meta_idx["topic_set"])
        return matched_topic if matched_topic else ""

    if s in meta_idx["person_alias"]:
        return meta_idx["person_alias"][s]
    if s in meta_idx["object_alias"]:
        return meta_idx["object_alias"][s]
    if s in meta_idx["topic_alias"]:
        return meta_idx["topic_alias"][s]
    if s in meta_idx["place_alias"]:
        return meta_idx["place_alias"][s]

    if re.fullmatch(r"[a-z]+", s) and s.capitalize() in meta_idx["person_set"]:
        return s.capitalize()

    obj = canonicalize_object(s)
    if obj and (obj in meta_idx["object_set"] or obj in {
        "container", "tripod", "phone", "hard drive", "cable", "bag", "marker",
        "power bank", "screw", "stopwatch", "glasses", "board", "clapperboard",
    }):
        return obj

    place = canonicalize_scene(s)
    if place and place in meta_idx["place_set"]:
        return place

    topic = canonicalize_topic(s)
    if topic and (topic in meta_idx["topic_set"] or not is_low_value_topic(topic)):
        return topic

    if is_low_value_entity_text(s):
        return ""

    return s


def entity_type(entity: str, meta_idx: Dict[str, Any]) -> str:
    if entity in meta_idx["person_set"]:
        return "Person"
    if entity in meta_idx["place_set"]:
        return "Place"
    if entity in meta_idx["object_set"] or entity in {
        "container", "tripod", "phone", "hard drive", "cable", "bag", "marker",
        "power bank", "screw", "stopwatch", "glasses", "board", "clapperboard",
    }:
        return "Object"
    if entity in meta_idx["topic_set"] or canonicalize_topic(entity):
        return "Topic"
    if entity == "group":
        return "Entity"
    return "Entity"


def normalize_tail_by_relation(raw_t: str, current_t: str, relation: str, meta_idx: Dict[str, Any]) -> str:
    raw_t = str(raw_t).strip()
    cur = str(current_t).strip()

    if relation in {"move", "hold", "use", "inspect", "organize", "place_on", "take_from", "assemble", "fit_into", "write_on"}:
        place = canonicalize_scene(raw_t)
        if place and place in meta_idx["place_set"]:
            return place
        obj = canonicalize_object(raw_t)
        if obj:
            return obj
        if canonicalize_object(cur):
            return canonicalize_object(cur)
        if is_low_value_entity_text(cur):
            return ""

    if relation in {"ask_about", "confirm", "explain", "offer", "say_about", "discuss", "invite"}:
        topic_match = best_topic_match(raw_t, meta_idx["topic_set"])
        if topic_match:
            return topic_match
        topic = canonicalize_topic(raw_t)
        if topic:
            return topic
        obj = canonicalize_object(raw_t)
        if obj:
            return obj
        obj2 = canonicalize_object(cur)
        if obj2:
            return obj2

    return cur


def valid_relation_type(h: str, r: str, t: str, meta_idx: Dict[str, Any]) -> bool:
    h_type = entity_type(h, meta_idx)
    t_type = entity_type(t, meta_idx)

    if r == "hand_to":
        return h_type == "Person" and t_type == "Person"
    if r in {"hold", "use", "inspect", "assemble", "fit_into", "write_on"}:
        return h_type == "Person" and t_type in {"Object", "Entity", "Place"}
    if r in {"place_on", "take_from", "move", "organize"}:
        return h_type == "Person" and t_type in {"Object", "Place", "Entity"}
    if r in {"ask_about", "confirm", "say_about", "discuss", "explain", "offer", "invite"}:
        return h_type in {"Person", "Entity"} and t_type in {"Topic", "Object", "Entity"}
    if r == "occurs_in":
        return t_type == "Place"
    return False


# =========================================================
# Structured payloads
# =========================================================


def build_30sec_triplet_payload(unit: Dict[str, Any], person_name: str) -> Dict[str, Any]:
    return {
        "wearer_name": title_case_name(person_name),
        "fine_caption": str(unit.get("fine_caption", "")).strip(),
        "caption_text": str(unit.get("caption_text", "")).strip(),
        "transcript_text": str(unit.get("transcript_text", "")).strip(),
        "speakers": ensure_list(unit.get("speakers", [])),
        "main_actions": ensure_list(unit.get("main_actions", [])),
        "salient_objects": ensure_list(unit.get("salient_objects", [])),
        "conversation_focus": ensure_list(unit.get("conversation_focus", [])),
    }


def build_highlevel_triplet_payload(unit: Dict[str, Any], person_name: str, scale: str) -> Dict[str, Any]:
    scene_summary = get_scene_summary(unit)
    visual_summary = get_visual_summary(unit)
    return {
        "wearer_name": title_case_name(person_name),
        "doc_id": get_doc_id(unit),
        "level": scale,
        "summary_text": get_unit_text(unit),
        "critical_speech_lines": unique_keep_order(ensure_list(unit.get("critical_speech_lines", [])))[:10],
        "action_threads": thread_canonical_values(unit.get("action_threads", []), ["action"]),
        "object_threads": thread_canonical_values(unit.get("object_threads", []), ["object"]),
        "topic_threads": thread_canonical_values(unit.get("topic_threads", []), ["topic"]),
        "speaker_names": speaker_names_from_stats(unit.get("speaker_stats", [])) or unique_keep_order(ensure_list(unit.get("speakers", []))),
        "scene_summary": scene_summary,
        "visual_summary": visual_summary,
    }


# =========================================================
# Extraction
# =========================================================


def dedup_triples(triples: List[List[str]]) -> List[List[str]]:
    seen = set()
    out: List[List[str]] = []
    for tri in triples:
        if not isinstance(tri, list) or len(tri) != 3:
            continue
        tri = [str(x).strip() for x in tri]
        if not tri[0] or not tri[1] or not tri[2]:
            continue
        key = tuple(tri)
        if key in seen:
            continue
        seen.add(key)
        out.append(tri)
    return out


def dedup_entities(entities: List[str]) -> List[str]:
    seen = set()
    out = []
    for ent in entities:
        s = str(ent).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def run_direct_30sec_triplet_extraction(llm_model: LLMModel, unit: Dict[str, Any], person_name: str) -> Tuple[List[str], List[List[str]]]:
    payload = build_30sec_triplet_payload(unit, person_name)
    messages = [
        {"role": "system", "content": DIRECT_30SEC_TRIPLET_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    response = llm_model.generate(messages, text_format=DirectTripletRawOutput)
    entities = dedup_entities(payload["speakers"] + payload["salient_objects"] + payload["conversation_focus"])
    return entities, dedup_triples(response.triplets)


def run_highlevel_triplet_extraction(llm_model: LLMModel, unit: Dict[str, Any], person_name: str, scale: str) -> Tuple[List[str], List[List[str]]]:
    payload = build_highlevel_triplet_payload(unit, person_name, scale)
    messages = [
        {"role": "system", "content": HIGH_LEVEL_TRIPLET_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    response = llm_model.generate(messages, text_format=HighLevelTripletRawOutput)
    return dedup_entities(response.entities), dedup_triples(response.triplets)


def run_structured_batch(
    units: List[Dict[str, Any]],
    llm_model: LLMModel,
    person_name: str,
    scale: str,
    max_workers: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    ner_results: Dict[str, List[str]] = {}
    triple_results: Dict[str, List[List[str]]] = {}

    total = len(units)
    logger.info("Starting structured triplet extraction | scale=%s | units=%d | workers=%d", scale, total, max_workers)

    def _process(unit: Dict[str, Any]):
        doc_id = get_doc_id(unit)
        try:
            if scale == "30sec":
                entities, triplets = run_direct_30sec_triplet_extraction(llm_model, unit, person_name)
            else:
                entities, triplets = run_highlevel_triplet_extraction(llm_model, unit, person_name, scale)
            return doc_id, entities, triplets, None
        except Exception as e:
            logger.warning("[%s] extraction failed: %s", doc_id, e)
            return doc_id, [], [], str(e)

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process, unit) for unit in units]
        desc = "Direct 30sec triplet extraction" if scale == "30sec" else f"Structured {scale} triplet extraction"
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
            doc_id, ner, triples, err = fut.result()
            ner_results[doc_id] = ner
            triple_results[doc_id] = triples
            completed += 1
            if completed % max(1, OPENIE_LOG_EVERY) == 0 or completed == total:
                logger.info("Extraction progress | scale=%s | completed=%d/%d", scale, completed, total)

    logger.info("Finished structured triplet extraction | scale=%s | units=%d", scale, total)
    return ner_results, triple_results


def save_openie_like_results(output_dir: str, model_name: str, ner_results: Dict[str, List[str]], triple_results: Dict[str, List[List[str]]]):
    data = {"ner_results": ner_results, "triple_results": triple_results}
    path = os.path.join(output_dir, f"openie_results_{model_name}.json")
    save_json(path, data)
    logger.info("Saved openie-like results to %s", path)


# =========================================================
# Triplet cleaning / filtering
# =========================================================


def should_keep_triplet(h: str, r: str, t: str, meta_idx: Dict[str, Any], raw_relation: str = "") -> bool:
    if not h or not r or not t:
        return False
    if h == t:
        return False
    if looks_like_bad_head(h) or looks_like_bad_head(t):
        return False
    if r in RELATION_STOPLIST:
        return False
    if r in BAD_RELATIONS:
        return False
    if relation_is_too_long(raw_relation or r, r):
        return False
    if r not in ALLOWED_RELATIONS:
        return False
    if looks_like_utterance_span(h) or looks_like_utterance_span(t):
        return False
    if is_low_value_entity_text(t):
        return False
    if r in {"say_about", "discuss", "explain", "offer", "invite"} and is_low_value_topic(t):
        return False
    if not valid_relation_type(h, r, t, meta_idx):
        return False
    if r in {"hand_to", "ask_about", "confirm", "say_about", "hold", "use", "inspect", "explain", "offer", "write_on", "assemble", "fit_into", "invite"}:
        return True
    head_known = h in meta_idx["person_set"] or h in meta_idx["object_set"] or h in meta_idx["topic_set"] or h in meta_idx["place_set"] or h == "group"
    tail_known = t in meta_idx["person_set"] or t in meta_idx["object_set"] or t in meta_idx["topic_set"] or t in meta_idx["place_set"] or t == "group"
    return head_known or tail_known


def normalize_and_filter_triplets(raw_triplets: List[List[str]], unit: Dict[str, Any], person_name: str) -> List[List[str]]:
    meta_idx = collect_metadata_index(unit, person_name)
    kept: List[List[str]] = []
    seen = set()

    for tri in raw_triplets:
        if len(tri) != 3:
            continue
        raw_h, raw_r, raw_t = tri
        h = normalize_entity(raw_h, meta_idx)
        r = normalize_relation(raw_r)
        t = normalize_entity(raw_t, meta_idx)
        raw_r_norm = canonicalize_text(raw_r)

        if re.search(r"hand.*to|pass.*to|give.*to|return.*to", raw_r_norm):
            h_guess = normalize_entity(raw_h, meta_idx)
            t_guess = normalize_entity(raw_t, meta_idx)
            if h_guess and t_guess:
                if entity_type(h_guess, meta_idx) == "Person" and entity_type(t_guess, meta_idx) == "Person":
                    h, r, t = h_guess, "hand_to", t_guess

        t = normalize_tail_by_relation(raw_t, t, r, meta_idx)

        if r == "discuss" and t == "group":
            continue

        if r in {"ask_about", "confirm", "say_about", "discuss", "explain", "offer", "invite"}:
            if not t:
                continue
            if r in {"say_about", "discuss", "explain", "offer", "invite"} and entity_type(t, meta_idx) not in {"Topic", "Object", "Entity"}:
                continue
            if entity_type(t, meta_idx) == "Topic" and is_low_value_topic(t):
                continue

        if not should_keep_triplet(h, r, t, meta_idx, raw_relation=str(raw_r)):
            continue

        key = (h, r, t)
        if key in seen:
            continue
        seen.add(key)
        kept.append([h, r, t])

    return kept


# =========================================================
# Graph building
# =========================================================


def make_node_id(node_type: str, label: str) -> str:
    safe_label = re.sub(r"[^a-zA-Z0-9_:/.-]+", "_", label.strip())
    return f"{node_type.lower()}::{safe_label}"


def add_node(nodes: Dict[str, Dict[str, Any]], node_id: str, node_type: str, label: str, **attrs):
    if node_id not in nodes:
        nodes[node_id] = {"id": node_id, "type": node_type, "label": label, **attrs}


def add_edge(edges: Dict[Tuple[str, str, str, str], Dict[str, Any]], source: str, target: str, edge_type: str, event_id: str, **attrs):
    key = (source, target, edge_type, event_id)
    if key not in edges:
        edges[key] = {"source": source, "target": target, "type": edge_type, "event_id": event_id, **attrs}


def build_event_centric_graph(
    units: List[Dict[str, Any]],
    triplet_map: Dict[str, List[List[str]]],
    person_name: str,
    scale: str,
    source_file: str,
) -> Dict[str, Any]:
    units = sort_units(units)

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    ordered_event_ids: List[str] = []
    doc_id_to_event_id: Dict[str, str] = {}

    for unit in units:
        doc_id = get_doc_id(unit)
        event_node_id = make_node_id("Event", doc_id)
        ordered_event_ids.append(event_node_id)
        doc_id_to_event_id[doc_id] = event_node_id

        resolved_visual_summary = get_visual_summary(unit)
        resolved_scene_summary = get_scene_summary(unit)

        add_node(
            nodes,
            event_node_id,
            "Event",
            doc_id,
            doc_id=doc_id,
            scale=scale,
            date=normalize_day_str(unit.get("date", "")),
            start_time=normalize_time_str(unit.get("start_time", "")),
            end_time=normalize_time_str(unit.get("end_time", "")),
            text=get_unit_text(unit),
            visual_summary=resolved_visual_summary,
            action_threads=unit.get("action_threads", unit.get("main_actions", [])),
            object_threads=unit.get("object_threads", unit.get("salient_objects", [])),
            topic_threads=unit.get("topic_threads", unit.get("conversation_focus", [])),
            speaker_stats=unit.get("speaker_stats", unit.get("speakers", [])),
            critical_speech_lines=unit.get("critical_speech_lines", []),
            scene_summary=resolved_scene_summary,
            visual_object_threads=unit.get("visual_object_threads", unit.get("visual_objects", [])),
            child_ids=unit.get("child_ids", []),
            source_doc_ids=unit.get("source_doc_ids", [doc_id]),
            scene=str(unit.get("scene", "")).strip(),
            keyframe_caption=str(unit.get("keyframe_caption", "")).strip(),
        )

        meta_idx = collect_metadata_index(unit, person_name)
        mentioned_nodes: Set[Tuple[str, str]] = set()

        for h, r, t in triplet_map.get(doc_id, []):
            h_type = entity_type(h, meta_idx)
            t_type = entity_type(t, meta_idx)
            h_id = make_node_id(h_type, h)
            t_id = make_node_id(t_type, t)
            add_node(nodes, h_id, h_type, h)
            add_node(nodes, t_id, t_type, t)
            add_edge(edges, h_id, t_id, r, event_node_id, doc_id=doc_id, scale=scale, edge_source="triplet")
            mentioned_nodes.add((h_id, h_type))
            mentioned_nodes.add((t_id, t_type))

        for ent_id, ent_type in mentioned_nodes:
            attach_type = "mentions"
            if ent_type == "Person":
                attach_type = "involves"
            elif ent_type == "Place":
                attach_type = "occurs_in"
            elif ent_type == "Topic":
                attach_type = "about"
            elif ent_type == "Object":
                attach_type = "mentions_object"
            add_edge(edges, event_node_id, ent_id, attach_type, event_node_id, doc_id=doc_id, scale=scale, edge_source="attachment")

        scene_summary = get_scene_summary(unit)
        dominant_scene = ""
        if isinstance(scene_summary, dict):
            dominant_scene = canonicalize_scene(scene_summary.get("dominant_scene", ""))
        if dominant_scene:
            place_id = make_node_id("Place", dominant_scene)
            add_node(nodes, place_id, "Place", dominant_scene)
            add_edge(edges, event_node_id, place_id, "occurs_in", event_node_id, doc_id=doc_id, scale=scale, edge_source="metadata")

    for i in range(len(ordered_event_ids) - 1):
        add_edge(edges, ordered_event_ids[i], ordered_event_ids[i + 1], "before", ordered_event_ids[i], edge_source="temporal", scale=scale)

    graph = {
        "graph_type": "event_centric_episodic_graph",
        "scale": scale,
        "source_file": source_file,
        "doc_id_to_event_id": doc_id_to_event_id,
        "event_order": ordered_event_ids,
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "stats": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "num_events": sum(1 for n in nodes.values() if n["type"] == "Event"),
        },
    }
    return graph


# =========================================================
# Main processing
# =========================================================


def load_units(input_file: str) -> List[Dict[str, Any]]:
    data = load_json(input_file)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of memory units")
    return data


def create_episodic_triplet_results(
    units: List[Dict[str, Any]],
    ner_results: Dict[str, List[str]],
    triple_results: Dict[str, List[List[str]]],
    person_name: str,
    scale: str,
    source_file: str,
) -> Dict[str, Any]:
    unit_results = []
    triplet_map = {}

    for unit in sort_units(units):
        doc_id = get_doc_id(unit)
        raw_triples = triple_results.get(doc_id, [])
        clean_triples = normalize_and_filter_triplets(raw_triples, unit, person_name)
        resolved_scene_summary = get_scene_summary(unit)
        resolved_visual_summary = get_visual_summary(unit)

        unit_result = {
            "doc_id": doc_id,
            "date": normalize_day_str(unit.get("date", "")),
            "start_time": normalize_time_str(unit.get("start_time", "")),
            "end_time": normalize_time_str(unit.get("end_time", "")),
            "text": get_unit_text(unit),
            "visual_summary": resolved_visual_summary,
            "scene_summary": resolved_scene_summary,
            "critical_speech_lines": unit.get("critical_speech_lines", []),
            "scene": str(unit.get("scene", "")).strip(),
            "keyframe_caption": str(unit.get("keyframe_caption", "")).strip(),
            "visual_objects": unit.get("visual_objects", []) or [],
            "ner": ner_results.get(doc_id, []),
            "openie_results": raw_triples,
            "episodic_triplets": clean_triples,
        }
        unit_results.append(unit_result)
        triplet_map[doc_id] = clean_triples

    return {
        "scale": scale,
        "source_file": source_file,
        "units": unit_results,
        "triplet_map": triplet_map,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract episodic triplets and build event-centric episodic graph")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--person", default="A1_Jake")
    parser.add_argument("--model", default=OPENAI_MODEL)
    parser.add_argument("--max-workers", type=int, default=OPENIE_MAX_WORKERS)
    args = parser.parse_args()

    person_name = args.person.split("_")[-1] if "_" in args.person else args.person
    person_name = title_case_name(person_name)
    max_workers = infer_worker_count(args.max_workers)

    units = load_units(args.input_file)
    scale = infer_scale(args.input_file, units)
    logger.info("Loaded %d units from %s (scale=%s)", len(units), args.input_file, scale)
    logger.info("Resolved person_name=%s | max_workers=%d", person_name, max_workers)

    llm_model = LLMModel(model_name=args.model)

    ner_results, triple_results = run_structured_batch(
        units=units,
        llm_model=llm_model,
        person_name=person_name,
        scale=scale,
        max_workers=max_workers,
    )
    save_openie_like_results(args.output_dir, args.model, ner_results, triple_results)

    total_entities = sum(len(v) for v in ner_results.values())
    total_raw_triples = sum(len(v) for v in triple_results.values())
    logger.info("Total entities extracted: %d", total_entities)
    logger.info("Total raw triples extracted: %d", total_raw_triples)

    triplet_results = create_episodic_triplet_results(
        units=units,
        ner_results=ner_results,
        triple_results=triple_results,
        person_name=person_name,
        scale=scale,
        source_file=args.input_file,
    )

    clean_total_triples = sum(len(u["episodic_triplets"]) for u in triplet_results["units"])
    logger.info("Total cleaned episodic triples: %d", clean_total_triples)

    triplets_out = os.path.join(args.output_dir, f"episodic_triplets_{scale}_{args.model}.json")
    save_json(triplets_out, triplet_results)
    logger.info("Saved episodic triplets to %s", triplets_out)

    graph = build_event_centric_graph(
        units=units,
        triplet_map=triplet_results["triplet_map"],
        person_name=person_name,
        scale=scale,
        source_file=args.input_file,
    )

    graph_out = os.path.join(args.output_dir, f"episodic_graph_{scale}_{args.model}.json")
    save_json(graph_out, graph)
    logger.info("Saved episodic graph to %s", graph_out)
    logger.info("Done. Graph stats: %s", graph["stats"])


if __name__ == "__main__":
    main()
