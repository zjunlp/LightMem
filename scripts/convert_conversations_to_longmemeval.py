import json
import os
from datetime import datetime

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_iso(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return datetime.now().date().isoformat()

def build_session(mapping):
    nodes = mapping or {}
    seq = []
    chain = []
    cur = nodes.get("root")
    while cur and cur.get("children"):
        nid = cur["children"][0]
        nxt = nodes.get(nid)
        if not nxt:
            break
        chain.append(nxt)
        cur = nxt
    messages = []
    i = 1
    for node in chain:
        msg = node.get("message") or {}
        frags = msg.get("fragments") or []
        user_txt = None
        assistant_txt = None
        for fr in frags:
            if fr.get("type") == "REQUEST" and not user_txt:
                user_txt = fr.get("content")
            if fr.get("type") not in ("REQUEST","THINK") and not assistant_txt:
                assistant_txt = fr.get("content")
        if user_txt:
            messages.append({"role":"user","content":user_txt,"sequence_number":i}); i+=1
            messages.append({"role":"assistant","content":assistant_txt or "","sequence_number":i}); i+=1
    return messages

def convert(user_path, conv_path, out_path):
    user = load_json(user_path)
    convs = load_json(conv_path)
    uid = user.get("user_id") or "unknown_user"
    out = []
    for c in convs:
        qid = c.get("id") or f"conv_{len(out)+1}"
        title = c.get("title") or ""
        inserted = c.get("inserted_at") or c.get("updated_at") or datetime.now().isoformat()
        date = to_iso(inserted)
        session = build_session(c.get("mapping"))
        if not session:
            continue
        last_assistant = ""
        for m in reversed(session):
            if m.get("role") == "assistant":
                last_assistant = m.get("content") or ""
                break
        item = {
            "question_id": qid,
            "question": title or session[0]["content"],
            "answer": last_assistant,
            "question_type": "single-session-user",
            "question_date": date,
            "haystack_sessions": [session],
            "haystack_dates": [date]
        }
        out.append(item)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    user_path = os.path.join(base, "user.json")
    conv_path = os.path.join(base, "conversations.json")
    out_path = os.path.join(base, "data", "longmemeval_converted.json")
    convert(user_path, conv_path, out_path)
