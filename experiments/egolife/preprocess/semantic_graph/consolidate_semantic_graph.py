import hashlib
import argparse
import json
import logging
import os
from typing import Any, Dict, List
from tqdm import tqdm

from em2mem.embedding import EmbeddingModel
from em2mem.llm import LLMModel
from em2mem.memory.semantic_graph.semantic_consolidation import SemanticConsolidation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ordered_unique(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def scale_rank(scale: str) -> int:
    return {
        "30sec": 0,
        "3min": 1,
        "10min": 2,
        "1h": 3,
        "grouped_30sec": 10,
    }.get(scale, 99)


def sort_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        items,
        key=lambda x: (
            str(x.get("date", "")),
            str(x.get("start_time", "")),
            str(x.get("end_time", "")),
            scale_rank(str(x.get("scale", ""))),
            str(x.get("chunk_id", x.get("doc_id", ""))),
        ),
    )


def triple_fingerprint(triple: List[str]) -> str:
    raw = "||".join([str(x).strip().lower() for x in triple])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def fact_id(triple: List[str]) -> str:
    return f"sf_{triple_fingerprint(triple)}"


def consolidate(items: List[Dict[str, Any]], consolidator) -> Dict[str, Any]:
    items = sort_items(items)

    # Metadata for original 30sec root units
    root_metadata_map: Dict[str, Dict[str, Any]] = {}
    for item in items:
        for unit_meta in item.get("source_units", []) or []:
            doc_id = str(unit_meta.get("doc_id", "")).strip()
            if doc_id:
                root_metadata_map[doc_id] = {
                    "doc_id": doc_id,
                    "scale": unit_meta.get("scale", "30sec"),
                    "date": unit_meta.get("date", ""),
                    "start_time": unit_meta.get("start_time", ""),
                    "end_time": unit_meta.get("end_time", ""),
                    "provenance_root_ids": unit_meta.get("provenance_root_ids", []) or [doc_id],
                }

    accumulated_semantic_triples: List[List[str]] = []
    accumulated_episodic_evidence: List[List[str]] = []
    timeline: Dict[str, Any] = {}

    for i, item in tqdm(
        enumerate(items),
        total=len(items),
        desc="Building semantic memory",
        leave=True,
    ):
        chunk_id = item.get("chunk_id", item.get("doc_id", ""))
        current_triples = item.get("semantic_triples", []) or []
        # IMPORTANT: use projected refs from extraction, not group-local indices
        current_evidence = item.get("episodic_evidence_refs", []) or []

        existing_results = (
            accumulated_semantic_triples.copy(),
            accumulated_episodic_evidence.copy(),
        )
        new_results = (current_triples, current_evidence)

        consolidated_triples, consolidated_evidence, triples_to_remove = (
            consolidator.batch_semantic_consolidation(
                existing_results,
                new_results,
            )
        )

        remove_set = {
            (tuple(triple), tuple(evidence))
            for triple, evidence in triples_to_remove
        }

        kept_triples: List[List[str]] = []
        kept_evidence: List[List[str]] = []
        removed_fact_ids: List[str] = []

        for old_triple, old_evidence in zip(
            accumulated_semantic_triples,
            accumulated_episodic_evidence,
        ):
            key = (tuple(old_triple), tuple(old_evidence))
            if key in remove_set:
                removed_fact_ids.append(fact_id(old_triple))
                continue
            kept_triples.append(old_triple)
            kept_evidence.append(old_evidence)

        accumulated_semantic_triples = kept_triples
        accumulated_episodic_evidence = kept_evidence

        added_fact_ids: List[str] = []
        for new_triple, new_evidence in zip(consolidated_triples, consolidated_evidence):
            deduped_evidence = _ordered_unique(new_evidence)
            accumulated_semantic_triples.append(new_triple)
            accumulated_episodic_evidence.append(deduped_evidence)
            added_fact_ids.append(fact_id(new_triple))

        active_fact_ids = sorted({fact_id(t) for t in accumulated_semantic_triples})

        active_root_ids: List[str] = []
        for evidence_refs in accumulated_episodic_evidence:
            for ref in evidence_refs:
                root_doc_id = ref.split("#", 1)[0]
                active_root_ids.append(root_doc_id)

        timeline[chunk_id] = {
            "order": i,
            "chunk_id": chunk_id,
            "date": item.get("date", ""),
            "start_time": item.get("start_time", ""),
            "end_time": item.get("end_time", ""),
            "scale": item.get("scale", ""),
            "group_period": item.get("group_period", None),
            "source_doc_ids": item.get("source_doc_ids", []),
            "added_fact_ids": sorted(set(added_fact_ids)),
            "removed_fact_ids": sorted(set(removed_fact_ids)),
            "active_fact_ids": active_fact_ids,
            "active_provenance_root_ids": _ordered_unique(active_root_ids),
        }

    facts: List[Dict[str, Any]] = []
    for triple, evidence_refs in zip(
        accumulated_semantic_triples,
        accumulated_episodic_evidence,
    ):
        evidence_refs = _ordered_unique(evidence_refs)
        support_docs = _ordered_unique([ref.split("#", 1)[0] for ref in evidence_refs])

        support_days: List[str] = []
        support_scales: List[str] = []
        for doc_id in support_docs:
            meta = root_metadata_map.get(doc_id, {})
            day = str(meta.get("date", ""))
            scale = str(meta.get("scale", ""))
            if day and day not in support_days:
                support_days.append(day)
            if scale and scale not in support_scales:
                support_scales.append(scale)

        provenance_root_ids = support_docs.copy()

        first_doc_meta = root_metadata_map.get(support_docs[0], {}) if support_docs else {}
        last_doc_meta = root_metadata_map.get(support_docs[-1], {}) if support_docs else {}

        facts.append(
            {
                "fact_id": fact_id(triple),
                "triple": triple,
                "support_count": len(evidence_refs),
                "support_doc_count": len(support_docs),
                "support_root_count": len(provenance_root_ids),
                "support_docs": support_docs,
                "support_days": support_days,
                "support_scales": support_scales,
                "evidence_refs": evidence_refs,
                "provenance_root_ids": provenance_root_ids,
                "first_seen": {
                    "doc_id": first_doc_meta.get("doc_id", support_docs[0] if support_docs else ""),
                    "date": first_doc_meta.get("date", ""),
                    "start_time": first_doc_meta.get("start_time", ""),
                    "end_time": first_doc_meta.get("end_time", ""),
                },
                "last_seen": {
                    "doc_id": last_doc_meta.get("doc_id", support_docs[-1] if support_docs else ""),
                    "date": last_doc_meta.get("date", ""),
                    "start_time": last_doc_meta.get("start_time", ""),
                    "end_time": last_doc_meta.get("end_time", ""),
                },
            }
        )

    facts.sort(key=lambda x: (-x["support_count"], x["fact_id"]))
    return {
        "facts": facts,
        "timeline": timeline,
    }


def run_semantic_consolidation(
    semantic_file: str,
    output_dir: str,
    model_name: str = "gpt-5-mini",
    llm_model: LLMModel | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    semantic_results = load_json(semantic_file)

    if "items" not in semantic_results:
        raise ValueError("Expected semantic extraction file with an `items` field.")

    items = semantic_results["items"]
    logger.info("Loaded %d extraction items", len(items))

    if embedding_model is None:
        embedding_model = EmbeddingModel(text_model_name="Qwen/Qwen3-Embedding-4B")
        embedding_model.load_model(model_type="text")

    if llm_model is None:
        llm_model = LLMModel(model_name=model_name)

    consolidator = SemanticConsolidation(llm_model, embedding_model)
    output = consolidate(items, consolidator)

    output_file = os.path.join(output_dir, f"semantic_graph_{model_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved semantic memory to %s", output_file)
    logger.info("Final semantic fact count: %d", len(output.get("facts", [])))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate semantic candidates into semantic graph.")
    parser.add_argument("--semantic-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    args = parser.parse_args()

    run_semantic_consolidation(
        semantic_file=args.semantic_file,
        output_dir=args.output_dir,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()

