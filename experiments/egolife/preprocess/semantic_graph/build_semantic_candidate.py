import argparse
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Tuple

from em2mem.llm import LLMModel
from em2mem.memory.semantic_graph.semantic_extraction import SemanticExtraction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_PERIOD = 10


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_atom(x: Any) -> str:
    text = str(x).strip()
    return " ".join(text.split())


def _normalize_triple(triple: Any) -> List[str] | None:
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        return None
    s, p, o = (_clean_atom(v) for v in triple)
    if not s or not p or not o:
        return None
    return [s, p, o]


def _dedupe_triples(triples: Iterable[Any]) -> List[List[str]]:
    out: List[List[str]] = []
    seen = set()
    for triple in triples:
        norm = _normalize_triple(triple)
        if norm is None:
            continue
        key = tuple(norm)
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def _ordered_unique(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def sort_units(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        units,
        key=lambda x: (
            str(x.get("date", "")),
            str(x.get("start_time", "")),
            str(x.get("end_time", "")),
            str(x.get("doc_id", "")),
        ),
    )


def choose_source_triples(unit: Dict[str, Any], source_field: str) -> Tuple[List[List[str]], str]:
    if source_field == "openie_results":
        return _dedupe_triples(unit.get("openie_results", [])), "openie_results"
    if source_field == "episodic_triplets":
        return _dedupe_triples(unit.get("episodic_triplets", [])), "episodic_triplets"

    openie_triples = _dedupe_triples(unit.get("openie_results", []))
    if openie_triples:
        return openie_triples, "openie_results"

    episodic_triples = _dedupe_triples(unit.get("episodic_triplets", []))
    return episodic_triples, "episodic_triplets"


def _normalize_root_ids(unit: Dict[str, Any], unit_scale: str) -> List[str]:
    roots = unit.get("provenance_root_ids", []) or []
    roots = [str(x).strip() for x in roots if str(x).strip()]
    if roots:
        return roots
    # native 30sec unit: itself is the root
    if unit_scale == "30sec":
        doc_id = str(unit.get("doc_id", "")).strip()
        return [doc_id] if doc_id else []
    return []


def build_grouped_payload_batch(
    episodic_file: str,
    source_field: str = "openie_results",
    period: int = DEFAULT_PERIOD,
    min_group_triples: int = 1,
    expected_scale: str = "30sec",
) -> Dict[str, Dict[str, Any]]:
    """
    Strict paper-style grouping:
    - group consecutive atomic units by fixed period
    - concatenate triples within each group
    - preserve exact source triple pointer mapping back to atomic units
    """
    data = load_json(episodic_file)
    file_scale = str(data.get("scale", "")).strip()
    units = sort_units(data.get("units", []))

    if expected_scale and file_scale != expected_scale:
        logger.warning(
            "Input file scale is %s, expected %s. "
            "For exact back-projection to 30sec, you should use the 30sec file.",
            file_scale,
            expected_scale,
        )

    payload_batch: Dict[str, Dict[str, Any]] = {}

    for i in range(0, len(units), period):
        group_units = units[i:i + period]
        if not group_units:
            continue

        first_item = group_units[0]
        last_item = group_units[-1]

        group_triples: List[List[str]] = []
        triple_pointer_map: List[str] = []
        source_doc_ids: List[str] = []
        source_units: List[Dict[str, Any]] = []
        used_source_fields: List[str] = []

        for unit in group_units:
            unit_scale = str(unit.get("scale", file_scale) or file_scale)
            doc_id = str(unit["doc_id"]).strip()
            source_doc_ids.append(doc_id)

            roots = _normalize_root_ids(unit, unit_scale)
            source_units.append(
                {
                    "doc_id": doc_id,
                    "scale": unit_scale,
                    "date": str(unit.get("date", "")),
                    "start_time": str(unit.get("start_time", "")),
                    "end_time": str(unit.get("end_time", "")),
                    "provenance_root_ids": roots,
                }
            )

            triples, used_source_field = choose_source_triples(unit, source_field)
            used_source_fields.append(used_source_field)

            for triple_idx, triple in enumerate(triples):
                group_triples.append(triple)
                # exact pointer to original atomic unit triple
                triple_pointer_map.append(f"{doc_id}#{triple_idx}")

        if len(group_triples) < min_group_triples:
            continue

        # still “last-item anchored” like the original paper, but make it unique/stable
        group_id = (
            f"{last_item['date']}_"
            f"{str(first_item['start_time']).zfill(8)}_"
            f"{str(last_item['end_time']).zfill(8)}_grp{period}"
        )

        all_group_roots: List[str] = []
        for unit_meta in source_units:
            roots = unit_meta.get("provenance_root_ids", []) or []
            if roots:
                all_group_roots.extend(roots)
            else:
                all_group_roots.append(unit_meta["doc_id"])

        payload_batch[group_id] = {
            "triples": group_triples,
            "metadata": {
                "doc_id": group_id,
                "group_id": group_id,
                "scale": f"grouped_{expected_scale or file_scale}",
                "base_scale": expected_scale or file_scale,
                "group_period": period,
                "date": str(last_item.get("date", "")),
                "start_time": str(first_item.get("start_time", "")),
                "end_time": str(last_item.get("end_time", "")),
                "source_field": source_field,
                "source_file": episodic_file,
                "source_doc_ids": source_doc_ids,
                "source_units": source_units,
                "used_source_fields": used_source_fields,
                "provenance_root_ids": _ordered_unique(all_group_roots),
                "triple_pointer_map": triple_pointer_map,
            },
        }

    return payload_batch


def project_group_evidence_to_root_refs(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert per-group local evidence indices into exact atomic evidence refs:
    episodic_evidence: [[0, 5], [2]]
    -> episodic_evidence_refs: [["DAY1_xxx#0", "DAY1_yyy#1"], ["DAY1_zzz#0"]]
    """
    for item in results.get("items", []):
        pointer_map = item.get("triple_pointer_map", []) or []
        evidence_lists = item.get("episodic_evidence", []) or []

        episodic_evidence_refs: List[List[str]] = []
        evidence_root_ids: List[List[str]] = []

        for evidence in evidence_lists:
            refs: List[str] = []
            roots: List[str] = []
            for idx in evidence:
                try:
                    idx_int = int(idx)
                except Exception:
                    continue
                if 0 <= idx_int < len(pointer_map):
                    ref = pointer_map[idx_int]
                    refs.append(ref)
                    roots.append(ref.split("#", 1)[0])

            episodic_evidence_refs.append(_ordered_unique(refs))
            evidence_root_ids.append(_ordered_unique(roots))

        item["episodic_evidence_refs"] = episodic_evidence_refs
        item["evidence_root_ids"] = evidence_root_ids

    results["episodic_evidence_refs"] = {
        item["chunk_id"]: item.get("episodic_evidence_refs", [])
        for item in results.get("items", [])
    }
    results["evidence_root_ids"] = {
        item["chunk_id"]: item.get("evidence_root_ids", [])
        for item in results.get("items", [])
    }
    return results


def run_semantic_extraction_from_grouped_30sec(
    episodic_file: str,
    output_dir: str,
    model_name: str = "gpt-5-mini",
    source_field: str = "openie_results",
    period: int = DEFAULT_PERIOD,
    min_group_triples: int = 1,
    llm_model: LLMModel | None = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    payload_batch = build_grouped_payload_batch(
        episodic_file=episodic_file,
        source_field=source_field,
        period=period,
        min_group_triples=min_group_triples,
        expected_scale="30sec",
    )
    logger.info("Prepared %d grouped chunks for semantic extraction", len(payload_batch))

    total_source_triples = sum(len(v["triples"]) for v in payload_batch.values())
    logger.info("Total grouped source triples: %d", total_source_triples)

    if llm_model is None:
        llm_model = LLMModel(model_name=model_name)

    extractor = SemanticExtraction(llm_model)
    results = extractor.batch_semantic_extraction_with_metadata(payload_batch)

    # overwrite saved extraction file with root-projected evidence refs
    results = project_group_evidence_to_root_refs(results)

    output_path = os.path.join(
        output_dir,
        f"semantic_candidates_{llm_model.model_name}.json",
    )
    extractor.save_results(results, output_path)

    total_semantic_triples = sum(len(item.get("semantic_triples", [])) for item in results.get("items", []))
    logger.info("Total semantic triples extracted: %d", total_semantic_triples)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-style semantic extraction from grouped 30sec episodic units."
    )
    parser.add_argument("--episodic-file", type=str, required=True, help="Path to 30sec episodic_triplets JSON.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--source-field",
        type=str,
        default="openie_results",
        choices=["openie_results", "episodic_triplets", "auto"],
    )
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD)
    parser.add_argument("--min-group-triples", type=int, default=1)
    args = parser.parse_args()

    run_semantic_extraction_from_grouped_30sec(
        episodic_file=args.episodic_file,
        output_dir=args.output_dir,
        model_name=args.model,
        source_field=args.source_field,
        period=args.period,
        min_group_triples=args.min_group_triples,
    )


if __name__ == "__main__":
    main()


