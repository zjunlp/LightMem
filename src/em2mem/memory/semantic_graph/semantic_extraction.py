import json
import os
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from .utils import SemanticRawOutput, SemanticOutput
from ...llm import LLMModel, PromptTemplateManager
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SemanticExtraction:
    """
    LLM-based semantic extraction from episodic triples.

    Keeps metadata from episodic units, including provenance_root_ids,
    so downstream semantic facts can project back to 10sec/30sec roots.

    Added robustness:
    - structured retries
    - raw-text fallback
    - JSON repair / malformed-entry cleanup
    """

    def __init__(
        self,
        llm_model: LLMModel,
        max_retries: int = 2,
    ):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model
        self.max_retries = max_retries

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def semantic_extraction(self, chunk_key: str, episodic_triples: List[List[str]]) -> SemanticOutput:
        formatted_triples = "\n".join(f"{i}. {triple}" for i, triple in enumerate(episodic_triples))
        base_messages = self.prompt_template_manager.render(
            name="semantic_extraction",
            episodic_triples=formatted_triples,
        )

        if not isinstance(base_messages, list):
            logger.warning("Prompt render for %s did not return chat messages; got %s", chunk_key, type(base_messages))
            return SemanticOutput(
                chunk_id=chunk_key,
                semantic_triples=[],
                episodic_evidence=[],
            )

        # 1) Structured attempts with retries
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            messages = base_messages if attempt == 0 else self._build_retry_messages(base_messages, attempt, last_error)

            try:
                response = self.llm_model.generate(messages, text_format=SemanticRawOutput)

                repaired = self._repair_payload(
                    {
                        "semantic_triples": getattr(response, "semantic_triples", []),
                        "episodic_evidence": getattr(response, "episodic_evidence", []),
                    },
                    num_input_triples=len(episodic_triples),
                )

                return SemanticOutput(
                    chunk_id=chunk_key,
                    semantic_triples=repaired["semantic_triples"],
                    episodic_evidence=repaired["episodic_evidence"],
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    "Structured semantic extraction failed for %s on attempt %d/%d: %s",
                    chunk_key,
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                )

        # 2) Raw-text fallback + repair
        try:
            repaired = self._generate_with_raw_fallback(
                base_messages=base_messages,
                num_input_triples=len(episodic_triples),
                last_error=last_error,
            )
            if repaired["semantic_triples"] or repaired["episodic_evidence"]:
                logger.info("Recovered semantic extraction for %s via raw fallback", chunk_key)
                return SemanticOutput(
                    chunk_id=chunk_key,
                    semantic_triples=repaired["semantic_triples"],
                    episodic_evidence=repaired["episodic_evidence"],
                )
        except Exception as e:
            logger.warning("Raw fallback semantic extraction failed for %s: %s", chunk_key, e)

        logger.warning("Semantic extraction failed for %s; returning empty output", chunk_key)
        return SemanticOutput(
            chunk_id=chunk_key,
            semantic_triples=[],
            episodic_evidence=[],
        )

    def batch_semantic_extraction(
        self,
        episodic_triples_batch: Dict[str, List[List[str]]]
    ) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[List[int]]]]:
        payload_batch = {
            chunk_key: {"triples": triples, "metadata": {}}
            for chunk_key, triples in episodic_triples_batch.items()
        }
        combined_results = self.batch_semantic_extraction_with_metadata(payload_batch)
        return combined_results["semantic_triples"], combined_results["episodic_evidence"]

    def batch_semantic_extraction_with_metadata(
        self,
        episodic_payload_batch: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        results: List[Tuple[str, SemanticOutput]] = []

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.semantic_extraction, chunk_key, payload["triples"]): chunk_key
                for chunk_key, payload in episodic_payload_batch.items()
            }

            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting semantic triples",
                leave=True,
            )

            for future in pbar:
                chunk_key = futures[future]
                result = future.result()
                results.append((chunk_key, result))

        ordered_keys = list(episodic_payload_batch.keys())
        semantic_triples_map = {chunk_key: res.semantic_triples for chunk_key, res in results}
        episodic_evidence_map = {chunk_key: res.episodic_evidence for chunk_key, res in results}

        items: List[Dict[str, Any]] = []
        for chunk_key in ordered_keys:
            payload = episodic_payload_batch[chunk_key]
            metadata = dict(payload.get("metadata", {}))
            triples = payload.get("triples", [])
            item = {
                "chunk_id": chunk_key,
                **metadata,
                "source_triples": triples,
                "semantic_triples": semantic_triples_map.get(chunk_key, []),
                "episodic_evidence": episodic_evidence_map.get(chunk_key, []),
            }
            items.append(item)

        combined_results: Dict[str, Any] = {
            "items": items,
            "semantic_triples": {
                item["chunk_id"]: item["semantic_triples"] for item in items
            },
            "episodic_evidence": {
                item["chunk_id"]: item["episodic_evidence"] for item in items
            },
            "metadata": {
                item["chunk_id"]: {
                    k: v
                    for k, v in item.items()
                    if k not in {"semantic_triples", "episodic_evidence", "source_triples"}
                }
                for item in items
            },
        }

        return combined_results

    # -------------------------------------------------------------------------
    # Retry helpers
    # -------------------------------------------------------------------------
    def _build_retry_messages(
        self,
        base_messages: List[Dict[str, str]],
        attempt: int,
        last_error: Exception | None,
    ) -> List[Dict[str, str]]:
        retry_instruction = (
            "Your previous output was invalid.\n"
            "Return ONLY valid JSON with exactly two keys: semantic_triples and episodic_evidence.\n"
            "Rules:\n"
            "- Never output an empty item like [] inside semantic_triples.\n"
            "- Every semantic triple must be exactly 3 non-empty strings: [subject, predicate, object].\n"
            "- Never output 2-slot triples such as ['Jake', 'holds_phone_with_both_hands'].\n"
            "- If a candidate triple is malformed, OMIT it instead of keeping it.\n"
            "- episodic_evidence must align exactly with semantic_triples.\n"
            "- Every evidence list must contain only valid 0-based integer indices.\n"
        )
        if last_error is not None:
            retry_instruction += f"\nPrevious validation error:\n{str(last_error)}\n"

        return list(base_messages) + [
            {"role": "user", "content": retry_instruction}
        ]

    # -------------------------------------------------------------------------
    # Raw fallback
    # -------------------------------------------------------------------------
    def _generate_with_raw_fallback(
        self,
        base_messages: List[Dict[str, str]],
        num_input_triples: int,
        last_error: Exception | None,
    ) -> Dict[str, List]:
        messages = self._build_retry_messages(base_messages, attempt=self.max_retries + 1, last_error=last_error)

        raw_response = self.llm_model.generate(messages)
        raw_text = self._extract_text_from_response(raw_response)
        parsed = self._parse_json_from_text(raw_text)

        return self._repair_payload(parsed, num_input_triples=num_input_triples)

    def _extract_text_from_response(self, raw_response: Any) -> str:
        """
        Best-effort extraction of text from different possible SDK wrappers.
        Adjust here if your LLM wrapper returns a different shape.
        """
        if raw_response is None:
            raise ValueError("Raw response is None")

        if isinstance(raw_response, str):
            return raw_response

        if isinstance(raw_response, dict):
            for key in ("output_text", "text", "content", "response", "raw_text"):
                value = raw_response.get(key)
                if isinstance(value, str):
                    return value

        for attr in ("output_text", "text", "content", "response", "raw_text"):
            if hasattr(raw_response, attr):
                value = getattr(raw_response, attr)
                if isinstance(value, str):
                    return value

        # last resort
        return str(raw_response)

    def _parse_json_from_text(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Empty raw text from model")

        cleaned = text.strip()

        # Remove fenced code block wrappers if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # First try direct parse
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Then try extracting the first top-level JSON object substring
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed

        raise ValueError("Could not parse a JSON object from raw model output")

    # -------------------------------------------------------------------------
    # Output repair
    # -------------------------------------------------------------------------
    def _repair_payload(
        self,
        payload: Dict[str, Any],
        num_input_triples: int,
    ) -> Dict[str, List]:
        semantic_triples = payload.get("semantic_triples", [])
        episodic_evidence = payload.get("episodic_evidence", [])

        if not isinstance(semantic_triples, list):
            semantic_triples = []
        if not isinstance(episodic_evidence, list):
            episodic_evidence = []

        repaired_triples: List[List[str]] = []
        repaired_evidence: List[List[int]] = []

        pair_count = min(len(semantic_triples), len(episodic_evidence))
        dropped_count = 0

        for i in range(pair_count):
            triple = semantic_triples[i]
            evidence = episodic_evidence[i]

            repaired_triple = self._repair_triple(triple)
            repaired_ev = self._repair_evidence(evidence, num_input_triples=num_input_triples)

            if repaired_triple is None:
                dropped_count += 1
                continue
            if not repaired_ev:
                dropped_count += 1
                continue

            repaired_triples.append(repaired_triple)
            repaired_evidence.append(repaired_ev)

        if len(semantic_triples) != len(episodic_evidence):
            logger.warning(
                "semantic_triples and episodic_evidence length mismatch: %d vs %d; truncated to %d",
                len(semantic_triples),
                len(episodic_evidence),
                pair_count,
            )

        if dropped_count > 0:
            logger.warning("Dropped %d malformed semantic entries during repair", dropped_count)

        return {
            "semantic_triples": repaired_triples,
            "episodic_evidence": repaired_evidence,
        }

    def _repair_triple(self, triple: Any) -> List[str] | None:
        if not isinstance(triple, list):
            return None

        # valid case
        if len(triple) == 3 and all(isinstance(x, str) for x in triple):
            cleaned = [self._clean_text(x) for x in triple]
            if all(cleaned):
                return cleaned
            return None

        return None

    def _repair_evidence(self, evidence: Any, num_input_triples: int) -> List[int]:
        if not isinstance(evidence, list):
            return []

        repaired: List[int] = []
        seen = set()

        for item in evidence:
            try:
                idx = int(item)
            except Exception:
                continue
            if 0 <= idx < num_input_triples and idx not in seen:
                seen.add(idx)
                repaired.append(idx)

        repaired.sort()
        return repaired

    def _clean_text(self, text: str) -> str:
        return " ".join(text.strip().split())


    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)