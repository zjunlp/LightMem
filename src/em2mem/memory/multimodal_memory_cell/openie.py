import json
import os
from typing import Dict, Any, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

from .utils import compute_mdhash_id, filter_invalid_triples, NerRawOutput, TripleRawOutput, NerOutput, TripleOutput
from ...llm import dynamic_retry_decorator, LLMModel, PromptTemplateManager

logger = logging.getLogger(__name__)


class OpenIE:
    def __init__(self, llm_model: LLMModel):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model

    @dynamic_retry_decorator
    def _execute_ner_call(self, ner_input_message) -> List[str]:
        """Retryable helper that runs the full NER try-block logic (so the whole block is retried)."""
        response = self.llm_model.generate(ner_input_message, text_format=NerRawOutput)
        return response.named_entities

    @dynamic_retry_decorator
    def _execute_triples_call(self, messages) -> List[List[str]]:
        """Retryable helper that runs the full triple-extraction try-block logic (so the whole block is retried)."""
        response = self.llm_model.generate(messages, text_format=TripleRawOutput)
        triples = filter_invalid_triples(response.triples)
        return triples

    def ner(self, chunk_key: str, passage: str) -> NerOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        metadata = {}
        try:
            # LLM INFERENCE (entire try-block is retried by the decorator)
            unique_entities = self._execute_ner_call(ner_input_message)

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerOutput(
                chunk_id=chunk_key,
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerOutput(
            chunk_id=chunk_key,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleOutput:
        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )
        metadata = {}
        try:
            # LLM INFERENCE (entire try-block is retried by the decorator)
            triplets = self._execute_triples_call(messages)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleOutput(
                chunk_id=chunk_key,
                triples=[],
                metadata=metadata,
            )

        return TripleOutput(
            chunk_id=chunk_key,
            metadata=metadata,
            triples=triplets
        )
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "."):
        """
        Save OpenIE results to a JSON file.
        
        Args:
            results: The results dictionary to save
            output_dir: Output directory path.
        """

        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if hasattr(value, '__dict__'):
                json_results[key] = value.__dict__
            else:
                json_results[key] = value
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"openie_results_{self.llm_model.model_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_dir}/openie_results_{self.llm_model.model_name}.json")

    def openie(self, passage: str) -> Dict[str, Any]:
        chunk_key = compute_mdhash_id(passage, prefix="chunk-")
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output.unique_entities, "triplets": triple_output.triples}

    # def batch_openie(self, chunks: Union[List[str], Dict[str, Any]], output_dir: str = ".") -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    #     """
    #     Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

    #     Args:
    #         chunks (Union[List[str], Dict[str, Any]]): List of text passages or a dict of chunk_id to chunk data to be processed.
    #         output_dir (str): Directory to save output file.

    #     Returns:
    #         Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    #             - A dict with keys as the chunk ids (mdhash) and values as the NER result.
    #             - A dict with keys as the chunk ids (mdhash) and values as the triple extraction result.
    #     """

    #     if isinstance(chunks, List):
    #         # Compute chunk ids in the same order as input chunks to preserve sequence
    #         chunk_keys = [compute_mdhash_id(chunk, prefix="chunk-") for chunk in chunks]
    #         # Map chunk_id -> passage for quick lookup
    #         chunk_passages = {key: chunks[i] for i, key in enumerate(chunk_keys)}
    #     elif isinstance(chunks, Dict):
    #         chunk_keys = list(chunks.keys())
    #         chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

    #     ner_results_list = []

    #     with ThreadPoolExecutor() as executor:
    #         # Create NER futures for each chunk (submission order doesn't matter)
    #         ner_futures = {
    #             executor.submit(self.ner, chunk_key, chunk_passages[chunk_key]): chunk_key
    #             for chunk_key in chunk_keys
    #         }

    #         pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
    #         for future in pbar:
    #             result = future.result()
    #             ner_results_list.append(result)

    #     triple_results_list = []
    #     with ThreadPoolExecutor() as executor:
    #         # Create triple extraction futures for each chunk using outputs from NER
    #         re_futures = {
    #             executor.submit(self.triple_extraction, ner_result.chunk_id,
    #                             chunk_passages[ner_result.chunk_id],
    #                             ner_result.unique_entities): ner_result.chunk_id
    #             for ner_result in ner_results_list
    #         }
    #         # Collect triple extraction results with progress bar
    #         pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
    #         for future in pbar:
    #             result = future.result()
    #             triple_results_list.append(result)

    #     # Build maps from chunk_id to results (these may be in completion order)
    #     ner_map = {res.chunk_id: res.unique_entities for res in ner_results_list}
    #     triple_map = {res.chunk_id: res.triples for res in triple_results_list}

    #     # Build ordered dicts that follow the original input chunk order
    #     ordered_ner = {key: ner_map.get(key, []) for key in chunk_keys}
    #     ordered_triples = {key: triple_map.get(key, []) for key in chunk_keys}

    #     # Save the results (will preserve order in the output JSON)
    #     combined_results = {
    #         "ner_results": ordered_ner,
    #         "triple_results": ordered_triples
    #     }
    #     self.save_results(combined_results, output_dir)

    #     return ordered_ner, ordered_triples

    def batch_openie(
        self,
        chunks: Union[List[str], Dict[str, Any]],
        output_dir: str = ".",
        max_workers: int = 4,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Union[List[str], Dict[str, Any]]): List of text passages or a dict of chunk_id to chunk data to be processed.
            output_dir (str): Directory to save output file.
            max_workers (int): Maximum number of worker threads.

        Returns:
            Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
                - A dict with keys as the chunk ids and values as the NER result.
                - A dict with keys as the chunk ids and values as the triple extraction result.
        """

        if isinstance(chunks, List):
            chunk_keys = [compute_mdhash_id(chunk, prefix="chunk-") for chunk in chunks]
            chunk_passages = {key: chunks[i] for i, key in enumerate(chunk_keys)}
        elif isinstance(chunks, Dict):
            chunk_keys = list(chunks.keys())
            chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        else:
            raise ValueError("chunks must be either a list or a dict")

        ner_results_list = []

        with ThreadPoolExecutor() as executor:
            ner_futures = {
                executor.submit(self.ner, chunk_key, chunk_passages[chunk_key]): chunk_key
                for chunk_key in chunk_keys
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)

        triple_results_list = []
        with ThreadPoolExecutor() as executor:
            re_futures = {
                executor.submit(
                    self.triple_extraction,
                    ner_result.chunk_id,
                    chunk_passages[ner_result.chunk_id],
                    ner_result.unique_entities
                ): ner_result.chunk_id
                for ner_result in ner_results_list
            }

            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)

        ner_map = {res.chunk_id: res.unique_entities for res in ner_results_list}
        triple_map = {res.chunk_id: res.triples for res in triple_results_list}

        ordered_ner = {key: ner_map.get(key, []) for key in chunk_keys}
        ordered_triples = {key: triple_map.get(key, []) for key in chunk_keys}

        combined_results = {
            "ner_results": ordered_ner,
            "triple_results": ordered_triples
        }
        self.save_results(combined_results, output_dir)

        return ordered_ner, ordered_triples
