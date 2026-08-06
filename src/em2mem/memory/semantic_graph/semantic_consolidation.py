import json
import os
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .utils import ConsolidationRawOutput
from ...llm import LLMModel, PromptTemplateManager
from ...embedding import EmbeddingModel

logger = logging.getLogger(__name__)

class SemanticConsolidation:
    def __init__(self, llm_model: LLMModel, embedding_model: EmbeddingModel):
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.6  # Threshold for finding relevant existing triples

    def find_relevant_triples(self, new_triples: List[List[str]], 
                                existing_triples: List[Tuple[List[str], List[str]]], 
                                top_k: int = 20) -> List[List[Tuple[List[str], List[str]]]]:
        """
        Find existing triples that are relevant to a new triple using semantic similarity.
        
        Args:
            new_triples: List of new semantic triples [subject, predicate, object]
            existing_triples: List of tuples (triple, evidence_list) where evidence is in "{timestamp}_{idx}" format
            top_k: Maximum number of relevant triples to return
            
        Returns:
            List of List of tuples (existing_triple, evidence_list) for relevant matches
        """
        if not existing_triples:
            return [[] for _ in new_triples]
        
        # Convert triples to text for embedding
        new_triple_texts = [" ".join(triple) for triple in new_triples]
        existing_triple_texts = [" ".join(triple) for triple, _ in existing_triples]
        
        # Get embeddings using text modality
        new_embeddings = self.embedding_model.encode(new_triple_texts, modality="text")
        existing_embeddings = self.embedding_model.encode(existing_triple_texts, modality="text")
        
        # Compute cosine similarity matrix (new_triples x existing_triples)
        similarities = cosine_similarity(new_embeddings, existing_embeddings)
        
        # Get top-k indices for each new triple (sorted by similarity descending)
        top_k_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        
        # Return relevant triples for each new triple
        return [
            [existing_triples[idx] for idx in indices if similarities[i, idx] >= self.similarity_threshold]
            for i, indices in enumerate(top_k_indices)
        ]

    def consolidate_triple(self, new_triple: List[str], 
                            relevant_existing_triples: List[List[str]]) -> Tuple[List[str], List[int]]:
        """
        Consolidate a new semantic triple with relevant existing ones.
        
        Args:
            new_triple (List[str]): A single new semantic triple [subject, predicate, object]
            relevant_existing_triples (List[List[str]]): List of relevant existing triples
            
        Returns:
            Tuple of (updated_new_triple, triple_indices_to_remove)
        """
        if not relevant_existing_triples:
            return new_triple, []

        formatted_existing_triples = "\n".join(f"{i}. {triple}" for i, triple in enumerate(relevant_existing_triples))
        messages = self.prompt_template_manager.render(
            name='semantic_consolidation',
            new_triple=new_triple,
            existing_triples=formatted_existing_triples
        )

        try:
            # LLM INFERENCE
            # Ensure messages is a list for chat-based templates
            if isinstance(messages, str):
                raise ValueError("Expected chat template to return List[Dict], got string")
            response = self.llm_model.generate(messages, text_format=ConsolidationRawOutput)

        except Exception as e:
            logger.warning(e)
            return new_triple, []
        
        return response.updated_triple, response.triples_to_remove

    def batch_semantic_consolidation(self, existing_semantic_results: Tuple[List[List[str]], List[List[str]]], 
                                      new_semantic_results: Tuple[List[List[str]], List[List[str]]]) -> Tuple[List[List[str]], List[List[str]], List[Tuple[List[str], List[str]]]]:
        """
        Conduct semantic consolidation for a single timestamp against existing results.
        
        Args:
            existing_semantic_results: Tuple of (existing_semantic_triples, existing_episodic_evidence) 
                                     where existing_episodic_evidence is already in "{timestamp}_{idx}" format
            new_semantic_results: Tuple of (new_semantic_triples, new_episodic_evidence)
                                where new_episodic_evidence is also in "{timestamp}_{idx}" format
            
        Returns:
            Tuple of (consolidated_semantic_triples, consolidated_episodic_evidence, triples_to_remove)
            where triples_to_remove is a list of (triple, evidence) tuples that should be removed from accumulated state
        """
        existing_semantic_triples, existing_episodic_evidence = existing_semantic_results
        new_semantic_triples, new_episodic_evidence = new_semantic_results
        
        if not new_semantic_triples:
            # No new triples to consolidate, return empty results
            return [], [], []
        
        # Create accumulated triples structure from existing results
        # Each tuple is (triple, evidence_list)
        accumulated_triples: List[Tuple[List[str], List[str]]] = []
        for triple, evidence in zip(existing_semantic_triples, existing_episodic_evidence):
            accumulated_triples.append((triple, evidence))
        
        # Process all new triples concurrently
        consolidated_results = self._process_timestamp_triples_concurrent(
            new_semantic_triples, new_episodic_evidence, accumulated_triples
        )
        
        # Extract results
        consolidated_triples = [result["updated_triple"] for result in consolidated_results]
        consolidated_evidence = [result["merged_evidence"] for result in consolidated_results]
        
        # Collect all triples to be removed across all consolidations
        all_triples_to_remove: List[Tuple[List[str], List[str]]] = []
        for result in consolidated_results:
            all_triples_to_remove.extend(result["triples_to_remove"])
        
        return consolidated_triples, consolidated_evidence, all_triples_to_remove

    def _process_timestamp_triples_concurrent(self, current_triples: List[List[str]], 
                                            current_evidence: List[List[str]],
                                            accumulated_triples: List[Tuple[List[str], List[str]]]) -> List[Dict[str, Any]]:
        """
        Process all triples at a timestamp concurrently.
        
        Args:
            current_triples: New semantic triples for this timestamp
            current_evidence: Evidence in "{timestamp}_{idx}" format
            accumulated_triples: Existing triples from previous timestamps
            
        Returns:
            List of consolidation results with updated triples and merged evidence
        """
        # Find relevant triples for ALL new triples at once (optimization)
        all_relevant_existing_data = self.find_relevant_triples(current_triples, accumulated_triples)
        
        def process_single_triple(triple_idx: int) -> Dict[str, Any]:
            new_triple = current_triples[triple_idx]
            new_evidence = current_evidence[triple_idx]
            
            # Get precomputed relevant triples for this specific triple
            relevant_existing = all_relevant_existing_data[triple_idx]
            
            if not relevant_existing:
                # No relevant existing triples, return as-is
                return {
                    "updated_triple": new_triple,
                    "triples_to_remove": [],
                    "merged_evidence": new_evidence,
                    "triple_idx": triple_idx
                }
            
            relevant_triples_only = [triple for triple, _ in relevant_existing]
            
            # Consolidate with LLM
            updated_triple, indices_to_remove = self.consolidate_triple(new_triple, relevant_triples_only)
            
            # Merge evidence from removed triples
            merged_evidence = new_evidence.copy()
            triples_to_remove_data = []
            
            for remove_idx in indices_to_remove:
                if remove_idx < len(relevant_existing):
                    removed_triple, removed_evidence = relevant_existing[remove_idx]
                    merged_evidence.extend(removed_evidence)
                    triples_to_remove_data.append((removed_triple, removed_evidence))
            
            return {
                "updated_triple": updated_triple,
                "triples_to_remove": triples_to_remove_data,
                "merged_evidence": merged_evidence,
                "triple_idx": triple_idx
            }
        
        # Process all triples concurrently
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_single_triple, i): i for i in range(len(current_triples))}
            pbar = tqdm(as_completed(futures), total=len(futures), 
                       desc=f"Consolidating triples", leave=False)
            
            for future in pbar:
                result = future.result()
                results.append(result)
        
        # Sort results by original triple index to maintain order
        results.sort(key=lambda x: x["triple_idx"])
        return results