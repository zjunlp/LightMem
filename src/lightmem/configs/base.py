import os
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, Optional, Literal
from pydantic import ValidationError
from lightmem.configs.pre_compressor.base import PreCompressorConfig
from lightmem.configs.topic_segmenter.base import TopicSegmenterConfig
from lightmem.configs.memory_manager.base import MemoryManagerConfig
from lightmem.configs.text_embedder.base import TextEmbedderConfig
from lightmem.configs.multimodal_embedder.base import MMEmbedderConfig
from lightmem.configs.retriever.contextretriever.base import ContextRetrieverConfig
from lightmem.configs.retriever.embeddingretriever.base import EmbeddingRetrieverConfig
from lightmem.configs.logging.base import LoggingConfig

lightmem_dir = ""

class BaseMemoryConfigs(BaseModel):
    pre_compress: Optional[bool] = Field(
        default=False,
        description="If True, enable pre-compression; otherwise disable it."
    )
    pre_compressor: Optional[PreCompressorConfig] = Field(
        default=None,
        description="Configuration for the compress model (active only if pre_compress=True)."
    )
    topic_segment: Optional[bool] = Field(
        default=False,
        description="If True, enable topic segmentation; otherwise disable it."
    )
    precomp_topic_shared: Optional[bool] = Field(
        default=False,
        description="",
    )
    topic_segmenter: Optional[TopicSegmenterConfig] = Field(
        default=None,
        description="Configuration for the topic segmenter (active only if topic_segment=True)."
    )
    messages_use: Optional[Literal["user_only", "assistant_only", "hybrid"]] = Field(
        default="user_only",
        description="Specifies which messages to use for subsequent metadata generation and text summarization. "
                    "Choices: "
                    "'user_only' - only user messages are processed; "
                    "'assistant_only' - only assistant messages are processed; "
                    "'hybrid' - both user and assistant messages are processed."
    )
    metadata_generate: Optional[bool] = Field(
        default=True,
        description="If True, extract metadata for optimized memory management and faster subsequent retrieval; if False, skip metadata generation."
    )
    text_summary: Optional[bool] = Field(
        default=True,
        description="When enabled, stores a summarized version in memory; otherwise only stores the original text unmodified."
    )
    memory_manager: MemoryManagerConfig = Field(
        description="Configuration for the memory management model, primarily used for metadata generation and text summarization",
        default_factory=MemoryManagerConfig
    )
    extract_threshold: float = Field(
        default=0.5,
    )
    index_strategy: Optional[Literal["embedding", "context", "hybrid"]] = Field(
        default=None,
        description="Indexing strategy to use. Choices: "
                "embedding|text|hybrid"
    )
    text_embedder: Optional[TextEmbedderConfig] = Field(
        description="Configuration for the text embedding model",
        default=None
    )
    multimodal_embedder: Optional[MMEmbedderConfig] = Field(
        description="Configuration for the image embedding model",
        default=None,
    )
    history_db_path: Optional[str] = Field(
        description="Path to the history database",
        default=os.path.join(lightmem_dir, "history.db"),
    )
    retrieve_strategy: Optional[Literal["context", "embedding", "hybrid"]] = Field(
        default=None,
        description="Retrieving strategy to use. Choices: "
        "embedding|context|hybrid. When omitted, the strategy is inferred "
        "from whichever retriever configs are supplied; 'hybrid' is never "
        "inferred and must be requested explicitly. Passing None explicitly "
        "is honoured as 'no retrieval' rather than inferred."
    )
    context_retriever: Optional[ContextRetrieverConfig] = Field(
        description="Configuration for the context-based retriever (active only if retrieve_strategy is 'context' or 'hybrid')",
        default=None,
    )
    embedding_retriever: Optional[EmbeddingRetrieverConfig] = Field(
        description="Configuration for the embedding-based retriever (active only if retrieve_strategy is 'embedding' or 'hybrid')",
        default=None,
    )
    summary_retriever: Optional[EmbeddingRetrieverConfig] = Field(
        description="Configuration for the summary retriever (optional, for storing and retrieving memory summaries)",
        default=None,
    )
    update: Optional[Literal["online","offline"]] = Field(
        description="'online'=immediate during execution, 'offline'=scheduled updates",
        default="offline",
    )
    kv_cache: Optional[bool] = Field(
        default=False,
        description="If True, enable precompute KV cache; otherwise disable it."
    )
    kv_cache_path: Optional[str] = Field(
        description="Path to the KV cache base",
        default=os.path.join(lightmem_dir, "kv_cache.db"),
    )
    graph_mem: Optional[bool] = Field(
        default=False,
        description="If True, enable topic segmentation; otherwise disable it."
    )
    version: Optional[str] = Field(
        description="The version of the API",
        default="v1.1",
    )
    logging: Optional[LoggingConfig] = Field(
        default=None,
        description="Logging configuration for the LightMem system."
    )
    extraction_mode: Optional[Literal["flat", "event"]] = Field(
        default="flat",
        description="Memory extraction mode:\n"
                    "- 'flat': Extract factual entries only (independent units)\n"
                    "- 'event': Extract event-level structure (factual + relational, temporally bound)"
    )

    @model_validator(mode="after")
    def infer_and_validate_retrieval_strategy(self):
        """Infer a retrieval strategy and validate explicit configuration."""
        if "retrieve_strategy" not in self.model_fields_set:
            # Only an omitted field is inferred. An explicit
            # retrieve_strategy=None keeps meaning "no retrieval", exactly as
            # it did before this field became optional.
            #
            # An omitted strategy never resolves to "hybrid": a configuration
            # that supplies both retrievers resolved to "embedding" before this
            # change, and keeps resolving to it. Hybrid retrieval stays opt-in
            # through an explicit retrieve_strategy="hybrid".
            if self.embedding_retriever:
                self.retrieve_strategy = "embedding"
            elif self.context_retriever:
                self.retrieve_strategy = "context"
            return self
        if self.retrieve_strategy is None:
            return self
        needs_context = self.retrieve_strategy in {"context", "hybrid"}
        needs_embedding = self.retrieve_strategy in {"embedding", "hybrid"}
        if needs_context and not self.context_retriever:
            raise ValueError(
                "context_retriever is required for context or hybrid retrieval"
            )
        if needs_embedding and not self.embedding_retriever:
            raise ValueError(
                "embedding_retriever is required for embedding or hybrid retrieval"
            )
        return self
