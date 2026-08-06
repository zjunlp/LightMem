"""FluxMem: Rethinking Memory as Continuously Evolving Connectivity"""

from .agent import FluxMemAgent
from .config import FluxMemConfig
from .graph.memory_graph import MemoryGraph
from .graph.nodes import SemanticNode, EpisodicNode, ProceduralNode, NodeType
from .interfaces.llm import BaseLLM, OpenAILLM
from .interfaces.embedder import BaseEmbedder, OpenAIEmbedder
from .interfaces.vectorstore import BaseVectorStore, FAISSVectorStore
from .stages import StageI, StageII, StageIII
from .metrics import PEMSCalculator

__version__ = "0.1.0"