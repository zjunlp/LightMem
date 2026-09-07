"""FluxMem: Rethinking Memory as Continuously Evolving Connectivity"""

from .agent import FluxMemAgent
from .config import FluxMemConfig
from .graph.memory_graph import MemoryGraph
from .graph.nodes import EpisodicNode, NodeType, ProceduralNode, SemanticNode
from .interfaces.embedder import BaseEmbedder, OpenAIEmbedder
from .interfaces.llm import BaseLLM, OpenAILLM
from .interfaces.vectorstore import BaseVectorStore, FAISSVectorStore
from .metrics import PEMSCalculator
from .stages import StageI, StageII, StageIII

__version__ = "0.1.0"
