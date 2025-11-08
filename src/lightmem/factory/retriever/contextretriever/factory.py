from typing import Dict, Optional
from importlib import import_module
from lightmem.configs.retriever.contextretriever.base import ContextRetrieverConfig


class ContextRetrieverFactory:
    _MODEL_MAPPING: Dict[str, str] = {
        "BM25": "lightmem.factory.retriever.contextretriever.bm25.BM25",
    }

    @classmethod
    def from_config(cls, config: ContextRetrieverConfig):
        """
        Instantiate a retriever by dynamically importing the class based on config.
        
        Args:
            config: ContextRetrieverConfig or specific retriever config (e.g., BM25Config)
        
        Returns:
            An instance of the requested retriever model
        
        Raises:
            ValueError: If model name is not supported or instantiation fails
            ImportError: If the module or class cannot be imported
        """
    
        model_name = getattr(config, "model_name", "BM25")

        if model_name not in cls._MODEL_MAPPING:
            raise ValueError(
                f"Unsupported retriever model: {model_name}. "
                f"Supported models are: {list(cls._MODEL_MAPPING.keys())}"
            )

        class_path = cls._MODEL_MAPPING[model_name]
        
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = import_module(module_path)
            retriever_class = getattr(module, class_name)
            if hasattr(config, "configs") and config.configs is not None:
                return retriever_class(config=config.configs)
            else:
                return retriever_class()
            
        except ImportError as e:
            raise ImportError(
                f"Could not import retriever class '{class_path}': {str(e)}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Maybe class '{class_name}' not found in module '{module_path}': {str(e)}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate {model_name} retriever: {str(e)}"
            ) from e
