from typing import Dict, Optional
from importlib import import_module
from lightmem.configs.retriever.embeddingretriever.base import EmbeddingRetrieverConfig

class EmbeddingRetrieverFactory:
    _MODEL_MAPPING: Dict[str, str] = {
        "qdrant": "lightmem.factory.retriever.embeddingretriever.qdrant.Qdrant",
    }

    @classmethod
    def from_config(cls, config: EmbeddingRetrieverConfig):
        """
        Instantiate a compressor by dynamically importing the class based on config.
        
        Args:
            config: PreCompressorConfig containing model name and specific configs
            
        Returns:
            An instance of the requested compressor model
            
        Raises:
            ValueError: If model name is not supported or instantiation fails
            ImportError: If the module or class cannot be imported
        """
        model_name = config.model_name
        
        if model_name not in cls._MODEL_MAPPING:
            raise ValueError(
                f"Unsupported compressor model: {model_name}. "
                f"Supported models are: {list(cls._MODEL_MAPPING.keys())}"
            )

        class_path = cls._MODEL_MAPPING[model_name]
        
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = import_module(module_path)
            compressor_class = getattr(module, class_name)
            if config.configs is None:
                return compressor_class()
            else:
                return compressor_class(config=config.configs)
            
        except ImportError as e:
            raise ImportError(
                f"Could not import compressor class '{class_path}': {str(e)}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Class '{class_name}' not found in module '{module_path}': {str(e)}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate {model_name} compressor: {str(e)}"
            ) from e