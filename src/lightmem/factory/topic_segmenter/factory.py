from typing import Dict, Optional
from importlib import import_module
from lightmem.configs.topic_segmenter.base import TopicSegmenterConfig

class TopicSegmenterFactory:
    _MODEL_MAPPING: Dict[str, str] = {
        "llmlingua-2": "lightmem.factory.topic_segmenter.llmlingua_2.LlmLingua2Segmenter",
    }

    @classmethod
    def from_config(cls, config: TopicSegmenterConfig, shared: bool, compressor: None):
        """
        Instantiate a topic segmenter by dynamically importing the class based on config.
        
        Args:
            config: TopicSegmenterConfig containing model name and specific configs
            
        Returns:
            An instance of the requested segmenter model
            
        Raises:
            ValueError: If model name is not supported or instantiation fails
            ImportError: If the module or class cannot be imported
        """
        model_name = config.model_name
        
        if model_name not in cls._MODEL_MAPPING:
            raise ValueError(
                f"Unsupported segmenter model: {model_name}. "
                f"Supported models are: {list(cls._MODEL_MAPPING.keys())}"
            )

        class_path = cls._MODEL_MAPPING[model_name]
        
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = import_module(module_path)
            segmenter_class = getattr(module, class_name)
            if config.configs is None:
                return segmenter_class()
            else:
                return segmenter_class(config=config.configs, shared = shared, compressor = compressor)
            
        except ImportError as e:
            raise ImportError(
                f"Could not import segmenter class '{class_path}': {str(e)}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Maybe class '{class_name}' not found in module '{module_path}': {str(e)}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate {model_name} segmenter: {str(e)}"
            ) from e