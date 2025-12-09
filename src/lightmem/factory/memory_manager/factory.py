from typing import Dict, Optional
from importlib import import_module
from lightmem.configs.memory_manager.base import MemoryManagerConfig


class MemoryManagerFactory:
    _MODEL_MAPPING: Dict[str, str] = {
        "deepseek": "lightmem.factory.memory_manager.deepseek.DeepseekManager",
        "openai": "lightmem.factory.memory_manager.openai.OpenaiManager",
        "transformers": "lightmem.factory.memory_manager.transformers.TransformersManager",
        "ollama": "lightmem.factory.memory_manager.ollama.OllamaManager",
        "vllm": "lightmem.factory.memory_manager.vllm.VllmManager",
        "vllm_offline": "lightmem.factory.memory_manager.vllm_offline.VllmOfflineManager",
    }

    @classmethod
    def from_config(cls, config: MemoryManagerConfig):
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
                f"Unsupported manager model: {model_name}. "
                f"Supported models are: {list(cls._MODEL_MAPPING.keys())}"
            )

        class_path = cls._MODEL_MAPPING[model_name]
        
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = import_module(module_path)
            manager_class = getattr(module, class_name)
            if config.configs is None:
                return manager_class()
            else:
                return manager_class(config=config.configs)
            
        except ImportError as e:
            raise ImportError(
                f"Could not import manager'{class_path}': {str(e)}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Maybe class '{class_name}' not found in module '{module_path}': {str(e)}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate {model_name} manager: {str(e)}"
            ) from e
