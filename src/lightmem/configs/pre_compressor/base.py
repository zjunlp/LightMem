from pydantic import BaseModel, Field, model_validator
from typing import Dict, Optional, Type, Any, Union, ClassVar
from importlib import import_module

class PreCompressorConfig(BaseModel):
    model_name: str = Field(
        description="The Compressor model or Deployment platform (e.g., 'openai', 'deepseek', 'ollama', 'llmlingua-2')",
        default="llmlingua-2",
    )

    _model_configs: ClassVar[Dict[str, str]] = {
        "llmlingua-2": "lightmem.configs.pre_compressor.llmlingua_2.LlmLingua2Config",  
        "entropy_compress": "lightmem.configs.pre_compressor.entropy_compress.EntropyCompressor"
    }

    configs: Dict[str, Any] = Field(
        description="Configuration for the specific Compressor model", 
        default=None
    )

    @model_validator(mode='before')
    def validate_model_name(cls, values):
        default_model = cls.__pydantic_fields__["model_name"].default
        model_name = values.get("model_name", default_model)

        if model_name not in cls._model_configs:
            raise ValueError(f"Unsupported model: {model_name}.")

        values["model_name"] = model_name
        return values

    @model_validator(mode='after')
    def load_config_class(self) -> 'PreCompressorConfig':
        config_path = self._model_configs[self.model_name]
        module_path, class_name = config_path.rsplit('.', 1)
        
        try:
            module = import_module(module_path)
            config_class = getattr(module, class_name)
            if self.configs is None:
                self.configs = config_class()  
                print(self.configs)
            elif isinstance(self.configs, Dict):
                self.configs = config_class(**self.configs)
            print(f"pre_compressor:{self.model_name}")
            print(f"pre_compressor:{self.configs}")
                
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not load config class '{config_path}': {e}")
        return self