from importlib import import_module
from typing import ClassVar, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class ContextRetrieverConfig(BaseModel):
    model_name: str = Field(description="The Context retriever or algorithm (e.g., 'BM25')", default="BM25")

    configs: Optional[Dict] = Field(description="Configuration for the context retriever or algorithm", default={})

    _model_list: ClassVar[Dict[str, str]] = {
        "BM25": "lightmem.configs.retriever.bm25.BM25Config",
    }

    @model_validator(mode="before")
    def validate_model_name(cls, values):
        field_info = cls.model_fields.get("model_name")
        default_val = field_info.default if field_info else None
        model_name = values.get("model_name", default_val)
        if model_name not in cls._model_list:
            raise ValueError(
                f"Unsupported model: {model_name}."
            )
        values["model_name"] = model_name
        return values

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "ContextRetrieverConfig":
        config_path = self._model_list[self.model_name]
        module_path, class_name = config_path.rsplit(".", 1)
        module = import_module(module_path)
        config_class = getattr(module, class_name)

        if self.configs is None:
            self.configs = config_class()
        elif isinstance(self.configs, Dict):
            self.configs = config_class(**self.configs)
        return self
