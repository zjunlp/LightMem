import importlib
from collections import OrderedDict
from collections.abc import Iterator
from typing import (
    Any, 
    Type, 
    Union,
)

# Mapping of memory layer types to their class names
MEMORY_LAYERS_MAPPING_NAMES = OrderedDict[str, str](
    [
        ("A-MEM", "AMEMLayer"),
        ("LangMem", "LangMemLayer"),
        ("MemZero", "MemZeroLayer"),
        ("MemZeroGraph", "MemZeroLayer"),
        ("FullContext", "FullContextLayer"),
        ("NaiveRAG", "NaiveRAGLayer"),
    ]
)

# Mapping of memory config types to their class names
CONFIG_MAPPING_NAMES = OrderedDict[str, str](
    [
        ("A-MEM", "AMEMConfig"),
        ("LangMem", "LangMemConfig"),
        ("MemZero", "MemZeroConfig"),
        ("MemZeroGraph", "MemZeroConfig"),
        ("FullContext", "FullContextConfig"),
        ("NaiveRAG", "NaiveRAGConfig"),
    ]
)

# Mapping of dataset types to their class names
DATASET_MAPPING_NAMES = OrderedDict[str, str](
    [
        ("LongMemEval", "LongMemEval"),
        ("LoCoMo", "LoCoMo"),
        ("MobileBench", "MobileBench"),
    ]
)

def type_to_module_name(key: str, mapping_type: str) -> str:
    """
    Converts a type key to the corresponding module path.

    Parameters
    ----------
    key : str
        The type key (e.g., "A-MEM", "LongMemEval")
    mapping_type : str
        The type of mapping ("layer", "config", or "dataset")

    Returns
    -------
    str
        The module path relative to memories package
    """
    match mapping_type:
        case "layer" | "config":
            match key:
                case "A-MEM":
                    return "layers.amem"
                case "LangMem":
                    return "layers.langmem"
                case "MemZero":
                    return "layers.memzero"
                case "MemZeroGraph":
                    return "layers.memzero"
                case "NaiveRAG":
                    return "layers.naive_rag"
                case "FullContext":
                    return "layers.full_context"
        case "dataset":
            match key:
                case "LongMemEval":
                    return "datasets.longmemeval"
                case "LoCoMo":
                    return "datasets.locomo"
                case "MobileBench":
                    return "datasets.mobilebench"
    # Default: convert key to module name
    return key.lower().replace("-", "_")

class _LazyMapping(OrderedDict):
    """
    A dictionary that lazily loads its values when they are requested.
    Inspired by [Hugging Face Transformers' lazy loading mechanism](https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/auto/configuration_auto.py).
    """
    
    def __init__(self, mapping: OrderedDict[str, str], mapping_type: str) -> None:
        """Initialize the lazy mapping."""
        self._mapping = mapping
        self._mapping_type = mapping_type
        self._extra_content = {}
        self._modules = {}
    
    def __getitem__(self, key: str) -> Type[Any]:
        """Lazily load and return the requested class."""
        if key in self._extra_content:
            return self._extra_content[key]
        
        if key not in self._mapping:
            raise KeyError(
                f"'{key}' not found. Available keys: {list(self._mapping.keys())}"
            )
        
        class_name = self._mapping[key]
        module_name = type_to_module_name(key, self._mapping_type)
        
        # Cache the module if not already loaded
        if module_name not in self._modules:
            try:
                self._modules[module_name] = importlib.import_module(
                    f".{module_name}", 
                    "memories"
                )
            except ImportError as e:
                raise ImportError(
                    f"Failed to import {module_name} for {key}: {e}"
                )
        
        # Get the class from the module
        if hasattr(self._modules[module_name], class_name):
            return getattr(self._modules[module_name], class_name)
        
        raise AttributeError(
            f"Module {module_name} does not have class {class_name}"
        )
    
    def keys(self) -> list[str]:
        """Return all available keys."""
        return list(self._mapping.keys()) + list(self._extra_content.keys())
    
    def values(self) -> list[Type[Any]]:
        """Return all values (loads them if necessary)."""
        return [self[k] for k in self.keys()]
    
    def items(self) -> list[tuple[str, Type[Any]]]:
        """Return all key-value pairs."""
        return [(k, self[k]) for k in self.keys()]
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self.keys())
    
    def __contains__(self, item: object) -> bool:
        """Check if a key exists in the mapping."""
        return item in self._mapping or item in self._extra_content
    
    def __len__(self) -> int:
        """Return the number of items in the mapping."""
        return len(self._mapping) + len(self._extra_content)
    
    def register(self, key: str, value: Type[Any], exist_ok: bool = False) -> None:
        """
        Register a new class in this mapping.
        
        Parameters
        ----------
        key : str
            The key to register the class under
        value : Type[Any]
            The class to register
        exist_ok : bool
            If True, allows overwriting existing keys
        
        Raises
        ------
        ValueError
            If the key already exists and exist_ok is False
        """
        if key in self._mapping and not exist_ok:
            raise ValueError(
                f"'{key}' is already registered in {self._mapping_type} mapping. "
                f"Use exist_ok=True to overwrite."
            )
        self._extra_content[key] = value
    
    def get(self, key: str, default: Any = None) -> Union[Type[Any], Any]:
        """Get a value with a default fallback."""
        try:
            return self[key]
        except (KeyError, ImportError, AttributeError):
            return default

MEMORY_LAYERS_MAPPING = _LazyMapping(MEMORY_LAYERS_MAPPING_NAMES, "layer")
CONFIG_MAPPING = _LazyMapping(CONFIG_MAPPING_NAMES, "config")
DATASET_MAPPING = _LazyMapping(DATASET_MAPPING_NAMES, "dataset")

# Export public API
__all__ = [
    "CONFIG_MAPPING",
    "MEMORY_LAYERS_MAPPING", 
    "DATASET_MAPPING",
]