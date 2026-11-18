"""Regression coverage for optional LightMemory retrieval configuration."""

from unittest.mock import Mock

import pytest

from lightmem.configs.base import BaseMemoryConfigs
from lightmem.configs.retriever.contextretriever.base import ContextRetrieverConfig
from lightmem.configs.retriever.embeddingretriever.base import EmbeddingRetrieverConfig
from lightmem.memory.lightmem import LightMemory


def _context_config():
    return ContextRetrieverConfig.model_construct(model_name="BM25", configs={})


def _embedding_config():
    return EmbeddingRetrieverConfig.model_construct(
        model_name="qdrant", configs={"path": "/tmp/qdrant"}
    )


def _stub_lightmem_dependencies(monkeypatch, lightmem_module):
    """Stub everything LightMemory.__init__ builds except the retrievers."""
    manager = type("Manager", (), {"config": type("Config", (), {"model": "test"})()})()
    monkeypatch.setattr(
        lightmem_module.MemoryManagerFactory,
        "from_config",
        lambda _config: manager,
    )
    monkeypatch.setattr(
        lightmem_module, "ShortMemBufferManager", lambda **_kwargs: object()
    )
    context_factory = Mock()
    embedding_factory = Mock()
    monkeypatch.setattr(
        lightmem_module.ContextRetrieverFactory, "from_config", context_factory
    )
    monkeypatch.setattr(
        lightmem_module.EmbeddingRetrieverFactory, "from_config", embedding_factory
    )
    return context_factory, embedding_factory


def test_minimal_config_has_no_retriever_strategy():
    assert BaseMemoryConfigs().retrieve_strategy is None


def test_lightmemory_constructs_without_retrievers(monkeypatch):
    """Minimal construction avoids optional retriever factories."""
    import lightmem.memory.lightmem as lightmem_module

    context_factory, embedding_factory = _stub_lightmem_dependencies(
        monkeypatch, lightmem_module
    )

    memory = LightMemory(BaseMemoryConfigs())

    assert memory.retrieve_strategy is None
    assert not hasattr(memory, "context_retriever")
    assert not hasattr(memory, "embedding_retriever")
    context_factory.assert_not_called()
    embedding_factory.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "strategy"),
    [
        ({"context_retriever": _context_config()}, "context"),
        ({"embedding_retriever": _embedding_config()}, "embedding"),
        (
            {
                "context_retriever": _context_config(),
                "embedding_retriever": _embedding_config(),
            },
            "embedding",
        ),
    ],
)
def test_retriever_configs_infer_strategy(kwargs, strategy):
    assert BaseMemoryConfigs(**kwargs).retrieve_strategy == strategy


def test_both_retrievers_without_strategy_still_build_embedding_only(monkeypatch):
    """Omitting the strategy must not turn a working config into hybrid.

    Such a configuration resolved to 'embedding' before retrieve_strategy became
    optional, and built an embedding-only LightMemory. Inferring 'hybrid' here
    would call ContextRetrieverFactory and change that.
    """
    import lightmem.memory.lightmem as lightmem_module

    context_factory, embedding_factory = _stub_lightmem_dependencies(
        monkeypatch, lightmem_module
    )

    memory = LightMemory(
        BaseMemoryConfigs(
            context_retriever=_context_config(),
            embedding_retriever=_embedding_config(),
        )
    )

    assert memory.retrieve_strategy == "embedding"
    context_factory.assert_not_called()
    embedding_factory.assert_called_once()
    assert not hasattr(memory, "context_retriever")


def test_explicit_hybrid_still_builds_both_retrievers(monkeypatch):
    """Hybrid remains reachable, but only when it is asked for explicitly."""
    import lightmem.memory.lightmem as lightmem_module

    context_factory, embedding_factory = _stub_lightmem_dependencies(
        monkeypatch, lightmem_module
    )

    memory = LightMemory(
        BaseMemoryConfigs(
            retrieve_strategy="hybrid",
            context_retriever=_context_config(),
            embedding_retriever=_embedding_config(),
        )
    )

    assert memory.retrieve_strategy == "hybrid"
    context_factory.assert_called_once()
    embedding_factory.assert_called_once()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"context_retriever": _context_config()},
        {"embedding_retriever": _embedding_config()},
        {
            "context_retriever": _context_config(),
            "embedding_retriever": _embedding_config(),
        },
    ],
)
def test_explicit_none_strategy_is_not_inferred(kwargs):
    """An explicit None keeps meaning "no retrieval", as it did before.

    Only an omitted field is inferred from the retriever configs; a caller who
    spells the absence out gets the strategy they asked for.
    """
    config = BaseMemoryConfigs(retrieve_strategy=None, **kwargs)

    assert config.retrieve_strategy is None


def test_explicit_none_with_context_retriever_still_constructs(monkeypatch):
    """The retrieval-less instance an explicit None built before still builds.

    Before retrieve_strategy became optional, an explicit None was the only way
    to get a LightMemory with no retriever, and such a config could still be
    carrying a context_retriever from when BM25 existed. Inferring 'context'
    for it would send it into ContextRetrieverFactory, which constructs for no
    configuration on this commit.
    """
    import lightmem.memory.lightmem as lightmem_module

    context_factory, embedding_factory = _stub_lightmem_dependencies(
        monkeypatch, lightmem_module
    )

    memory = LightMemory(
        BaseMemoryConfigs(
            retrieve_strategy=None,
            context_retriever=_context_config(),
        )
    )

    assert memory.retrieve_strategy is None
    assert not hasattr(memory, "context_retriever")
    assert not hasattr(memory, "embedding_retriever")
    context_factory.assert_not_called()
    embedding_factory.assert_not_called()


@pytest.mark.parametrize("strategy", ["context", "embedding", "hybrid"])
def test_explicit_strategy_requires_its_retriever_configs(strategy):
    with pytest.raises(ValueError):
        BaseMemoryConfigs(retrieve_strategy=strategy)
