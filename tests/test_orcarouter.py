import pytest

from lightmem.configs.memory_manager.base import MemoryManagerConfig
from lightmem.factory.memory_manager.factory import MemoryManagerFactory
from lightmem.factory.memory_manager.orcarouter import OrcaRouterManager


def test_orcarouter_is_accepted_model_name():
    cfg = MemoryManagerConfig(
        model_name="orcarouter", configs={"api_key": "sk-orca-test"}
    )
    assert cfg.model_name == "orcarouter"


def test_orcarouter_factory_instantiates():
    cfg = MemoryManagerConfig(
        model_name="orcarouter",
        configs={"api_key": "sk-orca-test"},
    )
    manager = MemoryManagerFactory.from_config(cfg)
    assert isinstance(manager, OrcaRouterManager)


def test_orcarouter_defaults():
    cfg = MemoryManagerConfig(
        model_name="orcarouter",
        configs={"api_key": "sk-orca-test"},
    )
    manager = MemoryManagerFactory.from_config(cfg)
    assert manager.config.model == "orcarouter/auto"
    assert manager.base_url == "https://api.orcarouter.ai/v1"


def test_orcarouter_custom_base_url():
    cfg = MemoryManagerConfig(
        model_name="orcarouter",
        configs={
            "api_key": "sk-orca-test",
            "orcarouter_base_url": "https://custom.example/v1",
        },
    )
    manager = MemoryManagerFactory.from_config(cfg)
    assert manager.base_url == "https://custom.example/v1"


def test_orcarouter_requires_api_key(monkeypatch):
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    cfg = MemoryManagerConfig(model_name="orcarouter")
    with pytest.raises(ValueError):
        MemoryManagerFactory.from_config(cfg)
