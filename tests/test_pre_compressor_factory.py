from types import SimpleNamespace

from lightmem.configs.pre_compressor.base import PreCompressorConfig
from lightmem.configs.pre_compressor.entropy_compress import EntropyCompressorConfig
from lightmem.factory.pre_compressor.factory import PreCompressorFactory


def test_entropy_compressor_config_resolves_to_its_config_class():
    config = PreCompressorConfig(model_name="entropy_compress")

    assert isinstance(config.configs, EntropyCompressorConfig)


def test_entropy_compressor_factory_resolves_to_its_implementation(monkeypatch):
    compressor_config = object()
    imported = {}

    class FakeEntropyCompressor:
        def __init__(self, config=None):
            self.config = config

    def fake_import_module(module_path):
        imported["module_path"] = module_path
        return SimpleNamespace(EntropyCompressor=FakeEntropyCompressor)

    monkeypatch.setattr(
        "lightmem.factory.pre_compressor.factory.import_module",
        fake_import_module,
    )

    compressor = PreCompressorFactory.from_config(
        SimpleNamespace(model_name="entropy_compress", configs=compressor_config)
    )

    assert imported["module_path"] == "lightmem.factory.pre_compressor.entropy_compress"
    assert isinstance(compressor, FakeEntropyCompressor)
    assert compressor.config is compressor_config
