from types import SimpleNamespace
from unittest.mock import patch

from lightmem.configs.base import BaseMemoryConfigs
from lightmem.memory.lightmem import LightMemory


def test_topic_segmentation_initializes_without_precompressor():
    config = BaseMemoryConfigs(
        pre_compress=False,
        topic_segment=True,
        topic_segmenter={"model_name": "llmlingua-2", "configs": {}},
        index_strategy="embedding",
        text_embedder={"model_name": "huggingface", "configs": {}},
        retrieve_strategy=None,
    )
    segmenter = SimpleNamespace(buffer_len=16, tokenizer=object())
    manager = SimpleNamespace(
        tokenizer=object(),
        config=SimpleNamespace(model="gpt-4o-mini"),
    )

    with (
        patch(
            "lightmem.memory.lightmem.TopicSegmenterFactory.from_config",
            return_value=segmenter,
        ) as create_segmenter,
        patch(
            "lightmem.memory.lightmem.MemoryManagerFactory.from_config",
            return_value=manager,
        ),
        patch(
            "lightmem.memory.lightmem.TextEmbedderFactory.from_config",
            return_value=object(),
        ),
    ):
        memory = LightMemory(config)

    assert memory.compressor is None
    assert memory.segmenter is segmenter
    create_segmenter.assert_called_once_with(config.topic_segmenter, False, None)
