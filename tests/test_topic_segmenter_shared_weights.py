import pytest

torch = pytest.importorskip("torch")

from lightmem.factory.topic_segmenter.llmlingua_2 import LlmLingua2Segmenter


def test_shared_without_compressor_raises_clear_error():
    """shared=True has no weights to share when no compressor is supplied."""
    with pytest.raises(ValueError, match="precomp_topic_shared=True requires a pre-compressor"):
        LlmLingua2Segmenter(config={"model_name": "dummy"}, shared=True, compressor=None)


def test_shared_with_compressor_reuses_its_weights():
    """The sharing path must still adopt the compressor's model and tokenizer."""

    class FakeModelConfig:
        max_position_embeddings = 128

    class FakeModel:
        config = FakeModelConfig()

    class FakeInner:
        model = FakeModel()
        tokenizer = object()

    class FakeCompressor:
        inner_compressor = FakeInner()

    compressor = FakeCompressor()
    segmenter = LlmLingua2Segmenter(config={"model_name": "dummy"}, shared=True, compressor=compressor)

    assert segmenter.model is compressor.inner_compressor.model
    assert segmenter.tokenizer is compressor.inner_compressor.tokenizer
    assert segmenter.buffer_len == 128
