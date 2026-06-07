from lightmem.factory.memory_buffer.sensory_memory import SenMemBufferManager


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeSegmenter:
    def propose_cut(self, buffer_texts):
        return []


class FakeEmbedder:
    def embed(self, text):
        return [1.0, 0.0]


def test_oversized_single_user_message_is_consumed():
    manager = SenMemBufferManager(max_tokens=3, tokenizer=FakeTokenizer())
    messages = [
        {"role": "user", "content": "one two three four"},
        {"role": "assistant", "content": "ok"},
    ]

    segments = manager.add_messages(messages, FakeSegmenter(), FakeEmbedder())

    assert segments == [messages]
    assert manager.big_buffer == []
    assert manager.buffer == []
    assert manager.token_count == 0


def test_force_segment_flushes_remaining_buffer():
    manager = SenMemBufferManager(max_tokens=10, tokenizer=FakeTokenizer())
    manager.buffer = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    manager.token_count = 1

    segments = manager.cut_with_segmenter(FakeSegmenter(), FakeEmbedder(), force_segment=True)

    assert segments == [[
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]]
    assert manager.buffer == []
    assert manager.token_count == 0
