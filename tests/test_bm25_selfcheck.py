from lightmem.factory.retriever.contextretriever.bm25 import BM25
from lightmem.factory.retriever.contextretriever.factory import ContextRetrieverFactory
from lightmem.configs.retriever.bm25 import BM25Config

def _mk_corpus():
    return [
        "I love machine learning and neural networks",
        "Apples are tasty and healthy",
        "BM25 is a lexical retrieval baseline",
        "Neural retrievers use embeddings for semantic similarity",
        "I like apples and oranges",
    ]

def test_index_then_retrieve_basic():
    bm = BM25()
    corpus = _mk_corpus()
    bm.index(corpus)
    out = bm.retrieve("machine learning", top_k=3)
    assert len(out) == 3
    scores = [s for _, s in out]
    assert all(s >= 0 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert out[0][0] == "I love machine learning and neural networks"

def test_unseen_terms_not_crash():
    bm = BM25()
    bm.index(_mk_corpus())
    out = bm.retrieve("quantum entanglement", top_k=2)
    assert len(out) == 2

def test_topk_clip():
    bm = BM25()
    c = _mk_corpus()
    bm.index(c)
    out = bm.retrieve("apples", top_k=len(c) + 10)
    assert len(out) == len(c)

def test_index_required_error():
    bm = BM25()
    try:
        bm.retrieve("anything")
        assert False, "should raise before index()"
    except RuntimeError:
        pass

def test_factory_from_config_default():
    cfg = BM25Config()
    bm = ContextRetrieverFactory.from_config(cfg)
    bm.index(_mk_corpus())
    out = bm.retrieve("lexical retrieval", top_k=2)
    assert len(out) == 2

def test_param_sensitivity():
    c = _mk_corpus()
    bm1 = BM25(k1=1.2, b=0.5); bm1.index(c)
    bm2 = BM25(k1=2.0, b=0.9); bm2.index(c)
    r1 = bm1.retrieve("neural embeddings", top_k=3)
    r2 = bm2.retrieve("neural embeddings", top_k=3)
    assert len(r1) == 3 and len(r2) == 3
    assert all(isinstance(x[0], str) and isinstance(x[1], float) for x in r1 + r2)

def test_stability_same_query_twice():
    bm = BM25()
    c = _mk_corpus()
    bm.index(c)
    r1 = bm.retrieve("apples", top_k=4)
    r2 = bm.retrieve("apples", top_k=4)
    assert r1 == r2

def test_config_validation_bounds():
    _ = BM25Config(k1=1.0, b=0.7)
    try:
        _ = BM25Config(b=1.5)
        assert False, "b>1 should raise"
    except ValueError:
        pass
    try:
        _ = BM25Config(k1=-0.1)
        assert False, "k1<=0 should raise"
    except ValueError:
        pass

if __name__ == "__main__":
    test_index_then_retrieve_basic()
    test_unseen_terms_not_crash()
    test_topk_clip()
    test_index_required_error()
    test_factory_from_config_default()
    test_param_sensitivity()
    test_stability_same_query_twice()
    test_config_validation_bounds()
    print("ALL TESTS PASSED ✅")
