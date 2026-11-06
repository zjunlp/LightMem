import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from service.multimodal_cache import QueryCache


PARTS = [{'type': 'text', 'text': 'same query'}]


def test_identity_user_scope_and_immutable_results():
    cache = QueryCache()
    calls = []
    def encode(parts):
        calls.append(parts)
        return np.array([[1., 0.]]), ['text']
    first = cache.get_or_encode('model', 'alice', PARTS, encode)
    assert cache.get_or_encode('model', 'alice', PARTS, encode) is not None
    assert len(calls) == 1
    cache.get_or_encode('model', 'bob', PARTS, encode)
    cache.get_or_encode('other-model', 'alice', PARTS, encode)
    assert len(calls) == 3
    with pytest.raises(ValueError):
        first[0][0, 0] = 2
    assert first[1] == ('text',)
    assert all(isinstance(k, bytes) and len(k) == 32 for k in cache.entries)


def test_concurrent_singleflight_and_shared_failure_then_retry():
    cache = QueryCache()
    started = threading.Event()
    release = threading.Event()
    calls = []
    def encode(parts):
        calls.append(1)
        started.set()
        assert release.wait(3)
        return np.array([[1., 0.]]), ['text']
    with ThreadPoolExecutor(max_workers=6) as pool:
        owner = pool.submit(cache.get_or_encode, 'm', 'u', PARTS, encode)
        assert started.wait(3)
        others = [pool.submit(cache.get_or_encode, 'm', 'u', PARTS, encode) for _ in range(5)]
        release.set()
        results = [f.result(timeout=3) for f in [owner] + others]
    assert len(calls) == 1
    assert all(r[0] is results[0][0] for r in results)
    cache.clear()
    def fail(parts):
        raise RuntimeError('failed')
    with pytest.raises(RuntimeError, match='failed'):
        cache.get_or_encode('m', 'u', PARTS, fail)
    assert not cache.entries and not cache.pending
    cache.get_or_encode('m', 'u', PARTS, encode)
    assert len(calls) == 2


def test_expiry_lru_memory_budget_oversize_and_disable():
    now = [0.]
    cache = QueryCache(max_entries=2, clock=lambda: now[0], ttl=10)
    calls = []
    def encode(parts):
        calls.append(1)
        return np.array([[1., 0.]]), ['text']
    for user in ['a', 'b', 'a', 'c', 'a']:
        cache.get_or_encode('m', user, PARTS, encode)
    assert len(calls) == 3
    cache.get_or_encode('m', 'b', PARTS, encode)
    assert len(calls) == 4
    now[0] = 11
    cache.get_or_encode('m', 'a', PARTS, encode)
    assert len(calls) == 5 and len(cache.entries) == 1
    # One entry fits, two entries must evict by estimated memory.
    cache = QueryCache(max_bytes=800)
    for user in ['a', 'b']:
        cache.get_or_encode('m', user, PARTS, encode)
    assert len(cache.entries) == 1 and cache.bytes <= 800
    for cache in [QueryCache(max_bytes=1), QueryCache(max_bytes=0)]:
        cache.get_or_encode('m', 'a', PARTS, encode)
        cache.get_or_encode('m', 'a', PARTS, encode)
        assert not cache.entries
    cache.close()
    assert cache.bytes == 0


def test_failures_shared_by_waiters_not_cached():
    cache = QueryCache()
    started, release = threading.Event(), threading.Event()
    def encode(parts):
        started.set()
        assert release.wait(3)
        raise RuntimeError('503 unavailable')
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.get_or_encode, 'm', 'u', PARTS, encode)
        assert started.wait(3)
        second = pool.submit(cache.get_or_encode, 'm', 'u', PARTS, encode)
        # Wait until the second is waiting on the same future, without timing assumptions.
        # ThreadPoolExecutor executes it promptly; either failure path is equivalent.
        release.set()
        for future in [first, second]:
            with pytest.raises(RuntimeError, match='503 unavailable'):
                future.result(timeout=3)
    assert not cache.entries and not cache.pending
