"""User-scoped bounded cache of query vectors with duplicate work suppression."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future

import numpy as np


class QueryCache:
    def __init__(self, max_bytes: int = 64 * 1024 * 1024, ttl: float = 300,
                 max_entries: int = 128, clock=time.monotonic):
        if max_bytes < 0 or ttl < 0 or max_entries < 0 or not np.isfinite(ttl):
            raise ValueError('invalid query cache limits')
        self.max_bytes, self.ttl, self.max_entries = max_bytes, ttl, max_entries
        self.clock = clock
        self.lock = threading.Lock()
        self.entries = OrderedDict()
        self.pending = {}
        self.bytes = 0
        self.closed = False

    @staticmethod
    def key(signature, user_id, parts):
        payload = json.dumps([signature, user_id, parts], ensure_ascii=False,
                             sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode()).digest()

    def get_or_encode(self, signature, user_id, parts, encode):
        if not self.max_bytes or not self.ttl or not self.max_entries:
            return encode(parts)
        key = self.key(signature, user_id, parts)
        with self.lock:
            if self.closed:
                raise RuntimeError('query cache is closed')
            now = self.clock()
            for old_key, (_, _, expires, size) in list(self.entries.items()):
                if expires <= now:
                    self.entries.pop(old_key)
                    self.bytes -= size
            if key in self.entries:
                matrix, types, _, _ = self.entries[key]
                self.entries.move_to_end(key)
                return matrix, types
            future = self.pending.get(key)
            owner = future is None
            if owner:
                future = Future()
                self.pending[key] = future
        if not owner:
            return future.result()
        try:
            matrix, types = encode(parts)
            matrix = np.array(matrix, copy=True)
            matrix.setflags(write=False)
            types = tuple(types)
            # Include array, modality strings, digest and conservative object overhead.
            size = matrix.nbytes + sum(len(kind) * 4 + 64 for kind in types) + 512
            result = (matrix, types)
            with self.lock:
                if not self.closed and size <= self.max_bytes:
                    while self.entries and (self.bytes + size > self.max_bytes or len(self.entries) >= self.max_entries):
                        _, (_, _, _, evicted_size) = self.entries.popitem(last=False)
                        self.bytes -= evicted_size
                    self.entries[key] = (matrix, types, self.clock() + self.ttl, size)
                    self.bytes += size
                future.set_result(result)
                self.pending.pop(key, None)
            return result
        except BaseException as exc:
            with self.lock:
                future.set_exception(exc)
                self.pending.pop(key, None)
            raise

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.bytes = 0

    def close(self):
        with self.lock:
            self.closed = True
            self.entries.clear()
            self.bytes = 0
