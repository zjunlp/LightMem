"""Idle-time retrieval preparation with an always-available synchronous fallback."""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager

from .retrieval import PreparedCorpus, prepare_corpus
from .store import RawMemoryStore


class _Stopped(Exception):
    pass


class BackgroundIndex:
    def __init__(
        self, store: RawMemoryStore, *, idle_seconds: float = 0.25,
        max_bytes: int = 512 * 1024 * 1024, max_users: int = 32,
        ttl_seconds: float = 300,
    ) -> None:
        if (
            not math.isfinite(idle_seconds) or idle_seconds < 0
            or max_bytes < 1 or max_users < 1
            or not math.isfinite(ttl_seconds) or ttl_seconds <= 0
        ):
            raise ValueError("Invalid background index resource limits")
        self.store = store
        self.idle_seconds = idle_seconds
        self.max_bytes = max_bytes
        self.max_users = max_users
        self.ttl_seconds = ttl_seconds
        self._condition = threading.Condition()
        self._pending: OrderedDict[str, None] = OrderedDict()
        self._cache: OrderedDict[str, tuple[PreparedCorpus, float]] = OrderedDict()
        self._oversized: OrderedDict[str, tuple[tuple[int, int], float]] = OrderedDict()
        self._active = 0
        self._last_activity = time.monotonic()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self.builds = 0
        self.failures = 0

    def start(self) -> None:
        with self._condition:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._worker, name="memory-index", daemon=True,
                )
                self._thread.start()

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._pending.clear()
            self._cache.clear()
            self._oversized.clear()
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @contextmanager
    def foreground(self):
        with self._condition:
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._last_activity = time.monotonic()
                self._condition.notify_all()

    def schedule(self, user_id: str) -> None:
        with self._condition:
            if self._stopped:
                return
            self._pending[user_id] = None
            self._pending.move_to_end(user_id)
            while len(self._pending) > 256:
                self._pending.popitem(last=False)
            self._condition.notify_all()

    def _expire(self) -> None:
        now = time.monotonic()
        for user in list(self._cache):
            if now - self._cache[user][1] >= self.ttl_seconds:
                del self._cache[user]
        for user in list(self._oversized):
            if now - self._oversized[user][1] >= self.ttl_seconds:
                del self._oversized[user]

    def get(self, user_id: str) -> PreparedCorpus | None:
        state = self.store.user_state(user_id)
        with self._condition:
            self._expire()
            entry = self._cache.get(user_id)
            if entry is not None and entry[0].state == state:
                self._cache[user_id] = (entry[0], time.monotonic())
                self._cache.move_to_end(user_id)
                return entry[0]
            if state[0] == 0:
                self._cache.pop(user_id, None)
                self._oversized.pop(user_id, None)
                return None
            if user_id in self._oversized and self._oversized[user_id][0] == state:
                return None
        self.schedule(user_id)
        return None

    def _checkpoint(self) -> None:
        with self._condition:
            while True:
                if self._stopped:
                    raise _Stopped()
                remaining = self.idle_seconds - (time.monotonic() - self._last_activity)
                if not self._active and remaining <= 0:
                    return
                self._condition.wait(max(0.01, remaining) if not self._active else None)

    def _build(self, user: str) -> None:
        state = self.store.user_state(user)
        if not state[0]:
            with self._condition:
                self._cache.pop(user, None)
            return
        with self._condition:
            entry = self._cache.get(user)
            prefix = entry[0] if entry else None
        if prefix is not None and prefix.state == state:
            return
        # Reuse token statistics for immutable old messages on append. Rebuild
        # after deletions; the synchronous Search path never uses stale state.
        if prefix is not None:
            delta = self.store.list_user(user, after_sequence=prefix.state[1])
            if len(prefix.memories) + len(delta) == state[0]:
                memories = list(prefix.memories) + delta
            else:
                prefix = None
        if prefix is None:
            memories = self.store.list_user(user)
        if len(memories) != state[0] or memories[-1].sequence != state[1]:
            self.schedule(user)
            return
        try:
            snapshot = prepare_corpus(
                memories, state=state, prefix=prefix, checkpoint=self._checkpoint,
                max_bytes=self.max_bytes,
            )
        except BufferError:
            with self._condition:
                self._oversized[user] = (state, time.monotonic())
                self._oversized.move_to_end(user)
                while len(self._oversized) > 256:
                    self._oversized.popitem(last=False)
            return
        if self.store.user_state(user) != state:
            self.schedule(user)
            return
        with self._condition:
            if self._stopped:
                return
            self._expire()
            self._oversized.pop(user, None)
            self._cache[user] = (snapshot, time.monotonic())
            self._cache.move_to_end(user)
            while (
                len(self._cache) > self.max_users
                or sum(item[0].estimated_bytes for item in self._cache.values()) > self.max_bytes
            ):
                self._cache.popitem(last=False)
            self.builds += 1
            self._condition.notify_all()

    def _worker(self) -> None:
        # Linux nice values apply per thread: lower only this worker's priority.
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        except (AttributeError, OSError):
            pass
        while True:
            try:
                with self._condition:
                    self._expire()
                    if self._stopped:
                        return
                    if not self._pending:
                        self._condition.wait(min(self.ttl_seconds, 30))
                        continue
                self._checkpoint()
                with self._condition:
                    if self._stopped:
                        return
                    user, _ = self._pending.popitem(last=False)
                self._build(user)
            except _Stopped:
                return
            except Exception as error:
                self.failures += 1
                logging.getLogger(__name__).warning(
                    "Background index failed; synchronous retrieval remains available (%s)",
                    type(error).__name__,
                )
