"""Serial job queue and server-sent event stream."""
from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from .settings import MAX_JOB_EVENTS, MAX_JOBS

PENDING = "pending"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    stage: Optional[str] = None
    events: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=MAX_JOB_EVENTS))
    seq: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "stage": self.stage,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": (
                (self.finished_at or time.time()) - self.started_at
                if self.started_at
                else None
            ),
            "error": self.error,
        }

    def detail(self) -> Dict[str, Any]:
        d = self.summary()
        d["params"] = self.params
        d["result"] = self.result
        d["traceback"] = self.traceback
        d["events"] = list(self.events)
        return d


class _LogBridge(logging.Handler):
    """Forwards LightMem log records onto the running job's event stream."""

    def __init__(self, manager: "JobManager"):
        super().__init__(level=logging.DEBUG)
        self.manager = manager

    def emit(self, record: logging.LogRecord) -> None:
        job = self.manager.current_job
        if job is None:
            return
        try:
            message = record.getMessage()
        except Exception:  # a broken format string should never kill a job
            message = str(record.msg)
        self.manager.emit(
            job,
            "log",
            {
                "level": record.levelname,
                "logger": record.name,
                "message": message,
                "ts": record.created,
            },
        )


class _StdoutBridge:
    """Mirror upstream stdout messages to the active job."""

    def __init__(self, manager: "JobManager", original):
        self.manager = manager
        self.original = original
        self._buffer = ""

    def write(self, text: str) -> int:
        self.original.write(text)
        job = self.manager.current_job
        if job is not None:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    self.manager.emit(
                        job,
                        "log",
                        {"level": "WARNING", "logger": "stdout", "message": line, "ts": time.time()},
                    )
        return len(text)

    def flush(self) -> None:
        self.original.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name):
        return getattr(self.original, name)


class JobManager:
    def __init__(self) -> None:
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._lock = threading.RLock()
        self._current: Optional[Job] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: Dict[str, Set["asyncio.Queue[Dict[str, Any]]"]] = {}
        self._worker = threading.Thread(target=self._run, name="lightmem-jobs", daemon=True)
        self._started = False
        self.bridge = _LogBridge(self)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        if not self._started:
            self._worker.start()
            self._started = True
        self.install_log_bridge()

    def install_log_bridge(self) -> None:
        """Attach handlers removed by LightMemory logging setup."""
        for name in ("LightMemory", "lightmem"):
            logger = logging.getLogger(name)
            if self.bridge not in logger.handlers:
                logger.addHandler(self.bridge)
        logging.getLogger("LightMemory").setLevel(logging.DEBUG)
        logging.getLogger("lightmem").setLevel(logging.DEBUG)

    @property
    def current_job(self) -> Optional[Job]:
        return self._current

    def emit(self, job: Job, kind: str, payload: Dict[str, Any]) -> None:
        job.seq += 1
        event = {"seq": job.seq, "kind": kind, **payload}
        job.events.append(event)
        loop = self._loop
        if loop is None:
            return
        with self._lock:
            subs = list(self._subscribers.get(job.id, ()))
        for q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:  # loop closed during shutdown
                pass

    def stage(self, job: Job, name: str, detail: Optional[Dict[str, Any]] = None) -> None:
        job.stage = name
        self.emit(job, "stage", {"stage": name, "detail": detail or {}})

    def subscribe(self, job_id: str) -> "asyncio.Queue[Dict[str, Any]]":
        q: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, set()).add(q)
        return q

    def unsubscribe(self, job_id: str, q: "asyncio.Queue[Dict[str, Any]]") -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if subs:
                subs.discard(q)
                if not subs:
                    self._subscribers.pop(job_id, None)

    def submit(
        self,
        job_type: str,
        fn: Callable[[Job], Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], type=job_type, params=params or {})
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_JOBS:
                old_id, old = self._jobs.popitem(last=False)
                self._subscribers.pop(old_id, None)
        self._queue.put((job, fn))
        self.emit(job, "status", {"status": job.status})
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.summary() for j in reversed(jobs)][:limit]

    def cancel(self, job_id: str) -> bool:
        """Only pending jobs can be cancelled; a running LightMem call cannot
        be interrupted safely."""
        job = self.get(job_id)
        if job is None or job.status != PENDING:
            return False
        job.status = CANCELLED
        job.finished_at = time.time()
        self.emit(job, "status", {"status": job.status})
        return True

    @property
    def busy(self) -> bool:
        return self._current is not None

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _run(self) -> None:
        original_stdout = sys.stdout
        sys.stdout = _StdoutBridge(self, original_stdout)  # type: ignore[assignment]
        while True:
            item = self._queue.get()
            if item is None:
                break
            job, fn = item
            if job.status == CANCELLED:
                continue
            self._current = job
            job.status = RUNNING
            job.started_at = time.time()
            self.emit(job, "status", {"status": job.status})
            try:
                job.result = fn(job)
                job.status = SUCCEEDED
            except Exception as exc:
                job.status = FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc()
                self.emit(job, "log", {"level": "ERROR", "logger": "web", "message": job.error, "ts": time.time()})
            finally:
                job.finished_at = time.time()
                self._current = None
                self.emit(
                    job,
                    "status",
                    {"status": job.status, "result": job.result, "error": job.error},
                )
                self.emit(job, "done", {"status": job.status})


manager = JobManager()
