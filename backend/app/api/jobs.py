"""In-memory background jobs for videos (a video can take minutes; HTTP requests must not block).

One worker thread: the models are shared, so jobs run one after another (queued jobs show "queued").
Phase 3 moves job records into PostgreSQL; the interface stays the same.
"""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    filename: str
    status: str = "queued"          # queued | running | done | failed
    progress: float = 0.0
    run_id: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-job")

    def submit(self, path: Path, work: Callable[[Path, Callable[[float], None]], dict]) -> Job:
        job = Job(uuid.uuid4().hex[:12], path.name)
        with self._lock:
            self._jobs[job.job_id] = job

        def run():
            job.status = "running"
            try:
                report = work(path, lambda p: setattr(job, "progress", round(p, 3)))
                job.run_id, job.status, job.progress = report["run_id"], "done", 1.0
            except Exception as e:                     # the job fails, the server keeps running
                log.exception("job %s failed", job.job_id)
                job.status, job.error = "failed", str(e)

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
