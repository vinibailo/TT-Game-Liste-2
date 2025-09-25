"""Background job tracking and orchestration utilities."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)


MAX_BACKGROUND_JOBS = 50


def _job_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


JOB_STATUS_PENDING = 'pending'
JOB_STATUS_RUNNING = 'running'
JOB_STATUS_SUCCESS = 'success'
JOB_STATUS_ERROR = 'error'
JOB_ACTIVE_STATUSES = {JOB_STATUS_PENDING, JOB_STATUS_RUNNING}
JOB_TERMINAL_STATUSES = {JOB_STATUS_SUCCESS, JOB_STATUS_ERROR}


@dataclass
class BackgroundJob:
    id: str
    job_type: str
    status: str = JOB_STATUS_PENDING
    message: str = ''
    progress_current: int = 0
    progress_total: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: str = ''
    updated_at: str = ''
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class BackgroundJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, BackgroundJob] = {}
        self._lock = Lock()

    def _serialize_job(self, job: BackgroundJob) -> dict[str, Any]:
        return {
            'id': job.id,
            'job_type': job.job_type,
            'status': job.status,
            'message': job.message,
            'progress_current': job.progress_current,
            'progress_total': job.progress_total,
            'data': dict(job.data),
            'result': dict(job.result),
            'error': job.error,
            'created_at': job.created_at,
            'updated_at': job.updated_at,
            'started_at': job.started_at,
            'finished_at': job.finished_at,
        }

    def _find_active_job_locked(self, job_type: str) -> Optional[BackgroundJob]:
        for job in self._jobs.values():
            if job.job_type == job_type and job.status in JOB_ACTIVE_STATUSES:
                return job
        return None

    def _prune_jobs_locked(self) -> None:
        if len(self._jobs) <= MAX_BACKGROUND_JOBS:
            return
        removable: list[BackgroundJob] = [
            job
            for job in self._jobs.values()
            if job.status in JOB_TERMINAL_STATUSES
        ]
        removable.sort(key=lambda j: j.finished_at or j.updated_at)
        while len(self._jobs) > MAX_BACKGROUND_JOBS and removable:
            victim = removable.pop(0)
            self._jobs.pop(victim.id, None)

    def list_jobs(self, job_type: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
            if job_type:
                jobs = [job for job in jobs if job.job_type == job_type]
            jobs.sort(key=lambda job: job.created_at)
            return [self._serialize_job(job) for job in jobs]

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return self._serialize_job(job)

    def get_active_job(self, job_type: str) -> Optional[dict[str, Any]]:
        with self._lock:
            job = self._find_active_job_locked(job_type)
            if job is None:
                return None
            return self._serialize_job(job)

    def start_job(
        self,
        job_type: str,
        runner: Callable[[Callable[..., None]], Optional[dict[str, Any]]],
        *,
        description: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        with self._lock:
            existing = self._find_active_job_locked(job_type)
            if existing is not None:
                return self._serialize_job(existing), False
            job_id = uuid.uuid4().hex
            timestamp = _job_timestamp()
            job = BackgroundJob(
                id=job_id,
                job_type=job_type,
                message=description or '',
                created_at=timestamp,
                updated_at=timestamp,
            )
            self._jobs[job_id] = job
            self._prune_jobs_locked()

        thread = Thread(
            target=self._run_job,
            args=(job_id, runner),
            name=f'job-{job_type}-{job_id}',
            daemon=True,
        )
        thread.start()
        return self.get_job(job_id), True

    def _set_job_running(self, job_id: str) -> None:
        timestamp = _job_timestamp()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_STATUS_RUNNING
            job.started_at = timestamp
            job.updated_at = timestamp
            if not job.message:
                job.message = 'Running…'

    def _update_job(
        self,
        job_id: str,
        *,
        progress_current: int | None = None,
        progress_total: int | None = None,
        message: str | None = None,
        data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        timestamp = _job_timestamp()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if progress_current is not None:
                job.progress_current = max(int(progress_current), 0)
            if progress_total is not None:
                job.progress_total = max(int(progress_total), 0)
            if message is not None:
                job.message = str(message)
            if data:
                for key, value in data.items():
                    job.data[key] = value
            job.updated_at = timestamp

    def _finalize_job(
        self,
        job_id: str,
        status: str,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        timestamp = _job_timestamp()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.error = error
            job.result = dict(result or {})
            job.finished_at = timestamp
            job.updated_at = timestamp

    def _run_job(
        self,
        job_id: str,
        runner: Callable[[Callable[..., None]], Optional[dict[str, Any]]],
    ) -> None:
        def progress_callback(
            current: int | None = None,
            total: int | None = None,
            message: str | None = None,
            *,
            data: Optional[Mapping[str, Any]] = None,
            **extra: Any,
        ) -> None:
            merged: dict[str, Any] = {}
            if data:
                merged.update(dict(data))
            if extra:
                merged.update({k: v for k, v in extra.items() if v is not None})
            self._update_job(
                job_id,
                progress_current=current,
                progress_total=total,
                message=message,
                data=merged or None,
            )

        self._set_job_running(job_id)
        try:
            result = runner(progress_callback)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception('Background job %s failed', job_id)
            self._finalize_job(job_id, JOB_STATUS_ERROR, error=str(exc))
            return
        self._finalize_job(job_id, JOB_STATUS_SUCCESS, result=result)


_JOB_MANAGER: BackgroundJobManager | None = None


def get_job_manager() -> BackgroundJobManager:
    """Return the process-wide :class:`BackgroundJobManager` instance."""

    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        _JOB_MANAGER = BackgroundJobManager()
    return _JOB_MANAGER


__all__ = [
    'BackgroundJob',
    'BackgroundJobManager',
    'JOB_ACTIVE_STATUSES',
    'JOB_STATUS_ERROR',
    'JOB_STATUS_PENDING',
    'JOB_STATUS_RUNNING',
    'JOB_STATUS_SUCCESS',
    'JOB_TERMINAL_STATUSES',
    'MAX_BACKGROUND_JOBS',
    'get_job_manager',
]
