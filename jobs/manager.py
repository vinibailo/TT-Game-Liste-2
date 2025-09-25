"""Background job tracking and orchestration utilities."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, Callable, Mapping, Optional

try:  # pragma: no cover - optional dependency fallback
    from celery import Celery, states
    from celery.app.task import Task
except ModuleNotFoundError:  # pragma: no cover - fallback for offline tests
    from jobs.celery_stub import Celery, Task, states

try:  # pragma: no cover - optional dependency fallback
    from redis import Redis
    from redis.exceptions import RedisError
except ModuleNotFoundError:  # pragma: no cover - fallback for offline tests
    from jobs.redis_stub import Redis, RedisError

logger = logging.getLogger(__name__)


MAX_BACKGROUND_JOBS = 50


def _job_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _json_default(value: Any) -> str:
    try:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
    except Exception:  # pragma: no cover - defensive
        return repr(value)


CELERY_DEFAULT_URL = 'redis://localhost:6379/0'


celery_app = Celery('tt_game_liste')


def _configure_celery(app: Celery) -> None:
    broker_url = os.environ.get('CELERY_BROKER_URL', CELERY_DEFAULT_URL)
    backend_url = os.environ.get('CELERY_RESULT_BACKEND', broker_url)
    eager = _coerce_truthy(os.environ.get('CELERY_TASK_ALWAYS_EAGER'))
    app.conf.update(
        broker_url=broker_url,
        result_backend=backend_url,
        task_track_started=True,
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        task_always_eager=eager,
        task_eager_propagates=eager,
    )


_configure_celery(celery_app)


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
    task_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'job_type': self.job_type,
            'status': self.status,
            'message': self.message,
            'progress_current': self.progress_current,
            'progress_total': self.progress_total,
            'data': dict(self.data),
            'result': dict(self.result),
            'error': self.error,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'task_id': self.task_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'BackgroundJob':
        data = dict(payload)
        return cls(
            id=str(data.get('id', '')),
            job_type=str(data.get('job_type', '')),
            status=str(data.get('status', JOB_STATUS_PENDING)),
            message=str(data.get('message', '')),
            progress_current=int(data.get('progress_current', 0) or 0),
            progress_total=int(data.get('progress_total', 0) or 0),
            data=dict(data.get('data') or {}),
            result=dict(data.get('result') or {}),
            error=data.get('error'),
            created_at=str(data.get('created_at', '')),
            updated_at=str(data.get('updated_at', '')),
            started_at=data.get('started_at'),
            finished_at=data.get('finished_at'),
            task_id=data.get('task_id'),
        )


class BackgroundJobManager:
    _JOB_INDEX_KEY = 'background_jobs:index'

    def __init__(self) -> None:
        redis_url = os.environ.get('JOB_REDIS_URL') or celery_app.conf.result_backend
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self._celery = celery_app

    def _job_key(self, job_id: str) -> str:
        return f'background_jobs:data:{job_id}'

    def _serialize(self, job: BackgroundJob) -> str:
        return json.dumps(job.to_dict(), default=_json_default)

    def _deserialize(self, raw: str | None) -> Optional[BackgroundJob]:
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            logger.error('Unable to decode job payload: %s', raw)
            return None
        return BackgroundJob.from_dict(payload)

    def _store_job(self, job: BackgroundJob) -> None:
        payload = self._serialize(job)
        try:
            with self._redis.pipeline() as pipe:
                pipe.set(self._job_key(job.id), payload)
                pipe.zadd(self._JOB_INDEX_KEY, {job.id: time.time()})
                pipe.execute()
        except RedisError as exc:  # pragma: no cover - defensive
            logger.error('Failed to persist job %s: %s', job.id, exc)
            raise

    def _load_job(self, job_id: str) -> Optional[BackgroundJob]:
        try:
            raw = self._redis.get(self._job_key(job_id))
        except RedisError as exc:  # pragma: no cover - defensive
            logger.error('Failed to load job %s: %s', job_id, exc)
            return None
        return self._deserialize(raw)

    def _save_job(self, job: BackgroundJob) -> None:
        try:
            self._redis.set(self._job_key(job.id), self._serialize(job))
        except RedisError as exc:  # pragma: no cover - defensive
            logger.error('Failed to save job %s: %s', job.id, exc)
            raise

    def _find_active_job(self, job_type: str) -> Optional[BackgroundJob]:
        try:
            job_ids = self._redis.zrange(self._JOB_INDEX_KEY, 0, -1)
        except RedisError as exc:  # pragma: no cover - defensive
            logger.error('Failed to query job index: %s', exc)
            return None
        for job_id in job_ids:
            job = self._load_job(job_id)
            if job and job.job_type == job_type and job.status in JOB_ACTIVE_STATUSES:
                return job
        return None

    def _prune_jobs(self) -> None:
        try:
            total = self._redis.zcard(self._JOB_INDEX_KEY)
        except RedisError as exc:  # pragma: no cover - defensive
            logger.error('Failed to read job index size: %s', exc)
            return
        if total <= MAX_BACKGROUND_JOBS:
            return
        try:
            job_ids = self._redis.zrange(self._JOB_INDEX_KEY, 0, -1)
        except RedisError as exc:  # pragma: no cover - defensive
            logger.error('Failed to read job ids for pruning: %s', exc)
            return
        removable: list[tuple[str, BackgroundJob]] = []
        for job_id in job_ids:
            job = self._load_job(job_id)
            if job and job.status in JOB_TERMINAL_STATUSES:
                removable.append((job_id, job))
        removable.sort(key=lambda item: item[1].finished_at or item[1].updated_at)
        for job_id, _job in removable:
            if total <= MAX_BACKGROUND_JOBS:
                break
            try:
                removed = self._redis.zrem(self._JOB_INDEX_KEY, job_id)
            except RedisError as exc:  # pragma: no cover - defensive
                logger.error('Failed to prune job %s: %s', job_id, exc)
                continue
            if removed:
                self._redis.delete(self._job_key(job_id))
                total -= 1

    def list_jobs(self, job_type: str | None = None) -> list[dict[str, Any]]:
        try:
            job_ids = self._redis.zrange(self._JOB_INDEX_KEY, 0, -1)
        except RedisError as exc:  # pragma: no cover - defensive
            logger.error('Failed to list jobs: %s', exc)
            return []
        jobs: list[BackgroundJob] = []
        for job_id in job_ids:
            job = self._load_job(job_id)
            if job is None:
                continue
            if job_type and job.job_type != job_type:
                continue
            jobs.append(job)
        jobs.sort(key=lambda job: job.created_at)
        return [job.to_dict() for job in jobs]

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        job = self._load_job(job_id)
        return job.to_dict() if job else None

    def get_active_job(self, job_type: str) -> Optional[dict[str, Any]]:
        job = self._find_active_job(job_type)
        return job.to_dict() if job else None

    def enqueue_job(
        self,
        job_type: str,
        runner_path: str,
        *,
        description: str | None = None,
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> tuple[dict[str, Any], bool]:
        existing = self._find_active_job(job_type)
        if existing is not None:
            return existing.to_dict(), False
        job_id = uuid.uuid4().hex
        timestamp = _job_timestamp()
        job = BackgroundJob(
            id=job_id,
            job_type=job_type,
            message=description or '',
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._store_job(job)
        self._prune_jobs()
        task_kwargs = dict(kwargs or {})
        signature = run_background_job.s(
            job_id=job_id,
            job_type=job_type,
            runner_path=runner_path,
            runner_kwargs=task_kwargs,
        )
        async_result = signature.apply_async(task_id=job_id)
        job.task_id = async_result.id
        job.updated_at = _job_timestamp()
        self._save_job(job)
        return job.to_dict(), True

    def _set_job_running(self, job_id: str, *, task_id: str | None = None) -> None:
        timestamp = _job_timestamp()
        job = self._load_job(job_id)
        if job is None:
            return
        job.status = JOB_STATUS_RUNNING
        job.started_at = timestamp
        job.updated_at = timestamp
        if task_id:
            job.task_id = task_id
        if not job.message:
            job.message = 'Running…'
        self._save_job(job)

    def _update_job(
        self,
        job_id: str,
        *,
        progress_current: int | None = None,
        progress_total: int | None = None,
        message: str | None = None,
        data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        job = self._load_job(job_id)
        if job is None:
            return
        if progress_current is not None:
            try:
                job.progress_current = max(int(progress_current), 0)
            except (TypeError, ValueError):
                job.progress_current = 0
        if progress_total is not None:
            try:
                job.progress_total = max(int(progress_total), 0)
            except (TypeError, ValueError):
                job.progress_total = 0
        if message is not None:
            job.message = str(message)
        if data:
            for key, value in data.items():
                job.data[key] = value
        job.updated_at = _job_timestamp()
        self._save_job(job)

    def _finalize_job(
        self,
        job_id: str,
        status: str,
        result: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        job = self._load_job(job_id)
        if job is None:
            return
        job.status = status
        job.error = error
        job.result = dict(result or {})
        timestamp = _job_timestamp()
        job.finished_at = timestamp
        job.updated_at = timestamp
        self._save_job(job)


def _resolve_runner(runner_path: str) -> Callable[[Callable[..., None]], Any]:
    if ':' in runner_path:
        module_path, attr = runner_path.split(':', 1)
    else:
        module_path, attr = runner_path.rsplit('.', 1)
    module = import_module(module_path)
    runner = getattr(module, attr)
    if not callable(runner):
        raise RuntimeError(f'Runner {runner_path} is not callable')
    return runner


@celery_app.task(name='jobs.run_background_job', bind=True)
def run_background_job(
    self: Task,
    *,
    job_id: str,
    job_type: str,
    runner_path: str,
    runner_kwargs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    manager = get_job_manager()
    manager._set_job_running(job_id, task_id=getattr(self.request, 'id', None))

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
        manager._update_job(
            job_id,
            progress_current=current,
            progress_total=total,
            message=message,
            data=merged or None,
        )
        job_data = manager.get_job(job_id)
        if job_data is not None:
            self.update_state(state='PROGRESS', meta=job_data)

    try:
        runner = _resolve_runner(runner_path)
        result = runner(progress_callback, **(runner_kwargs or {}))
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception('Background job %s (%s) failed', job_id, job_type)
        manager._finalize_job(job_id, JOB_STATUS_ERROR, error=str(exc))
        self.update_state(state=states.FAILURE, meta={'exc_message': str(exc)})
        raise

    normalized_result: dict[str, Any]
    if isinstance(result, Mapping):
        normalized_result = dict(result)
    elif result is None:
        normalized_result = {}
    else:
        normalized_result = {'result': result}
    manager._finalize_job(job_id, JOB_STATUS_SUCCESS, result=normalized_result)
    final_job = manager.get_job(job_id) or {}
    self.update_state(state=states.SUCCESS, meta=final_job)
    return final_job


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
