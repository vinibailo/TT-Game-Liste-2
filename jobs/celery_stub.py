"""Minimal Celery stub for environments without the real dependency."""

from __future__ import annotations

import types
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping


class _States:
    FAILURE = 'FAILURE'
    SUCCESS = 'SUCCESS'
    PENDING = 'PENDING'


states = _States()


@dataclass
class _Config:
    data: Dict[str, Any] = field(default_factory=dict)

    def update(self, *dicts: Mapping[str, Any], **kwargs: Any) -> None:
        for mapping in dicts:
            self.data.update(mapping)
        if kwargs:
            self.data.update(kwargs)

    def __getattr__(self, item: str) -> Any:
        return self.data.get(item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == 'data':
            super().__setattr__(key, value)
        else:
            self.data[key] = value


class AsyncResult:
    def __init__(self, task_id: str, result: Any = None) -> None:
        self.id = task_id
        self.result = result


class _TaskWrapper:
    def __init__(self, app: 'Celery', func: Callable[..., Any], *, name: str, bind: bool) -> None:
        self.app = app
        self.func = func
        self.__name__ = func.__name__
        self.name = name
        self.bind = bind
        self.request = types.SimpleNamespace(id=None)
        self._state: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.bind:
            return self.func(self, *args, **kwargs)
        return self.func(*args, **kwargs)

    def update_state(self, *, state: str | None = None, meta: Any = None) -> None:
        self._state = {'state': state, 'meta': meta}

    def s(self, *args: Any, **kwargs: Any) -> '_Signature':
        return _Signature(self, args=args, kwargs=kwargs)

    def apply_async(
        self,
        args: Iterable[Any] | None = None,
        kwargs: MutableMapping[str, Any] | None = None,
        task_id: str | None = None,
    ) -> AsyncResult:
        signature = self.s(*(args or ()), **(kwargs or {}))
        return signature.apply_async(task_id=task_id)


class _Signature:
    def __init__(self, task: _TaskWrapper, *, args: Iterable[Any], kwargs: Mapping[str, Any]) -> None:
        self.task = task
        self.args = tuple(args)
        self.kwargs = dict(kwargs)

    def apply_async(self, *, task_id: str | None = None) -> AsyncResult:
        request_id = task_id or uuid.uuid4().hex
        self.task.request = types.SimpleNamespace(id=request_id)
        result = self.task(*self.args, **self.kwargs)
        return AsyncResult(request_id, result)


class Task(_TaskWrapper):
    pass


class Celery:
    def __init__(self, name: str, broker: str | None = None, backend: str | None = None) -> None:
        self.main = name
        self.conf = _Config({'broker_url': broker, 'result_backend': backend})
        self.tasks: dict[str, _TaskWrapper] = {}

    def task(self, name: str | None = None, bind: bool = False) -> Callable[[Callable[..., Any]], Task]:
        def decorator(func: Callable[..., Any]) -> Task:
            task_name = name or func.__name__
            wrapper = Task(self, func, name=task_name, bind=bind)
            self.tasks[task_name] = wrapper
            return wrapper

        return decorator


__all__ = ['Celery', 'Task', 'states']
