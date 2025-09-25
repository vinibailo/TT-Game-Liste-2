"""In-memory Redis stub for environments without the redis package."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


class RedisError(Exception):
    """Fallback Redis error type."""


@dataclass
class _SortedEntry:
    score: float
    member: str


class _Pipeline:
    def __init__(self, redis: 'Redis') -> None:
        self._redis = redis
        self._operations: list[Tuple[str, tuple, dict]] = []

    def set(self, key: str, value: str) -> '_Pipeline':
        self._operations.append(('set', (key, value), {}))
        return self

    def zadd(self, key: str, mapping: Mapping[str, float]) -> '_Pipeline':
        self._operations.append(('zadd', (key, mapping), {}))
        return self

    def execute(self) -> None:
        for name, args, kwargs in self._operations:
            getattr(self._redis, name)(*args, **kwargs)
        self._operations.clear()

    # Context manager support
    def __enter__(self) -> '_Pipeline':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            self.execute()


class Redis:
    def __init__(self, *, decode_responses: bool = False) -> None:
        self._strings: Dict[str, str] = {}
        self._sorted_sets: Dict[str, list[_SortedEntry]] = {}
        self.decode_responses = decode_responses

    @classmethod
    def from_url(cls, _url: str, *, decode_responses: bool = False) -> 'Redis':
        return cls(decode_responses=decode_responses)

    def set(self, key: str, value: str) -> None:
        self._strings[key] = value

    def get(self, key: str) -> str | None:
        return self._strings.get(key)

    def delete(self, key: str) -> None:
        self._strings.pop(key, None)
        self._sorted_sets.pop(key, None)

    def zadd(self, key: str, mapping: Mapping[str, float]) -> None:
        entries = self._sorted_sets.setdefault(key, [])
        for member, score in mapping.items():
            for idx, entry in enumerate(entries):
                if entry.member == member:
                    entries.pop(idx)
                    break
            bisect.insort(entries, _SortedEntry(float(score), member))

    def zrange(self, key: str, start: int, end: int) -> list[str]:
        entries = self._sorted_sets.get(key, [])
        if end == -1:
            subset = entries[start:]
        else:
            subset = entries[start : end + 1]
        return [entry.member for entry in subset]

    def zcard(self, key: str) -> int:
        return len(self._sorted_sets.get(key, []))

    def zrem(self, key: str, member: str) -> int:
        entries = self._sorted_sets.get(key, [])
        for idx, entry in enumerate(entries):
            if entry.member == member:
                entries.pop(idx)
                return 1
        return 0

    def pipeline(self) -> _Pipeline:
        return _Pipeline(self)


__all__ = ['Redis', 'RedisError']
