"""Tests covering navigator state persistence on MariaDB."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects.mysql import dialect as mysql_dialect

from processed.navigator import GameNavigator


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTransaction:
    def __init__(self, connection: "FakeMariaDBConnection"):
        self._connection = connection

    def __enter__(self):
        self._connection._begin_calls += 1
        return self._connection

    def __exit__(self, exc_type, exc, tb):
        self._connection._begin_calls -= 1
        return False


class FakeMariaDBConnection:
    def __init__(self):
        self.engine = SimpleNamespace(dialect=SimpleNamespace(name="mariadb"))
        self.executed_sql: list[str] = []
        self.executed_params: list[dict[str, object]] = []
        self.auto_increment_counter = 0
        self.state_row: dict[str, object] | None = None
        self._begin_calls = 0

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self)

    def execute(self, stmt):
        compiled = stmt.compile(
            dialect=mysql_dialect(),
            compile_kwargs={"render_postcompile": True},
        )
        sql = str(compiled)
        params = dict(compiled.params)
        self.executed_sql.append(sql)
        self.executed_params.append(params)

        normalized = sql.lower()
        if "on duplicate key update" in normalized and params.get("id") == 1:
            if self.state_row is None:
                self.auto_increment_counter += 1
                self.state_row = params
            else:
                self.state_row.update(
                    {
                        "current_index": params["current_index"],
                        "seq_index": params["seq_index"],
                        "skip_queue": params["skip_queue"],
                    }
                )
        else:  # pragma: no cover - defensive branch for unexpected SQL
            self.auto_increment_counter += 1
            self.state_row = params


class FakeHandle:
    def __init__(self, connection: FakeMariaDBConnection):
        self._connection = connection
        self.engine = connection.engine

    def sa_connection(self):
        connection = self._connection

        class _Context:
            def __enter__(self):
                return connection

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Context()


@pytest.fixture
def fake_navigator():
    connection = FakeMariaDBConnection()
    handle = FakeHandle(connection)

    navigator = GameNavigator(
        db_lock=DummyLock(),
        get_db=lambda: handle,
        is_processed_game_done=lambda *_args, **_kwargs: True,
    )
    navigator.current_index = 5
    navigator.seq_index = 7
    navigator.skip_queue = [{"index": 9, "countdown": 3}]
    return navigator, connection


def test_save_inserts_singleton_row_on_mariadb(fake_navigator):
    navigator, connection = fake_navigator

    navigator._save()

    assert connection.auto_increment_counter == 1
    assert connection.state_row == {
        "id": 1,
        "current_index": 5,
        "seq_index": 7,
        "skip_queue": json.dumps(navigator.skip_queue),
    }
    assert any(
        "on duplicate key update" in sql.lower() for sql in connection.executed_sql
    )


def test_save_updates_existing_row_without_incrementing_id(fake_navigator):
    navigator, connection = fake_navigator

    navigator._save()
    navigator.current_index = 6
    navigator.seq_index = 8
    navigator.skip_queue = []

    navigator._save()

    assert connection.auto_increment_counter == 1
    assert connection.state_row == {
        "id": 1,
        "current_index": 6,
        "seq_index": 8,
        "skip_queue": json.dumps(navigator.skip_queue),
    }
    assert all(
        "on duplicate key update" in sql.lower() for sql in connection.executed_sql
    )

