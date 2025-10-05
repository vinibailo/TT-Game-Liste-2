from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from igdb import cache as igdb_cache


class _RecordingSAConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.parameters: list[object] = []

    def execute(self, statement, params=None):  # type: ignore[override]
        sql_text = getattr(statement, "text", str(statement)).strip()
        self.statements.append(sql_text)
        self.parameters.append(params)
        return SimpleNamespace()

    def in_transaction(self) -> bool:  # pragma: no cover - tests never start tx
        return False


@contextmanager
def _fake_sa_connection_factory(recordings: list[_RecordingSAConnection]):
    recorder = _RecordingSAConnection()
    recordings.append(recorder)
    yield recorder, False


def test_mariadb_cache_tables_use_compatible_types(monkeypatch):
    recordings: list[_RecordingSAConnection] = []

    monkeypatch.setattr(igdb_cache, "_dialect_name", lambda _conn: "mariadb")
    monkeypatch.setattr(igdb_cache, "_sa_connection", lambda conn: _fake_sa_connection_factory(recordings))

    igdb_cache._ensure_cache_state_table(object())
    igdb_cache._ensure_games_table(object())

    assert len(recordings) == 2

    state_statements = "\n".join(recordings[0].statements).upper()
    assert "CHECK" not in state_statements
    assert "VARCHAR(64)" in state_statements

    game_statements = recordings[1].statements
    create_sql = next(stmt for stmt in game_statements if "CREATE TABLE" in stmt.upper())
    assert "VARCHAR(255)" in create_sql
    assert "LONGTEXT" in create_sql
    index_sql = next(stmt for stmt in game_statements if "CREATE INDEX" in stmt.upper())
    assert "IF NOT EXISTS" not in index_sql.upper()


def test_set_cached_total_uses_mariadb_upsert(monkeypatch):
    recordings: list[_RecordingSAConnection] = []

    monkeypatch.setattr(igdb_cache, "_dialect_name", lambda _conn: "mariadb")
    monkeypatch.setattr(igdb_cache, "_sa_connection", lambda conn: _fake_sa_connection_factory(recordings))

    igdb_cache.set_cached_total(object(), 42, synced_at="2024-01-01T00:00:00Z")

    assert len(recordings) == 2
    upsert_sql = "\n".join(recordings[1].statements).upper()
    assert "ON DUPLICATE KEY UPDATE" in upsert_sql


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeMariaDBCacheConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.rows: dict[int, tuple[object, ...]] = {}
        self.engine = SimpleNamespace(dialect=SimpleNamespace(name="mariadb"))

    def execute(self, sql: str, params=None):  # type: ignore[override]
        parameters = tuple(params or ())
        self.calls.append((sql, parameters))
        normalized = " ".join(sql.split()).lower()
        if normalized.startswith("select") and "where igdb_id" in normalized:
            igdb_id = int(parameters[0])
            return _FakeCursor(self.rows.get(igdb_id))
        if normalized.startswith("insert"):
            igdb_id = int(parameters[0])
            self.rows[igdb_id] = tuple(parameters[1:-1])
            return _FakeCursor(None)
        if normalized.startswith("update"):
            igdb_id = int(parameters[-1])
            self.rows[igdb_id] = tuple(parameters[:-2])
            return _FakeCursor(None)
        raise AssertionError(f"Unexpected SQL for MariaDB cache test: {sql}")


def test_upsert_igdb_games_handles_mariadb(monkeypatch):
    monkeypatch.setattr(igdb_cache, "_ensure_games_table", lambda _conn: None)

    connection = _FakeMariaDBCacheConnection()
    payload = {
        "id": 1,
        "name": "Example",
        "summary": "Summary",
        "updated_at": 100,
        "first_release_date": 200,
        "category": 2,
        "cover": {"image_id": "abc"},
        "rating_count": 5,
        "developers": ["Dev"],
        "publishers": ["Pub"],
        "genres": ["Genre"],
        "platforms": ["Platform"],
        "game_modes": ["Mode"],
    }
    payload_updated = dict(payload, summary="Updated summary")

    inserted, updated, unchanged = igdb_cache.upsert_igdb_games(
        connection,
        [payload, payload, payload_updated],
    )

    assert inserted == 1
    assert unchanged == 1
    assert updated == 1
    executed_sql = " ".join(sql for sql, _ in connection.calls).upper()
    assert "INSERT INTO" in executed_sql
    assert "UPDATE" in executed_sql

