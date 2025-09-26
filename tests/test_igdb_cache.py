import json

from igdb import cache as igdb_cache
from igdb import diff as igdb_diff


def test_persist_payloads_creates_files(tmp_path):
    payload_a = {"id": 101, "name": "First Game", "genres": ["Action"]}
    payload_b = {"id": "202", "name": "Second Game"}

    summary = igdb_cache.persist_igdb_payloads(
        tmp_path,
        [payload_a, payload_b],
        timestamp_factory=lambda: "2024-01-01T00:00:00+00:00",
    )

    assert summary["inserted"] == 2
    assert summary["updated"] == 0
    assert summary["unchanged"] == 0
    assert len(summary["entries"]) == 2

    entry_a = tmp_path / "101.json"
    assert entry_a.exists()
    data = json.loads(entry_a.read_text())
    assert data["payload"]["name"] == "First Game"

    repeat = igdb_cache.persist_igdb_payloads(
        tmp_path,
        [payload_a],
        timestamp_factory=lambda: "2024-02-02T00:00:00+00:00",
    )
    assert repeat["inserted"] == 0
    assert repeat["updated"] == 0
    assert repeat["unchanged"] == 1

    modified = dict(payload_a)
    modified["name"] = "Updated Game"
    changed = igdb_cache.persist_igdb_payloads(tmp_path, [modified])
    assert changed["updated"] == 1

    loaded = igdb_cache.load_cached_entry(tmp_path, 101)
    assert loaded is not None
    assert loaded.igdb_id == "101"
    assert loaded.payload["name"] == "Updated Game"


def test_fetch_cached_updates_with_diff(tmp_path):
    payload = {"id": 777, "name": "Original", "genres": ["Action"]}
    igdb_cache.persist_igdb_payloads(tmp_path, [payload])

    new_payload = {"id": 777, "name": "Updated", "genres": ["Action", "Adventure"]}
    updates = igdb_cache.fetch_cached_updates(tmp_path, [new_payload])
    assert len(updates) == 1
    entry = updates[0]
    assert entry["igdb_id"] == "777"
    assert entry["has_diff"] is True
    assert entry["diff"]["name"]["added"] == "Updated"
    assert entry["diff"]["genres"]["added"] == ["Adventure"]


def test_build_diff_report_handles_various_types():
    cached = {
        "name": "Game Name",
        "first_release_date": 1_600_000_000,
        "developers": ["Company A"],
        "publishers": [{"name": "Publisher"}],
    }
    new = {
        "name": "Game Name 2",
        "first_release_date": "1600500000",
        "developers": ["Company A", "Studio B"],
        "publishers": ["Publisher"],
    }

    diff = igdb_diff.build_diff_report(cached, new)
    assert diff["name"]["added"] == "Game Name 2"
    assert diff["first_release_date"]["added"].startswith("2020")
    assert diff["developers"]["added"] == ["Studio B"]
    assert "publishers" not in diff
