from __future__ import annotations

import pandas as pd
import pytest

from processed import source_index_cache


def _canonical(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(text))
    except (TypeError, ValueError):
        return text


@pytest.fixture(autouse=True)
def clear_source_index_cache() -> None:
    source_index_cache.invalidate_cache()
    yield
    source_index_cache.invalidate_cache()


def test_build_source_index_mappings_uses_cache() -> None:
    df = pd.DataFrame(
        {
            "Source Index": ["10", "20", "30"],
            "Name": ["First", "Second", "Third"],
        }
    )

    mapping, reverse = source_index_cache.build_source_index_mappings(
        df, canonicalize=_canonical
    )
    cached_mapping, cached_reverse = source_index_cache.build_source_index_mappings(
        df, canonicalize=_canonical
    )

    assert mapping is cached_mapping
    assert reverse is cached_reverse
    assert mapping == {0: "10", 1: "20", 2: "30"}
    assert reverse == {"10": 0, "20": 1, "30": 2}


def test_invalidate_cache_refreshes_dataframe_entries() -> None:
    df = pd.DataFrame(
        {
            "Source Index": ["10", "20"],
            "Name": ["First", "Second"],
        }
    )

    mapping, _ = source_index_cache.build_source_index_mappings(
        df, canonicalize=_canonical
    )
    df.loc[1, "Source Index"] = "25"
    source_index_cache.invalidate_cache(df)
    updated_mapping, updated_reverse = source_index_cache.build_source_index_mappings(
        df, canonicalize=_canonical
    )

    assert updated_mapping is not mapping
    assert updated_mapping == {0: "10", 1: "25"}
    assert updated_reverse["25"] == 1


def test_mappings_without_source_index_column() -> None:
    df = pd.DataFrame({"Name": ["First", "Second", "Third"]})

    mapping, reverse = source_index_cache.build_source_index_mappings(
        df, canonicalize=_canonical
    )

    assert mapping == {0: "0", 1: "1", 2: "2"}
    assert reverse == {"0": 0, "1": 1, "2": 2}


def test_invalidate_cache_only_clears_target_dataframe() -> None:
    df_one = pd.DataFrame({"Source Index": ["10"]})
    df_two = pd.DataFrame({"Source Index": ["20"]})

    mapping_one, _ = source_index_cache.build_source_index_mappings(
        df_one, canonicalize=_canonical
    )
    mapping_two, _ = source_index_cache.build_source_index_mappings(
        df_two, canonicalize=_canonical
    )

    source_index_cache.invalidate_cache(df_one)

    refreshed_mapping_one, _ = source_index_cache.build_source_index_mappings(
        df_one, canonicalize=_canonical
    )
    cached_mapping_two, _ = source_index_cache.build_source_index_mappings(
        df_two, canonicalize=_canonical
    )

    assert refreshed_mapping_one is not mapping_one
    assert cached_mapping_two is mapping_two
