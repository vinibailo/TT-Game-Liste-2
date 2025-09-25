"""Catalog state management for processed games navigation."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Callable, Iterable, Mapping

import pandas as pd

from . import navigator as processed_navigator


@dataclass
class CatalogState:
    """Track the in-memory games catalog and navigator integration."""

    navigator_factory: Callable[[], processed_navigator.GameNavigator]
    category_labels: Mapping[int, str] | None = None
    logger: logging.Logger | None = None

    games_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_games: int = 0
    categories_list: list[str] = field(default_factory=list)
    platforms_list: list[str] = field(default_factory=list)
    _category_values: set[str] = field(default_factory=set)
    _navigator: processed_navigator.GameNavigator | None = None

    def set_games_dataframe(
        self,
        df: pd.DataFrame | None,
        *,
        rebuild_metadata: bool = True,
        rebuild_navigator: bool = True,
    ) -> None:
        """Replace the cached games dataframe and refresh derived state."""

        if df is None:
            df = pd.DataFrame()
        self.games_df = df
        self.total_games = len(df)

        if rebuild_metadata:
            self._rebuild_metadata(df)

        if rebuild_navigator:
            self.ensure_navigator().set_games_df(df, rebuild_state=True)
        else:
            navigator = self._navigator
            if navigator is not None:
                navigator.set_games_df(df, rebuild_state=False)

    def ensure_navigator(self) -> processed_navigator.GameNavigator:
        """Return the shared navigator instance, creating it if needed."""

        if self._navigator is None:
            self._navigator = self.navigator_factory()
            try:
                self._navigator.set_games_df(self.games_df, rebuild_state=True)
            except Exception:
                if self.logger:
                    self.logger.exception("Failed to initialize navigator state")
                raise
        return self._navigator

    def ensure_navigator_dataframe(
        self, *, rebuild_state: bool = False
    ) -> processed_navigator.GameNavigator:
        """Synchronize the navigator with the current dataframe."""

        navigator = self.ensure_navigator()
        df = self.games_df if self.games_df is not None else pd.DataFrame()
        if navigator.games_df is not df or navigator.total != len(df):
            navigator.set_games_df(df, rebuild_state=rebuild_state)
        return navigator

    def reset_source_index_cache(self) -> None:
        self.ensure_navigator().reset_source_index_cache()

    def get_source_index_for_position(self, position: int) -> str:
        navigator = self.ensure_navigator_dataframe(rebuild_state=False)
        return navigator.get_source_index_for_position(position)

    def get_position_for_source_index(self, value: object) -> int | None:
        navigator = self.ensure_navigator_dataframe(rebuild_state=False)
        return navigator.get_position_for_source_index(value)

    def get_category_values(self) -> set[str]:
        return set(self._category_values)

    def get_categories(self) -> list[str]:
        return list(self.categories_list)

    def get_platforms(self) -> list[str]:
        return list(self.platforms_list)

    def set_navigator(self, navigator: processed_navigator.GameNavigator | None) -> None:
        self._navigator = navigator

    def _rebuild_metadata(self, df: pd.DataFrame) -> None:
        category_values: set[str] = set()
        if not df.empty and 'Category' in df.columns:
            for raw_category in df['Category'].dropna():
                text = str(raw_category).strip()
                if text:
                    category_values.add(text)
        if self.category_labels:
            category_values.update(
                label for label in self.category_labels.values() if label
            )
        self._category_values = category_values
        self.categories_list = sorted(category_values, key=str.casefold)

        platform_values: set[str] = set()
        if not df.empty and 'Platforms' in df.columns:
            for raw_platforms in df['Platforms'].dropna():
                for entry in str(raw_platforms).split(','):
                    text = entry.strip()
                    if text:
                        platform_values.add(text)
        self.platforms_list = sorted(platform_values, key=str.casefold)


__all__ = ["CatalogState"]

