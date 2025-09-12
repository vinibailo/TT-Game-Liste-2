#!/usr/bin/env python3
"""Remove leading zeros from processed_covers filenames."""
from pathlib import Path

def rename_covers(directory: Path) -> None:
    for path in directory.glob("0000*.jpg"):
        new_name = f"{int(path.stem)}{path.suffix}"
        path.rename(directory / new_name)

if __name__ == "__main__":
    rename_covers(Path("processed_covers"))
