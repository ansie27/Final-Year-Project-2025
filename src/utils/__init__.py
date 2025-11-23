from __future__ import annotations

import sys
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def ensure_directory(path: PathLike) -> Path:
    """Create the directory (and parents) if it does not already exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def print_section_header(title: str) -> None:
    """Print a formatted section header to improve console readability."""
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def print_progress(message: str, *, step: int | None = None, total: int | None = None) -> None:
    """Print a consistent progress message with optional step/total metadata."""
    prefix = ""
    if step is not None and total is not None:
        prefix = f"[{step}/{total}] "
    elif step is not None:
        prefix = f"[{step}] "
    print(f"{prefix}{message}")
    sys.stdout.flush()

