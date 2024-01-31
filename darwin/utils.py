"""Utility functions for the Darwin package."""

from __future__ import annotations

from pathlib import Path

from darwin.defaults import basepath


def normalize_whitespace(text: str) -> str:
    """Remove redundant whitespace from a string."""
    return " ".join(text.split())


def remove_nonalphanumerics(string: str) -> str:
    """Remove non-alphanumeric characters from a string.

    Args:
        string: String to remove non-alphanumeric characters from.

    Returns:
        String with non-alphanumeric characters removed.
    """
    return "".join(ch for ch in string if ch.isalnum())


def glob_files(path: str | Path, pattern: str) -> list[Path]:
    """Glob files in a directory.

    Args:
        path: Path to directory to glob files in.
        pattern: Glob pattern to match.

    Returns:
        List of paths to files matching the glob pattern.
    """
    if isinstance(path, str):
        return list(Path(path).glob(pattern))
    if not path.is_absolute():
        path = basepath / path
    return list(path.glob(pattern))
