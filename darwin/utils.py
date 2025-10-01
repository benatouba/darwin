"""Utility functions for the Darwin package."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pint

from darwin.defaults import basepath, default_end, default_start

nino_anomaly: pd.DataFrame = pd.read_csv(
    "./darwin/nino12_anomaly_long.csv", index_col=0, parse_dates=True
).loc[default_start:default_end]


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


def glob_files(path: Path | str, pattern: str) -> list[Path]:
    """Glob files in a directory.

    Args:
        path: Path to directory to glob files in.
        pattern: Glob pattern to match.

    Returns:
        List of paths to files matching the glob pattern.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.is_absolute():
        path = basepath / path
    files = list(path.glob(pattern))
    if len(files) == 0:
        msg = f"No file found for {pattern} in {path}."
        raise FileNotFoundError(msg)
    return files


# Create a UnitRegistry object
ureg = pint.UnitRegistry()

# Superscript dictionary for common exponents
superscripts: dict[str, str] = {
    "-": "⁻",
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
}

# Define time slots
time_slots = [
    (1, 1, 3, 31),  # January 1 to March 31
    (4, 1, 6, 30),  # April 1 to June 30
    (7, 1, 9, 30),  # July 1 to September 30
    (10, 1, 12, 31),  # October 1 to December 31
]


def superscript_exponent(exp: int) -> str:
    return "".join(superscripts.get(char, char) for char in str(exp))


def format_units(unit: pint.Unit) -> str:
    numerator = " ".join([ureg.get_symbol(part) for part in unit._units if unit._units[part] > 0])
    denominator = " ".join(
        [
            f"{ureg.get_symbol(part)}{superscript_exponent(int(unit._units[part]))}"
            for part in unit._units
            if unit._units[part] < 0
        ]
    )
    return f"{numerator} {denominator}".strip()


# Function to calculate days in a given slot for a specific year
def days_in_slot(start_month: int, start_day: int, end_month: int, end_day: int, year: int) -> int:
    start_date = datetime(year, start_month, start_day, tzinfo=UTC)
    end_date = datetime(year, end_month, end_day, tzinfo=UTC)
    return (end_date - start_date).days + 1
