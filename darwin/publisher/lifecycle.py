"""Lifecycle transitions, enforced in exactly one place."""

from __future__ import annotations

ALLOWED: dict[str, frozenset[str]] = {
    "candidate": frozenset({"published", "failed"}),
    "published": frozenset({"superseded"}),
    "superseded": frozenset(),
    "failed": frozenset(),
}


def check_transition(old: str, new: str) -> None:
    """Raise ValueError unless `old -> new` is a legal lifecycle move."""
    if new not in ALLOWED.get(old, frozenset()):
        raise ValueError(f"illegal lifecycle transition: {old!r} -> {new!r}")
