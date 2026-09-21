"""In-memory catalogue enforcing the product lifecycle.

candidate -> published -> superseded, plus failed. Publishing moves the
current-version pointer; older immutable versions stay addressable.
"""

from __future__ import annotations

import dataclasses

from darwin.publisher.candidate import Candidate
from darwin.publisher.lifecycle import check_transition


class Catalogue:
    """In-memory product catalogue; the single enforcement point for lifecycle moves."""

    def __init__(self) -> None:
        self._status: dict[str, str] = {}
        self._candidates: dict[str, Candidate] = {}
        self._reason: dict[str, str] = {}
        self._current: dict[str, str] = {}

    def register(self, candidate: Candidate) -> None:
        """Register a candidate; it enters the lifecycle as ``candidate``."""
        self._candidates[candidate.candidate_id] = candidate
        self._status[candidate.candidate_id] = "candidate"

    def next_version(self, product_id: str, release: str) -> str:
        """Return the next immutable version label for a product and release."""
        seq = sum(
            1
            for candidate in self._candidates.values()
            if candidate.product_id == product_id
        )
        return f"{release}.{seq + 1}"

    def get(self, product_id: str, version: str) -> Candidate:
        """Return one immutable version with its current lifecycle status attached."""
        for candidate in self._candidates.values():
            if candidate.product_id == product_id and candidate.version == version:
                return dataclasses.replace(
                    candidate, status=self._status[candidate.candidate_id]
                )
        raise LookupError(f"unknown product version: {product_id} {version}")

    def current(self, product_id: str) -> Candidate:
        """Return the currently published version; old versions stay addressable via get."""
        try:
            version = self._current[product_id]
        except KeyError:
            raise LookupError(f"no published version: {product_id}") from None
        return self.get(product_id, version)

    def publish(self, candidate_id: str) -> Candidate:
        """Publish a candidate and move the current-version pointer to it.

        A previously current version is marked ``superseded`` but remains
        addressable, so existing permalinks keep resolving.
        """
        self._move(candidate_id, "published")
        candidate = self._candidates[candidate_id]
        previous = self._current.get(candidate.product_id)
        if previous is not None and previous != candidate.version:
            old_id = self._find(candidate.product_id, previous)
            self._move(old_id, "superseded")
        self._current[candidate.product_id] = candidate.version
        return self.get(candidate.product_id, candidate.version)

    def fail(self, candidate_id: str, reason: str) -> Candidate:
        """Mark a candidate ``failed`` with a safe reason; the current pointer never moves here."""
        self._move(candidate_id, "failed")
        self._reason[candidate_id] = reason
        candidate = self._candidates[candidate_id]
        return self.get(candidate.product_id, candidate.version)

    def failure_reason(self, candidate_id: str) -> str:
        """Return the retained safe failure reason for a failed candidate."""
        return self._reason[candidate_id]

    def _find(self, product_id: str, version: str) -> str:
        for candidate in self._candidates.values():
            if candidate.product_id == product_id and candidate.version == version:
                return candidate.candidate_id
        raise LookupError(f"unknown product version: {product_id} {version}")

    def _move(self, candidate_id: str, new_status: str) -> None:
        try:
            old_status = self._status[candidate_id]
        except KeyError:
            raise LookupError(f"unknown candidate: {candidate_id}") from None
        check_transition(old_status, new_status)
        self._status[candidate_id] = new_status
