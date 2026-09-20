"""Failure handling, admin status and institutional-email alerts.

On any import or publication failure the previous published version keeps
serving: recording a failure never moves the catalogue's current-version
pointer. A safe failure report (identifiers plus a sanitized message — no
tracebacks, no raw exception objects, no file contents) is retained, the
admin status view tracks last success and failure per operation kind, and
an email alert is queued for the institutional admin address.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OPERATION_KINDS = ("import", "publication", "job")


def sanitize_message(message: str, limit: int = 500) -> str:
    """Collapse a raw exception message to one safe, bounded line."""
    cleaned = "".join(
        ch if ch.isprintable() and ch not in "\n\r\t" else " " for ch in message
    )
    collapsed = " ".join(cleaned.split())
    return collapsed[:limit]


@dataclass(frozen=True)
class FailureReport:
    """Immutable, safe-by-construction record of one failure."""

    kind: str
    identifier: str
    error_type: str
    safe_message: str


class AdminStatus:
    """Last success and failure identifiers per operation kind."""

    def __init__(self) -> None:
        self._last_success: dict[str, str] = {}
        self._last_failure: dict[str, str] = {}

    def record_success(self, kind: str, identifier: str) -> None:
        self._last_success[kind] = identifier

    def record_failure(self, kind: str, identifier: str) -> None:
        self._last_failure[kind] = identifier

    def view(self) -> dict[str, dict[str, str]]:
        return {
            "last_success": dict(self._last_success),
            "last_failure": dict(self._last_failure),
        }


class EmailAlerter:
    """Queues institutional-email alerts; sending is done by operations."""

    def __init__(self, recipients: list[str]) -> None:
        self.recipients = list(recipients)
        self.outbox: list[tuple[str, str]] = []

    def alert(self, report: FailureReport) -> None:
        subject = f"DARWIN {report.kind} failure: {report.identifier}"
        body = (
            f"To: {', '.join(self.recipients)}\n"
            f"Subject: {subject}\n\n"
            f"Operation: {report.kind}\n"
            f"Identifier: {report.identifier}\n"
            f"Error: {report.error_type}: {report.safe_message}\n"
        )
        self.outbox.append((subject, body))


def record_failure(sink: object, *, kind: str, identifier: str, error: BaseException) -> FailureReport:
    """Record one failure: retain a safe report, update status, alert.

    The sink is duck-typed so catalogue retention, admin status and email
    alerts compose in one call: a Catalogue keeps the safe report (for
    ``publication`` failures against a registered candidate), an
    AdminStatus tracks the identifier, and an EmailAlerter queues the
    institutional alert. The current-version pointer is never moved here.
    """
    if kind not in OPERATION_KINDS:
        raise ValueError(f"unknown operation kind: {kind}")
    report = FailureReport(
        kind=kind,
        identifier=identifier,
        error_type=type(error).__name__,
        safe_message=sanitize_message(str(error)),
    )
    fail = getattr(sink, "fail", None)
    if callable(fail) and kind == "publication":
        fail(identifier, f"{report.error_type}: {report.safe_message}")
    record = getattr(sink, "record_failure", None)
    if callable(record):
        record(kind, identifier)
    alert = getattr(sink, "alert", None)
    if callable(alert):
        alert(report)
    return report


def record_success(sink: object, *, kind: str, identifier: str) -> None:
    """Record one success on an AdminStatus-compatible sink."""
    if kind not in OPERATION_KINDS:
        raise ValueError(f"unknown operation kind: {kind}")
    record = getattr(sink, "record_success", None)
    if not callable(record):
        raise TypeError("success sink must provide record_success(kind, identifier)")
    record(kind, identifier)
