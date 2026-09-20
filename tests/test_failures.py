"""RED tests for D9: failure handling, admin status and alerts.

On failure the previous published version must keep serving, a safe
failure report must be retained, the admin view must show last success
and failure with identifiers, and an institutional-email alert is raised.
"""

from darwin.publisher.candidate import Candidate
from darwin.publisher.catalogue import Catalogue
from darwin.publisher.failures import (
    AdminStatus,
    EmailAlerter,
    FailureReport,
    record_failure,
    record_success,
)


def _candidate(product_id: str, version: str) -> Candidate:
    return Candidate(
        candidate_id=f"{product_id}@{version}",
        product_id=product_id,
        version=version,
        manifest_revision="rev-1",
        source_entries=(),
    )


def _boom():
    raise RuntimeError("simulated publisher crash")


def test_failed_publication_preserves_previous_published_version():
    catalogue = Catalogue()
    v1 = _candidate("gar-d02-t2-max", "R1.1")
    catalogue.register(v1)
    catalogue.publish(v1.candidate_id)
    assert catalogue.current("gar-d02-t2-max").version == "R1.1"

    v2 = _candidate("gar-d02-t2-max", "R1.2")
    catalogue.register(v2)
    try:
        _boom()
    except RuntimeError as exc:
        report = record_failure(
            catalogue,
            kind="publication",
            identifier=v2.candidate_id,
            error=exc,
        )
    # Previous published version keeps serving.
    assert catalogue.current("gar-d02-t2-max").version == "R1.1"
    assert isinstance(report, FailureReport)
    assert report.identifier == v2.candidate_id


def test_safe_report_retained_without_raw_exception_content():
    catalogue = Catalogue()
    candidate = _candidate("gar-d02-prcp-sum", "R1.1")
    catalogue.register(candidate)
    try:
        _boom()
    except RuntimeError as exc:
        report = record_failure(
            catalogue,
            kind="publication",
            identifier=candidate.candidate_id,
            error=exc,
        )
    stored = catalogue.failure_reason(candidate.candidate_id)
    assert stored is not None
    assert "traceback" not in stored.lower()
    assert report.error_type == "RuntimeError"
    assert "\n" not in report.safe_message


def test_admin_view_shows_last_success_and_failure_with_identifiers():
    status = AdminStatus()
    record_success(status, kind="import", identifier="aws-minas-rojas@2026-05-20")
    record_success(status, kind="publication", identifier="gar-d02-t2-max@1")
    record_failure(
        status,
        kind="job",
        identifier="job-42",
        error=RuntimeError("worker lost"),
    )
    view = status.view()
    assert view["last_success"]["import"] == "aws-minas-rojas@2026-05-20"
    assert view["last_success"]["publication"] == "gar-d02-t2-max@1"
    assert view["last_failure"]["job"] == "job-42"


def test_failure_raises_institutional_email_alert():
    alerter = EmailAlerter(recipients=["admin@example.org"])
    record_failure(
        alerter,
        kind="publication",
        identifier="gar-d02-prcp-sum@2",
        error=RuntimeError("checksum mismatch"),
    )
    assert len(alerter.outbox) == 1
    subject, _body = alerter.outbox[0]
    assert "publication" in subject
    assert "gar-d02-prcp-sum@2" in subject


def test_failure_message_is_sanitized():
    status = AdminStatus()
    try:
        raise ValueError("bad value\nsecond line\x00with control")
    except ValueError as exc:
        report = record_failure(
            status, kind="job", identifier="job-7", error=exc
        )
    assert "\n" not in report.safe_message
    assert "\x00" not in report.safe_message
