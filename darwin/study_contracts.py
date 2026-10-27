"""Consumer validation of setups-owned study requirements and acceptance records.

``darwin-wrf-setups`` owns ``study-requirements/1`` and ``acceptance-assessment/1``.
Darwin vendors their schemas byte-for-byte and checks exact bindings. Validation
never grants acceptance and cannot establish that an authorization is genuine.
Identities in the ``synthetic:`` namespace are refused outside explicit test mode.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from functools import cache
from importlib.resources import files

import jsonschema

REQUIREMENTS_CONTRACT = "darwin-wrf-setups.study-requirements/1"
ASSESSMENT_CONTRACT = "darwin-wrf-setups.acceptance-assessment/1"
SYNTHETIC_PREFIX = "synthetic:"


class StudyContractError(ValueError):
    """A malformed, mismatched or untrusted setups-owned record."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def sha256(data: bytes) -> str:
    """Digest of exact bytes, never of a re-encoding."""
    return hashlib.sha256(data).hexdigest()


@cache
def _schema(name: str) -> dict:
    return json.loads(files("darwin").joinpath(f"setups-{name}-v1.schema.json").read_text())


def _parse(data: bytes, contract: str, schema: str, invalid: str) -> dict:
    try:
        document = json.loads(data)
    except (ValueError, TypeError) as error:
        raise StudyContractError(invalid, str(error)) from error
    if not isinstance(document, dict) or document.get("contract") != contract:
        raise StudyContractError("unsupported_contract", f"expected {contract}")
    try:
        jsonschema.Draft202012Validator(_schema(schema)).validate(document)
    except jsonschema.ValidationError as error:
        raise StudyContractError(invalid, error.message) from error
    return document


def refuse_synthetic(identifiers: Iterable[str], allow_synthetic: bool) -> None:
    """Refuse fixture-only identities unless the caller is explicitly in test mode."""
    if not allow_synthetic and any(i.startswith(SYNTHETIC_PREFIX) for i in identifiers):
        raise StudyContractError(
            "synthetic_not_trusted", "synthetic fixture identities are refused outside test mode"
        )


def parse_requirements(data: bytes, *, allow_synthetic: bool = False) -> dict:
    """Validated requirements document; structure and identities only."""
    document = _parse(data, REQUIREMENTS_CONTRACT, "study-requirements", "invalid_requirements")
    acceptance = document["acceptance"]
    for records in (
        document["sources"],
        acceptance["criteria"],
        acceptance["authorized_authorities"],
    ):
        ids = [record["id"] for record in records]
        if len(ids) != len(set(ids)):
            raise StudyContractError("ambiguous_identity", "duplicate identities in requirements")
    refuse_synthetic(
        (
            identifier
            for authority in acceptance["authorized_authorities"]
            for identifier in (authority["id"], authority["authorization"]["id"])
        ),
        allow_synthetic,
    )
    return document


def inspect_requirements(data: bytes, *, allow_synthetic: bool = False) -> dict:
    """Exact identity and review readiness of requirements bytes; not a decision."""
    document = parse_requirements(data, allow_synthetic=allow_synthetic)
    acceptance = document["acceptance"]
    return {
        "requirements": {
            "id": document["id"],
            "revision": document["revision"],
            "sha256": sha256(data),
        },
        "review_status": document["review_status"],
        "ready": document["review_status"] == "reviewed"
        and bool(acceptance["criteria"])
        and bool(acceptance["authorized_authorities"]),
        "scientific_status": "unassessed",
    }


def _identities(document: dict) -> list[str]:
    identifiers = [document["id"], document["dataset"]["id"], document["requirements"]["id"]]
    identifiers += [record["artifact"]["id"] for record in document["evidence"]]
    if document["authority"]:
        identifiers += [document["authority"]["id"], document["authority"]["authorization"]["id"]]
    if document["supersedes"]:
        identifiers.append(document["supersedes"]["id"])
    return identifiers


def validate_assessment(
    assessment: bytes,
    *,
    requirements: bytes,
    dataset: Mapping,
    evidence: Mapping[str, bytes],
    prior: bytes | None = None,
    allow_synthetic: bool = False,
) -> dict:
    """Check one record's exact bindings and report its state; never approve science."""
    document = _parse(
        assessment, ASSESSMENT_CONTRACT, "acceptance-assessment", "invalid_assessment"
    )
    synthetic = document.get("synthetic") is True or any(
        identifier.startswith(SYNTHETIC_PREFIX) for identifier in _identities(document)
    )
    if synthetic and not allow_synthetic:
        raise StudyContractError(
            "synthetic_not_trusted", "synthetic fixture records are refused outside test mode"
        )
    inspected = inspect_requirements(requirements, allow_synthetic=allow_synthetic)
    rules = json.loads(requirements)
    if document["requirements"] != inspected["requirements"]:
        raise StudyContractError(
            "requirements_mismatch", "record is not bound to these exact requirements bytes"
        )
    if document["dataset"] != dict(dataset):
        raise StudyContractError("dataset_mismatch", "record is bound to a different dataset")
    criteria = {c["id"]: c["evidence_type"] for c in rules["acceptance"]["criteria"]}
    for record in document["evidence"]:
        if criteria.get(record["criterion"]) != record["evidence_type"]:
            raise StudyContractError(
                "unknown_criterion", f"evidence cites unknown criterion {record['criterion']}"
            )
        artifact = record["artifact"]
        if artifact["id"] not in evidence:
            raise StudyContractError("evidence_unavailable", f"no bytes for {artifact['id']}")
        if sha256(evidence[artifact["id"]]) != artifact["sha256"]:
            raise StudyContractError("evidence_mismatch", f"evidence {artifact['id']} was altered")
    if document["status"] != "unassessed":
        if not inspected["ready"]:
            raise StudyContractError(
                "requirements_not_reviewed", "only reviewed requirements can carry a decision"
            )
        if document["authority"] not in rules["acceptance"]["authorized_authorities"]:
            raise StudyContractError(
                "unauthorized_authority", "authority is not authorized by these requirements"
            )
    if document["status"] == "accepted":
        if {record["criterion"] for record in document["evidence"]} != set(criteria):
            raise StudyContractError(
                "incomplete_evidence", "acceptance needs evidence for every criterion"
            )
    if document["supersedes"]:
        _check_prior(document, prior)
    return {
        "assessment": {"id": document["id"], "sha256": sha256(assessment)},
        "status": document["status"],
        "dataset": document["dataset"],
        "requirements": document["requirements"],
        "authority": document["authority"],
        "evidence": document["evidence"],
        "supersedes": document["supersedes"],
        "synthetic": synthetic,
    }


def _check_prior(document: dict, prior: bytes | None) -> None:
    if prior is None:
        raise StudyContractError("prior_unavailable", "superseded record bytes are required")
    reference = document["supersedes"]
    if sha256(prior) != reference["sha256"]:
        raise StudyContractError("prior_mismatch", "superseded record bytes were altered")
    earlier = _parse(prior, ASSESSMENT_CONTRACT, "acceptance-assessment", "invalid_assessment")
    if earlier["id"] != reference["id"] or earlier["dataset"] != document["dataset"]:
        raise StudyContractError("prior_mismatch", "superseded record is for another dataset")
    if any(record not in document["evidence"] for record in earlier["evidence"]):
        raise StudyContractError(
            "prior_evidence_dropped", "a reassessment must retain every prior evidence entry"
        )
