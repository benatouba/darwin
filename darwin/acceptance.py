"""Portable scientific acceptance: checks, evidence and authority-bound decisions.

Three separate things, never one flag:

* **Evidence.** :func:`run_checks` runs the checks named by one exact
  setups requirements revision (``darwin-wrf-setups.study-requirements/1``)
  against one exact dataset identity and returns ``darwin.assessment-evidence/1``
  receipts. A receipt reports a check outcome; it carries no authority and is
  never acceptance.
* **Decision.** :func:`record_assessment` writes a
  ``darwin-wrf-setups.acceptance-assessment/1`` record (setups owns that
  contract; Darwin produces and consumes it). ``accepted``/``rejected`` need a
  reviewer or policy that the exact requirements list as authorized;
  ``unassessed`` has no authority. A reassessment supersedes its prior record
  by exact bytes and keeps every prior evidence entry.
* **Validation.** :func:`validate_assessment` checks a record's exact
  bindings (requirements bytes, dataset identity, evidence bytes, prior
  record, authority). It never grants acceptance and cannot prove that an
  authorization is genuine.

Execution completeness and publication are recorded elsewhere (the product
handoff and the dashboard catalogue); nothing here reads or changes them.

Identities in the ``synthetic:`` namespace exist only for interoperability
fixtures and are refused unless the caller passes ``allow_synthetic=True``.
The setups schemas are vendored verbatim under ``darwin/contracts/setups``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from importlib.resources import files

import jsonschema

ASSESSMENT_CONTRACT = "darwin-wrf-setups.acceptance-assessment/1"
REQUIREMENTS_CONTRACT = "darwin-wrf-setups.study-requirements/1"
EVIDENCE_CONTRACT = "darwin.assessment-evidence/1"
SYNTHETIC_PREFIX = "synthetic:"
STATUSES = ("accepted", "rejected", "unassessed")
OUTCOMES = ("pass", "fail")


class AcceptanceError(ValueError):
    """A malformed, mismatched, unauthorized or untrusted acceptance input."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _schema(name: str) -> dict:
    return json.loads(files("darwin").joinpath("contracts", name).read_text())


def _validate_schema(document: object, name: str, code: str) -> None:
    try:
        jsonschema.Draft202012Validator(_schema(name)).validate(document)
    except jsonschema.ValidationError as error:
        raise AcceptanceError(code, error.message) from error


def sha256_bytes(data: bytes) -> str:
    """Lowercase hex sha256 of exact bytes (no re-encoding)."""
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(document: object) -> bytes:
    """The exact bytes Darwin writes for a portable JSON record."""
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _is_synthetic(identifier: str) -> bool:
    return identifier.startswith(SYNTHETIC_PREFIX)


def _refuse_synthetic(identifiers: Iterable[str], allow_synthetic: bool) -> None:
    if not allow_synthetic and any(map(_is_synthetic, identifiers)):
        raise AcceptanceError(
            "synthetic_not_trusted", "synthetic fixture identities are refused outside test mode"
        )


def inspect_requirements(data: bytes, *, allow_synthetic: bool = False) -> dict:
    """Return the exact identity of requirements bytes and their parsed rules.

    ``reviewed`` is true only for ``review_status: reviewed`` requirements that
    name criteria and authorized authorities; only those can carry a decision.
    """
    try:
        document = json.loads(data)
    except ValueError as error:
        raise AcceptanceError("invalid_requirements", str(error)) from error
    if not isinstance(document, dict) or document.get("contract") != REQUIREMENTS_CONTRACT:
        raise AcceptanceError("unsupported_contract", f"expected {REQUIREMENTS_CONTRACT}")
    _validate_schema(document, "setups/study-requirements-v1.schema.json", "invalid_requirements")
    rules = document["acceptance"]
    for records in (document["sources"], rules["criteria"], rules["authorized_authorities"]):
        ids = [record["id"] for record in records]
        if len(ids) != len(set(ids)):
            raise AcceptanceError("ambiguous_identity", "duplicate identities in requirements")
    _refuse_synthetic(
        (
            identifier
            for authority in rules["authorized_authorities"]
            for identifier in (authority["id"], authority["authorization"]["id"])
        ),
        allow_synthetic,
    )
    return {
        "identity": {
            "id": document["id"],
            "revision": document["revision"],
            "sha256": sha256_bytes(data),
        },
        "reviewed": document["review_status"] == "reviewed"
        and bool(rules["criteria"])
        and bool(rules["authorized_authorities"]),
        "criteria": rules["criteria"],
        "authorities": rules["authorized_authorities"],
    }


def _parse_assessment(data: bytes) -> dict:
    try:
        document = json.loads(data)
    except ValueError as error:
        raise AcceptanceError("invalid_assessment", str(error)) from error
    if not isinstance(document, dict) or document.get("contract") != ASSESSMENT_CONTRACT:
        raise AcceptanceError("unsupported_contract", f"expected {ASSESSMENT_CONTRACT}")
    _validate_schema(document, "setups/acceptance-assessment-v1.schema.json", "invalid_assessment")
    return document


def _assessment_identities(document: dict) -> list[str]:
    identifiers = [document["id"], document["dataset"]["id"], document["requirements"]["id"]]
    identifiers += [record["artifact"]["id"] for record in document["evidence"]]
    if document["authority"]:
        identifiers += [document["authority"]["id"], document["authority"]["authorization"]["id"]]
    if document["supersedes"]:
        identifiers.append(document["supersedes"]["id"])
    return identifiers


def validate_assessment(  # noqa: PLR0913
    assessment: bytes,
    *,
    requirements: bytes,
    dataset: Mapping,
    evidence: Mapping[str, bytes],
    prior: bytes | None = None,
    allow_synthetic: bool = False,
) -> dict:
    """Check one assessment record's exact bindings and return its state.

    ``requirements`` are the exact requirements bytes, ``dataset`` the identity
    the record must be attached to, ``evidence`` the exact bytes of each cited
    artifact keyed by id, and ``prior`` the exact bytes of a superseded record.
    Validation never approves science.
    """
    document = _parse_assessment(assessment)
    synthetic = document.get("synthetic") is True or any(
        map(_is_synthetic, _assessment_identities(document))
    )
    if synthetic and not allow_synthetic:
        raise AcceptanceError(
            "synthetic_not_trusted", "synthetic fixture records are refused outside test mode"
        )
    inspected = inspect_requirements(requirements, allow_synthetic=allow_synthetic)
    if document["requirements"] != inspected["identity"]:
        raise AcceptanceError(
            "requirements_mismatch", "record is not bound to these exact requirements bytes"
        )
    if document["dataset"] != dict(dataset):
        raise AcceptanceError("dataset_mismatch", "record is bound to a different dataset")
    criteria = {c["id"]: c["evidence_type"] for c in inspected["criteria"]}
    for record in document["evidence"]:
        if criteria.get(record["criterion"]) != record["evidence_type"]:
            raise AcceptanceError(
                "unknown_criterion", f"evidence cites unknown criterion {record['criterion']}"
            )
        artifact = record["artifact"]
        if artifact["id"] not in evidence:
            raise AcceptanceError("evidence_unavailable", f"no bytes for {artifact['id']}")
        if sha256_bytes(evidence[artifact["id"]]) != artifact["sha256"]:
            raise AcceptanceError("evidence_mismatch", f"evidence {artifact['id']} was altered")
    if document["status"] != "unassessed":
        _check_decision(document, inspected)
    if document["supersedes"]:
        _check_prior(document, prior)
    return {
        "assessment": {"id": document["id"], "sha256": sha256_bytes(assessment)},
        "status": document["status"],
        "dataset": document["dataset"],
        "requirements": document["requirements"],
        "authority": document["authority"],
        "evidence": document["evidence"],
        "supersedes": document["supersedes"],
        "synthetic": synthetic,
    }


def _check_decision(document: dict, inspected: dict) -> None:
    if not inspected["reviewed"]:
        raise AcceptanceError(
            "requirements_not_reviewed", "only reviewed requirements can carry a decision"
        )
    if document["authority"] not in inspected["authorities"]:
        raise AcceptanceError(
            "unauthorized_authority", "authority is not authorized by these requirements"
        )
    if document["status"] == "accepted":
        covered = {record["criterion"] for record in document["evidence"]}
        if covered != {criterion["id"] for criterion in inspected["criteria"]}:
            raise AcceptanceError(
                "incomplete_evidence", "acceptance needs evidence for every criterion"
            )


def _check_prior(document: dict, prior: bytes | None) -> None:
    """A reassessment cites its prior record exactly and keeps all prior evidence."""
    if prior is None:
        raise AcceptanceError("prior_unavailable", "superseded record bytes are required")
    reference = document["supersedes"]
    if sha256_bytes(prior) != reference["sha256"]:
        raise AcceptanceError("prior_mismatch", "superseded record bytes were altered")
    earlier = _parse_assessment(prior)
    if earlier["id"] != reference["id"] or earlier["dataset"] != document["dataset"]:
        raise AcceptanceError("prior_mismatch", "superseded record is for another record/dataset")
    if any(record not in document["evidence"] for record in earlier["evidence"]):
        raise AcceptanceError(
            "prior_evidence_dropped", "a reassessment must retain every prior evidence entry"
        )


CheckFunction = Callable[[dict, dict, Mapping], dict]
"""``check(dataset, parameters, inputs) -> {"outcome": "pass"|"fail", "details": {...}}``."""


def _synthetic_test_only(_dataset: dict, _parameters: dict, inputs: Mapping) -> dict:
    """Interoperability stand-in: reports the outcome the test supplies, nothing else."""
    return {"outcome": inputs["criterion_outcome"], "details": {"synthetic": True}}


# Darwin has no reviewed scientific check bound to a requirements revision yet.
# Requirements naming any other check fail with ``unsupported_check``; a
# supported check is added here only together with its scientific review.
CHECKS: dict[str, CheckFunction] = {"synthetic.test-only": _synthetic_test_only}


def _check_dataset(dataset: Mapping) -> dict:
    identity = dict(dataset)
    if (
        set(identity) != {"id", "version", "sha256"}
        or not all(isinstance(value, str) and value for value in identity.values())
        or identity["version"] in {"latest", "current", "HEAD"}
        or len(identity["sha256"]) != 64  # noqa: PLR2004
        or identity["sha256"].strip("0123456789abcdef")
    ):
        raise AcceptanceError(
            "invalid_dataset", "dataset needs an exact id, pinned version and lowercase sha256"
        )
    return identity


def _software() -> dict:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        darwin_version = version("darwin")
    except PackageNotFoundError:  # pragma: no cover - source tree without metadata
        darwin_version = "unknown"
    implementation = files("darwin").joinpath("acceptance.py").read_bytes()
    return {"darwin": darwin_version, "check_implementation_sha256": sha256_bytes(implementation)}


def run_checks(
    dataset: Mapping,
    requirements: bytes,
    *,
    inputs: Mapping,
    allow_synthetic: bool = False,
) -> list[dict]:
    """Run every criterion's named check and return one evidence receipt each.

    ``requirements`` are the exact setups requirements bytes; ``inputs`` hold the
    deployment-local data the checks read. Receipts are evidence only. Write
    them with :func:`canonical_bytes` and cite them by those bytes' sha256.
    """
    identity = _check_dataset(dataset)
    inspected = inspect_requirements(requirements, allow_synthetic=allow_synthetic)
    _refuse_synthetic([identity["id"]], allow_synthetic)
    software = _software()
    receipts = []
    for criterion in inspected["criteria"]:
        check = CHECKS.get(criterion["check"])
        if check is None:
            raise AcceptanceError(
                "unsupported_check",
                f"criterion {criterion['id']} names check {criterion['check']!r},"
                " which Darwin does not implement",
            )
        synthetic = criterion["check"].startswith("synthetic.")
        _refuse_synthetic(["synthetic:"] if synthetic else [], allow_synthetic)
        outcomes = inputs.get("synthetic:outcomes", {}) if synthetic else {}
        result = check(
            identity,
            criterion["parameters"],
            {**inputs, "criterion_outcome": outcomes.get(criterion["id"], "pass")},
        )
        receipt = {
            "contract": EVIDENCE_CONTRACT,
            "dataset": identity,
            "requirements": inspected["identity"],
            "criterion": criterion["id"],
            "evidence_type": criterion["evidence_type"],
            "check": criterion["check"],
            "parameters": criterion["parameters"],
            "outcome": result["outcome"],
            "details": result["details"],
            "software": software,
            "synthetic": synthetic or _is_synthetic(identity["id"]),
        }
        # Content-addressed: a rerun with another outcome is a different receipt.
        receipt["id"] = f"darwin-evidence:{sha256_bytes(canonical_bytes(receipt))}"
        _validate_schema(receipt, "assessment-evidence-v1.schema.json", "invalid_evidence")
        receipts.append(receipt)
    return receipts


def evidence_reference(receipt_bytes: bytes) -> dict:
    """Cite one exact Darwin evidence receipt in an acceptance-assessment record."""
    try:
        receipt = json.loads(receipt_bytes)
    except ValueError as error:
        raise AcceptanceError("invalid_evidence", str(error)) from error
    if not isinstance(receipt, dict) or receipt.get("contract") != EVIDENCE_CONTRACT:
        raise AcceptanceError("unsupported_contract", f"expected {EVIDENCE_CONTRACT}")
    _validate_schema(receipt, "assessment-evidence-v1.schema.json", "invalid_evidence")
    return {
        "criterion": receipt["criterion"],
        "evidence_type": receipt["evidence_type"],
        "artifact": {"id": receipt["id"], "version": "1", "sha256": sha256_bytes(receipt_bytes)},
    }


def record_assessment(  # noqa: PLR0913
    *,
    assessment_id: str,
    dataset: Mapping,
    requirements: bytes,
    status: str,
    evidence: Iterable[bytes],
    authority: Mapping | None,
    prior: bytes | None = None,
    prior_evidence: Mapping[str, bytes] | None = None,
    allow_synthetic: bool = False,
) -> bytes:
    """Return the exact bytes of a new acceptance-assessment record.

    ``accepted``/``rejected`` need ``authority``: a reviewer or explicitly
    authorized policy listed by these exact requirements. Darwin never infers
    one, and execution success or check outcomes alone never set a status.
    ``evidence`` are exact Darwin receipt bytes for this dataset and
    requirements. A reassessment passes the exact ``prior`` record bytes and
    the bytes of the evidence it cites (``prior_evidence``); the new record
    supersedes it and keeps every prior evidence entry. The prior record is
    never modified. Store the result as a new immutable artifact.
    """
    if status not in STATUSES:
        raise AcceptanceError("invalid_assessment", f"status must be one of {STATUSES}")
    if status != "unassessed" and authority is None:
        raise AcceptanceError(
            "missing_authority", "only a reviewer or authorized policy can accept or reject"
        )
    identity = _check_dataset(dataset)
    inspected = inspect_requirements(requirements, allow_synthetic=allow_synthetic)
    cited: dict[str, bytes] = dict(prior_evidence or {})
    entries: list[dict] = []
    if prior is not None:
        entries.extend(_parse_assessment(prior)["evidence"])
    for receipt_bytes in evidence:
        receipt = json.loads(receipt_bytes)
        reference = evidence_reference(receipt_bytes)
        if receipt["dataset"] != identity:
            raise AcceptanceError("dataset_mismatch", "evidence is for a different dataset")
        if receipt["requirements"] != inspected["identity"]:
            raise AcceptanceError(
                "requirements_mismatch", "evidence was checked against other requirements"
            )
        cited[reference["artifact"]["id"]] = receipt_bytes
        if reference not in entries:
            entries.append(reference)
    synthetic = _is_synthetic(assessment_id) or _is_synthetic(identity["id"])
    document = {
        "contract": ASSESSMENT_CONTRACT,
        "id": assessment_id,
        **({"synthetic": True} if synthetic else {}),
        "dataset": identity,
        "requirements": inspected["identity"],
        "status": status,
        "evidence": entries,
        "authority": dict(authority) if authority is not None else None,
        "supersedes": None
        if prior is None
        else {"id": json.loads(prior)["id"], "sha256": sha256_bytes(prior)},
    }
    data = canonical_bytes(document)
    validate_assessment(
        data,
        requirements=requirements,
        dataset=identity,
        evidence=cited,
        prior=prior,
        allow_synthetic=allow_synthetic,
    )
    return data
