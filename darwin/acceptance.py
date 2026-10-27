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
* **Validation.** :func:`validate_assessment` (from
  :mod:`darwin.study_contracts`) checks a record's exact bindings. It never
  grants acceptance and cannot prove that an authorization is genuine.

Execution completeness and publication are recorded elsewhere (the product
handoff and the dashboard catalogue); nothing here reads or changes them.
Identities in the ``synthetic:`` namespace exist only for interoperability
fixtures and are refused unless the caller passes ``allow_synthetic=True``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from importlib.resources import files

import jsonschema

from darwin.study_contracts import (
    ASSESSMENT_CONTRACT,
    REQUIREMENTS_CONTRACT,
    SYNTHETIC_PREFIX,
    inspect_requirements,
    parse_requirements,
    refuse_synthetic,
    validate_assessment,
)
from darwin.study_contracts import StudyContractError as AcceptanceError
from darwin.study_contracts import sha256 as sha256_bytes

__all__ = [
    "ASSESSMENT_CONTRACT",
    "EVIDENCE_CONTRACT",
    "REQUIREMENTS_CONTRACT",
    "AcceptanceError",
    "canonical_bytes",
    "evidence_reference",
    "record_assessment",
    "run_checks",
    "sha256_bytes",
    "validate_assessment",
]

EVIDENCE_CONTRACT = "darwin.assessment-evidence/1"
STATUSES = ("accepted", "rejected", "unassessed")


def validate_schema(document: object, name: str, code: str) -> None:
    """Validate against a Darwin-owned schema in ``darwin/contracts``."""
    schema = json.loads(files("darwin").joinpath("contracts", name).read_text())
    try:
        jsonschema.Draft202012Validator(schema).validate(document)
    except jsonschema.ValidationError as error:
        raise AcceptanceError(code, error.message) from error


def canonical_bytes(document: object) -> bytes:
    """The exact bytes Darwin writes for a portable JSON record."""
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _is_synthetic(identifier: str) -> bool:
    return identifier.startswith(SYNTHETIC_PREFIX)


def _requirements(data: bytes, *, allow_synthetic: bool) -> tuple[dict, list[dict]]:
    """Exact identity and criteria of requirements bytes."""
    identity = inspect_requirements(data, allow_synthetic=allow_synthetic)["requirements"]
    criteria = parse_requirements(data, allow_synthetic=allow_synthetic)["acceptance"]["criteria"]
    return identity, criteria


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
    requirements_identity, criteria = _requirements(requirements, allow_synthetic=allow_synthetic)
    refuse_synthetic([identity["id"]], allow_synthetic)
    software = _software()
    receipts = []
    for criterion in criteria:
        check = CHECKS.get(criterion["check"])
        if check is None:
            raise AcceptanceError(
                "unsupported_check",
                f"criterion {criterion['id']} names check {criterion['check']!r},"
                " which Darwin does not implement",
            )
        synthetic = criterion["check"].startswith("synthetic.")
        refuse_synthetic(["synthetic:"] if synthetic else [], allow_synthetic)
        outcomes = inputs.get("synthetic:outcomes", {}) if synthetic else {}
        result = check(
            identity,
            criterion["parameters"],
            {**inputs, "criterion_outcome": outcomes.get(criterion["id"], "pass")},
        )
        receipt = {
            "contract": EVIDENCE_CONTRACT,
            "dataset": identity,
            "requirements": requirements_identity,
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
        validate_schema(receipt, "assessment-evidence-v1.schema.json", "invalid_evidence")
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
    validate_schema(receipt, "assessment-evidence-v1.schema.json", "invalid_evidence")
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
    requirements_identity, _ = _requirements(requirements, allow_synthetic=allow_synthetic)
    cited: dict[str, bytes] = dict(prior_evidence or {})
    entries: list[dict] = []
    if prior is not None:
        try:
            entries.extend(json.loads(prior)["evidence"])
        except (ValueError, KeyError, TypeError) as error:
            raise AcceptanceError("invalid_assessment", "prior record is malformed") from error
    for receipt_bytes in evidence:
        receipt = json.loads(receipt_bytes)
        reference = evidence_reference(receipt_bytes)
        if receipt["dataset"] != identity:
            raise AcceptanceError("dataset_mismatch", "evidence is for a different dataset")
        if receipt["requirements"] != requirements_identity:
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
        "requirements": requirements_identity,
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
