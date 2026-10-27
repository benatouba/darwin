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

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping
from importlib.resources import files
from pathlib import Path

import jsonschema

from darwin.study_contracts import (
    ASSESSMENT_CONTRACT,
    EVIDENCE_CONTRACT,
    REQUIREMENTS_CONTRACT,
    SYNTHETIC_PREFIX,
    ArtifactKey,
    inspect_requirements,
    parse_requirements,
    refuse_synthetic,
    schema_validator,
    validate_assessment,
    validate_evidence_schema,
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
    "main",
    "read_receipt",
    "record_assessment",
    "run_checks",
    "sha256_bytes",
    "validate_assessment",
]

RECEIPT_VERSION = "1"
"""Version under which Darwin receipts are cited; their ids are content addresses."""
STATUSES = ("accepted", "rejected", "unassessed")


def validate_schema(document: object, name: str, code: str) -> None:
    """Validate against a Darwin-owned schema in ``darwin/contracts``."""
    try:
        schema_validator("contracts", name).validate(document)
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
        or identity["version"].lower() in {"latest", "current", "head"}
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
        if synthetic and not allow_synthetic:
            raise AcceptanceError(
                "synthetic_not_trusted", "synthetic checks run only in explicit test mode"
            )
        check_inputs = dict(inputs)
        if synthetic:
            # Test mode states every synthetic outcome; an unlisted one defaults to pass
            # only because the fixture criterion exists solely to exercise the record flow.
            check_inputs["criterion_outcome"] = inputs.get("synthetic:outcomes", {}).get(
                criterion["id"], "pass"
            )
        result = check(identity, criterion["parameters"], check_inputs)
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
        validate_evidence_schema(receipt)
        receipts.append(receipt)
    return receipts


def read_receipt(receipt_bytes: bytes) -> dict:
    """Parse and schema-check exact ``darwin.assessment-evidence/1`` receipt bytes."""
    try:
        receipt = json.loads(receipt_bytes)
    except (ValueError, TypeError) as error:
        raise AcceptanceError("invalid_evidence", f"receipt is not JSON: {error}") from error
    if not isinstance(receipt, dict) or receipt.get("contract") != EVIDENCE_CONTRACT:
        raise AcceptanceError("unsupported_contract", f"expected {EVIDENCE_CONTRACT}")
    validate_evidence_schema(receipt)
    return receipt


def evidence_reference(receipt_bytes: bytes) -> dict:
    """Cite one exact Darwin evidence receipt in an acceptance-assessment record."""
    receipt = read_receipt(receipt_bytes)
    return {
        "criterion": receipt["criterion"],
        "evidence_type": receipt["evidence_type"],
        "artifact": {
            "id": receipt["id"],
            "version": RECEIPT_VERSION,
            "sha256": sha256_bytes(receipt_bytes),
        },
    }


def record_assessment(  # noqa: PLR0913
    *,
    assessment_id: str,
    dataset: Mapping,
    requirements: bytes,
    status: str,
    evidence: Iterable[bytes],
    authority: Mapping | None,
    authorization: bytes | None = None,
    prior: bytes | None = None,
    prior_evidence: Mapping[ArtifactKey, bytes] | None = None,
    allow_synthetic: bool = False,
) -> bytes:
    """Return the exact bytes of a new acceptance-assessment record.

    ``accepted``/``rejected`` need ``authority``: a reviewer or explicitly
    authorized policy listed by these exact requirements, plus ``authorization``,
    the exact bytes of the authorization artifact whose sha256 the requirements
    pin for it. Darwin never infers either, and execution success or check
    outcomes alone never set a status. Holding those bytes is not proof that
    the reviewer or policy issued them: authenticating an authorization is a
    deployment/human gate outside Darwin (see docs/integration-spec.md).
    ``evidence`` are exact Darwin receipt bytes for this dataset and
    requirements. A reassessment passes the exact ``prior`` record bytes and
    the bytes of the evidence it cites (``prior_evidence``, keyed by
    ``(id, version)``); the new record
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
    cited: dict[ArtifactKey, bytes] = dict(prior_evidence or {})
    entries: list[dict] = []
    prior_id = None
    if prior is not None:
        try:
            prior_document = json.loads(prior)
            entries.extend(prior_document["evidence"])
            prior_id = prior_document["id"]
        except (ValueError, KeyError, TypeError) as error:
            raise AcceptanceError("invalid_assessment", "prior record is malformed") from error
    for receipt_bytes in evidence:
        receipt = read_receipt(receipt_bytes)
        reference = evidence_reference(receipt_bytes)
        if receipt["dataset"] != identity:
            raise AcceptanceError("dataset_mismatch", "evidence is for a different dataset")
        if receipt["requirements"] != requirements_identity:
            raise AcceptanceError(
                "requirements_mismatch", "evidence was checked against other requirements"
            )
        cited[(reference["artifact"]["id"], RECEIPT_VERSION)] = receipt_bytes
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
        "supersedes": None if prior is None else {"id": prior_id, "sha256": sha256_bytes(prior)},
    }
    data = canonical_bytes(document)
    validate_assessment(
        data,
        requirements=requirements,
        dataset=identity,
        evidence=cited,
        authorizations={}
        if authority is None or authorization is None
        else {
            (authority["authorization"]["id"], authority["authorization"]["version"]): (
                authorization
            )
        },
        prior=prior,
        allow_synthetic=allow_synthetic,
    )
    return data


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise AcceptanceError("invalid_arguments", f"{path}: {error}") from error


def _read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as error:
        raise AcceptanceError("invalid_arguments", f"{path}: {error}") from error


def _write_new(path: Path, data: bytes) -> None:
    """Write a new immutable record; an existing file is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as error:
        raise AcceptanceError("record_exists", f"{path} already exists") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m darwin.acceptance", description=__doc__.splitlines()[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    checks = commands.add_parser("run-checks", help="write one evidence receipt per criterion")
    checks.add_argument("--inputs", help="JSON deployment-local check inputs")
    checks.add_argument("--out-dir", type=Path, required=True)
    record = commands.add_parser("record", help="write a new acceptance-assessment record")
    record.add_argument("--assessment-id", required=True)
    record.add_argument("--status", required=True, choices=STATUSES)
    record.add_argument("--evidence", nargs="*", default=[], help="receipt files")
    record.add_argument("--authority", help="JSON authority entry from the requirements")
    record.add_argument("--authorization", help="exact authorization artifact bytes")
    record.add_argument("--prior", help="exact prior record bytes")
    record.add_argument("--prior-evidence", nargs="*", default=[], help="receipts the prior cites")
    record.add_argument("--output", type=Path, required=True)
    for command in (checks, record):
        command.add_argument("--dataset", required=True, help="JSON {id, version, sha256}")
        command.add_argument("--requirements", required=True, help="exact requirements bytes")
        command.add_argument("--test-mode", action="store_true", help="accept synthetic fixtures")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """CLI over the same functions: ``run-checks`` and ``record``.

    Validation of an existing record is ``python -m darwin.handoff validate``
    (as part of a handoff) or :func:`validate_assessment` from Python.
    """
    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        dataset = _read_json(args.dataset)
        requirements = _read_bytes(args.requirements)
        if args.command == "run-checks":
            receipts = run_checks(
                dataset,
                requirements,
                inputs=_read_json(args.inputs) if args.inputs else {},
                allow_synthetic=args.test_mode,
            )
            written = []
            for receipt in receipts:
                data = canonical_bytes(receipt)
                path = args.out_dir / f"{sha256_bytes(data)}.json"
                if not (path.exists() and path.read_bytes() == data):
                    _write_new(path, data)
                written.append(str(path))
            result = {"contract": EVIDENCE_CONTRACT, "receipts": written}
        else:
            prior_evidence = {}
            for path in args.prior_evidence:
                data = _read_bytes(path)
                prior_evidence[(read_receipt(data)["id"], RECEIPT_VERSION)] = data
            data = record_assessment(
                assessment_id=args.assessment_id,
                dataset=dataset,
                requirements=requirements,
                status=args.status,
                evidence=[_read_bytes(path) for path in args.evidence],
                authority=_read_json(args.authority) if args.authority else None,
                authorization=_read_bytes(args.authorization) if args.authorization else None,
                prior=_read_bytes(args.prior) if args.prior else None,
                prior_evidence=prior_evidence,
                allow_synthetic=args.test_mode,
            )
            _write_new(args.output, data)
            result = {"contract": ASSESSMENT_CONTRACT, "sha256": sha256_bytes(data)}
    except AcceptanceError as error:
        sys.stderr.write(json.dumps({"code": error.code, "message": str(error)}) + "\n")
        raise SystemExit(2) from error
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
