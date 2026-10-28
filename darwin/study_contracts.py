"""Consumer validation of setups-owned study requirements and acceptance records.

``darwin-wrf-setups`` owns ``study-requirements/1`` and ``acceptance-assessment/1``.
Darwin vendors their schemas byte-for-byte and checks exact bindings, including
the detached SSH signature (SSHSIG, namespace ``darwin-acceptance``) that an
``accepted``/``rejected`` decision needs from a key the reviewed requirements pin
for its authority. Validation never grants acceptance. Identities in the
``synthetic:`` namespace are refused outside explicit test mode.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from functools import cache
from importlib.resources import files
from pathlib import Path

import jsonschema

from darwin.decision_signature import SignatureError, verify_signature
from darwin.errors import CodedError

REQUIREMENTS_CONTRACT = "darwin-wrf-setups.study-requirements/1"
ASSESSMENT_CONTRACT = "darwin-wrf-setups.acceptance-assessment/1"
SYNTHETIC_PREFIX = "synthetic:"
EVIDENCE_CONTRACT = "darwin.assessment-evidence/1"
EVIDENCE_SCHEMA = "assessment-evidence-v1.schema.json"

ArtifactKey = tuple[str, str]
"""``(id, version)`` of a cited artifact whose exact bytes the caller supplies."""


class StudyContractError(CodedError):
    """A malformed, mismatched or untrusted setups-owned record."""


def sha256(data: bytes) -> str:
    """Digest of exact bytes, never of a re-encoding."""
    return hashlib.sha256(data).hexdigest()


@cache
def schema_validator(*parts: str) -> jsonschema.Draft202012Validator:
    """Cached validator for a schema shipped inside the ``darwin`` package."""
    return jsonschema.Draft202012Validator(json.loads(files("darwin").joinpath(*parts).read_text()))


def _parse(data: bytes, contract: str, schema: str, invalid: str) -> dict:
    try:
        document = json.loads(data)
    except (ValueError, TypeError) as error:
        raise StudyContractError(invalid, str(error)) from error
    if not isinstance(document, dict) or document.get("contract") != contract:
        raise StudyContractError("unsupported_contract", f"expected {contract}")
    try:
        schema_validator(f"setups-{schema}-v1.schema.json").validate(document)
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


def _parse_assessment(data: bytes) -> dict:
    return _parse(data, ASSESSMENT_CONTRACT, "acceptance-assessment", "invalid_assessment")


def _refuse_untrusted(document: dict, allow_synthetic: bool) -> bool:
    """Return whether a record is synthetic, refusing it outside test mode."""
    synthetic = document.get("synthetic") is True or any(
        identifier.startswith(SYNTHETIC_PREFIX) for identifier in _identities(document)
    )
    if synthetic and not allow_synthetic:
        raise StudyContractError(
            "synthetic_not_trusted", "synthetic fixture records are refused outside test mode"
        )
    return synthetic


def _cited_bytes(identity: Mapping, available: Mapping[ArtifactKey, bytes], kind: str) -> bytes:
    """The exact bytes of a cited ``{id, version, sha256}`` identity, present and unaltered."""
    key = (identity["id"], identity["version"])
    if key not in available:
        raise StudyContractError(f"{kind}_unavailable", f"no bytes for {key[0]} version {key[1]}")
    data = available[key]
    if sha256(data) != identity["sha256"]:
        raise StudyContractError(
            f"{kind}_mismatch", f"{kind} {key[0]} version {key[1]} was altered"
        )
    return data


def validate_evidence_schema(receipt: object) -> None:
    """Validate a ``darwin.assessment-evidence/1`` receipt against its Darwin schema."""
    try:
        schema_validator("contracts", EVIDENCE_SCHEMA).validate(receipt)
    except jsonschema.ValidationError as error:
        raise StudyContractError("invalid_evidence", error.message) from error


def _darwin_receipt(data: bytes) -> dict | None:
    """A cited Darwin receipt, or ``None`` for evidence of another (opaque) type."""
    try:
        receipt = json.loads(data)
    except ValueError, TypeError:
        return None
    if not isinstance(receipt, dict) or receipt.get("contract") != EVIDENCE_CONTRACT:
        return None
    validate_evidence_schema(receipt)
    return receipt


def _check_receipt(receipt: dict, record: Mapping, document: Mapping) -> None:
    """A Darwin receipt reports a check of this dataset, requirements and criterion."""
    if receipt["id"] != record["artifact"]["id"]:
        raise StudyContractError(
            "evidence_mismatch", f"receipt {receipt['id']} is cited under another id"
        )
    if receipt["dataset"] != document["dataset"]:
        raise StudyContractError(
            "evidence_dataset_mismatch", f"evidence {receipt['id']} checked another dataset"
        )
    if receipt["requirements"] != document["requirements"]:
        raise StudyContractError(
            "evidence_requirements_mismatch",
            f"evidence {receipt['id']} was checked against other requirements",
        )
    if (receipt["criterion"], receipt["evidence_type"]) != (
        record["criterion"],
        record["evidence_type"],
    ):
        raise StudyContractError(
            "evidence_criterion_mismatch", f"evidence {receipt['id']} reports another criterion"
        )


def _evidence_outcomes(
    document: dict, criteria: Mapping[str, str], evidence: Mapping[ArtifactKey, bytes]
) -> dict[str, list[str | None]]:
    """Check every cited evidence entry; return Darwin outcomes (``None``: opaque) by criterion."""
    outcomes: dict[str, list[str | None]] = {}
    for record in document["evidence"]:
        if record["criterion"] not in criteria:
            raise StudyContractError(
                "unknown_criterion", f"evidence cites unknown criterion {record['criterion']}"
            )
        if criteria[record["criterion"]] != record["evidence_type"]:
            raise StudyContractError(
                "evidence_type_mismatch",
                f"criterion {record['criterion']} needs {criteria[record['criterion']]} evidence",
            )
        receipt = _darwin_receipt(_cited_bytes(record["artifact"], evidence, "evidence"))
        if receipt is not None:
            _check_receipt(receipt, record, document)
        outcomes.setdefault(record["criterion"], []).append(
            None if receipt is None else receipt["outcome"]
        )
    return outcomes


def _check_acceptance(outcomes: Mapping[str, list[str | None]], criteria: Mapping) -> None:
    """Acceptance covers every criterion and never rests only on failed Darwin checks."""
    if set(outcomes) != set(criteria):
        raise StudyContractError(
            "incomplete_evidence", "acceptance needs evidence for every criterion"
        )
    failing = sorted(
        criterion
        for criterion, results in outcomes.items()
        if None not in results and "pass" not in results
    )
    if failing:
        raise StudyContractError("failing_evidence", f"no cited check passed for {failing}")


def validate_assessment(  # noqa: PLR0913
    assessment: bytes,
    *,
    requirements: bytes,
    dataset: Mapping,
    evidence: Mapping[ArtifactKey, bytes],
    authorizations: Mapping[ArtifactKey, bytes] | None = None,
    prior: bytes | None = None,
    signature: bytes | None = None,
    allow_synthetic: bool = False,
) -> dict:
    """Check one record's exact bindings and signature and report its state.

    ``dataset`` is the caller's verified product identity (never taken from the
    record). ``evidence`` and ``authorizations`` hold the exact bytes of every
    cited artifact keyed by ``(id, version)``; ``prior`` the exact bytes of a
    superseded record; ``signature`` the detached armored SSHSIG over
    ``assessment`` (the ``<record>.sig`` file ``ssh-keygen -Y sign -n
    darwin-acceptance`` writes).

    A decision must cite authorization bytes whose sha256 the requirements pin
    and carry a signature over its exact bytes by one of the ``signing_keys``
    the reviewed requirements pin for its authority. Refusals:
    ``signature_unavailable``, ``signature_invalid``,
    ``signature_namespace_mismatch``, ``signer_not_authorized`` (including a key
    pinned for another authority) and ``signature_mismatch``. An unassessed
    record carries no signature (``unexpected_signature``). The report's
    ``signer`` is the pinned key that signed, or ``None``. Trust rests on the
    review that pinned the keys; validation never judges the science.

    Cited Darwin receipts (``darwin.assessment-evidence/1``) must report this
    dataset, requirements and criterion, and an ``accepted`` record needs a
    passing receipt for every criterion whose cited evidence is all Darwin
    receipts (a failed attempt may stay cited next to a passing rerun). Other
    evidence types are opaque and checked by exact bytes only.
    """
    return check_assessment(
        assessment,
        requirements=requirements,
        dataset=dataset,
        evidence=evidence,
        authorizations=authorizations,
        prior=prior,
        signature=signature,
        allow_synthetic=allow_synthetic,
    )


def check_assessment(  # noqa: PLR0913
    assessment: bytes,
    *,
    requirements: bytes,
    dataset: Mapping,
    evidence: Mapping[ArtifactKey, bytes],
    authorizations: Mapping[ArtifactKey, bytes] | None = None,
    prior: bytes | None = None,
    signature: bytes | None = None,
    allow_synthetic: bool = False,
    verify_signature_bytes: bool = True,
) -> dict:
    """:func:`validate_assessment`, optionally without the decision signature step.

    ``verify_signature_bytes=False`` exists only for
    :func:`darwin.acceptance.record_assessment`, which checks a decision before the
    reviewer has signed it; a record checked that way is not a valid decision.
    """
    document = _parse_assessment(assessment)
    synthetic = _refuse_untrusted(document, allow_synthetic)
    inspected = inspect_requirements(requirements, allow_synthetic=allow_synthetic)
    rules = json.loads(requirements)
    if document["requirements"] != inspected["requirements"]:
        raise StudyContractError(
            "requirements_mismatch", "record is not bound to these exact requirements bytes"
        )
    if document["dataset"] != dict(dataset):
        raise StudyContractError("dataset_mismatch", "record is bound to a different dataset")
    criteria = {c["id"]: c["evidence_type"] for c in rules["acceptance"]["criteria"]}
    outcomes = _evidence_outcomes(document, criteria, evidence)
    signer = None
    if document["status"] != "unassessed":
        if not inspected["ready"]:
            raise StudyContractError(
                "requirements_not_reviewed", "only reviewed requirements can carry a decision"
            )
        authority = authorized_authority(document["authority"], rules)
        _cited_bytes(document["authority"]["authorization"], authorizations or {}, "authorization")
        if verify_signature_bytes:
            signer = _check_signature(assessment, signature, authority["signing_keys"])
    elif signature is not None:
        raise StudyContractError(
            "unexpected_signature",
            "an unassessed record is not a decision and carries no signature",
        )
    if document["status"] == "accepted":
        _check_acceptance(outcomes, criteria)
    if document["supersedes"]:
        _check_prior(document, prior, allow_synthetic)
    elif prior is not None:
        raise StudyContractError(
            "unexpected_prior", "record supersedes nothing but a prior was given"
        )
    return {
        "assessment": {"id": document["id"], "sha256": sha256(assessment)},
        "status": document["status"],
        "dataset": document["dataset"],
        "requirements": document["requirements"],
        "authority": document["authority"],
        "evidence": document["evidence"],
        "supersedes": document["supersedes"],
        "signer": signer,
        "synthetic": synthetic,
    }


def authorized_authority(authority: Mapping, rules: Mapping) -> dict:
    """The requirements entry naming exactly this authority (kind, id, authorization)."""
    for entry in rules["acceptance"]["authorized_authorities"]:
        if all(entry[field] == authority[field] for field in ("kind", "id", "authorization")):
            return entry
    raise StudyContractError(
        "unauthorized_authority", "authority is not authorized by these requirements"
    )


def _check_signature(assessment: bytes, signature: bytes | None, keys: list[str]) -> str:
    """A decision's detached signature covers its exact bytes by a key pinned for it."""
    if signature is None:
        raise StudyContractError("signature_unavailable", "a decision needs its detached signature")
    try:
        return verify_signature(assessment, signature, keys)
    except SignatureError as error:
        raise StudyContractError(error.code, str(error)) from error


def _check_prior(document: dict, prior: bytes | None, allow_synthetic: bool) -> None:
    """A reassessment cites its prior exactly and keeps every prior evidence entry.

    The prior's own bindings are not re-checked here; validate it first with
    its own inputs (:func:`darwin.handoff.prepare_handoff` validates a history
    oldest first).
    """
    if prior is None:
        raise StudyContractError("prior_unavailable", "superseded record bytes are required")
    reference = document["supersedes"]
    if sha256(prior) != reference["sha256"]:
        raise StudyContractError("prior_mismatch", "superseded record bytes were altered")
    earlier = _parse_assessment(prior)
    _refuse_untrusted(earlier, allow_synthetic)
    if earlier["id"] != reference["id"] or earlier["dataset"] != document["dataset"]:
        raise StudyContractError("prior_mismatch", "superseded record is for another dataset")
    if any(record not in document["evidence"] for record in earlier["evidence"]):
        raise StudyContractError(
            "prior_evidence_dropped", "a reassessment must retain every prior evidence entry"
        )


def _keyed_bytes(entries: Iterable[Mapping], root: Path) -> dict[ArtifactKey, bytes]:
    return {(e["id"], e["version"]): (root / e["path"]).read_bytes() for e in entries}


def load_assessment_bundle(path: str | Path) -> dict:
    """Read one assessment bundle file into :func:`validate_assessment` inputs.

    The bundle is JSON with paths relative to its own directory, in the setups
    fixture-manifest shape: ``{"assessment", "requirements"?, "evidence":
    [{id, version, path}], "authorizations": [{id, version, path}], "prior"?,
    "signature"?}``. ``signature`` names the detached ``<record>.sig`` file a
    decision needs; it is loaded as ``signature`` bytes (``None`` when absent).
    """
    path = Path(path)
    try:
        bundle = json.loads(path.read_text())
        root = path.parent
        loaded = {
            "assessment": (root / bundle["assessment"]).read_bytes(),
            "evidence": _keyed_bytes(bundle.get("evidence", []), root),
            "authorizations": _keyed_bytes(bundle.get("authorizations", []), root),
            "prior": (root / bundle["prior"]).read_bytes() if bundle.get("prior") else None,
            "signature": (
                (root / bundle["signature"]).read_bytes() if bundle.get("signature") else None
            ),
        }
        if bundle.get("requirements"):
            loaded["requirements"] = (root / bundle["requirements"]).read_bytes()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StudyContractError("invalid_bundle", f"{path}: {error}") from error
    return loaded
