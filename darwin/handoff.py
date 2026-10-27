"""Publication-ready product handoff, contract ``darwin.product-handoff/2``.

Darwin prepares products; the dashboard catalogue decides publication, access
and current-version pointers. :func:`prepare_handoff` therefore works without
any catalogue and never publishes, moves a pointer or grants visibility. Each
product binds:

* ``artifacts``: exact identity-bound artifacts (``darwin.artifact-identity/1``
  records, re-verified byte for byte before the handoff is written);
* ``dataset``: ``{id: product_id, version, sha256}``, where ``sha256`` is the
  sha256 of the canonical JSON of ``artifacts``. Acceptance records bind this
  identity, so they never transfer to another version or artifact set;
* ``sources`` and ``recipe``: the reviewed release manifest's pinned sources
  and its exact revision;
* ``execution_completeness`` (``complete``/``partial``, or ``null`` when not
  reported) and its ``execution_evidence`` identity;
* ``scientific_status`` (``accepted``/``rejected``/``unassessed``) and every
  validated ``acceptance`` reference, oldest first. The status is the latest
  record's status, or ``unassessed`` without records. It is independent of
  execution completeness and of publication.

Every legacy (unversioned) handoff field except ``status`` is kept on each
product, so the dashboard's existing importer still reads a v2 document.
:func:`validate_handoff` is the consumer-side check of the contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from darwin import acceptance
from darwin.publisher.candidate import plan_candidates
from darwin.publisher.immutable import (
    ArtifactIntegrityError,
    identity_path,
    staged_artifact,
    verify_artifact,
)
from darwin.publisher.manifest import load_manifest
from darwin.publisher.smoke import CREATOR, legacy_product_entry
from darwin.study_contracts import SYNTHETIC_PREFIX

CONTRACT = "darwin.product-handoff/2"
SCHEMA = "product-handoff-v2.schema.json"


class HandoffError(ValueError):
    """A handoff Darwin refuses to prepare or a consumer must refuse to read."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def artifact_entry(path: str | Path) -> dict:
    """Re-verify one published artifact and return its handoff entry."""
    path = Path(path)
    if not identity_path(path).exists():
        raise HandoffError("unbound_artifact", f"{path.name} has no identity record")
    try:
        record = verify_artifact(path)
    except ArtifactIntegrityError as error:
        raise HandoffError("artifact_mismatch", str(error)) from error
    artifact = record["artifact"]
    return {
        "name": artifact["name"],
        "kind": artifact["kind"],
        "sha256": artifact["sha256"],
        "size_bytes": artifact["size_bytes"],
        "identity_record_sha256": acceptance.sha256_bytes(identity_path(path).read_bytes()),
    }


def dataset_identity(product_id: str, version: str, artifacts: Sequence[dict]) -> dict:
    """Return the exact dataset identity acceptance records must bind."""
    digest = acceptance.sha256_bytes(
        json.dumps(list(artifacts), sort_keys=True, separators=(",", ":")).encode()
    )
    return {"id": product_id, "version": version, "sha256": digest}


def _acceptance_references(
    dataset: dict, records: Sequence[Mapping], *, allow_synthetic: bool
) -> list[dict]:
    """Validate a product's assessment history against its exact dataset identity."""
    references: list[dict] = []
    for record in records:
        state = acceptance.validate_assessment(
            record["assessment"],
            requirements=record["requirements"],
            dataset=dataset,
            evidence=record.get("evidence", {}),
            prior=record.get("prior"),
            allow_synthetic=allow_synthetic,
        )
        previous = references[-1] if references else None
        expected = (
            None if previous is None else {"id": previous["id"], "sha256": previous["sha256"]}
        )
        if state["supersedes"] != expected:
            raise HandoffError(
                "broken_acceptance_chain",
                f"assessment {state['assessment']['id']} must supersede the previous record",
            )
        references.append(
            {
                "contract": acceptance.ASSESSMENT_CONTRACT,
                "id": state["assessment"]["id"],
                "sha256": state["assessment"]["sha256"],
                "status": state["status"],
                "requirements": state["requirements"],
                "authority": state["authority"],
                "supersedes": state["supersedes"],
            }
        )
    return references


def prepare_handoff(  # noqa: PLR0913
    manifest_path: str | Path,
    out_path: str | Path,
    *,
    artifacts: Mapping[str, Sequence[str | Path]],
    execution: Mapping[str, Mapping] | None = None,
    assessments: Mapping[str, Sequence[Mapping]] | None = None,
    creator: str = CREATOR,
    created_at: str | None = None,
    allow_synthetic: bool = False,
) -> dict:
    """Prepare a ``darwin.product-handoff/2`` document and write it immutably.

    ``artifacts`` maps every manifest product to its published artifact paths.
    ``execution`` optionally maps a product to ``{"completeness":
    "complete"|"partial", "evidence": identity}``. ``assessments`` maps a
    product to its acceptance-assessment history, oldest first, each item
    ``{"assessment", "requirements", "evidence", "prior"}`` as exact bytes;
    every record is validated against the product's exact dataset identity.
    Nothing is published, no current pointer moves and no access is granted.
    """
    manifest = load_manifest(manifest_path)
    stamp = created_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    specs = {spec.product_id: spec for spec in manifest.products}
    execution = execution or {}
    assessments = assessments or {}
    unknown = (set(execution) | set(assessments) | set(artifacts)) - set(specs)
    if unknown:
        raise HandoffError("unknown_product", f"not in the manifest: {sorted(unknown)}")
    sources = [{"id": entry.path, "sha256": entry.sha256} for entry in manifest.sources]
    recipe = {
        "id": f"release-manifest:{manifest.name}",
        "version": manifest.revision,
        "sha256": manifest.revision,
    }
    products = []
    synthetic = False
    for candidate in plan_candidates(manifest):
        paths = artifacts.get(candidate.product_id)
        if not paths:
            raise HandoffError(
                "missing_artifacts", f"{candidate.product_id} names no exact artifacts"
            )
        entries = [artifact_entry(path) for path in paths]
        dataset = dataset_identity(candidate.product_id, candidate.version, entries)
        references = _acceptance_references(
            dataset, assessments.get(candidate.product_id, ()), allow_synthetic=allow_synthetic
        )
        synthetic = synthetic or any(
            identifier.startswith(SYNTHETIC_PREFIX)
            for identifier in [candidate.product_id, *(ref["id"] for ref in references)]
        )
        run = execution.get(candidate.product_id)
        products.append(
            {
                **legacy_product_entry(
                    specs[candidate.product_id],
                    candidate,
                    manifest,
                    creator=creator,
                    created_at=stamp,
                ),
                "lifecycle": "prepared",
                "dataset": dataset,
                "artifacts": entries,
                "sources": sources,
                "recipe": recipe,
                "execution_completeness": run["completeness"] if run else None,
                "execution_evidence": dict(run["evidence"]) if run else None,
                "scientific_status": references[-1]["status"] if references else "unassessed",
                "acceptance": references,
            }
        )
    document = {
        "contract": CONTRACT,
        "release": manifest.release,
        "manifest": manifest.name,
        "manifest_revision": manifest.revision,
        "creator": creator,
        "created_at": stamp,
        "synthetic": synthetic,
        "products": products,
    }
    data = acceptance.canonical_bytes(document)
    validate_handoff(data, allow_synthetic=allow_synthetic)
    provenance = {
        "builder": "darwin.handoff.prepare_handoff",
        "recipe": {"manifest_revision": manifest.revision},
        "sources": [
            {"id": f"release-manifest:{manifest.name}", "sha256": manifest.revision},
            *sources,
            *(
                {"id": f"{p['product_id']}/{a['name']}", "sha256": a["sha256"]}
                for p in products
                for a in p["artifacts"]
            ),
        ],
        "parameters": {"creator": creator, "created_at": stamp},
    }
    with staged_artifact(out_path, provenance=provenance) as staged:
        staged.write_bytes(data)
    return document


def validate_handoff(
    data: bytes, *, artifact_root: str | Path | None = None, allow_synthetic: bool = False
) -> dict:
    """Validate a v2 handoff; with ``artifact_root`` also re-verify artifact bytes.

    Checks the schema, that each ``dataset`` identity matches its artifacts and
    that ``scientific_status`` is backed by the latest acceptance reference.
    Synthetic fixture handoffs are refused unless ``allow_synthetic``.
    Acceptance record bytes are validated separately with
    :func:`darwin.acceptance.validate_assessment`.
    """
    try:
        document = json.loads(data)
    except ValueError as error:
        raise HandoffError("invalid_handoff", str(error)) from error
    if not isinstance(document, dict) or document.get("contract") != CONTRACT:
        raise HandoffError("unsupported_contract", f"expected {CONTRACT}")
    try:
        acceptance.validate_schema(document, SCHEMA, "invalid_handoff")
    except acceptance.AcceptanceError as error:
        raise HandoffError(error.code, str(error)) from error
    synthetic = document["synthetic"] or any(
        identifier.startswith(SYNTHETIC_PREFIX)
        for product in document["products"]
        for identifier in [product["product_id"], *(ref["id"] for ref in product["acceptance"])]
    )
    if synthetic and not allow_synthetic:
        raise HandoffError(
            "synthetic_not_trusted", "synthetic fixture handoffs are refused outside test mode"
        )
    for product in document["products"]:
        expected = dataset_identity(product["product_id"], product["version"], product["artifacts"])
        if product["dataset"] != expected:
            raise HandoffError(
                "dataset_mismatch", f"{product['product_id']}: dataset does not match artifacts"
            )
        references = product["acceptance"]
        status = references[-1]["status"] if references else "unassessed"
        if product["scientific_status"] != status:
            raise HandoffError(
                "status_mismatch",
                f"{product['product_id']}: scientific_status is not backed by its latest record",
            )
        if artifact_root is not None:
            for artifact in product["artifacts"]:
                if artifact_entry(Path(artifact_root) / artifact["name"]) != artifact:
                    raise HandoffError(
                        "artifact_mismatch", f"{artifact['name']} differs from the handoff"
                    )
    return document


def main(argv: Sequence[str] | None = None) -> None:
    """CLI over the same functions: ``prepare`` and ``validate``."""
    parser = argparse.ArgumentParser(prog="python -m darwin.handoff", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="prepare a handoff; never publishes")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument(
        "--artifacts", type=Path, required=True, help="JSON {product_id: [artifact paths]}"
    )
    prepare.add_argument("--creator", default=CREATOR)
    check = commands.add_parser("validate", help="validate a handoff")
    check.add_argument("handoff", type=Path)
    check.add_argument("--artifact-root", type=Path)
    check.add_argument("--test-mode", action="store_true", help="accept synthetic fixtures")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            document = prepare_handoff(
                args.manifest,
                args.output,
                artifacts=json.loads(args.artifacts.read_text()),
                creator=args.creator,
            )
        else:
            document = validate_handoff(
                args.handoff.read_bytes(),
                artifact_root=args.artifact_root,
                allow_synthetic=args.test_mode,
            )
    except (HandoffError, acceptance.AcceptanceError) as error:
        sys.stderr.write(json.dumps({"code": error.code, "message": str(error)}) + "\n")
        raise SystemExit(2) from error
    print(json.dumps({"contract": document["contract"], "products": len(document["products"])}))


if __name__ == "__main__":
    main()
