"""Local, no-replace commit of fully written, identity-bound artifacts.

Directory commits require Linux renameat2(RENAME_NOREPLACE). Unsupported
filesystems fail closed. Object storage needs its own conditional-write adapter.

An artifact committed with ``provenance`` gets a sibling identity record
(``<name>.identity.json``, contract ``darwin.artifact-identity/1``) binding the
exact bytes of every file to the builder, parameters and pinned source
checksums. The record carries no timestamps, so an honest retry reproduces it
byte for byte.
"""

from __future__ import annotations

import ctypes
import errno
import filecmp
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

IDENTITY_CONTRACT = "darwin.artifact-identity/1"
IDENTITY_SUFFIX = ".identity.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1 << 20


class ArtifactConflict(FileExistsError):
    """An existing artifact differs from (or cannot be proven equal to) the new one."""


class ArtifactIntegrityError(ValueError):
    """Committed bytes do not match their identity record, or no record exists."""


def _commit(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace commit requires renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(code, os.strerror(code), str(destination))
        raise OSError(code, os.strerror(code), str(destination))


def identity_path(destination: str | Path) -> Path:
    """Return the identity-record path that belongs to ``destination``."""
    destination = Path(destination)
    return destination.with_name(destination.name + IDENTITY_SUFFIX)


def _regular_files(path: Path) -> list[tuple[str, Path]]:
    """List regular files by POSIX relative path; refuse links and special files."""
    mode = path.lstat().st_mode
    if stat.S_ISREG(mode):
        return [(".", path)]
    if not stat.S_ISDIR(mode):
        raise ArtifactIntegrityError(f"{path}: artifacts contain only files and directories")
    files = []
    for root, directories, names in os.walk(path):
        for name in directories + names:
            child = Path(root) / name
            child_mode = child.lstat().st_mode
            if stat.S_ISREG(child_mode):
                files.append((child.relative_to(path).as_posix(), child))
            elif not stat.S_ISDIR(child_mode):
                raise ArtifactIntegrityError(f"{child}: links and special files are not allowed")
    return sorted(files)


def file_sha256(path: str | Path) -> str:
    """sha256 of one file's exact bytes, read in chunks."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def artifact_digest(path: str | Path) -> dict:
    """Canonical identity of a file or directory tree.

    ``sha256`` covers, per file in sorted relative-path order, the line
    ``<path>\\0<size>\\0<file sha256>\\n``. A single file uses path ``.``.
    """
    path = Path(path)
    kind = "file" if stat.S_ISREG(path.lstat().st_mode) else "directory"
    entries = []
    digest = hashlib.sha256()
    for relative, file in _regular_files(path):
        entry = {"path": relative, "size_bytes": file.stat().st_size, "sha256": file_sha256(file)}
        digest.update(f"{relative}\0{entry['size_bytes']}\0{entry['sha256']}\n".encode())
        entries.append(entry)
    return {
        "kind": kind,
        "sha256": digest.hexdigest(),
        "size_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }


def _validated_provenance(provenance: dict) -> dict:
    builder = provenance.get("builder")
    if not isinstance(builder, str) or not builder:
        raise ValueError("provenance needs a builder identifier")
    sources = provenance.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("derived artifacts need at least one pinned source")
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            raise ValueError("each source needs an id")
        if not _SHA256.fullmatch(str(source.get("sha256", ""))):
            raise ValueError(f"source {source.get('id')!r} needs a lowercase sha256")
    if provenance.get("artifact_class", "derived") != "derived":
        raise ValueError("identity-bound artifacts are derived products, never original outputs")
    record = {**provenance, "artifact_class": "derived"}
    json.dumps(record, sort_keys=True, allow_nan=False)
    return record


def _refuse_source_overlap(destination: Path, provenance: dict) -> None:
    """Refuse a destination equal to, inside or containing a source (links resolved).

    Sources named only by id and sha256 carry no path and cannot be checked here.
    """
    destination = destination.resolve()
    for source in provenance["sources"]:
        if "path" not in source:
            continue
        original = Path(source["path"]).resolve()
        if destination == original or original in destination.parents or (
            destination in original.parents
        ):
            raise ValueError(f"{destination}: derived artifact overlaps source {original}")


def _identity_bytes(name: str, digest: dict, provenance: dict) -> bytes:
    record = {
        "contract": IDENTITY_CONTRACT,
        "artifact": {"name": name, **digest},
        "provenance": provenance,
        "scientific_status": "unassessed",
    }
    return (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _byte_identical(left: Path, right: Path) -> bool:
    """Prove equality by full byte comparison, never by size/mtime shortcuts."""
    try:
        if stat.S_IFMT(left.lstat().st_mode) != stat.S_IFMT(right.lstat().st_mode):
            return False
        left_files, right_files = _regular_files(left), _regular_files(right)
    except ArtifactIntegrityError:
        return False
    if [name for name, _ in left_files] != [name for name, _ in right_files]:
        return False
    return all(
        filecmp.cmp(a, b, shallow=False)
        for (_, a), (_, b) in zip(left_files, right_files, strict=True)
    )


def _commit_or_match(staged: Path, destination: Path, retry: str) -> None:
    try:
        _commit(staged, destination)
    except FileExistsError:
        if retry == "verify-identical" and _byte_identical(staged, destination):
            return
        raise ArtifactConflict(
            errno.EEXIST, "existing artifact differs; choose a new version", str(destination)
        ) from None


def _unbound(destination: Path) -> bool:
    return os.path.lexists(destination) and not os.path.lexists(identity_path(destination))


@contextmanager
def staged_artifact(
    destination: str | Path,
    *,
    provenance: dict | None = None,
    retry: Literal["refuse", "verify-identical"] = "refuse",
):
    """Yield a private path; commit only on success, never replacing a target.

    ``retry="refuse"`` (default) rejects every existing target before writing.
    ``retry="verify-identical"`` writes, then treats an existing target as an
    idempotent success only when it is provably byte-identical (full comparison
    of every file and of the identity record); any difference raises
    ``ArtifactConflict`` and leaves the existing bytes untouched. Concurrent
    writers have exactly one winner; identical losers succeed in that mode.

    With ``provenance`` the identity record is committed first, then the
    artifact, so equal bytes are never attested to another provenance: an
    existing artifact without an identity record is refused (its writer made
    no provenance claim), and a crash between the two leaves an identity
    record without an artifact. ``verify_artifact`` rejects that state; only a
    ``verify-identical`` retry with the same bytes and provenance (the record
    must match byte for byte) completes it. Anything else conflicts, and the
    operator recovers by committing under a new name or, because no artifact
    bytes were ever committed there, removing the orphan record. Cleanup
    touches only this invocation's staging directory; a killed process can
    leave hidden staging, never a partial final artifact.
    """
    destination = Path(destination).absolute()
    if retry not in ("refuse", "verify-identical"):
        raise ValueError(f"unknown retry policy {retry!r}")
    record = None
    if provenance is not None:
        record = _validated_provenance(provenance)
        _refuse_source_overlap(destination, record)
    targets = [destination] + ([identity_path(destination)] if record is not None else [])
    if retry == "refuse":
        for target in targets:
            if os.path.lexists(target):
                raise ArtifactConflict(errno.EEXIST, "artifact already exists", str(target))
    elif record is not None and _unbound(destination):
        raise ArtifactConflict(
            errno.EEXIST, "an artifact without identity record exists", str(destination)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    staged = temporary / destination.name
    try:
        yield staged
        if record is not None:
            staged_identity = temporary / identity_path(destination).name
            staged_identity.write_bytes(
                _identity_bytes(destination.name, artifact_digest(staged), record)
            )
        if record is not None:
            _commit_or_match(staged_identity, identity_path(destination), retry)
        _commit_or_match(staged, destination, retry)
    finally:
        shutil.rmtree(temporary)


def verify_artifact(destination: str | Path) -> dict:
    """Re-hash a committed artifact against its identity record and return the record."""
    destination = Path(destination)
    try:
        record = json.loads(identity_path(destination).read_bytes())
    except FileNotFoundError:
        raise ArtifactIntegrityError(f"{destination}: no identity record") from None
    if record.get("contract") != IDENTITY_CONTRACT:
        raise ArtifactIntegrityError(f"{destination}: unsupported identity contract")
    artifact = record.get("artifact", {})
    expected = {key: value for key, value in artifact.items() if key != "name"}
    if artifact.get("name") != destination.name or artifact_digest(destination) != expected:
        raise ArtifactIntegrityError(f"{destination}: bytes do not match identity record")
    return record
