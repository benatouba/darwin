"""Verify detached OpenSSH (SSHSIG) signatures over exact acceptance-decision bytes.

A reviewer signs the exact record file with ``ssh-keygen -Y sign -f KEY -n
darwin-acceptance RECORD``, which writes ``RECORD.sig``. Verification follows
OpenSSH ``PROTOCOL.sshsig`` and needs no ``ssh-keygen`` at runtime: it parses the
armored signature, checks the namespace, requires the embedded public key to be
one of the keys the reviewed requirements pin for the cited authority, and checks
the Ed25519 signature over ``H(record bytes)``. Only ``ssh-ed25519`` keys are
supported.

Copied from ``darwin-wrf-setups`` ``tooling/decision_signature.py`` (the producer of
the acceptance contract) and kept in parity with it by the copied fixtures in
``tests/fixtures/setups-contracts``; Darwin never imports setups. Darwin only
verifies: it never holds a reviewer's private key.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Iterable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from darwin.errors import CodedError

NAMESPACE = "darwin-acceptance"
"""The SSHSIG namespace every acceptance decision is signed under."""

KEY_TYPE = "ssh-ed25519"
_MAGIC = b"SSHSIG"
_VERSION = 1
_BEGIN = "-----BEGIN SSH SIGNATURE-----"
_END = "-----END SSH SIGNATURE-----"
_HASHES = {"sha256": hashlib.sha256, "sha512": hashlib.sha512}


class SignatureError(CodedError):
    """A missing, malformed, unauthorized or non-matching decision signature."""


def _string(value: bytes) -> bytes:
    return len(value).to_bytes(4, "big") + value


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.offset = data, 0

    def take(self, size: int) -> bytes:
        if self.offset + size > len(self.data):
            raise SignatureError("signature_invalid", "truncated SSH signature")
        chunk = self.data[self.offset : self.offset + size]
        self.offset += size
        return chunk

    def string(self) -> bytes:
        return self.take(int.from_bytes(self.take(4), "big"))

    def done(self) -> None:
        if self.offset != len(self.data):
            raise SignatureError("signature_invalid", "trailing bytes in SSH signature")


def public_key_blob(public_key: str) -> bytes:
    """The SSH wire blob of an ``ssh-ed25519 <base64>`` public key line."""
    fields = public_key.split()
    if len(fields) < 2 or fields[0] != KEY_TYPE:
        raise SignatureError("invalid_signing_key", f"only {KEY_TYPE} keys are supported")
    try:
        blob = base64.b64decode(fields[1], validate=True)
    except binascii.Error as error:
        raise SignatureError("invalid_signing_key", "public key is not base64") from error
    reader = _Reader(blob)
    try:
        if reader.string() != KEY_TYPE.encode() or len(reader.string()) != 32:
            raise SignatureError("invalid_signing_key", "malformed ssh-ed25519 public key")
        reader.done()
    except SignatureError as error:
        raise SignatureError("invalid_signing_key", str(error)) from error
    return blob


def _signed_data(message: bytes, namespace: str, reserved: bytes, hash_name: str) -> bytes:
    digest = _HASHES[hash_name](message).digest()
    return (
        _MAGIC
        + _string(namespace.encode())
        + _string(reserved)
        + _string(hash_name.encode())
        + _string(digest)
    )


def _unarmor(signature: bytes) -> bytes:
    try:
        lines = signature.decode("ascii").strip().splitlines()
    except UnicodeDecodeError as error:
        raise SignatureError("signature_invalid", "signature is not ASCII armor") from error
    if len(lines) < 3 or lines[0] != _BEGIN or lines[-1] != _END:
        raise SignatureError("signature_invalid", "not an armored SSH signature")
    try:
        return base64.b64decode("".join(lines[1:-1]), validate=True)
    except binascii.Error as error:
        raise SignatureError("signature_invalid", "signature body is not base64") from error


def verify_signature(
    message: bytes, signature: bytes, allowed_keys: Iterable[str], namespace: str = NAMESPACE
) -> str:
    """Verify an armored SSHSIG over exact ``message`` bytes; return the pinned signer key.

    Refusals: ``signature_invalid`` (unparseable or unsupported), ``signature_namespace_
    mismatch``, ``signer_not_authorized`` (key not in ``allowed_keys``) and
    ``signature_mismatch`` (the signature does not cover these bytes).
    """
    allowed = {public_key_blob(key): " ".join(key.split()[:2]) for key in allowed_keys}
    reader = _Reader(_unarmor(signature))
    if reader.take(6) != _MAGIC or int.from_bytes(reader.take(4), "big") != _VERSION:
        raise SignatureError("signature_invalid", "not an SSHSIG version 1 signature")
    key_blob = reader.string()
    signed_namespace = reader.string()
    reserved = reader.string()
    hash_name = reader.string().decode("ascii", "replace")
    wrapped = reader.string()
    reader.done()
    if hash_name not in _HASHES:
        raise SignatureError("signature_invalid", f"unsupported hash algorithm {hash_name}")
    if signed_namespace != namespace.encode():
        raise SignatureError(
            "signature_namespace_mismatch", f"signature is not in namespace {namespace}"
        )
    if key_blob not in allowed:
        raise SignatureError(
            "signer_not_authorized", "signing key is not pinned for this authority"
        )
    inner = _Reader(wrapped)
    if inner.string() != KEY_TYPE.encode():
        raise SignatureError("signature_invalid", f"only {KEY_TYPE} signatures are supported")
    raw = inner.string()
    inner.done()
    # A pinned blob was validated by public_key_blob: its last 32 bytes are the raw key.
    key = Ed25519PublicKey.from_public_bytes(key_blob[-32:])
    try:
        key.verify(raw, _signed_data(message, namespace, reserved, hash_name))
    except InvalidSignature as error:
        raise SignatureError(
            "signature_mismatch", "signature does not cover these exact record bytes"
        ) from error
    return allowed[key_blob]
