"""Test-only SSHSIG signing with the setups fixture tree's TEST-ONLY keys.

Darwin itself only verifies decision signatures. Tests sign records the way a
reviewer does with ``ssh-keygen -Y sign -n darwin-acceptance``; the output is
byte-identical (Ed25519 is deterministic, sha512 message hash, 70-column armor).
The keys are published synthetic fixture keys, never real ones.
"""

from __future__ import annotations

import base64
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_ssh_private_key

from darwin.decision_signature import KEY_TYPE, NAMESPACE

KEYS = (
    Path(__file__).parent
    / "fixtures"
    / "setups-contracts"
    / "fixtures"
    / "acceptance-assessment-v1"
    / "signing-keys"
)


def _string(value: bytes) -> bytes:
    return len(value).to_bytes(4, "big") + value


def test_key(name: str) -> Ed25519PrivateKey:
    """A TEST-ONLY fixture key: ``reviewer``, ``policy`` or ``unpinned``."""
    return load_ssh_private_key((KEYS / f"TEST-ONLY-{name}").read_bytes(), password=None)


test_key.__test__ = False  # a helper, not a pytest test


def sign(message: bytes, key: str = "reviewer", namespace: str = NAMESPACE) -> bytes:
    """Armored detached SSHSIG over exact bytes by a TEST-ONLY key."""
    import hashlib  # noqa: PLC0415

    private = test_key(key)
    public = private.public_key().public_bytes_raw()
    digest = hashlib.sha512(message).digest()
    signed = (
        b"SSHSIG"
        + _string(namespace.encode())
        + _string(b"")
        + _string(b"sha512")
        + _string(digest)
    )
    blob = (
        b"SSHSIG"
        + (1).to_bytes(4, "big")
        + _string(_string(KEY_TYPE.encode()) + _string(public))
        + _string(namespace.encode())
        + _string(b"")
        + _string(b"sha512")
        + _string(_string(KEY_TYPE.encode()) + _string(private.sign(signed)))
    )
    body = base64.b64encode(blob).decode()
    lines = [body[i : i + 70] for i in range(0, len(body), 70)]
    return (
        "\n".join(["-----BEGIN SSH SIGNATURE-----", *lines, "-----END SSH SIGNATURE-----"]) + "\n"
    ).encode()


def public_key(key: str = "reviewer") -> str:
    """The bare ``ssh-ed25519 <base64>`` line of a TEST-ONLY key."""
    return " ".join((KEYS / f"TEST-ONLY-{key}.pub").read_text().split()[:2])
