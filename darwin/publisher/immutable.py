"""Local, no-replace publication of fully written artifacts.

Directory publication requires Linux renameat2(RENAME_NOREPLACE). Unsupported
filesystems fail closed. Object storage needs its own conditional-write adapter.
"""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


def _publish(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace publication requires renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


@contextmanager
def staged_artifact(destination: str | Path):
    """Yield a private path; publish only on success, refusing every existing target.

    Retries (even identical content) require a new destination or explicit external
    verification, never an overwrite. Concurrent writers have exactly one winner.
    Cleanup touches only this invocation's staging directory. A killed process can
    leave a hidden staging directory, never a partially visible final artifact.
    """
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(errno.EEXIST, "artifact already exists", str(destination))
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    staged = temporary / destination.name
    try:
        yield staged
        _publish(staged, destination)
    finally:
        shutil.rmtree(temporary)
