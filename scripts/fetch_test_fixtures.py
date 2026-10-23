"""Fetch only reviewed, size-bounded GAR fixtures; never download during pytest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "gar-sources.json"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024


def fixture_directory() -> Path:
    """Use the same configurable cache path as pytest."""
    return Path(os.environ.get("DARWIN_TEST_FIXTURES", ROOT / "tests" / ".cache" / "gar"))


def load_sources(path: Path = MANIFEST) -> list[dict]:
    """Validate the entire plan before creating files or making requests."""
    sources = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sources, list) or not sources:
        raise ValueError("fixture manifest must be a nonempty list")
    names: set[str] = set()
    total = 0
    for source in sources:
        name = source["path"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.nc", name):
            raise ValueError("fixture paths must be plain NetCDF filenames")
        if name in names:
            raise ValueError(f"duplicate fixture path: {name}")
        names.add(name)
        size = source["size"]
        if type(size) is not int or not 0 < size <= MAX_FILE_BYTES:
            raise ValueError(f"{name}: exceeds per-file size budget")
        total += size
        url = urlsplit(source["url"])
        if (
            url.scheme != "https"
            or url.netloc != "data.klima.tu-berlin.de"
            or not url.path.startswith("/GAR/")
            or url.query
            or url.fragment
        ):
            raise ValueError(f"{name}: URL must use the HTTPS GAR archive")
        if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
            raise ValueError(f"{name}: invalid sha256")
    if total > MAX_TOTAL_BYTES:
        raise ValueError("fixture manifest exceeds total size budget")
    return sources


def verify(path: Path, source: dict) -> None:
    """Reject stale or damaged cache entries without silently replacing them."""
    if not path.is_file() or path.stat().st_size != source["size"]:
        raise ValueError(f"{path.name}: missing fixture or unexpected size")
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != source["sha256"]:
        raise ValueError(f"{path.name}: fixture checksum mismatch")


def fetch(source: dict, directory: Path, *, opener=urlopen) -> Path:
    """Stream one bounded download, verify it, then atomically publish it."""
    destination = directory / source["path"]
    if destination.exists() or destination.is_symlink():
        verify(destination, source)
        return destination
    directory.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with opener(source["url"], timeout=30) as response:
            # Refuse redirects away from the reviewed endpoint as well.
            if response.geturl() != source["url"]:
                raise ValueError(f"{destination.name}: unexpected redirect")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != source["size"]:
                raise ValueError(f"{destination.name}: unexpected Content-Length")
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                received = 0
                while chunk := response.read(min(65536, source["size"] - received + 1)):
                    received += len(chunk)
                    if received > source["size"]:
                        raise ValueError(f"{destination.name}: download exceeds reviewed size")
                    handle.write(chunk)
        verify(temporary, source)
        temporary.replace(destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=fixture_directory())
    parser.add_argument("--check", action="store_true", help="verify cached bytes without network access")
    args = parser.parse_args()
    sources = load_sources()
    for source in sources:
        if args.check:
            path = args.directory / source["path"]
            verify(path, source)
        else:
            path = fetch(source, args.directory)
        print(f"Verified {path.name} ({source['size']} bytes)")


if __name__ == "__main__":
    main()
