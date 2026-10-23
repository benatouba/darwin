"""Download safeguards are tested offline, including misleading HTTP sizes."""

import hashlib
import io
import json

import pytest

from scripts import fetch_test_fixtures as fixtures


def _source(payload=b"fixture"):
    return {
        "path": "test.nc",
        "url": "https://data.klima.tu-berlin.de/GAR/test.nc",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class Response(io.BytesIO):
    def __init__(self, payload, source, length=None):
        super().__init__(payload)
        self.url = source["url"]
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def geturl(self):
        return self.url


def test_download_is_verified_and_cache_is_reused(tmp_path):
    source = _source()
    calls = []

    def opener(url, timeout):
        calls.append((url, timeout))
        return Response(b"fixture", source, length=7)

    path = fixtures.fetch(source, tmp_path, opener=opener)
    assert path.read_bytes() == b"fixture"
    assert fixtures.fetch(source, tmp_path, opener=opener) == path
    assert calls == [(source["url"], 30)]
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.parametrize(
    ("payload", "length", "message"),
    [
        (b"fixture!", None, "exceeds"),
        (b"fixture!", 7, "exceeds"),
        (b"fixture", 100000000, "Content-Length"),
        (b"short", None, "size"),
        (b"changed", None, "checksum"),
    ],
)
def test_bad_downloads_leave_no_partial_or_destination(tmp_path, payload, length, message):
    source = _source()
    with pytest.raises(ValueError, match=message):
        fixtures.fetch(
            source, tmp_path,
            opener=lambda *args, **kwargs: Response(payload, source, length),
        )
    assert list(tmp_path.iterdir()) == []


def test_interrupted_download_is_cleaned_up(tmp_path):
    source = _source()

    class Interrupted(Response):
        def read(self, size):
            raise OSError("connection lost")

    with pytest.raises(OSError, match="connection lost"):
        fixtures.fetch(source, tmp_path, opener=lambda *args, **kwargs: Interrupted(b"", source))
    assert list(tmp_path.iterdir()) == []


def test_redirect_is_rejected(tmp_path):
    source = _source()
    response = Response(b"fixture", source)
    response.url = "https://example.com/large.nc"
    with pytest.raises(ValueError, match="redirect"):
        fixtures.fetch(source, tmp_path, opener=lambda *args, **kwargs: response)
    assert list(tmp_path.iterdir()) == []


def test_corrupt_cache_is_not_overwritten(tmp_path):
    source = _source()
    path = tmp_path / source["path"]
    path.write_bytes(b"changed")

    def forbidden(*args, **kwargs):
        pytest.fail("cache validation must not download")

    with pytest.raises(ValueError, match="checksum"):
        fixtures.fetch(source, tmp_path, opener=forbidden)
    assert path.read_bytes() == b"changed"


@pytest.mark.parametrize("change", [
    {"path": "../escape.nc"},
    {"url": "http://data.klima.tu-berlin.de/GAR/test.nc"},
    {"url": "https://example.com/GAR/test.nc"},
    {"size": fixtures.MAX_FILE_BYTES + 1},
    {"size": -1},
    {"sha256": "not-a-hash"},
])
def test_invalid_download_plan_is_rejected(tmp_path, change):
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{**_source(), **change}]))
    with pytest.raises(ValueError):
        fixtures.load_sources(manifest)


def test_total_download_budget_is_enforced(tmp_path):
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([
        {**_source(), "path": f"test-{i}.nc", "size": fixtures.MAX_FILE_BYTES}
        for i in range(3)
    ]))
    with pytest.raises(ValueError, match="total size budget"):
        fixtures.load_sources(manifest)


def test_duplicate_fixtures_are_rejected(tmp_path):
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([_source(), _source()]))
    with pytest.raises(ValueError, match="duplicate"):
        fixtures.load_sources(manifest)


def test_real_fixture_plan_fits_budgets():
    sources = fixtures.load_sources()
    assert len(sources) == 3
    assert sum(source["size"] for source in sources) < fixtures.MAX_TOTAL_BYTES


def test_fixture_cache_path_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("DARWIN_TEST_FIXTURES", str(tmp_path))
    assert fixtures.fixture_directory() == tmp_path
