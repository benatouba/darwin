"""GAR product preparation (darwin#2, D1; migrated by darwin#18).

A reviewed, version-controlled YAML manifest goes in and immutable product
candidates come out; ``darwin.handoff.prepare_handoff`` describes them for
the dashboard. Source paths/checksums always come from the background
manifest; the publisher never scans the source archive on run or request
paths.

``Catalogue`` is a legacy in-memory lifecycle model kept for existing
callers and tests. It is not a publication authority: the dashboard
catalogue owns publication, access and current-version pointers, and no
Darwin builder publishes through ``Catalogue.publish`` any more.
"""

from darwin.publisher.candidate import Candidate, run_manifest
from darwin.publisher.catalogue import Catalogue
from darwin.publisher.manifest import Manifest, load_manifest

__all__ = ["Candidate", "Catalogue", "Manifest", "load_manifest", "run_manifest"]
