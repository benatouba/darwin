"""GAR product publication platform (darwin#2, D1).

The single seam for all publisher work: a reviewed, version-controlled
YAML manifest goes in, an immutable product candidate comes out, and the
catalogue enforces the candidate/published/superseded/failed lifecycle.
Source paths/checksums always come from the background manifest; the
publisher never scans the source archive on run or request paths.
"""

from darwin.publisher.candidate import Candidate, run_manifest
from darwin.publisher.catalogue import Catalogue
from darwin.publisher.manifest import Manifest, load_manifest

__all__ = ["Candidate", "Catalogue", "Manifest", "load_manifest", "run_manifest"]
