"""Non-map accessors must not fetch Salem's unrelated sample-data archive."""

import os
import subprocess
import sys
from pathlib import Path


def test_accessors_import_outside_checkout_without_network(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "MPLBACKEND": "Agg",
        "PYTHONPATH": str(root),
    }
    result = subprocess.run(
        [sys.executable, "-c", """
import socket
import sys

def forbidden(*args, **kwargs):
    raise AssertionError('import attempted network access')

socket.socket.connect = forbidden
socket.create_connection = forbidden
from darwin import accessors
assert 'salem' not in sys.modules
"""],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
