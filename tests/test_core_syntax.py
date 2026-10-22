"""Check core syntax without importing its plotting and data dependencies."""

from pathlib import Path


def test_core_compiles():
    source = Path(__file__).resolve().parents[1] / "darwin" / "core.py"
    compile(source.read_text(encoding="utf-8"), str(source), "exec")
