"""Structured ``{code, message}`` errors shared by Darwin's Python interface and CLIs."""

from __future__ import annotations

import json
import sys
from typing import NoReturn


class CodedError(ValueError):
    """A refusal with a stable machine-readable ``code`` and a human message."""

    def __init__(self, code: str, message: str) -> None:
        """Keep ``code`` next to the message so CLIs can report both."""
        super().__init__(message)
        self.code = code


def emit_error(code: str, message: str, status: int = 2) -> NoReturn:
    """Write one JSON error object to stderr and exit; the CLI error contract."""
    sys.stderr.write(json.dumps({"code": code, "message": message}) + "\n")
    raise SystemExit(status)
