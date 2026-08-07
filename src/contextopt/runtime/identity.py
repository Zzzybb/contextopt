"""Canonical fingerprints for resumable runtime configuration and operations."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data with a stable, provider-neutral encoding."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any) -> str:
    """Return the SHA-256 of :func:`canonical_json` for *value*."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
