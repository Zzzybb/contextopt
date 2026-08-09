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


def model_request_idempotency_key(
    *, run_id: str, turn: int, request_sha256: str
) -> str:
    """Return a stable provider-facing key for one logical model request.

    The request hash is part of the key so a changed payload cannot accidentally
    reuse a provider-side result. The key is only a hook: a provider must explicitly
    honor its idempotency header before it can provide duplicate suppression.
    """

    return "contextopt-" + stable_hash(
        {
            "kind": "model-request",
            "run_id": run_id,
            "turn": turn,
            "request_sha256": request_sha256,
        }
    )
