from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def merge_settings(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge settings; ``None`` means that the default should be inherited."""

    merged = dict(defaults)
    merged.update({key: value for key, value in overrides.items() if value})
    return merged
