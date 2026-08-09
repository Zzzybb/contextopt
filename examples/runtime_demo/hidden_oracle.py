from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast


def _load_merge_settings(workspace: Path) -> Callable[
    [dict[str, Any], dict[str, Any]], dict[str, Any]
]:
    source = workspace / "settings.py"
    spec = importlib.util.spec_from_file_location("runtime_demo_settings", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(Callable[..., dict[str, Any]], module.merge_settings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the runtime demo hidden oracle.")
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    merge_settings = _load_merge_settings(args.workspace.resolve(strict=True))

    defaults = {"enabled": True, "retries": 3, "label": "prod"}
    overrides: dict[str, Any] = {
        "enabled": False,
        "retries": 0,
        "label": "",
        "ignored": None,
    }
    defaults_before = deepcopy(defaults)
    overrides_before = deepcopy(overrides)
    actual = merge_settings(defaults, overrides)
    expected = {"enabled": False, "retries": 0, "label": ""}

    if actual != expected:
        raise AssertionError(f"unexpected result: {actual!r} != {expected!r}")
    if defaults != defaults_before or overrides != overrides_before:
        raise AssertionError("merge_settings mutated one of its inputs")
    print("hidden oracle passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
