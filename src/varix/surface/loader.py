"""Resolve a pipeline-target string into a concrete `Adapter` instance.

Two forms are supported:

  - **File path** — `agent.py`, `path/to/agent.py`, or any string with a
    path separator. The file is loaded as a module and its `adapter`
    attribute is returned.
  - **Import string** — `module.path:attribute`. The module is imported
    and the named attribute is returned. A bare module name (no colon)
    is also accepted; the implicit attribute is `adapter`.

If the resolved object is a class, it is instantiated with no arguments.
The result must satisfy the `Adapter` Protocol; otherwise a `TypeError` is
raised.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from varix.core import Adapter

_DEFAULT_ATTRIBUTE = "adapter"


def load_adapter(target: str) -> Adapter:
    """Resolve `target` into an Adapter instance."""
    if _looks_like_path(target):
        obj = _load_from_file(target)
    elif ":" in target:
        module_name, _, attr_name = target.partition(":")
        obj = _load_from_module(module_name, attr_name)
    else:
        obj = _load_from_module(target, _DEFAULT_ATTRIBUTE)

    if isinstance(obj, type):
        obj = obj()

    if not isinstance(obj, Adapter):
        raise TypeError(f"resolved object from {target!r} does not satisfy the Adapter protocol")
    return obj


def _looks_like_path(target: str) -> bool:
    return target.endswith(".py") or "/" in target or os.sep in target


def _load_from_file(target: str) -> Any:
    path = Path(target).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pipeline file not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, _DEFAULT_ATTRIBUTE):
        raise AttributeError(f"file {path} has no `{_DEFAULT_ATTRIBUTE}` attribute at module level")
    return getattr(module, _DEFAULT_ATTRIBUTE)


def _load_from_module(module_name: str, attr_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"could not import module {module_name!r}: {exc}") from exc
    if not hasattr(module, attr_name):
        raise AttributeError(f"module {module_name!r} has no attribute {attr_name!r}")
    return getattr(module, attr_name)
