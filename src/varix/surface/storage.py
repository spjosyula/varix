"""JSON artifact storage for `PipelineAnalysis`.

One JSON file per analysis at `<base_dir>/<analysis_id>.json`. Default
`base_dir` is `~/.varix/runs/`. Writes are atomic (write-then-rename) so
an interrupted save never leaves a partial file in place.

Compatibility policy is strict: an artifact whose `schema_version` is not
in `_KNOWN_VERSIONS` is refused at load time. See `docs/schema.md`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from varix.core import SCHEMA_VERSION, PipelineAnalysis, RefusalRequired

_KNOWN_VERSIONS: tuple[str, ...] = (SCHEMA_VERSION,)


def default_runs_dir() -> Path:
    """Return the per-user default directory for varix run artifacts."""
    return Path("~/.varix/runs").expanduser()


def save(analysis: PipelineAnalysis, *, base_dir: Path | None = None) -> Path:
    """Persist `analysis` as a JSON file. Returns the path written.

    The write is atomic: a sibling `.tmp` file receives the bytes and is
    then renamed into place via `os.replace` (which is atomic on POSIX and
    Windows). The parent directory is created if missing.
    """
    target_dir = _resolve_dir(base_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    final_path = target_dir / f"{analysis.analysis_id}.json"
    tmp_path = target_dir / f".{analysis.analysis_id}.json.tmp"

    serialized = json.dumps(analysis.to_dict(), indent=2, sort_keys=True)
    tmp_path.write_text(serialized, encoding="utf-8")
    os.replace(tmp_path, final_path)
    return final_path


def load(analysis_id: str, *, base_dir: Path | None = None) -> PipelineAnalysis:
    """Load an analysis by its ID from `base_dir`."""
    target_dir = _resolve_dir(base_dir)
    return load_path(target_dir / f"{analysis_id}.json")


def load_path(path: Path) -> PipelineAnalysis:
    """Load an analysis from an explicit file path."""
    raw = path.read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(raw)
    migrated = _read_with_schema_check(data)
    return PipelineAnalysis.from_dict(migrated)


def list_analyses(base_dir: Path | None = None) -> list[Path]:
    """Return all `.json` artifact paths in `base_dir`, sorted by name.

    Partial `.tmp` files (from an interrupted save) are ignored.
    """
    target_dir = _resolve_dir(base_dir)
    if not target_dir.is_dir():
        return []
    return sorted(target_dir.glob("*.json"))


def latest_analysis(base_dir: Path | None = None) -> Path | None:
    """Return the most recently modified artifact in `base_dir`, or None if empty."""
    paths = list_analyses(base_dir)
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _resolve_dir(base_dir: Path | None) -> Path:
    return Path(base_dir).expanduser() if base_dir is not None else default_runs_dir()


def _read_with_schema_check(data: dict[str, Any]) -> dict[str, Any]:
    raw_version = data.get("schema_version")
    if not raw_version:
        raise RefusalRequired("artifact is missing schema_version")
    artifact_version = str(raw_version)
    if artifact_version not in _KNOWN_VERSIONS:
        raise RefusalRequired(
            f"artifact schema_version {artifact_version!r} is not understood by "
            f"this varix (knows {list(_KNOWN_VERSIONS)})"
        )
    return _migrate_to_current(data, artifact_version)


def _migrate_to_current(data: dict[str, Any], from_version: str) -> dict[str, Any]:
    """Stepwise migration from `from_version` to the current schema.

    Schema 0.1 is currently the only shipped version, so this is a no-op.
    Future breaking changes register stepwise migrations here.
    """
    if from_version == SCHEMA_VERSION:
        return data
    raise RefusalRequired(f"no migration path from schema {from_version!r} to {SCHEMA_VERSION!r}")
