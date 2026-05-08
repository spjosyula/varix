"""varix.surface — file storage, CLI, reporter.

The only layer that touches stdin/stdout, the filesystem, the environment,
or external processes. Every other layer must remain pure.
"""

from varix.surface.storage import (
    default_runs_dir,
    latest_analysis,
    list_analyses,
    load,
    load_path,
    save,
)

__all__ = [
    "default_runs_dir",
    "latest_analysis",
    "list_analyses",
    "load",
    "load_path",
    "save",
]
