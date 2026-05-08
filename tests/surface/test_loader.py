"""Tests for surface.loader (pipeline-target resolution)."""

from __future__ import annotations

from pathlib import Path

import pytest

from varix.adapters import FakeAdapter
from varix.core import Adapter
from varix.surface.loader import load_adapter


def test_resolves_import_string_to_class_and_instantiates() -> None:
    adapter = load_adapter("varix.adapters:FakeAdapter")
    assert isinstance(adapter, Adapter)
    assert isinstance(adapter, FakeAdapter)


def test_resolves_bare_module_with_implicit_adapter_attribute(tmp_path: Path) -> None:
    module_path = tmp_path / "my_agent_pkg.py"
    module_path.write_text(
        "from varix.adapters import FakeAdapter\nadapter = FakeAdapter()\n",
        encoding="utf-8",
    )
    # Use the file path form since we can't easily put files on the import path.
    adapter = load_adapter(str(module_path))
    assert isinstance(adapter, Adapter)


def test_resolves_file_path_to_adapter_attribute(tmp_path: Path) -> None:
    module_path = tmp_path / "agent.py"
    module_path.write_text(
        "from varix.adapters import FakeAdapter\nadapter = FakeAdapter()\n",
        encoding="utf-8",
    )
    adapter = load_adapter(str(module_path))
    assert isinstance(adapter, Adapter)


def test_resolves_file_path_with_class_at_attribute(tmp_path: Path) -> None:
    module_path = tmp_path / "agent.py"
    module_path.write_text(
        "from varix.adapters import FakeAdapter\nadapter = FakeAdapter\n",
        encoding="utf-8",
    )
    # Class at module level → loader should instantiate it.
    adapter = load_adapter(str(module_path))
    assert isinstance(adapter, FakeAdapter)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_adapter(str(tmp_path / "nope.py"))


def test_file_without_adapter_attribute_raises_attribute_error(tmp_path: Path) -> None:
    module_path = tmp_path / "agent.py"
    module_path.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(AttributeError, match="adapter"):
        load_adapter(str(module_path))


def test_unknown_module_raises_import_error() -> None:
    with pytest.raises(ImportError):
        load_adapter("definitely_not_a_real_module_xyz:adapter")


def test_module_missing_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        load_adapter("varix.adapters:NotAClass")


def test_non_adapter_object_raises_type_error(tmp_path: Path) -> None:
    module_path = tmp_path / "agent.py"
    module_path.write_text("adapter = 'not an adapter'\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Adapter protocol"):
        load_adapter(str(module_path))


def test_sync_run_pipeline_method_rejected_at_load(tmp_path: Path) -> None:
    module_path = tmp_path / "agent.py"
    module_path.write_text(
        "from typing import Any\n"
        "from varix.core import AdapterCapabilities, PipelineRun, StepGraph, StepRun\n"
        "from datetime import UTC, datetime\n"
        "_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)\n"
        "\n"
        "class _BadlyTyped:\n"
        "    def capabilities(self) -> AdapterCapabilities:\n"
        "        return AdapterCapabilities()\n"
        "    async def pipeline_structure(self, pi: Any) -> StepGraph:\n"
        "        return StepGraph(steps=())\n"
        "    def run_pipeline(self, pi: Any, seed: int | None = None) -> PipelineRun:\n"
        "        return PipelineRun(run_id='r', step_runs=(),\n"
        "            started_at=_T, finished_at=_T)\n"
        "    async def replay_step(self, sid: str, fi: Any,\n"
        "        seed: int | None = None) -> StepRun:\n"
        "        raise NotImplementedError\n"
        "\n"
        "adapter = _BadlyTyped()\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match=r"run_pipeline.*async def"):
        load_adapter(str(module_path))


def test_sync_replay_step_method_rejected_at_load(tmp_path: Path) -> None:
    module_path = tmp_path / "agent.py"
    module_path.write_text(
        "from typing import Any\n"
        "from varix.core import AdapterCapabilities, PipelineRun, StepGraph, StepRun\n"
        "from datetime import UTC, datetime\n"
        "_T = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)\n"
        "\n"
        "class _BadlyTyped:\n"
        "    def capabilities(self) -> AdapterCapabilities:\n"
        "        return AdapterCapabilities()\n"
        "    async def pipeline_structure(self, pi: Any) -> StepGraph:\n"
        "        return StepGraph(steps=())\n"
        "    async def run_pipeline(self, pi: Any, seed: int | None = None) -> PipelineRun:\n"
        "        return PipelineRun(run_id='r', step_runs=(),\n"
        "            started_at=_T, finished_at=_T)\n"
        "    def replay_step(self, sid: str, fi: Any,\n"
        "        seed: int | None = None) -> StepRun:\n"
        "        raise NotImplementedError\n"
        "\n"
        "adapter = _BadlyTyped()\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match=r"replay_step.*async def"):
        load_adapter(str(module_path))


def test_fully_async_adapter_loads_cleanly() -> None:
    """Regression: the FakeAdapter (all async) still loads after the new check."""
    adapter = load_adapter("varix.adapters:FakeAdapter")
    assert isinstance(adapter, Adapter)
