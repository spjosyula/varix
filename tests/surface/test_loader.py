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
