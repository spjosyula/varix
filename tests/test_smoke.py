"""Smoke tests for the project skeleton.

These prove the toolchain works: the package imports, the version is set,
and the logging hierarchy is configured. Real behavior tests land alongside
the features they cover, starting at C1.1.
"""

import logging


def test_package_imports() -> None:
    import varix

    assert varix.__version__


def test_logger_has_null_handler() -> None:
    import varix  # noqa: F401  ensure package __init__ runs

    logger = logging.getLogger("varix")
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_child_loggers_inherit_from_root() -> None:
    import varix  # noqa: F401

    child = logging.getLogger("varix.core")
    assert child.parent is logging.getLogger("varix")
