"""Tests for VarianceMetric Protocol and ExactMatch."""

from __future__ import annotations

from varix.core import (
    Confidence,
    ExactMatch,
    Finding,
    LocalizationOutcome,
    VarianceMetric,
)


def test_exact_match_satisfies_protocol() -> None:
    assert isinstance(ExactMatch(), VarianceMetric)


def test_exact_match_name_is_stable() -> None:
    assert ExactMatch().name() == "exact"


def test_exact_match_equivalent_for_equal_strings() -> None:
    assert ExactMatch().equivalent("hello", "hello") is True


def test_exact_match_distinguishes_whitespace() -> None:
    assert ExactMatch().equivalent("hello", "hello ") is False
    assert ExactMatch().equivalent("hello", "Hello") is False


def test_exact_match_equivalent_for_equal_dicts() -> None:
    assert ExactMatch().equivalent({"a": 1, "b": 2}, {"a": 1, "b": 2}) is True


def test_exact_match_distinguishes_different_dicts() -> None:
    assert ExactMatch().equivalent({"a": 1}, {"a": 2}) is False


def test_exact_match_equivalent_for_equal_lists() -> None:
    assert ExactMatch().equivalent([1, 2, 3], [1, 2, 3]) is True


def test_exact_match_distinguishes_list_order() -> None:
    assert ExactMatch().equivalent([1, 2, 3], [3, 2, 1]) is False


def test_exact_match_handles_none() -> None:
    assert ExactMatch().equivalent(None, None) is True
    assert ExactMatch().equivalent(None, 0) is False


def test_metric_name_records_on_finding() -> None:
    metric = ExactMatch()
    finding = Finding(
        step_id="s1",
        localization=LocalizationOutcome.DETERMINISTIC,
        confidence=Confidence.HIGH,
        metric_name=metric.name(),
    )
    assert finding.metric_name == "exact"
