"""Tests for VarianceMetric Protocol and ExactMatch."""

from __future__ import annotations

import math

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


def test_nan_top_level_is_equivalent_to_itself() -> None:
    assert ExactMatch().equivalent(float("nan"), float("nan")) is True


def test_nan_distinct_from_non_nan_float() -> None:
    assert ExactMatch().equivalent(float("nan"), 0.0) is False
    assert ExactMatch().equivalent(float("nan"), 1.0) is False


def test_distinct_non_nan_floats_remain_unequal() -> None:
    assert ExactMatch().equivalent(1.0, 2.0) is False


def test_floats_with_tiny_precision_difference_still_unequal() -> None:
    assert ExactMatch().equivalent(0.1 + 0.2, 0.3) is False


def test_nan_inside_list_is_equivalent() -> None:
    a = [1.0, float("nan"), 3.0]
    b = [1.0, float("nan"), 3.0]
    assert ExactMatch().equivalent(a, b) is True


def test_nan_inside_tuple_is_equivalent() -> None:
    a = (1.0, float("nan"))
    b = (1.0, float("nan"))
    assert ExactMatch().equivalent(a, b) is True


def test_nan_inside_dict_value_is_equivalent() -> None:
    a = {"score": float("nan"), "label": "x"}
    b = {"score": float("nan"), "label": "x"}
    assert ExactMatch().equivalent(a, b) is True


def test_nan_deeply_nested_is_equivalent() -> None:
    a = {"results": [{"score": float("nan")}, {"score": 0.5}]}
    b = {"results": [{"score": float("nan")}, {"score": 0.5}]}
    assert ExactMatch().equivalent(a, b) is True


def test_list_length_mismatch_is_not_equivalent() -> None:
    assert ExactMatch().equivalent([1, 2], [1, 2, 3]) is False


def test_dict_key_mismatch_is_not_equivalent() -> None:
    assert ExactMatch().equivalent({"a": 1}, {"b": 1}) is False


def test_list_and_tuple_with_same_elements_remain_unequal() -> None:
    assert ExactMatch().equivalent([1, 2], (1, 2)) is False


def test_nan_vs_string_is_not_equivalent() -> None:
    assert ExactMatch().equivalent(float("nan"), "nan") is False


def test_math_isnan_baseline() -> None:
    assert math.isnan(float("nan"))
    assert ExactMatch().equivalent(2.5, 2.5) is True
    assert ExactMatch().equivalent(2.5, float("nan")) is False
