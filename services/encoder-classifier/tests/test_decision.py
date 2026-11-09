"""Юнит-тесты логики маршрутизации и softmax (без загрузки моделей).

Запуск: python -m pytest tests/ -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision import DIRECT, DISAMBIGUATE, FALLBACK, decide, multi_label_select, softmax


def test_direct_route_above_high_threshold():
    d = decide("3", "billing_refund", 0.93, tau_high=0.85, tau_low=0.50)
    assert d.action == DIRECT
    assert d.intent_name == "billing_refund"


def test_disambiguate_in_uncertainty_zone():
    d = decide("7", "order_status", 0.62, tau_high=0.85, tau_low=0.50)
    assert d.action == DISAMBIGUATE


def test_fallback_below_low_threshold():
    d = decide("1", "smalltalk", 0.31, tau_high=0.85, tau_low=0.50)
    assert d.action == FALLBACK


def test_boundary_values():
    assert decide("0", "a", 0.85, 0.85, 0.5).action == DIRECT
    assert decide("0", "a", 0.50, 0.85, 0.5).action == DISAMBIGUATE
    assert decide("0", "a", 0.49, 0.85, 0.5).action == FALLBACK


def test_inverted_thresholds_fall_back_to_defaults():
    d = decide("0", "a", 0.90, tau_high=0.2, tau_low=0.9)
    # tau_low > tau_high — некорректная настройка, применяются значения по умолчанию
    assert d.action == DIRECT


def test_softmax_sums_to_one_and_preserves_order():
    probs = softmax([2.0, 1.0, 0.1], temperature=1.0)
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[0] > probs[1] > probs[2]


def test_softmax_temperature_sharpens():
    sharp = softmax([2.0, 1.0], temperature=0.1)
    flat = softmax([2.0, 1.0], temperature=10.0)
    assert sharp[0] > flat[0]


def test_softmax_numerical_stability():
    probs = softmax([1000.0, 999.0], temperature=1.0)
    assert abs(sum(probs) - 1.0) < 1e-9


def test_multi_label_select():
    picked = multi_label_select({"a": 0.9, "b": 0.7, "c": 0.3}, threshold=0.5)
    assert picked == [("a", 0.9), ("b", 0.7)]
