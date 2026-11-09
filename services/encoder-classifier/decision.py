"""Правила маршрутизации по порогам уверенности.

Чистые функции без зависимости от сервиса. Покрыты тестами.
Логика из ТЗ, раздел 4.1:
  p >= tau_high            -> direct (жесткая маршрутизация)
  tau_low <= p < tau_high  -> disambiguate (уточняющий вопрос или LLM-фоллбэк)
  p < tau_low              -> fallback (общий RAG-маршрут)
"""

from dataclasses import dataclass

DIRECT = "direct"
DISAMBIGUATE = "disambiguate"
FALLBACK = "fallback"

DEFAULT_TAU_HIGH = 0.85
DEFAULT_TAU_LOW = 0.50


@dataclass(frozen=True)
class Decision:
    action: str
    intent_id: str
    intent_name: str
    confidence: float


def decide(
    top_intent_id: str,
    top_intent_name: str,
    confidence: float,
    tau_high: float = DEFAULT_TAU_HIGH,
    tau_low: float = DEFAULT_TAU_LOW,
) -> Decision:
    """Выбирает действие по двухуровневому порогу уверенности."""
    if tau_low > tau_high:
        # Некорректная конфигурация: считаем по умолчанию
        tau_high, tau_low = DEFAULT_TAU_HIGH, DEFAULT_TAU_LOW
    if confidence >= tau_high:
        action = DIRECT
    elif confidence >= tau_low:
        action = DISAMBIGUATE
    else:
        action = FALLBACK
    return Decision(action, top_intent_id, top_intent_name, confidence)


def multi_label_select(
    probabilities: dict[str, float],
    threshold: float = 0.5,
) -> list[tuple[str, float]]:
    """Для multi-label: возвращает все метки с p >= threshold, по убыванию."""
    return sorted(
        ((name, p) for name, p in probabilities.items() if p >= threshold),
        key=lambda item: item[1],
        reverse=True,
    )


def softmax(logits: "list[float]", temperature: float = 1.0) -> list[float]:
    """Численно стабильный softmax с масштабированием температуры."""
    import math

    scaled = [z / max(temperature, 1e-5) for z in logits]
    max_z = max(scaled)
    exps = [math.exp(z - max_z) for z in scaled]
    total = sum(exps)
    return [e / total for e in exps]
