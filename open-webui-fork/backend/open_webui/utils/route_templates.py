"""Чистые функции маршрутизации: шаблоны правил и решение о дизамбигуации.

Модуль не тянет зависимости open_webui — покрывается юнит-тестами напрямую
(корень монорепозитория, tests/test_route_templates.py).
"""

from __future__ import annotations

import re
from typing import Any

_TEMPLATE_TOKEN = re.compile(r'\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}')
_UNMATCHED_SLOT = re.compile(r'\{\{\s*slots\.[a-zA-Z0-9_]+\s*\}\}')
_DEFAULT_DISAMBIGUATION_DELTA = 0.15

DISAMBIGUATION_PROMPT = (
    'Намерение пользователя неоднозначно между сценариями: '
    '[Кандидат 1: {{top_1}}], [Кандидат 2: {{top_2}}]. '
    'Задай один короткий уточняющий вопрос, чтобы выбрать точный сценарий, '
    'прежде чем давать развернутый ответ.'
)


def render_route_template(
    template_str: str,
    user_message: str = '',
    intent: str = '',
    slots: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Подстановка {{user_message}}, {{intent}}, {{slots.<key>}} и доп. переменных.

    Незаполненные слоты очищаются пустой строкой — шаблон никогда не уходит
    наружу с сырыми скобками.
    """
    slots = slots or {}
    extra = extra or {}

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key == 'user_message':
            return user_message
        if key == 'intent':
            return intent
        if key.startswith('slots.'):
            slot_key = key[len('slots.') :]
            value = slots.get(slot_key)
            return str(value) if value is not None else ''
        if key in extra:
            return str(extra[key])
        return match.group(0)

    rendered = _TEMPLATE_TOKEN.sub(replace, template_str)
    # Ненайденные слоты с выравниванием ключа (регистр/пробелы) тоже чистим
    rendered = _UNMATCHED_SLOT.sub('', rendered)
    return rendered


def should_disambiguate(
    top_k_candidates: list[dict[str, Any]],
    delta_threshold: float = _DEFAULT_DISAMBIGUATION_DELTA,
) -> tuple[bool, str, str, float]:
    """Активная дизамбигуация в серой зоне.

    Возвращает (нужно_уточнить, top_1, top_2, delta). Уточнение требуется,
    когда разница вероятностей двух лидеров меньше порога.
    """
    if len(top_k_candidates) < 2:
        return False, '', '', 1.0
    top_1 = str(top_k_candidates[0].get('intent_name', ''))
    top_2 = str(top_k_candidates[1].get('intent_name', ''))
    p1 = float(top_k_candidates[0].get('confidence', 0.0))
    p2 = float(top_k_candidates[1].get('confidence', 0.0))
    delta = p1 - p2
    return delta < delta_threshold, top_1, top_2, delta


def build_disambiguation_prompt(
    top_k_candidates: list[dict[str, Any]],
    delta_threshold: float = _DEFAULT_DISAMBIGUATION_DELTA,
) -> str | None:
    """Готовая инструкция уточнения или None, если кандидаты различимы."""
    needed, top_1, top_2, _ = should_disambiguate(top_k_candidates, delta_threshold)
    if not needed:
        return None
    return render_route_template(DISAMBIGUATION_PROMPT, extra={'top_1': top_1, 'top_2': top_2})


def slots_from_spans(spans: list[dict[str, Any]]) -> dict[str, str]:
    """Словарь {label: text} из ответа /v1/extract-slots. Поздний спан побеждает."""
    slots: dict[str, str] = {}
    for span in spans or []:
        label = str(span.get('label') or '').strip()
        value = str(span.get('text') or '').strip()
        if label and value:
            slots[label] = value
    return slots


# Д3 (Stage 3): защита slot-filling от зацикливания

SLOT_CANCEL_PATTERNS = (
    r'\bотмен\w*\b',
    r'\bстоп\b',
    r'\bсброс\w*\b',
    r'\bпередума\w*\b',
    r'\bзабудь\b',
    r'\bсоедини\s+(меня\s+)?с\s+оператором\b',
    r'\bк\s+оператору\b',
    r'\bcancel\b',
    r'\bstop\b',
    r'\bspeak\s+to\s+(an?\s+)?(agent|human|operator)\b',
)

SLOT_MAX_RETRIES = 3


def is_slot_cancellation(text: str) -> bool:
    """Глобальный стоп-фильтр: пользователь выходит из сбора слотов."""
    lowered = (text or '').lower()
    return any(re.search(pattern, lowered) for pattern in SLOT_CANCEL_PATTERNS)


def mentions_slot_value(text: str, patterns: list[str]) -> bool:
    """Реплика содержит значение, потенциально удовлетворяющее схеме слотов."""
    return any(pattern and re.search(pattern, text or '') for pattern in patterns)


def slot_retry_exceeded(retries: int, max_retries: int = SLOT_MAX_RETRIES) -> bool:
    """Превышен лимит неудачных попыток валидации слота."""
    return retries >= max_retries
