"""Гибридный экстрактор слотов (К2, Stage 4).

Компилированные именованные паттерны работают даже без обученной
NER-модели: order_id, phone, email, date, amount, person, url, card.
Спаны паттернов объединяются со спанами ONNX-модели; при пересечении
приоритет у ONNX (обученный домен), паттерн заполняет пробелы.
"""

from __future__ import annotations

import re
from typing import Any

_PATTERN_DEFS: list[tuple[str, str]] = [
    ('EMAIL', r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    ('PHONE', r'(?:\+7|8)[\s\-(]?\d{3}[\s\-)]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b'),
    ('DATE', r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b'),
    ('DATE_WORD', r'\b\d{1,2}\s+(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\s+\d{4}\b'),
    ('AMOUNT', r'\b\d[\d\s]*(?:[.,]\d{1,2})?\s*(?:руб(?:л\w*)?|₽|тыс\.?\s*руб\w*|млн\.?\s*руб\w*)\b'),
    ('CARD', r'\b(?:\d[ -]?){15,16}\d\b'),
    ('URL', r'https?://[^\s,;)]+'),
    ('ORDER_ID', r'(?:№\s*|\bномер\w*\s+|\bзаказ\w*\s+)(\d{4,10})\b'),
    ('PERSON', r'\b[А-ЯЁ][а-яё]+(?:ов|ев|ин|ский|цкий)\s+[А-ЯЁ][а-яё]+\b'),
    ('DATETIME_ISO', r'\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b'),
]

_COMPILED = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _PATTERN_DEFS]

# Метки, которые распознает гибридный экстрактор (для label_map новой NER-модели)
SLOT_LABELS = [
    'O',
    'B-PERSON', 'I-PERSON',
    'B-ORDER_ID', 'I-ORDER_ID',
    'B-DATE', 'I-DATE',
    'B-EMAIL', 'I-EMAIL',
    'B-PHONE', 'I-PHONE',
    'B-AMOUNT', 'I-AMOUNT',
    'B-CARD', 'I-CARD',
]


def extract_pattern_spans(text: str) -> list[dict[str, Any]]:
    """Спаны по компилированным паттернам: text, label, start, end, confidence=1.0."""
    spans: list[dict[str, Any]] = []
    taken: list[tuple[int, int]] = []
    for name, pattern in _COMPILED:
        for match in pattern.finditer(text):
            start, end = match.span(1) if (name == 'ORDER_ID' and match.lastindex) else match.span()
            value = text[start:end].strip()
            if not value:
                continue
            # Не дублируем пересекающиеся спаны более приоритетных паттернов
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            taken.append((start, end))
            label = name.replace('_WORD', '').replace('_ISO', '')
            spans.append(
                {
                    'text': value,
                    'label': label,
                    'start': start,
                    'end': end,
                    'confidence': 1.0,
                    'source': 'regex',
                }
            )
    spans.sort(key=lambda s: s['start'])
    return spans


def merge_spans(onnx_spans: list[dict[str, Any]], regex_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Объединение: ONNX-спаны приоритетны при пересечении, паттерны — на пробелах."""
    merged: list[dict[str, Any]] = []
    for span in sorted(onnx_spans, key=lambda s: s.get('start', 0)):
        merged.append({**span, 'source': 'onnx'})
    for span in regex_spans:
        overlaps = any(
            span['start'] < other.get('end', 0) and span['end'] > other.get('start', 0)
            for other in merged
        )
        if not overlaps:
            merged.append(span)
    merged.sort(key=lambda s: s.get('start', 0))
    return merged
