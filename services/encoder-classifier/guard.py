"""Guardrails: детекция промпт-инъекций и джейлбрейков (киллер-фича №5).

Основной путь — ONNX-модель (каталог 'prompt-guard' с метками
INJECTION/SAFE). Без модели работает консервативная эвристика по
известным шаблонам инъекций на русском и английском.
"""

from __future__ import annotations

import re
from typing import Any

GUARD_MODEL_ID = 'prompt-guard'
INJECTION_LABEL = 'INJECTION'

_HEURISTIC_PATTERNS = [
    r'забуд(ь)?\s+(все|всё)\s+(свои\s+)?(инструкции|правила|предыдущ)',
    r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions',
    r'(представ(ь)?|представь)\s+себе,?\s+что\s+ты',
    r'act\s+as\s+(if\s+you\s+were\s+)?(a\s+)?(different|new)\s+(ai|assistant|system)',
    r'ты\s+теперь\s+(директор|администратор|система|разработчик)',
    r'you\s+are\s+now\s+(a|an|the)\s+(dan|admin|developer|root)',
    r'(режим|mode)\s+(без\s+ограничений|without\s+restrictions|developer)',
    r'раскрой\s+(сво(и|й)\s+)?(системн|инструкци|промпт)',
    r'reveal\s+(your\s+)?(system\s+)?(prompt|instructions)',
    r'одобри\s+(мне\s+)?(скидк|возврат|платеж)',
    r'отключи\s+(все\s+)?(фильтр|ограничен|проверк)',
    r'disable\s+(all\s+)?(safety|filters|restrictions)',
]

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in _HEURISTIC_PATTERNS]


async def detect_injection(
    registry, text: str, threshold: float
) -> dict[str, Any]:
    """Возвращает {'injection': bool, 'score': float, 'labels': [...]}.

    При наличии ONNX-модели prompt-guard — скор модели (softmax),
    иначе — эвристика: score = доля сработавших шаблонов (0.25 каждая),
    порог 0.5.
    """
    artifact = None
    try:
        artifact = registry.get(GUARD_MODEL_ID)
    except Exception:  # noqa: BLE001 — модель не установлена, это норма
        artifact = None

    if artifact is not None and artifact.task == 'sequence_classification':
        import numpy as np

        logits = artifact.classify(text)
        scaled = logits - logits.max()
        exp = np.exp(scaled)
        probs = exp / exp.sum()
        labels = artifact.labels or {}
        injection_idx = next(
            (idx for idx, name in labels.items() if str(name).upper() == INJECTION_LABEL),
            int(probs.argmax()),
        )
        score = float(probs[injection_idx])
        return {
            'injection': score >= threshold,
            'score': round(score, 4),
            'labels': {name: round(float(probs[idx]), 4) for idx, name in labels.items()},
            'engine': 'onnx',
        }

    hits = [pattern.pattern for pattern in _COMPILED if pattern.search(text)]
    score = min(1.0, 0.25 * len(hits)) if hits else 0.0
    return {
        'injection': score >= 0.5,
        'score': round(score, 4),
        'labels': {'matched_patterns': hits},
        'engine': 'heuristic',
    }
