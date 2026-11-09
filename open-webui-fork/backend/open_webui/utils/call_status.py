"""Реестр статусов звонков для диалплана Asterisk (Д5, Stage 3).

Основное хранилище — Redis (`call_transfer:{uuid}`, TTL 300 c, пишет бэкенд).
Этот модуль — внутрипроцессный fallback, когда Redis недоступен: эндпоинт
`/v1/call-status/{uuid}` голосового моста отдает очередь из этого реестра.
"""

from __future__ import annotations

import time
from typing import Optional

_transfers: dict[str, tuple[str, float]] = {}
_TTL = 300


def mark_transfer(call_uuid: str, queue: Optional[str]) -> None:
    """Фиксирует цель перевода для звонка."""
    if call_uuid:
        _transfers[call_uuid] = (queue or 'NONE', time.monotonic() + _TTL)


def get_transfer(call_uuid: str) -> Optional[str]:
    """Возвращает имя очереди или None (и подчищает истекшие записи)."""
    record = _transfers.get(call_uuid)
    if record is None:
        return None
    value, expires_at = record
    if expires_at < time.monotonic():
        _transfers.pop(call_uuid, None)
        return None
    return value
