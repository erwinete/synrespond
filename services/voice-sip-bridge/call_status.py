"""Реестр статусов звонков для Asterisk (Д5, Stage 3).

Хранилище: Redis (`call_transfer:{uuid}`, пишет бэкенд при HUMAN_HANDOFF)
с fallback на память процесса (эскалация по гневу внутри моста).
Диалплан Asterisk после выхода из AudioSocket запрашивает
GET /v1/call-status/{uuid} через CURL() и при непустом ответе
уходит в Queue(<имя>).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from config import REDIS_URL

log = logging.getLogger(__name__)

_memory: dict[str, tuple[str, float]] = {}
_TTL = 300
_redis = None


async def _get_redis():
    global _redis
    if _redis is None and REDIS_URL:
        try:
            import redis.asyncio as aioredis

            _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        except Exception as exc:  # noqa: BLE001
            log.warning('Redis unavailable for call status: %s', exc)
            _redis = False
    return _redis or None


async def mark_transfer(call_uuid: str, queue: str) -> None:
    """Эскалация внутри моста (например, по порогу гнева)."""
    if not call_uuid:
        return
    redis = await _get_redis()
    if redis:
        try:
            await redis.set(f'call_transfer:{call_uuid}', queue, ex=_TTL)
            return
        except Exception as exc:  # noqa: BLE001
            log.warning('Call status Redis write failed: %s', exc)
    _memory[call_uuid] = (queue, time.monotonic() + _TTL)


async def get_transfer(call_uuid: str) -> Optional[str]:
    """Имя очереди для перевода или None (звонок не требует оператора)."""
    redis = await _get_redis()
    if redis:
        try:
            value = await redis.get(f'call_transfer:{call_uuid}')
            if value and value != 'NONE':
                return value
            if value == 'NONE':
                return None
        except Exception as exc:  # noqa: BLE001
            log.warning('Call status Redis read failed: %s', exc)
    record = _memory.get(call_uuid)
    if record is None:
        return None
    queue, expires_at = record
    if expires_at < time.monotonic():
        _memory.pop(call_uuid, None)
        return None
    return queue
