"""Общий пул HTTP-соединений (httpx.AsyncClient) для низколатентных вызовов микросервисов.

Контракт ARCH-01: все вызовы ядра к encoder-classifier / ragas-worker / вебхукам
переиспользуют keep-alive соединения вместо нового сокета на каждый запрос.
"""

from __future__ import annotations

import httpx

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Возвращает синглтон AsyncClient с keep-alive пулом."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=2.0),
            limits=httpx.Limits(max_keepalive_connections=32, max_connections=128, keepalive_expiry=60.0),
        )
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
