"""Переписывание запроса перед поиском (ТЗ, раздел 5.2).

Этапы:
1. Деконтекстуализация — разрешение местоимений по истории диалога.
2. Декомпозиция — разбиение составного вопроса на 2-3 атомарных подзапроса.

LLM вызывается через OpenAI-совместимый эндпоинт из настроек RAG.
Модель задаётся конфигом rag.query_rewriting_model; без модели хук не активен.
Любая ошибка возвращает исходные запросы — поиск никогда не блокируется.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import aiohttp

from open_webui.env import AIOHTTP_CLIENT_SESSION_SSL, AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST

log = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """You are a search query optimizer for a RAG system.
Given the conversation history and the latest user question:
1. Rewrite the question as a standalone question (resolve pronouns and references to the history).
2. If the question is compound, split it into 2-3 atomic search queries.
Reply with strict JSON only: {"rewritten": "<standalone question>", "sub_queries": ["<q1>", "<q2>"]}
Do not add any commentary."""


async def _call_llm(
    base_url: str,
    api_key: str,
    model: str,
    history: list[dict[str, str]],
    timeout: int,
) -> Optional[dict[str, Any]]:
    """Один chat.completions вызов с JSON-ответом."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    payload = {
        'model': model,
        'messages': [*history, {'role': 'system', 'content': REWRITE_SYSTEM_PROMPT}],
        'temperature': 0.0,
        'stream': False,
    }
    timeout_config = aiohttp.ClientTimeout(total=timeout or 30)
    connector = aiohttp.TCPConnector(ssl=AIOHTTP_CLIENT_SESSION_SSL)
    async with aiohttp.ClientSession(timeout=timeout_config, connector=connector) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
    bracket_start = content.rfind('{')
    bracket_end = content.rfind('}') + 1
    if bracket_start == -1 or bracket_end <= bracket_start:
        return None
    parsed = json.loads(content[bracket_start:bracket_end])
    return parsed if isinstance(parsed, dict) else None


def _history_tail(messages: list[dict[str, Any]], limit: int = 6) -> list[dict[str, str]]:
    """Последние реплики диалога в формате OpenAI."""
    tail: list[dict[str, str]] = []
    for message in messages[-limit:]:
        content = message.get('content')
        if isinstance(content, str) and content.strip():
            tail.append({'role': message.get('role', 'user'), 'content': content[:2000]})
    return tail


async def rewrite_and_expand(
    request,
    queries: list[str],
    messages: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[str], str]:
    """Возвращает (запросы для поиска, переписанный запрос для трейса).

    При выключенном конфиге, отсутствующей модели или ошибке LLM — исходные запросы.
    """
    from open_webui.models.config import Config

    try:
        # Д3: Voice Fast-Path — в телефонии LLM-переписывание пропускается
        if getattr(request.state, 'voice_mode', False):
            return queries, ''
        enabled = await Config.get('rag.enable_query_rewriting')
        model = await Config.get('rag.query_rewriting_model')
        if not enabled or not model:
            return queries, ''
        base_url = await Config.get('rag.openai.api_base_url')
        api_key = await Config.get('rag.openai.api_key')
        if not base_url:
            return queries, ''

        history = _history_tail(messages or [])
        parsed = await _call_llm(
            base_url=base_url,
            api_key=api_key,
            model=model,
            history=history,
            timeout=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST if isinstance(AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST, int) else 30,
        )
        if not parsed:
            return queries, ''

        rewritten = str(parsed.get('rewritten') or '').strip()
        sub_queries = [str(q).strip() for q in (parsed.get('sub_queries') or []) if str(q).strip()]
        expanded = [rewritten, *sub_queries] if rewritten else sub_queries
        expanded = [q for q in expanded if q][:4]
        return (expanded or queries), rewritten
    except Exception as exc:  # noqa: BLE001 — ретривер не должен падать из-за рерайтера
        log.warning('Query rewriting skipped: %s', exc)
        return queries, ''
