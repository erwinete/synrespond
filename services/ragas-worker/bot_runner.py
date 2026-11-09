"""Прогон бота: вопрос -> полный RAG-пайплайн Open WebUI -> ответ и контексты."""

from __future__ import annotations

import logging
from typing import Any, Optional

import aiohttp

from config import OPEN_WEBUI_TOKEN, OPEN_WEBUI_URL
from evaluator import extract_contexts_from_sources

log = logging.getLogger(__name__)


async def run_bot_pipeline(
    question: str, bot_model_id: Optional[str] = None, rag_overrides: Optional[dict] = None
) -> dict[str, Any]:
    """Отправляет вопрос в /api/chat/completions (stream: false).

    Возвращает {'answer', 'contexts', 'error'}; контексты берутся из
    sources в ответе, которые бэкенд добавляет при RAG-поиске.
    rag_overrides — веса поиска для Auto-Tune (фича №4).
    """
    result: dict[str, Any] = {'answer': '', 'contexts': [], 'pool': {}, 'error': None}
    headers = {'Content-Type': 'application/json'}
    if OPEN_WEBUI_TOKEN:
        headers['Authorization'] = f'Bearer {OPEN_WEBUI_TOKEN}'
    payload = {
        'model': bot_model_id or 'default',
        'messages': [{'role': 'user', 'content': question}],
        'stream': False,
    }
    if rag_overrides:
        payload['metadata'] = {'rag_overrides': rag_overrides}
    try:
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f'{OPEN_WEBUI_URL}/api/chat/completions', json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                data = await response.json()
        choices = data.get('choices') or []
        result['answer'] = (choices[0].get('message', {}).get('content') or '') if choices else ''
        sources = data.get('sources') or []
        result['contexts'] = extract_contexts_from_sources(sources)
        # Д6: сырой пул чанков (тексты + метаданные pool_list/pool_rank) для Auto-Tune
        documents: list[str] = []
        metadatas: list[dict] = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            docs = source.get('document') or []
            metas = source.get('metadata') or []
            documents.extend(str(doc) for doc in docs if doc)
            for idx in range(len(docs)):
                metadatas.append(metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {})
        result['pool'] = {'documents': [documents], 'metadatas': [metadatas]}
        if not result['answer']:
            result['error'] = 'Empty answer from bot'
    except Exception as exc:  # noqa: BLE001
        result['error'] = str(exc)
    return result
