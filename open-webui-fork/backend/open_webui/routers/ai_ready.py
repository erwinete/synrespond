"""AI-Ready самообнаружение (Stage 4, раздел 4.2).

GET /llms.txt — компактное Markdown-руководство для контекста ИИ-агента.
GET /api/v1/ai-manifest.json — полная спецификация: дескрипторы настроек,
шаблон манифеста бота, список MCP-инструментов и энкодеров.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Response

from open_webui.models.config import Config
from open_webui.models.intents import Encoders
from open_webui.utils.ai_ready_schema import AI_PARAMETER_DESCRIPTORS, _empty_manifest
from open_webui.utils.auth import get_verified_user

log = logging.getLogger(__name__)

router = APIRouter()

LLMS_TXT = """# Enterprise AI Platform — AI-Ready Guide

Self-hosted платформа текстовых и голосовых ботов: маршрутизация интентов
на ONNX-энкодерах (5-20 мс), трехконтурный RAG (Dense + BM25 + HippoRAG),
контроль качества Ragas, три независимых канала.

## Канальные модули (работают автономно или вместе)

| Канал | config-ключ | Описание |
|---|---|---|
| chat | channels.chat.enabled | Текстовый чат и виджет, SSE, Reasoning Path |
| webrtc | channels.webrtc.enabled | Браузерный голос: WS /api/v1/voice/webrtc/stream, VAD, barge-in |
| sip | channels.sip.enabled | Телефония Asterisk через voice-sip-bridge (:8765), /v1/call-status |

## Подключение как MCP-клиент

POST /api/v1/mcp — JSON-RPC 2.0 (initialize, tools/list, tools/call,
resources/list, resources/read). SSE: GET /api/v1/mcp/sse.
Авторизация: Authorization: Bearer <API-ключ Open WebUI>.

## Шаблонные переменные

| Переменная | Где доступна |
|---|---|
| {{user_message}} | direct_response, system_prompt_template, webhook body |
| {{intent}} | там же |
| {{slots.<name>}} | после сбора required_slots |
| {{voice.anger_score}} | голосовые каналы (webrtc, sip) |
| {{channel.type}} | все каналы: chat / webrtc / sip |

## Playbook: создать бота за 4 шага

1. Прочитай /api/v1/ai-manifest.json (дескрипторы всех настроек).
2. Вызови MCP tools/call bot_apply_behavior_manifest с BotBehaviorManifest
   (dry_run=true сначала — семантический линтер вернет ошибки и
   agent_remediation_hint).
3. Дообучи энкодер: tools/call encoder_bootstrap_or_train с примерами
   [{"text","label"}] — Rollback Gate не пропустит деградацию F1.
4. Проверь сценарий: tools/call bot_simulate_conversation_turn для каждого
   канала (chat, webrtc, sip) — вернется детерминированный DecisionTrace.

## Инструменты MCP (8)

platform_inspect_status, platform_configure_modules,
bot_apply_behavior_manifest, bot_export_behavior_manifest,
intent_upsert_single_route, bot_simulate_conversation_turn,
encoder_bootstrap_or_train, knowledge_hipporag_manage.
"""


@router.get('/llms.txt', include_in_schema=True)
async def llms_txt():
    """Публичное руководство для ИИ-агентов (стандарт /llms.txt)."""
    return Response(content=LLMS_TXT, media_type='text/markdown; charset=utf-8')


@router.get('/ai-manifest.json')
async def ai_manifest(user=Depends(get_verified_user)):
    """Полная AI-Ready спецификация платформы."""
    settings = await Config.get_many(*[d['key'] for d in AI_PARAMETER_DESCRIPTORS])
    encoders = await Encoders.get_available_encoders()
    return {
        'name': 'enterprise-ai-platform',
        'version': '1.0.0',
        'channels': ['chat', 'webrtc', 'sip'],
        'mcp': {
            'endpoint': '/api/v1/mcp',
            'sse': '/api/v1/mcp/sse',
            'tools': [
                'platform_inspect_status',
                'platform_configure_modules',
                'bot_apply_behavior_manifest',
                'bot_export_behavior_manifest',
                'intent_upsert_single_route',
                'bot_simulate_conversation_turn',
                'encoder_bootstrap_or_train',
                'knowledge_hipporag_manage',
            ],
        },
        'parameter_descriptors': AI_PARAMETER_DESCRIPTORS,
        'current_config': settings,
        'bot_behavior_manifest_template': _empty_manifest(),
        'template_variables': [
            '{{user_message}}',
            '{{intent}}',
            '{{slots.<name>}}',
            '{{voice.anger_score}}',
            '{{channel.type}}',
        ],
        'encoders': [
            {'id': encoder.id, 'task': encoder.task, 'model_path': encoder.model_path}
            for encoder in encoders
        ],
    }
