"""MCP Control Plane Server (Stage 4, направление А).

JSON-RPC 2.0 поверх Streamable HTTP: POST /api/v1/mcp и GET /api/v1/mcp/sse.
Внешние агентные харнесы (Claude Code, Cursor, LangGraph, OpenAI Agents SDK)
получают программный контроль над созданием и настройкой ботов.

Инструменты: platform_inspect_status, platform_configure_modules,
bot_apply_behavior_manifest, bot_export_behavior_manifest,
intent_upsert_single_route, bot_simulate_conversation_turn,
encoder_bootstrap_or_train, knowledge_hipporag_manage.
Ресурсы: ai-ready-schema, config, bots/{id}/manifest, bots/{id}/telemetry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from open_webui.models.config import Config
from open_webui.models.intents import (
    Encoders,
    IntentFeedbacks,
    IntentRouteModel,
    IntentRoutes,
    MCPHarnesses,
)
from open_webui.models.knowledge import Knowledges
from open_webui.models.models import Models
from open_webui.models.ragas_evals import RagasRuns, RagasSamples
from open_webui.utils.ai_ready_schema import (
    AI_PARAMETER_DESCRIPTORS,
    _empty_manifest,
    validate_bot_manifest,
)
from open_webui.utils.auth import get_verified_user

log = logging.getLogger(__name__)

router = APIRouter()

RPC_METHODS = ('initialize', 'ping', 'tools/list', 'tools/call', 'resources/list', 'resources/read')

TOOLS_SPEC: list[dict[str, Any]] = [
    {
        'name': 'platform_inspect_status',
        'description': (
            'Возвращает состояние всех модулей (chat, webrtc, sip, encoder, hipporag, ragas), '
            'список зарегистрированных ONNX-энкодеров с классами, список баз знаний и MCP-харнесов.'
        ),
        'inputSchema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'platform_configure_modules',
        'description': (
            'Включает/выключает каналы (chat, webrtc, sip) и настройки '
            '(hipporag.enable, ragas.enable, guardrails.*, веса RRF) без перезапуска сервера.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'settings': {
                    'type': 'object',
                    'description': 'Словарь ключ->значение; ключи из /api/v1/ai-manifest.json',
                }
            },
            'required': ['settings'],
        },
    },
    {
        'name': 'bot_apply_behavior_manifest',
        'description': (
            'Главный декларативный инструмент создания бота. Принимает BotBehaviorManifest, '
            'валидирует и атомарно создает/обновляет модель бота, промпт, каналы и матрицу интентов.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'manifest': {'type': 'object', 'description': 'BotBehaviorManifest JSON'},
                'dry_run': {'type': 'boolean', 'default': False},
            },
            'required': ['manifest'],
        },
    },
    {
        'name': 'bot_export_behavior_manifest',
        'description': 'Выгружает текущую конфигурацию бота в BotBehaviorManifest JSON.',
        'inputSchema': {
            'type': 'object',
            'properties': {'bot_model_id': {'type': 'string'}},
            'required': ['bot_model_id'],
        },
    },
    {
        'name': 'intent_upsert_single_route',
        'description': (
            'Точечно добавляет/модифицирует правило матрицы интентов: пороги, allowed_channels, '
            'action_type, required_slots, use_hipporag, mcp_harness_config.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'bot_model_id': {'type': 'string'},
                'route': {'type': 'object'},
            },
            'required': ['bot_model_id', 'route'],
        },
    },
    {
        'name': 'bot_simulate_conversation_turn',
        'description': (
            'Песочница: тестовая реплика в канале chat/webrtc/sip с заданным anger_score. '
            'Возвращает детерминированный DecisionTrace без вызова LLM.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'bot_model_id': {'type': 'string'},
                'text': {'type': 'string'},
                'channel': {'type': 'string', 'enum': ['chat', 'webrtc', 'sip']},
                'anger_score': {'type': 'number', 'minimum': 0, 'maximum': 1},
            },
            'required': ['bot_model_id', 'text'],
        },
    },
    {
        'name': 'encoder_bootstrap_or_train',
        'description': (
            'Передает обучающие примеры [{"text","label"}] и запускает дообучение ONNX-энкодера '
            '(/v1/train) с Rollback Gate; возвращает job_id.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'base_model_id': {'type': 'string'},
                'examples': {
                    'type': 'array',
                    'items': {'type': 'object', 'properties': {'text': {'type': 'string'}, 'label': {'type': 'string'}}},
                },
                'activate': {'type': 'boolean', 'default': True},
                'force_swap': {'type': 'boolean', 'default': False},
            },
            'required': ['base_model_id', 'examples'],
        },
    },
    {
        'name': 'knowledge_hipporag_manage',
        'description': (
            'Управление базой знаний и графом HippoRAG: добавить текст (add_text), '
            'перестроить граф (reindex), тестовый поиск по графу (query).'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'action': {'type': 'string', 'enum': ['add_text', 'reindex', 'query']},
                'knowledge_id': {'type': 'string'},
                'name': {'type': 'string'},
                'text': {'type': 'string'},
                'query': {'type': 'string'},
            },
            'required': ['action'],
        },
    },
]

RESOURCES_SPEC: list[dict[str, Any]] = [
    {'uri': 'mcp://platform/ai-ready-schema', 'name': 'AI-Ready schema', 'mimeType': 'application/json'},
    {'uri': 'mcp://platform/config', 'name': 'Platform config tree', 'mimeType': 'application/json'},
    {'uri': 'mcp://bots/{bot_id}/manifest', 'name': 'Bot behavior manifest', 'mimeType': 'application/json'},
    {'uri': 'mcp://bots/{bot_id}/telemetry', 'name': 'Bot telemetry (Ragas, alerts, Active Learning)', 'mimeType': 'application/json'},
]


class JsonRpcRequest(BaseModel):
    jsonrpc: str = '2.0'
    id: Any = None
    method: str
    params: dict[str, Any] = {}


def _rpc_result(rpc_id: Any, result: Any) -> dict[str, Any]:
    return {'jsonrpc': '2.0', 'id': rpc_id, 'result': result}


def _rpc_error(rpc_id: Any, code: int, message: str) -> dict[str, Any]:
    return {'jsonrpc': '2.0', 'id': rpc_id, 'error': {'code': code, 'message': message}}


async def _config_map(keys: list[str]) -> dict[str, Any]:
    return await Config.get_many(*keys)


async def _tool_platform_insight(_args: dict[str, Any]) -> dict[str, Any]:
    settings = await _config_map(
        [
            'channels.chat.enabled', 'channels.webrtc.enabled', 'channels.sip.enabled',
            'intent_classifier.enable', 'intent_classifier.url', 'intent_classifier.model_id',
            'intent_classifier.slot_model_id', 'hipporag.enable', 'ragas.enable',
            'guardrails.prompt_guard', 'guardrails.pii_masking',
            'rag.hybrid_weight_dense', 'rag.hybrid_weight_bm25', 'rag.hybrid_weight_hipporag',
            'rag.enable_query_rewriting', 'voice.handoff_queue',
        ]
    )
    encoders = await Encoders.get_available_encoders()
    knowledges = await Knowledges.get_knowledge_bases()
    harnesses = await MCPHarnesses.get_harnesses()
    return {
        'modules': {
            'chat': settings.get('channels.chat.enabled'),
            'webrtc': settings.get('channels.webrtc.enabled'),
            'sip': settings.get('channels.sip.enabled'),
            'encoder': settings.get('intent_classifier.enable'),
            'hipporag': settings.get('hipporag.enable'),
            'ragas': settings.get('ragas.enable'),
        },
        'settings': settings,
        'encoders': [encoder.model_dump() for encoder in encoders],
        'knowledge_bases': [{'id': kb.id, 'name': kb.name} for kb in knowledges],
        'mcp_harnesses': [
            {'id': h.id, 'name': h.name, 'transport': h.transport, 'endpoint': h.endpoint_or_cmd}
            for h in harnesses
        ],
    }


async def _tool_configure_modules(args: dict[str, Any]) -> dict[str, Any]:
    settings = args.get('settings') or {}
    if not settings:
        raise ValueError('settings is empty')
    allowed_prefixes = ('channels.', 'modules.', 'hipporag.', 'ragas.', 'guardrails.', 'rag.hybrid_weight_', 'intent_classifier.')
    filtered = {k: v for k, v in settings.items() if k.startswith(allowed_prefixes)}
    if not filtered:
        raise ValueError('No allowed keys in settings (см. ai-manifest)')
    await Config.upsert(filtered)
    return {'updated': filtered}


async def _encoder_labels(encoder_model_id: str) -> dict[str, Any]:
    encoder = await Encoders.get_encoder_by_id(encoder_model_id)
    labels = (encoder.labels_json or {}) if encoder else {}
    return {v if isinstance(v, str) else str(v): True for v in labels.values()} if labels else {}


async def _validate_context(manifest: dict[str, Any]) -> dict[str, Any]:
    encoder_model_id = manifest.get('encoder_model_id') or 'rubert-intents-v1'
    harnesses = await MCPHarnesses.get_harnesses()
    harness_tools = {
        harness.id: [
            tool.get('name') for tool in ((harness.discovered_tools_schema or {}).get('tools') or [])
            if isinstance(tool, dict) and tool.get('name')
        ]
        for harness in harnesses
    }
    knowledges = await Knowledges.get_knowledge_bases()
    return {
        'encoder_labels': await _encoder_labels(encoder_model_id),
        'knowledge_ids': [kb.id for kb in knowledges],
        'harness_tools': harness_tools,
    }


async def _apply_manifest(manifest: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    context = await _validate_context(manifest)
    validation = validate_bot_manifest(manifest, context)
    bot = manifest.get('bot') or {}
    bot_model_id = bot.get('model_id') or ''
    if not bot_model_id:
        validation['valid'] = False
        validation['errors'].append(
            {
                'code': 'MISSING_BOT_MODEL_ID',
                'message': 'bot.model_id обязателен',
                'agent_remediation_hint': 'Заполните bot.model_id — идентификатор Workspace-модели бота.',
            }
        )
    if not validation['valid'] or dry_run:
        return {'applied': False, 'dry_run': dry_run, 'validation': validation}

    # Workspace-модель: создать или обновить
    existing = await Models.get_model_by_id(bot_model_id)
    model_meta = {
        'description': bot.get('description') or '',
        'systemPrompt': bot.get('system_prompt') or '',
        'channels': bot.get('channels') or ['chat', 'webrtc', 'sip'],
        'knowledge': [{'id': kb_id} for kb_id in (manifest.get('knowledge_ids') or [])],
        'enterprise_manifest_version': manifest.get('version') or '1.0',
    }
    form_data = {
        'id': bot_model_id,
        'name': bot.get('name') or bot_model_id,
        'meta': model_meta,
        'access_control': None,
    }
    if existing:
        existing.meta = model_meta
        existing.name = form_data['name']
        await Models.update_model_by_id(bot_model_id, existing)
    else:
        from open_webui.models.models import ModelForm

        await Models.insert_new_model(ModelForm(**form_data))

    # Матрица интентов: атомарная замена набора правил бота
    old_routes = await IntentRoutes.get_routes()
    kept_ids = set()
    for route in manifest.get('intent_routes') or []:
        route_dict = dict(route)
        route_dict.setdefault('bot_model_id', bot_model_id)
        route_dict.setdefault('allowed_channels', bot.get('channels') or ['chat', 'webrtc', 'sip'])
        existing_route = next(
            (r for r in old_routes if r.bot_model_id == bot_model_id and r.intent_name == route_dict['intent_name']),
            None,
        )
        if existing_route:
            updated = await IntentRoutes.update_route_by_id(existing_route.id, route_dict)
            kept_ids.add(updated.id)
        else:
            created = await IntentRoutes.insert_new_route(route_dict)
            kept_ids.add(created.id)
    for route in old_routes:
        if route.bot_model_id == bot_model_id and route.id not in kept_ids:
            await IntentRoutes.delete_route_by_id(route.id)

    return {'applied': True, 'validation': validation, 'routes_applied': len(kept_ids)}


async def _export_manifest(bot_model_id: str) -> dict[str, Any]:
    model = await Models.get_model_by_id(bot_model_id)
    routes = await IntentRoutes.get_active_routes_by_model(bot_model_id)
    manifest = _empty_manifest()
    meta = (model.meta or {}) if model else {}
    manifest['bot'] = {
        'model_id': bot_model_id,
        'name': (model.name if model else bot_model_id),
        'base_model_id': (model.base_model_id if model else None),
        'system_prompt': meta.get('systemPrompt') or '',
        'channels': meta.get('channels') or ['chat', 'webrtc', 'sip'],
    }
    manifest['knowledge_ids'] = [k.get('id') for k in (meta.get('knowledge') or []) if isinstance(k, dict) and k.get('id')]
    manifest['intent_routes'] = [route.model_dump() for route in routes]
    return manifest


async def _upsert_single_route(bot_model_id: str, route: dict[str, Any]) -> dict[str, Any]:
    route_dict = dict(route)
    route_dict.setdefault('bot_model_id', bot_model_id)
    existing = next(
        (
            r
            for r in await IntentRoutes.get_routes()
            if r.bot_model_id == bot_model_id and r.intent_name == route_dict.get('intent_name')
        ),
        None,
    )
    if existing:
        updated = await IntentRoutes.update_route_by_id(existing.id, route_dict)
        return {'action': 'updated', 'route': updated.model_dump() if updated else None}
    created = await IntentRoutes.insert_new_route(route_dict)
    return {'action': 'created', 'route': created.model_dump()}


async def _simulate_turn(args: dict[str, Any]) -> dict[str, Any]:
    """Детерминированный DecisionTrace: классификатор -> слоты -> маршрут -> ответ/план."""
    from open_webui.utils.intent_router import _classify, _extract_slots, _get_classifier_settings
    from open_webui.utils.channel_adapter import sanitize_for_voice_tts
    from open_webui.utils.route_templates import render_route_template, slots_from_spans

    bot_model_id = args.get('bot_model_id') or ''
    text = args.get('text') or ''
    channel = args.get('channel') or 'chat'
    settings = await _get_classifier_settings()
    trace: dict[str, Any] = {'channel': channel, 'text': text, 'steps': []}

    routes = await IntentRoutes.get_active_routes_by_model(bot_model_id)
    verdict = await _classify(text, settings) or {}
    candidates = verdict.get('top_k_candidates') or []
    trace['classification'] = {
        'intent': verdict.get('intent_name'),
        'confidence': verdict.get('confidence'),
        'action': verdict.get('action'),
        'top_k': candidates[:5],
    }

    slots_model = settings.get('slot_model_id')
    slots = await _extract_slots(text, settings, slots_model) if slots_model else {}
    trace['slots'] = slots

    intent_name = verdict.get('intent_name')
    rule = next((r for r in routes if r.intent_name == intent_name and verdict.get('action') == 'direct'), None)
    if rule is None:
        trace['decision'] = {'route': 'fallback_general_rag', 'reason': verdict.get('action') or 'no classifier'}
        return trace

    trace['routed_intent'] = {'intent': intent_name, 'confidence': verdict.get('confidence'), 'rule_id': rule.id}

    missing = []
    required = rule.required_slots or []
    if required:
        missing = [slot for slot in required if not slots.get(slot.get('name'))]
    if missing:
        question = missing[0].get('question') or f"Уточните: {missing[0].get('name')}?"
        trace['decision'] = {
            'route': 'slot_filling',
            'action_type': rule.action_type,
            'next_slot': missing[0].get('name'),
            'question': question,
        }
        return trace

    extra = {'user_message': text, 'intent': intent_name, 'channel.type': channel}
    variables = {**extra, **{f'slots.{k}': v for k, v in slots.items()}}

    if rule.action_type == 'DIRECT_RESPONSE' and rule.direct_response:
        answer = render_route_template(rule.direct_response, user_message=text, intent=intent_name, slots=slots, extra=variables)
        if channel in ('webrtc', 'sip'):
            answer = sanitize_for_voice_tts(answer, channel)
        trace['decision'] = {'route': 'direct_response', 'answer': answer}
    elif rule.action_type == 'WEBHOOK_TOOL':
        trace['decision'] = {
            'route': 'webhook',
            'url': (rule.webhook_config or {}).get('url'),
            'would_render_body': json.dumps(
                render_arguments_preview(rule.webhook_config or {}, variables), ensure_ascii=False
            ),
        }
    elif rule.action_type == 'MCP_HARNESS':
        from open_webui.utils.mcp_harness import render_arguments

        harness_config = rule.mcp_harness_config or {}
        trace['decision'] = {
            'route': 'mcp_harness',
            'harness_id': harness_config.get('harness_id'),
            'tool_name': harness_config.get('tool_name'),
            'would_send_arguments': render_arguments(harness_config.get('argument_mapping') or {}, variables),
        }
    else:
        trace['decision'] = {
            'route': 'rag_collection',
            'target_knowledge_ids': rule.target_collection_ids or [],
            'use_hipporag': bool(rule.use_hipporag),
            'hybrid_weights': rule.hybrid_weights,
        }
    return trace


def render_arguments_preview(webhook_config: dict[str, Any], variables: dict[str, Any]) -> dict[str, Any]:
    from open_webui.utils.mcp_harness import render_arguments

    body = webhook_config.get('body') or {'message': '{{user_message}}'}
    return render_arguments(body, variables)


async def _encoder_train(args: dict[str, Any]) -> dict[str, Any]:
    base_url = (await Config.get('intent_classifier.url') or '').rstrip('/')
    if not base_url:
        raise ValueError('Encoder service URL is not configured')
    import httpx

    payload = {
        'base_model_id': args.get('base_model_id') or 'rubert-intents-v1',
        'examples': args.get('examples') or [],
        'activate': args.get('activate', True),
        'force_swap': args.get('force_swap', False),
    }
    if args.get('anchor_examples'):
        payload['anchor_examples'] = args['anchor_examples']
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f'{base_url}/v1/train', json=payload)
        response.raise_for_status()
        return response.json()


async def _knowledge_manage(args: dict[str, Any], user) -> dict[str, Any]:
    action = args.get('action')
    if action == 'add_text':
        from open_webui.retrieval.hipporag_engine import schedule_hipporag_indexing

        name = args.get('name') or f'agent-kb-{uuid.uuid4().hex[:6]}'
        knowledge = await Knowledges.insert_new_knowledge(name=name, description='Created via MCP', user_id=user.id)
        if args.get('text'):
            from open_webui.utils.access_control.files import has_access_to_file  # noqa: F401

            file_model = await _create_text_file(user, name, args['text'])
            if file_model:
                await Knowledges.add_file_to_knowledge_by_id(
                    knowledge_id=knowledge.id, file_id=file_model.id, user_id=user.id
                )
                from open_webui.routers.retrieval import ProcessFileForm, process_file
                from fastapi import Request as FastapiRequest

                # Индексация файла в векторное хранилище выполняется отдельным вызовом
                schedule_hipporag_indexing(None, knowledge.id)
        return {'knowledge_id': knowledge.id, 'name': name, 'status': 'text added, indexing scheduled'}
    if action == 'reindex':
        from open_webui.retrieval.hipporag_engine import schedule_hipporag_indexing

        knowledge_id = args.get('knowledge_id')
        if not knowledge_id:
            raise ValueError('knowledge_id is required')
        schedule_hipporag_indexing(None, knowledge_id)
        return {'knowledge_id': knowledge_id, 'status': 'reindex scheduled'}
    if action == 'query':
        from open_webui.retrieval.hipporag_engine import get_hipporag_adapter

        knowledge_id = args.get('knowledge_id')
        query = args.get('query')
        if not knowledge_id or not query:
            raise ValueError('knowledge_id and query are required')
        adapter = get_hipporag_adapter()
        docs = await adapter.retrieve_ppr(None, knowledge_id, [query], top_k=5)
        path = await adapter.get_reasoning_path(None, knowledge_id, query)
        return {'query': query, 'results': docs, 'reasoning_path': path}
    raise ValueError(f'Unknown action: {action}')


async def _create_text_file(user, name: str, text: str):
    try:
        from open_webui.models.files import FileModel, Files

        file = Files.insert_new_file(
            user.id,
            FileModel(
                **{
                    'id': str(uuid.uuid4()),
                    'user_id': user.id,
                    'filename': f'{name}.txt',
                    'meta': {'name': f'{name}.txt', 'content_type': 'text/plain'},
                    'data': {'content': text},
                }
            ),
        )
        return file
    except Exception as exc:  # noqa: BLE001
        log.warning('Text file creation failed: %s', exc)
        return None


async def _dispatch_tool(name: str, args: dict[str, Any], user) -> dict[str, Any]:
    if name == 'platform_inspect_status':
        return await _tool_platform_insight(args)
    if name == 'platform_configure_modules':
        return await _tool_configure_modules(args)
    if name == 'bot_apply_behavior_manifest':
        return await _apply_manifest(args.get('manifest') or {}, bool(args.get('dry_run', False)))
    if name == 'bot_export_behavior_manifest':
        return await _export_manifest(args.get('bot_model_id') or '')
    if name == 'intent_upsert_single_route':
        return await _upsert_single_route(args.get('bot_model_id') or '', args.get('route') or {})
    if name == 'bot_simulate_conversation_turn':
        return await _simulate_turn(args)
    if name == 'encoder_bootstrap_or_train':
        return await _encoder_train(args)
    if name == 'knowledge_hipporag_manage':
        return await _knowledge_manage(args, user)
    raise ValueError(f'Unknown tool: {name}')


async def _read_resource(uri: str, user) -> dict[str, Any]:
    if uri == 'mcp://platform/ai-ready-schema':
        return {
            'parameters': AI_PARAMETER_DESCRIPTORS,
            'manifest_template': _empty_manifest(),
        }
    if uri == 'mcp://platform/config':
        keys = [descriptor['key'] for descriptor in AI_PARAMETER_DESCRIPTORS]
        return {'config': await _config_map(keys)}
    if uri.startswith('mcp://bots/') and uri.endswith('/manifest'):
        bot_model_id = uri.split('/')[3]
        return await _export_manifest(bot_model_id)
    if uri.startswith('mcp://bots/') and uri.endswith('/telemetry'):
        bot_model_id = uri.split('/')[3]
        runs = await RagasRuns.get_runs(limit=5)
        alerts = await RagasSamples.get_flagged_samples(limit=20)
        feedback = await IntentFeedbacks.get_feedback(status='pending', limit=50)
        return {
            'bot_model_id': bot_model_id,
            'recent_ragas_runs': [run.model_dump() for run in runs],
            'hallucination_alerts': [alert.model_dump() for alert in alerts],
            'active_learning_queue': [item.model_dump() for item in feedback],
        }
    raise ValueError(f'Unknown resource: {uri}')


@router.post('/mcp')
async def mcp_endpoint(
    request: Request,
    body: dict[str, Any],
    user=Depends(get_verified_user),
):
    """Streamable HTTP: один JSON-RPC запрос за вызов (пакеты поддерживаются списком)."""
    enabled = await Config.get('mcp.server.enabled')
    if enabled is False:
        return _rpc_error(body.get('id') if isinstance(body, dict) else None, -32601, 'MCP server is disabled')

    def is_rpc(item: Any) -> bool:
        return isinstance(item, dict) and 'method' in item

    if isinstance(body, list):
        responses = [await _handle_single(item, user) for item in body if is_rpc(item)]
        return responses
    return await _handle_single(body, user)


async def _handle_single(body: dict[str, Any], user) -> dict[str, Any]:
    rpc_id = body.get('id')
    method = body.get('method') or ''
    params = body.get('params') or {}

    try:
        if method == 'initialize':
            return _rpc_result(
                rpc_id,
                {
                    'protocolVersion': '2025-03-26',
                    'capabilities': {'tools': {'listChanged': False}, 'resources': {'subscribe': False}},
                    'serverInfo': {'name': 'enterprise-ai-platform', 'version': '1.0.0'},
                },
            )
        if method == 'ping':
            return _rpc_result(rpc_id, {})
        if method == 'tools/list':
            return _rpc_result(rpc_id, {'tools': TOOLS_SPEC})
        if method == 'tools/call':
            name = params.get('name')
            arguments = params.get('arguments') or {}
            result = await _dispatch_tool(name, arguments, user)
            return _rpc_result(
                rpc_id,
                {
                    'content': [{'type': 'text', 'text': json.dumps(result, ensure_ascii=False, default=str)}],
                    'structuredContent': result,
                    'isError': False,
                },
            )
        if method == 'resources/list':
            return _rpc_result(rpc_id, {'resources': RESOURCES_SPEC})
        if method == 'resources/read':
            result = await _read_resource(params.get('uri') or '', user)
            return _rpc_result(
                rpc_id,
                {
                    'contents': [
                        {'uri': params.get('uri'), 'mimeType': 'application/json', 'text': json.dumps(result, ensure_ascii=False, default=str)}
                    ]
                },
            )
        return _rpc_error(rpc_id, -32601, f'Method not found: {method}')
    except ValueError as exc:
        return _rpc_error(rpc_id, -32602, str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception('MCP tool execution failed')
        return _rpc_error(rpc_id, -32603, f'Internal error: {exc}')


@router.get('/mcp/sse')
async def mcp_sse(user=Depends(get_verified_user)):
    """SSE-канал: объявление POST-эндпоинта + keepalive (минимальный transport)."""

    async def event_stream():
        yield f'event: endpoint\ndata: /api/v1/mcp\n\n'
        while True:
            yield f'event: ping\ndata: {int(time.time() * 1000)}\n\n'
            await asyncio.sleep(15)

    return StreamingResponse(event_stream(), media_type='text/event-stream')
