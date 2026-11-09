"""Маршрутизация интентов через encoder-classifier.

Покрывает: классификацию, извлечение слотов (Д1), активную дизамбигуацию (Д2),
голосовые переменные и деэскалацию (Ф3), декларативный slot-filling (Ф2),
guardrails (Ф5), HUMAN_HANDOFF (Ф7) и отправку сэмплов в ragas-worker (ТЗ-6).

Хуки вызываются из utils.middleware:
  apply_intent_routing     — в process_chat_payload до RAG и инструментов;
  build_direct_response    — в main.process_chat для мгновенных ответов без LLM;
  dispatch_ragas_sampling  — в background_tasks_handler после ответа модели.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Any, Optional

import httpx

from open_webui.models.config import Config
from open_webui.models.intents import IntentRoutes
from open_webui.utils.channel_adapter import (
    detect_channel,
    filter_routes_by_channel,
    sanitize_for_voice_tts,
)
from open_webui.utils.misc import add_or_update_system_message, get_last_user_message
from open_webui.utils.route_templates import (
    build_disambiguation_prompt,
    is_slot_cancellation,
    mentions_slot_value,
    render_route_template,
    should_disambiguate,
    slot_retry_exceeded,
    slots_from_spans,
)

log = logging.getLogger(__name__)

_SLOT_SESSION_TTL = 1800
_ANGER_ESCALATION_TURNS = 2


async def _get_classifier_settings() -> dict[str, Any]:
    values = await Config.get_many(
        'intent_classifier.enable',
        'intent_classifier.url',
        'intent_classifier.model_id',
        'intent_classifier.timeout',
        'intent_classifier.tau_high',
        'intent_classifier.tau_low',
        'intent_classifier.slot_model_id',
        'intent_classifier.disambiguation_enable',
        'intent_classifier.disambiguation_delta',
        'guardrails.prompt_guard',
        'guardrails.injection_threshold',
        'guardrails.refusal_response',
        'guardrails.pii_masking',
        'voice.anger_threshold',
        'voice.deescalation_prompt',
    )
    return {
        'enable': bool(values.get('intent_classifier.enable')),
        'url': (values.get('intent_classifier.url') or '').rstrip('/'),
        'model_id': values.get('intent_classifier.model_id') or 'rubert-intents-v1',
        'timeout': float(values.get('intent_classifier.timeout') or 0.4),
        'tau_high': float(values.get('intent_classifier.tau_high') or 0.85),
        'tau_low': float(values.get('intent_classifier.tau_low') or 0.50),
        'slot_model_id': values.get('intent_classifier.slot_model_id') or '',
        'disambiguation_enable': bool(values.get('intent_classifier.disambiguation_enable', True)),
        'disambiguation_delta': float(values.get('intent_classifier.disambiguation_delta') or 0.15),
        'prompt_guard': bool(values.get('guardrails.prompt_guard')),
        'injection_threshold': float(values.get('guardrails.injection_threshold') or 0.75),
        'refusal_response': values.get('guardrails.refusal_response')
        or 'Я не могу выполнить эту инструкцию. Опишите ваш запрос по существу.',
        'pii_masking': bool(values.get('guardrails.pii_masking')),
        'pii_mask_llm_input': bool(values.get('guardrails.pii_mask_llm_input', True)),
        'anger_threshold': float(values.get('voice.anger_threshold') or 0.75),
        'deescalation_prompt': values.get('voice.deescalation_prompt') or '',
    }


# --- Состояние slot-filling и эмоций ---
# Д3 (Stage 3): состояние формы персистится в Redis (TTL 1800 c), чтобы
# переживать рестарты и работать с несколькими воркерами uvicorn.
# Fallback при недоступном Redis — память процесса.

_slot_sessions: dict[str, dict[str, Any]] = {}
_anger_streak: dict[str, int] = {}
_redis_client: Any = None


def _session_key(user, metadata: dict, model_id: str) -> str:
    chat_id = metadata.get('chat_id') or (getattr(user, 'id', None) or 'anon')
    return f'{model_id}:{chat_id}'


def _replace_last_user_message(form_data: dict, new_content: str) -> None:
    """Подменяет текст последней реплики пользователя (маскирование ПДн)."""
    for message in reversed(form_data.get('messages', [])):
        if message.get('role') == 'user':
            message['content'] = new_content
            return


async def _get_redis():
    global _redis_client
    if _redis_client is None:
        from open_webui.env import REDIS_URL

        if not REDIS_URL:
            _redis_client = False
        else:
            import redis.asyncio as aioredis

            _redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client or None


def _slot_state_key(key: str) -> str:
    return f'slot_filling_state:{key}'


async def _slot_state_load(key: str) -> Optional[dict[str, Any]]:
    redis = await _get_redis()
    if redis:
        try:
            raw = await redis.get(_slot_state_key(key))
            if raw:
                return json.loads(raw)
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning('Slot state Redis load failed: %s', exc)
    session = _slot_sessions.get(key)
    if session and time.monotonic() - session.get('updated', 0) > _SLOT_SESSION_TTL:
        session = None
    return session


async def _slot_state_save(key: str, state: dict[str, Any]) -> None:
    state['updated'] = time.monotonic()
    redis = await _get_redis()
    if redis:
        try:
            await redis.set(_slot_state_key(key), json.dumps(state, ensure_ascii=False), ex=_SLOT_SESSION_TTL)
            return
        except Exception as exc:  # noqa: BLE001
            log.warning('Slot state Redis save failed: %s', exc)
    _slot_sessions[key] = state


async def _slot_state_clear(key: str) -> None:
    redis = await _get_redis()
    _slot_sessions.pop(key, None)
    if redis:
        try:
            await redis.delete(_slot_state_key(key))
        except Exception as exc:  # noqa: BLE001
            log.warning('Slot state Redis clear failed: %s', exc)


# --- HTTP-вызовы классификатора ---


async def _classify(text: str, settings: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Вызов encoder-classifier. None — сервис недоступен или выключен."""
    if not settings['enable'] or not settings['url']:
        return None
    try:
        async with httpx.AsyncClient(timeout=settings['timeout']) as client:
            response = await client.post(
                f"{settings['url']}/v1/classify",
                json={
                    'text': text,
                    'model_id': settings['model_id'],
                    'tau_high': settings['tau_high'],
                    'tau_low': settings['tau_low'],
                },
            )
            response.raise_for_status()
            return response.json()
    except Exception as exc:  # noqa: BLE001 — классификатор не должен ломать чат
        log.warning('Encoder classifier unavailable: %s', exc)
        return None


async def _extract_slots(text: str, settings: dict[str, Any], slot_model_id: str) -> dict[str, str]:
    """NER-слоты через /v1/extract-slots. Пустой dict — модель не настроена или ошибка."""
    if not slot_model_id or not settings['url']:
        return {}
    try:
        async with httpx.AsyncClient(timeout=settings['timeout'] + 0.6) as client:
            response = await client.post(
                f"{settings['url']}/v1/extract-slots",
                json={'text': text, 'model_id': slot_model_id},
            )
            response.raise_for_status()
            return slots_from_spans(response.json().get('spans') or [])
    except Exception as exc:  # noqa: BLE001
        log.warning('Slot extraction failed: %s', exc)
        return {}


async def check_prompt_guard(text: str, settings: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Проверка джейлбрейка через /v1/guard. None — guard выключен или недоступен."""
    if not settings['prompt_guard'] or not settings['url']:
        return None
    try:
        async with httpx.AsyncClient(timeout=settings['timeout'] + 0.6) as client:
            response = await client.post(
                f"{settings['url']}/v1/guard",
                json={'text': text, 'threshold': settings['injection_threshold']},
            )
            response.raise_for_status()
            return response.json()
    except Exception as exc:  # noqa: BLE001
        log.warning('Prompt guard unavailable: %s', exc)
        return None


async def _set_call_transfer(call_uuid: str, queue: str) -> None:
    """Д5: отметка эскалации для моста/диалплана (Redis с TTL 300 c)."""
    redis = await _get_redis()
    if redis:
        try:
            await redis.set(f'call_transfer:{call_uuid}', queue or 'NONE', ex=300)
            return
        except Exception as exc:  # noqa: BLE001
            log.warning('Call transfer flag not stored: %s', exc)
    from open_webui.utils.call_status import mark_transfer

    mark_transfer(call_uuid, queue)


async def _call_webhook(webhook_config: dict[str, Any], variables: dict[str, Any]) -> Optional[str]:
    """Вызов внешнего REST-бэкенда. Возвращает текст ответа или None."""
    url = webhook_config.get('url')
    if not url:
        return None
    body = webhook_config.get('body') or {'message': '{{user_message}}'}
    rendered_body = json.loads(render_route_template(json.dumps(body, ensure_ascii=False), extra=variables))
    headers = webhook_config.get('headers') or {}
    timeout = float(webhook_config.get('timeout_s', 5))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=rendered_body, headers=headers)
            response.raise_for_status()
            if webhook_config.get('response_format') == 'text':
                return response.text
            data = response.json()
        path = webhook_config.get('response_path', 'answer')
        value: Any = data
        for part in str(path).split('.'):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break
        return str(value) if value is not None else None
    except Exception as exc:  # noqa: BLE001
        log.warning('Intent webhook failed: %s', exc)
        return None


# --- Слоты: обязательная схема и декларативный сбор (Ф2) ---


def _missing_required_slots(rule, collected: dict[str, str]) -> list[dict[str, Any]]:
    """Обязательные слоты, отсутствующие или не прошедшие валидацию."""
    missing: list[dict[str, Any]] = []
    for slot in rule.required_slots or []:
        name = slot.get('name')
        value = collected.get(name)
        if not value:
            missing.append(slot)
            continue
        pattern = slot.get('pattern')
        if pattern and not re.fullmatch(pattern, str(value)):
            missing.append(slot)
    return missing


def _merge_rule_slots(session: dict[str, Any], rule, new_slots: dict[str, str]) -> dict[str, str]:
    """Сливает накопленные и новые слоты в рамках правила."""
    if session.get('rule_id') != rule.id:
        session['slots'] = {}
        session['rule_id'] = rule.id
    session['slots'].update(new_slots)
    session['updated'] = time.monotonic()
    return session['slots']


def _apply_collection_targets(files: list[dict[str, Any]], rule) -> list[dict[str, Any]]:
    """Фильтрует RAG-коллекции по правилу интента и ставит флаг use_hipporag."""
    target_ids = rule.target_collection_ids or []
    kept: list[dict[str, Any]] = []
    for item in files:
        if item.get('type') == 'collection' or (item.get('legacy') and item.get('id')):
            continue  # коллекции заменяются целевым набором
        kept.append(item)
    for collection_id in target_ids:
        kept.append(
            {
                'type': 'collection',
                'id': collection_id,
                'name': collection_id,
                'use_hipporag': bool(rule.use_hipporag),
                'hybrid_weights': rule.hybrid_weights,
            }
        )
    return kept if target_ids else files


def _voice_variables(metadata: dict) -> dict[str, Any]:
    """Переменные голосового канала для шаблонов: {{voice.anger_score}} и т.п."""
    anger = metadata.get('voice_anger')
    return {
        'voice.anger_score': '' if anger is None else f'{float(anger):.2f}',
        'voice_mode': 'true' if metadata.get('voice_mode') else '',
    }


async def apply_intent_routing(request, form_data: dict, metadata: dict, user, model: dict) -> None:
    """Хук маршрутизации в process_chat_payload (после inlet-фильтров, до RAG).

    Мутирует form_data и metadata:
      - metadata['routed_intent'], metadata['intent_candidates'], metadata['disambiguation'];
      - metadata['direct_response'] = {'content', 'intent'} — ответ без LLM;
      - metadata['human_handoff'] — карточка эскалации на оператора;
      - фильтрация коллекций RAG, подмена системного промпта, slot-filling сессии.
    """
    settings = await _get_classifier_settings()
    if not settings['enable']:
        return

    routes = await IntentRoutes.get_active_routes_by_model(model.get('id', ''))
    if not routes:
        return

    # Stage 4: определение канала и матрица отключения контуров
    channel = detect_channel(request, form_data)
    metadata['channel'] = channel
    channel_enabled = await Config.get(f'channels.{channel}.enabled')
    if channel_enabled is False:
        return
    routes = filter_routes_by_channel(routes, channel)
    if not routes:
        return

    user_message = get_last_user_message(form_data.get('messages', [])) or ''
    if not user_message.strip():
        return

    session_key = _session_key(user, metadata, model.get('id', ''))

    # Д3 (Stage 3): активная форма сбора слотов — глобальный стоп-фильтр отмены.
    active_state = await _slot_state_load(session_key)
    if active_state and is_slot_cancellation(user_message):
        await _slot_state_clear(session_key)
        active_state = None
        metadata['slot_filling_cancelled'] = True

    # Ф5: guardrails — отсев джейлбрейков и промпт-инъекций
    guard = await check_prompt_guard(user_message, settings)
    if guard and guard.get('injection'):
        metadata['prompt_guard'] = guard
        metadata['direct_response'] = {'content': settings['refusal_response'], 'intent': '__guard__'}
        return

    verdict = await _classify(user_message, settings)
    if not verdict:
        return

    intent_name = verdict.get('intent_name')
    confidence = float(verdict.get('confidence') or 0.0)
    action = verdict.get('action')
    candidates = verdict.get('top_k_candidates') or []

    # Ф5/Д2 (Stage 3): обратимое маскирование входа. Вольт сессии нужен
    # дальше для размаскирования ответа (в т.ч. потоком, StreamingPIIUnmasker).
    if settings['pii_masking'] and settings.get('pii_mask_llm_input'):
        from open_webui.utils.pii_vault import get_vault

        vault = get_vault(session_key)
        masked_message = vault.mask(user_message)
        if masked_message != user_message:
            _replace_last_user_message(form_data, masked_message)
            metadata['pii_vault_session'] = session_key
            user_message = masked_message

    if action != 'direct':
        # Д2: серая зона — активная дизамбигуация, если лидеры различимы слабо
        metadata['intent_candidates'] = candidates
        if settings['disambiguation_enable']:
            prompt = build_disambiguation_prompt(candidates, settings['disambiguation_delta'])
            if prompt:
                form_data['messages'] = add_or_update_system_message(
                    prompt, form_data['messages'], append=True
                )
                _, top_1, top_2, delta = should_disambiguate(candidates, settings['disambiguation_delta'])
                metadata['disambiguation'] = {'top_1': top_1, 'top_2': top_2, 'delta': round(delta, 3)}
        # Ф1: Active Learning — фраза серой зоны попадает в очередь разметки
        asyncio.create_task(
            _enqueue_active_learning(model.get('id', ''), user_message, candidates)
        )
        return

    rule = next((r for r in routes if r.intent_name == intent_name), None)
    if rule is None:
        return

    metadata['routed_intent'] = {
        'rule_id': rule.id,
        'intent': intent_name,
        'confidence': confidence,
        'action_type': rule.action_type,
    }

    # Деэскалация голосового канала (Ф3): два хода гнева подряд -> промпт
    anger = metadata.get('voice_anger')
    if anger is not None and settings['deescalation_prompt']:
        streak = _anger_streak.get(session_key, 0)
        if float(anger) >= settings['anger_threshold']:
            streak += 1
        else:
            streak = 0
        _anger_streak[session_key] = streak
        if streak >= _ANGER_ESCALATION_TURNS:
            form_data['messages'] = add_or_update_system_message(
                settings['deescalation_prompt'], form_data['messages'], append=True
            )
            metadata['voice_deescalated'] = True
            _anger_streak[session_key] = 0

    from open_webui.utils.channel_adapter import channel_variables

    extra = {
        'user_message': user_message,
        'intent': intent_name,
        **_voice_variables(metadata),
        **channel_variables(channel),
    }

    # Ф7: эскалация на оператора
    if rule.action_type == 'HUMAN_HANDOFF':
        collected = {}
        if settings['slot_model_id']:
            collected = await _extract_slots(user_message, settings, settings['slot_model_id'])
        queue = (rule.webhook_config or {}).get('queue') or await Config.get('voice.handoff_queue')
        call_uuid = request.headers.get('x-call-uuid')
        metadata['human_handoff'] = {
            'intent': intent_name,
            'confidence': confidence,
            'slots': collected,
            'summary': user_message[:500],
            'queue': queue,
            'call_uuid': call_uuid,
        }
        if call_uuid:
            # Д5 (Stage 3): сигнал для Asterisk — диалплан проверяет статус
            # через CURL(http://bridge:8765/v1/call-status/{uuid}) после AudioSocket
            asyncio.create_task(_set_call_transfer(call_uuid, queue))
        if rule.webhook_config:
            asyncio.create_task(
                _call_webhook(
                    rule.webhook_config,
                    {**extra, 'slots': json.dumps(collected, ensure_ascii=False)},
                )
            )
        metadata['direct_response'] = {
            'content': render_route_template(
                rule.direct_response or 'Соединяю вас с оператором, секунду.',
                user_message=user_message,
                intent=intent_name,
                slots=collected,
                extra=extra,
            ),
            'intent': intent_name,
        }
        return

    # Ф2 + Д3 (Stage 3): декларативный slot-filling с защитой от зацикливания.
    # Активная форма переживает смену темы только до уверенного нового интента;
    # три неудачные валидации подряд переводят диалог в HUMAN_HANDOFF.
    needs_slots = rule.action_type in ('WEBHOOK_TOOL', 'DIRECT_RESPONSE', 'MCP_HARNESS') and bool(
        rule.required_slots
    )
    collected: dict[str, str] = {}
    if active_state and not needs_slots:
        # Пользователь в середине формы, но новая реплика классифицировалась
        # в интент без схемы слотов: escape — уверенный новый интент
        # без признаков ожидаемых значений прерывает старую форму.
        active_rule = next((r for r in routes if r.id == active_state.get('rule_id')), None)
        if active_rule is not None:
            patterns = [s.get('pattern') for s in (active_rule.required_slots or []) if s.get('pattern')]
            escape = (
                confidence >= max(settings['tau_high'], 0.90)
                and not mentions_slot_value(user_message, patterns)
            )
            if escape:
                await _slot_state_clear(session_key)
                active_state = None
            else:
                # Продолжаем сбор активной формы независимо от нового интента
                rule = active_rule
                intent_name = rule.intent_name
                needs_slots = rule.action_type in ('WEBHOOK_TOOL', 'DIRECT_RESPONSE') and bool(
                    rule.required_slots
                )

    if needs_slots:
        session = active_state or {'rule_id': rule.id, 'slots': {}, 'retries': 0}
        new_slots = await _extract_slots(user_message, settings, settings['slot_model_id'])
        collected = _merge_rule_slots(session, rule, new_slots)
        missing = _missing_required_slots(rule, collected)
        if missing:
            asked_slot = missing[0]
            # Неудачная попытка: реплика не закрыла текущий вопрос формы
            if new_slots.get(asked_slot.get('name')) is None:
                session['retries'] = int(session.get('retries', 0)) + 1
            else:
                session['retries'] = 0

            if slot_retry_exceeded(int(session.get('retries', 0))):
                # Лимит ошибок: сброс формы и эскалация с уже собранными полями
                await _slot_state_clear(session_key)
                metadata['slot_filling'] = {
                    'rule_id': rule.id,
                    'status': 'max_retries_handoff',
                    'collected': dict(collected),
                }
                metadata['human_handoff'] = {
                    'intent': intent_name,
                    'confidence': confidence,
                    'slots': dict(collected),
                    'summary': f'Не заполнены слоты {[s.get("name") for s in missing]} после 3 попыток',
                }
                metadata['direct_response'] = {
                    'content': render_route_template(
                        'Передаю вас специалисту. Уже собранные данные: {{slots}}',
                        user_message=user_message,
                        intent=intent_name,
                        slots=collected,
                        extra={'slots': ', '.join(f'{k}={v}' for k, v in collected.items()) or 'нет'},
                    ),
                    'intent': intent_name,
                }
                return

            await _slot_state_save(session_key, session)
            # Детерминированный уточняющий вопрос вместо LLM
            question = asked_slot.get('question') or f"Уточните: {asked_slot.get('name')}?"
            metadata['slot_filling'] = {
                'rule_id': rule.id,
                'missing': [slot.get('name') for slot in missing],
                'collected': dict(collected),
                'retries': int(session.get('retries', 0)),
            }
            metadata['direct_response'] = {'content': question, 'intent': intent_name}
            return
        # Все слоты собраны — форма закрыта
        await _slot_state_clear(session_key)

    if rule.action_type == 'DIRECT_RESPONSE' and rule.direct_response:
        metadata['direct_response'] = {
            'content': render_route_template(
                rule.direct_response, user_message=user_message, intent=intent_name,
                slots=collected, extra=extra,
            ),
            'intent': intent_name,
        }
        return

    if rule.action_type == 'WEBHOOK_TOOL' and rule.webhook_config:
        webhook_slots = collected or (
            await _extract_slots(user_message, settings, settings['slot_model_id'])
            if settings['slot_model_id']
            else {}
        )
        answer = await _call_webhook(
            rule.webhook_config,
            {**extra, 'slots': json.dumps(webhook_slots, ensure_ascii=False)},
        )
        if answer:
            metadata['direct_response'] = {'content': answer, 'intent': intent_name}
        return

    if rule.action_type == 'MCP_HARNESS' and rule.mcp_harness_config:
        # Stage 4, направление Б: вызов внешнего агентного харнеса по MCP.
        from open_webui.models.intents import MCPHarnesses
        from open_webui.utils.mcp_harness import call_tool, content_to_text, render_arguments

        config_map = rule.mcp_harness_config or {}
        harness = await MCPHarnesses.get_harness_by_id(config_map.get('harness_id') or '')
        if harness is None:
            metadata['mcp_harness_error'] = f"Unknown harness: {config_map.get('harness_id')}"
            return
        variables = {
            **extra,
            **{f'slots.{k}': v for k, v in (collected or {}).items()},
        }
        arguments = render_arguments(config_map.get('argument_mapping') or {}, variables)
        result = await call_tool(harness, config_map.get('tool_name') or '', arguments)
        metadata['mcp_harness_result'] = {'ok': result.get('ok'), 'tool': config_map.get('tool_name')}
        if result.get('ok'):
            answer_text = content_to_text(result.get('content')) or 'Готово.'
            metadata['direct_response'] = {
                'content': answer_text,
                'intent': intent_name,
            }
        elif config_map.get('fallback_to_llm'):
            # Не отдаём ошибку пользователю: уходим в обычный LLM-сценарий
            metadata.pop('direct_response', None)
        return

    # RAG_COLLECTION и SYSTEM_PROMPT_OVERRIDE: настраиваем поиск и промпт.
    if rule.system_prompt_template:
        form_data['messages'] = add_or_update_system_message(
            render_route_template(
                rule.system_prompt_template, user_message=user_message, intent=intent_name,
                slots=collected, extra=extra,
            ),
            form_data['messages'],
        )

    if rule.target_collection_ids or rule.use_hipporag:
        files = metadata.get('files')
        if isinstance(files, list):
            metadata['files'] = _apply_collection_targets(files, rule)
        files = form_data.get('files')
        if isinstance(files, list) and files:
            form_data['files'] = _apply_collection_targets(files, rule)


async def _enqueue_active_learning(bot_model_id: str, text: str, candidates: list[dict[str, Any]]) -> None:
    """Ф1: серая зона -> intent_feedback с LLM-подсказкой (если настроена)."""
    try:
        if not await Config.get('active_learning.enable'):
            return
        from open_webui.models.intents import IntentFeedbacks

        hint_intent, hint_confidence = await _llm_hint(text)
        await IntentFeedbacks.insert_new_feedback(
            {
                'bot_model_id': bot_model_id,
                'text': text[:2000],
                'predicted_intent': candidates[0].get('intent_name') if candidates else None,
                'confidence': candidates[0].get('confidence') if candidates else None,
                'top_candidates': candidates[:5],
                'hint_intent': hint_intent,
                'hint_confidence': hint_confidence,
                'source': 'gray_zone',
                'status': 'pending',
            }
        )
    except Exception as exc:  # noqa: BLE001
        log.debug('Active learning enqueue failed: %s', exc)


async def _llm_hint(text: str) -> tuple[Optional[str], Optional[float]]:
    """Подсказка от большой LLM: «с вероятностью X это интент Y».

    Модель берется из active_learning.hint_model, LLM-эндпоинт —
    OpenAI-совместимый из настроек RAG. Без модели — (None, None).
    """
    try:
        hint_model = await Config.get('active_learning.hint_model')
        if not hint_model:
            return None, None
        base_url = await Config.get('rag.openai.api_base_url')
        api_key = await Config.get('rag.openai.api_key')
        if not base_url:
            return None, None

        from open_webui.retrieval.query_rewriter import _call_llm

        parsed = await _call_llm(
            base_url=base_url,
            api_key=api_key,
            model=hint_model,
            history=[
                {
                    'role': 'system',
                    'content': (
                        'You are labeling user utterances for an intent classifier. '
                        'Reply with strict JSON only: {"intent": "<most likely intent label>", "confidence": 0.0-1.0}'
                    ),
                },
                {'role': 'user', 'content': text},
            ],
            timeout=6,
        )
        if not parsed:
            return None, None
        intent = str(parsed.get('intent') or '').strip() or None
        confidence = parsed.get('confidence')
        return intent, float(confidence) if confidence is not None else None
    except Exception as exc:  # noqa: BLE001
        log.debug('Active learning hint failed: %s', exc)
        return None, None


def build_direct_response(
    form_data: dict, direct_response: dict[str, Any], metadata: dict | None = None
):
    """Собирает ответ без вызова LLM в формате, который понимает process_chat_response.

    Streaming — SSE-чанки OpenAI-формата; иначе — dict в формате chat.completion.
    """
    from fastapi.responses import StreamingResponse

    from open_webui.utils.json_codec import JSONCodec

    content = direct_response.get('content', '')
    model_id = form_data.get('model', 'intent-router')
    created = 0

    # Stage 4: санитайзер голосовых каналов — TTS не читает Markdown
    channel = (metadata or {}).get('channel')
    if channel in ('webrtc', 'sip'):
        content = sanitize_for_voice_tts(content, channel)

    # Д2 (Stage 3): вольт сессии — токены ПДн размаскировываются в ответе
    vault = None
    session_key = (metadata or {}).get('pii_vault_session')
    if session_key:
        from open_webui.utils.pii_vault import get_vault

        vault = get_vault(session_key)

    if form_data.get('stream'):
        def sse_generator():
            nonlocal created
            import time

            created = int(time.time())
            from open_webui.utils.pii_vault import StreamingPIIUnmasker

            unmasker = StreamingPIIUnmasker(vault.mapping()) if vault is not None else None
            chunks = [content[i : i + 120] for i in range(0, len(content), 120)] or ['']
            for index, chunk_text in enumerate(chunks):
                if unmasker is not None:
                    chunk_text = unmasker.feed(chunk_text)
                yield (
                    'data: '
                    + JSONCodec.dumps(
                        {
                            'id': f'intent-{created}',
                            'object': 'chat.completion.chunk',
                            'created': created,
                            'model': model_id,
                            'choices': [
                                {
                                    'index': 0,
                                    'delta': (
                                        {'role': 'assistant', 'content': chunk_text}
                                        if index == 0
                                        else {'content': chunk_text}
                                    ),
                                    'finish_reason': None,
                                }
                            ],
                        }
                    )
                    + '\n\n'
                )
            yield (
                'data: '
                + JSONCodec.dumps(
                    {
                        'id': f'intent-{created}',
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': model_id,
                        'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                    }
                )
                + '\n\n'
            )
            if unmasker is not None:
                tail = unmasker.flush()
                if tail:
                    yield (
                        'data: '
                        + JSONCodec.dumps(
                            {
                                'id': f'intent-{created}',
                                'object': 'chat.completion.chunk',
                                'created': created,
                                'model': model_id,
                                'choices': [{'index': 0, 'delta': {'content': tail}, 'finish_reason': None}],
                            }
                        )
                        + '\n\n'
                    )
            yield 'data: [DONE]\n\n'

        return StreamingResponse(sse_generator(), media_type='text/event-stream')

    if vault is not None:
        content = vault.unmask(content)

    return {
        'id': f'intent-{created}',
        'object': 'chat.completion',
        'created': created,
        'model': model_id,
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': content},
                'finish_reason': 'stop',
            }
        ],
    }


async def dispatch_ragas_sampling(request, user, metadata: dict, messages: list[dict], assistant_content: str) -> None:
    """Отправка сэмпла живого диалога в ragas-worker (ТЗ, раздел 6.1, режим 2).

    Fire-and-forget: ошибки логируются и не влияют на ответ. ПДн маскируются
    перед отправкой (guardrails.pii_masking).
    """
    try:
        enabled = await Config.get('ragas.enable')
        sample_rate = float(await Config.get('ragas.live_sample_rate') or 0)
        worker_url = (await Config.get('ragas.worker_url') or '').rstrip('/')
        pii_masking = bool(await Config.get('guardrails.pii_masking'))
        if not enabled or not worker_url or sample_rate <= 0:
            return
        if random.random() >= sample_rate:
            return
        if not assistant_content:
            return

        question = get_last_user_message(messages) or ''
        if not question:
            return

        vault = None
        if pii_masking:
            from open_webui.utils.pii_vault import get_vault

            vault = get_vault(f"ragas:{metadata.get('chat_id') or metadata.get('user_id') or 'anon'}")

        sources = metadata.get('sources') or []
        contexts: list[str] = []
        for source in sources:
            documents = source.get('document') if isinstance(source, dict) else None
            if isinstance(documents, list):
                contexts.extend(str(doc) for doc in documents if doc)

        if vault is not None:
            question = vault.mask(question)
            assistant_content = vault.mask(assistant_content)
            contexts = [vault.mask(context) for context in contexts]

        payload = {
            'question': question,
            'answer': assistant_content,
            'contexts': contexts,
            'bot_model_id': metadata.get('model_id'),
            'chat_id': metadata.get('chat_id'),
            'message_id': metadata.get('message_id'),
            'rewritten_query': metadata.get('rewritten_query'),
            'retrieval_trace': metadata.get('retrieval_trace'),
        }

        async def post_sample() -> None:
            try:
                async with httpx.AsyncClient(timeout=4) as client:
                    await client.post(f'{worker_url}/v1/evaluate/live', json=payload)
            except Exception as exc:  # noqa: BLE001
                log.debug('Ragas live sample not sent: %s', exc)

        asyncio.create_task(post_sample())
    except Exception as exc:  # noqa: BLE001
        log.warning('Ragas sampling dispatch failed: %s', exc)
