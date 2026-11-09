"""100% AI-Ready: семантические паспорта настроек, манифест бота, линтер.

Stage 4, раздел 4. Любой ИИ-агент настраивает платформу без чтения исходников:
  - AI_PARAMETER_DESCRIPTORS: паспорт каждой настройки (тип, ограничения,
    физический смысл, влияние изменения, рекомендуемые значения по каналам);
  - BotBehaviorManifest: декларативный сценарий бота целиком;
  - validate_bot_manifest: 6 правил консистентности + agent_remediation_hint.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


class AIParameterDescriptor(dict):
    """Паспорт одной настройки для ИИ-агента."""

    def __init__(
        self,
        key: str,
        type: str,
        constraints: Optional[dict[str, Any]] = None,
        purpose_for_ai: str = '',
        tuning_impact: str = '',
        recommended_by_channel: Optional[dict[str, Any]] = None,
        default: Any = None,
    ):
        super().__init__(
            key=key,
            type=type,
            constraints=constraints or {},
            purpose_for_ai=purpose_for_ai,
            tuning_impact=tuning_impact,
            recommended_by_channel=recommended_by_channel or {},
            default=default,
        )


AI_PARAMETER_DESCRIPTORS: list[AIParameterDescriptor] = [
    AIParameterDescriptor(
        'intent_classifier.tau_high', 'float', {'min': 0.0, 'max': 1.0},
        'Порог прямой маршрутизации интента. Если p >= tau_high, выполняется действие правила без LLM.',
        'Увеличение с 0.80 до 0.92 снижает риск ложного срабатывания вебхука, но переводит больше запросов в серую зону уточнения. Уменьшение ускоряет ответы, но повышает ложные срабатывания.',
        {'chat': 0.85, 'webrtc': 0.80, 'sip': 0.80}, default=0.85,
    ),
    AIParameterDescriptor(
        'intent_classifier.tau_low', 'float', {'min': 0.0, 'max': 1.0},
        'Нижняя граница серой зоны. Ниже tau_low — fallback на общий RAG-сценарий.',
        'Увеличение расширяет fallback-зону: больше запросов идут в общий поиск. Уменьшение чаще запускает уточняющие вопросы.',
        {'chat': 0.50, 'webrtc': 0.45, 'sip': 0.45}, default=0.50,
    ),
    AIParameterDescriptor(
        'intent_classifier.disambiguation_delta', 'float', {'min': 0.0, 'max': 1.0},
        'Максимальная разница топ-1 и топ-2 вероятностей, при которой бот задает уточняющий вопрос.',
        'Увеличение делает уточнения чаще (бот осторожнее). Уменьшение — реже уточняет, чаще угадывает.',
        {'chat': 0.15, 'webrtc': 0.12, 'sip': 0.12}, default=0.15,
    ),
    AIParameterDescriptor(
        'intent_classifier.timeout', 'float', {'min': 0.05, 'max': 5.0},
        'Таймаут вызова ONNX-классификатора в секундах.',
        'Увеличение — устойчивость к медленному сервису, но рост задержки чата при его деградации.',
        {'chat': 0.4, 'webrtc': 0.3, 'sip': 0.3}, default=0.4,
    ),
    AIParameterDescriptor(
        'hipporag.enable', 'bool', {},
        'Третий контур поиска: граф знаний OpenIE + Personalized PageRank для multi-hop вопросов.',
        'Включение повышает Context Recall на многошаговых вопросах на 15-25%, но добавляет задержку поиска.',
        {'chat': True, 'webrtc': True, 'sip': False}, default=True,
    ),
    AIParameterDescriptor(
        'rag.hybrid_weight_dense', 'float', {'min': 0.0, 'max': 1.0},
        'Вес векторного (dense) контура в Weighted RRF.',
        'Увеличение усиливает семантический поиск (перефразировки), ослабляет точные ключевые слова.',
        {'chat': 0.4, 'webrtc': 0.4, 'sip': 0.4}, default=0.4,
    ),
    AIParameterDescriptor(
        'rag.hybrid_weight_bm25', 'float', {'min': 0.0, 'max': 1.0},
        'Вес BM25-контура (точные ключевые слова, артикулы, номера) в Weighted RRF.',
        'Увеличение помогает поискам по номерам заказов и артикулам, ослабляет смысловые перефразировки.',
        {'chat': 0.2, 'webrtc': 0.2, 'sip': 0.2}, default=0.2,
    ),
    AIParameterDescriptor(
        'rag.hybrid_weight_hipporag', 'float', {'min': 0.0, 'max': 1.0},
        'Вес графового контура HippoRAG (PPR) в Weighted RRF.',
        'Увеличение усиливает многошаговые связи между документами, при 0 граф не влияет на выдачу.',
        {'chat': 0.4, 'webrtc': 0.4, 'sip': 0.3}, default=0.4,
    ),
    AIParameterDescriptor(
        'rag.relevance_threshold', 'float', {'min': 0.0, 'max': 1.0},
        'Порог отсечения чанков реранкером перед подачей в LLM.',
        'Увеличение — меньше, но точнее контекст (меньше галлюцинаций). Уменьшение — больше контекста, выше риск шума.',
        {'chat': 0.0, 'webrtc': 0.1, 'sip': 0.1}, default=0.0,
    ),
    AIParameterDescriptor(
        'rag.enable_query_rewriting', 'bool', {},
        'LLM-переписывание и декомпозиция запроса перед поиском.',
        'Включение повышает Recall в диалогах с местоимениями, добавляет ~250 мс. В голосе пропускается автоматически.',
        {'chat': True, 'webrtc': False, 'sip': False}, default=True,
    ),
    AIParameterDescriptor(
        'ragas.live_sample_rate', 'float', {'min': 0.0, 'max': 1.0},
        'Доля живых диалогов, отправляемых на оценку Faithfulness в ragas-worker.',
        'Увеличение до 1.0 — сплошной аудит качества (дороже по Judge LLM). 0.15 — типовой прод-режим.',
        {'chat': 0.15, 'webrtc': 0.15, 'sip': 0.15}, default=0.15,
    ),
    AIParameterDescriptor(
        'ragas.hallucination_threshold', 'float', {'min': 0.0, 'max': 1.0},
        'Порог Faithfulness, ниже которого диалог помечается Hallucination Alert.',
        'Увеличение делает алерт чувствительнее (больше разборов). Уменьшение ловит только грубые галлюцинации.',
        {'chat': 0.7, 'webrtc': 0.7, 'sip': 0.7}, default=0.7,
    ),
    AIParameterDescriptor(
        'guardrails.prompt_guard', 'bool', {},
        'Проверка каждого входа на промпт-инъекции и джейлбрейки до LLM.',
        'Включение защищает от обхода инструкций, добавляет ~4 мс на эвристике.',
        {'chat': True, 'webrtc': True, 'sip': True}, default=True,
    ),
    AIParameterDescriptor(
        'guardrails.pii_masking', 'bool', {},
        'Обратимое маскирование ПДн (карты, телефоны, паспорта, ИНН, СНИЛС) в логах, Ragas и вебхуках.',
        'Включение закрывает утечки 152-ФЗ/GDPR; значения восстанавливаются из вольта перед ответом клиенту.',
        {'chat': True, 'webrtc': True, 'sip': True}, default=True,
    ),
    AIParameterDescriptor(
        'channels.chat.enabled', 'bool', {},
        'Контур 1: текстовый чат и виджет. Отключение не влияет на голосовые контуры.',
        'Выключайте, когда платформа используется только как голосовой/телефонный бот.',
        {'chat': True, 'webrtc': False, 'sip': False}, default=True,
    ),
    AIParameterDescriptor(
        'channels.webrtc.enabled', 'bool', {},
        'Контур 2: браузерный голосовой бот WebRTC. Работает без SIP-моста и Asterisk.',
        'Выключайте, если голос из браузера не нужен (например, только телефония).',
        {'chat': False, 'webrtc': True, 'sip': False}, default=True,
    ),
    AIParameterDescriptor(
        'channels.sip.enabled', 'bool', {},
        'Контур 3: телефония SIP через voice-sip-bridge и Asterisk/FreeSWITCH.',
        'Выключайте на стендах без телефонии; мост и диалплан остаются выключенными.',
        {'chat': False, 'webrtc': False, 'sip': True}, default=True,
    ),
    AIParameterDescriptor(
        'voice.anger_threshold', 'float', {'min': 0.0, 'max': 1.0},
        'Порог акустического гнева абонента: выше — включается деэскалация и возможен перевод на оператора.',
        'Уменьшение до 0.6 — ранняя эскалация (меньше потерянных клиентов, больше нагрузка на операторов).',
        {'chat': 1.0, 'webrtc': 0.75, 'sip': 0.75}, default=0.75,
    ),
    AIParameterDescriptor(
        'voice.handoff_queue', 'str', {'enum': None},
        'Очередь операторов Asterisk по умолчанию для HUMAN_HANDOFF и эскалаций по гневу.',
        'Изменение переводит все эскалации в другую очередь; индивидуальная очередь задается в правиле.',
        {'chat': None, 'webrtc': 'support_queue', 'sip': 'vip_support'}, default='support_queue',
    ),
    AIParameterDescriptor(
        'intent_classifier.slot_model_id', 'str', {},
        'ID NER-модели в каталоге encoder-classifier для извлечения слотов ({{slots.*}}).',
        'Смена модели меняет набор распознаваемых сущностей; гибридный экстрактор паттернов работает всегда.',
        {'chat': 'rubert-slots-v1', 'webrtc': 'rubert-slots-v1', 'sip': 'rubert-slots-v1'},
        default='rubert-slots-v1',
    ),
    AIParameterDescriptor(
        'active_learning.enable', 'bool', {},
        'Автоматическая запись фраз серой зоны в очередь разметки Active Learning.',
        'Включение питает дообучение энкодера силами аналитиков без ML-инженеров.',
        {'chat': True, 'webrtc': True, 'sip': True}, default=True,
    ),
]


# --- BotBehaviorManifest ---


def _empty_manifest() -> dict[str, Any]:
    return {
        'version': '1.0',
        'bot': {
            'model_id': '',
            'name': '',
            'system_prompt': '',
            'channels': ['chat', 'webrtc', 'sip'],
        },
        'encoder_model_id': 'rubert-intents-v1',
        'intent_routes': [],
        'knowledge_ids': [],
    }


MANIFEST_ROUTE_FIELDS = {
    'intent_name': str,
    'action_type': str,
}


def validate_bot_manifest(manifest: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Семантический линтер манифеста: 6 правил + agent_remediation_hint.

    context: {'encoder_labels': {intent: True}, 'knowledge_ids': [...], 'harness_tools': {harness_id: [tool,...]}}
    """
    context = context or {}
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    bot = manifest.get('bot') or {}
    routes = manifest.get('intent_routes') or []
    encoder_labels = context.get('encoder_labels') or {}
    knowledge_ids = set(context.get('knowledge_ids') or [])
    harness_tools = context.get('harness_tools') or {}

    for index, route in enumerate(routes):
        route_id = route.get('intent_name') or f'route[{index}]'
        tau_high = float(route.get('min_confidence', 0.80))
        tau_low = float(route.get('tau_low', 0.50))

        # Правило 6: пороги вероятности
        if not (0.0 <= tau_low < tau_high <= 1.0):
            errors.append(
                {
                    'code': 'INVALID_THRESHOLDS',
                    'route': route_id,
                    'message': f'tau_low ({tau_low}) должен быть меньше tau_high ({tau_high}) в диапазоне [0, 1]',
                    'agent_remediation_hint': (
                        f'Установите 0.0 <= tau_low < tau_high <= 1.0 для правила "{route_id}". '
                        f'Например: tau_low=0.5, tau_high=0.85.'
                    ),
                }
            )

        # Правило 1: слоты объявлены тогда же, когда используются в шаблонах
        templates = ' '.join(
            str(route.get(field) or '')
            for field in ('direct_response', 'system_prompt_template', 'webhook_config')
        ) + ' ' + json.dumps(route.get('mcp_harness_config') or {}, ensure_ascii=False)
        used_slots = set(re.findall(r'\{\{\s*slots\.([a-zA-Z0-9_]+)\s*\}\}', templates))
        declared_slots = {slot.get('name') for slot in (route.get('required_slots') or [])}
        for slot in used_slots - declared_slots:
            errors.append(
                {
                    'code': 'UNDECLARED_SLOT_IN_TEMPLATE',
                    'route': route_id,
                    'message': f'Шаблон использует {{{{slots.{slot}}}}}, но слот не объявлен в required_slots',
                    'agent_remediation_hint': (
                        f'Добавьте в required_slots правила "{route_id}" слот {{'
                        f'"name": "{slot}", "pattern": "...", "question": "Уточните {slot}?"}} '
                        'либо уберите подстановку из шаблонов.'
                    ),
                }
            )

        # Правило 2: валидность regex слотов
        for slot in route.get('required_slots') or []:
            pattern = slot.get('pattern')
            if not pattern:
                continue
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(
                    {
                        'code': 'INVALID_SLOT_REGEX',
                        'route': route_id,
                        'message': f'Slot "{slot.get("name")}": regex не компилируется: {exc}',
                        'agent_remediation_hint': (
                            f'Исправьте pattern слота "{slot.get("name")}" в правиле "{route_id}": '
                            f'текущее значение "{pattern}" — {exc}. Проверьте экранирование и скобки.'
                        ),
                    }
                )

        # Правило 3: метка интента известна энкодеру
        intent_name = route.get('intent_name')
        if encoder_labels and intent_name not in encoder_labels:
            warnings.append(
                {
                    'code': 'INTENT_MISSING_IN_ENCODER',
                    'route': route_id,
                    'message': f'Метка "{intent_name}" отсутствует в label_map энкодера',
                    'agent_remediation_hint': (
                        f'Вызовите MCP-инструмент encoder_bootstrap_or_train с примерами для метки '
                        f'"{intent_name}" либо включите zero-shot NLI-фоллбэк для этого правила.'
                    ),
                }
            )

        # Правило 4: голосовые каналы не должны получать Markdown
        voice_channels = (route.get('allowed_channels') or bot.get('channels') or ['chat'])
        if any(c in ('webrtc', 'sip') for c in voice_channels):
            voice_text = str(route.get('direct_response') or '') + str(route.get('system_prompt_template') or '')
            if len(voice_text) > 500 or re.search(r'(\|\s*---|```|\|[^\n]*\|)', voice_text):
                warnings.append(
                    {
                        'code': 'VOICE_UNFRIENDLY_TEMPLATE',
                        'route': route_id,
                        'message': 'Шаблон содержит Markdown-таблицы, код или длиннее 500 символов для голосового канала',
                        'agent_remediation_hint': (
                            f'Для правила "{route_id}" создайте короткий голосовой вариант ответа '
                            '(до 500 символов, без таблиц и кода) в direct_response, '
                            'Markdown-версию — в правиле только для канала chat.'
                        ),
                    }
                )

        # Правило 5: существование баз знаний и MCP-харнесов
        for kb_id in route.get('target_collection_ids') or manifest.get('knowledge_ids') or []:
            if knowledge_ids and kb_id not in knowledge_ids:
                warnings.append(
                    {
                        'code': 'UNKNOWN_KNOWLEDGE_BASE',
                        'route': route_id,
                        'message': f'База знаний "{kb_id}" не найдена',
                        'agent_remediation_hint': (
                            f'Создайте базу знаний инструментом knowledge_hipporag_manage '
                            f'или исправьте target_collection_ids правила "{route_id}".'
                        ),
                    }
                )
        harness_config = route.get('mcp_harness_config') or {}
        harness_id = harness_config.get('harness_id')
        if harness_id:
            tools = harness_tools.get(harness_id)
            if tools is None:
                errors.append(
                    {
                        'code': 'UNKNOWN_MCP_HARNESS',
                        'route': route_id,
                        'message': f'MCP-харнес "{harness_id}" не зарегистрирован',
                        'agent_remediation_hint': (
                            f'Зарегистрируйте харнес в блоке External MCP Harnesses (таблица mcp_harness) '
                            f'или исправьте harness_id правила "{route_id}".'
                        ),
                    }
                )
            elif harness_config.get('tool_name') and harness_config['tool_name'] not in tools:
                errors.append(
                    {
                        'code': 'UNKNOWN_MCP_TOOL',
                        'route': route_id,
                        'message': f'Инструмент "{harness_config["tool_name"]}" отсутствует в харнесе "{harness_id}"',
                        'agent_remediation_hint': (
                            f'Вызовите tools/list у харнеса "{harness_id}" (кнопка Discover Tools) '
                            f'и исправьте tool_name правила "{route_id}".'
                        ),
                    }
                )

    return {
        'valid': not errors,
        'errors': errors,
        'warnings': warnings,
        'checked_routes': len(routes),
    }
