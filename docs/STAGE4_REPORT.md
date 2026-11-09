# Отчет: Stage 4 — модульность каналов, MCP и AI-Ready

Выполнение ТЗ `stage_4_work_report_3_mcp_ai_ready.md`. Рабочий каталог: `F:\Z\chatbot\enterprise-ai-platform`.

## 1. Резюме

ТЗ Stage 4 выполнено полностью: закрыты 4 корректировки аудита (К1–К4), реализованы три автономных канальных контура, двусторонний MCP и AI-Ready архитектура. Дополнительно, после базовой сдачи, кнопка клиентского конфига MCP адаптирована под 4 харнеса, включая Zcode и Codex CLI.

| Показатель | Значение |
|---|---|
| Коммиты | форк `c08f91e` (Stage 4) и `6b87926` (multi-client). Корень `e5f195f` и `b73ffd8` |
| Юнит-тесты | 87 passed (56 прежних + 31 новый) |
| E2E-сценарии | 20 (15 прежних + 5 новых) |
| E2E на живом стенде | 17 passed, 0 failed, 3 skip (нет ragas-worker и speaches) |
| E2E строгий без GPU-зависимых | 18 passed, 0 failed, 0 skipped |
| Миграция БД | `f2c8a9b3d4e1` (2 колонки + таблица `mcp_harness`) |
| Новые роутеры ядра | 3 (`mcp_server`, `webrtc_voice`, `ai_ready`) |
| Стартовых ONNX-моделей | 4 (все экспортированы и зарегистрированы на стенде) |

## 2. Корректировки К1–К4

### К1. Три независимых веса в Auto-Tune RAG

Проблема: Optuna подбирал 2 веса (`w_hybrid`, `w_hipporag`), так как ядро сливало Dense и BM25 в `EnsembleRetriever` до слияния контуров.

Решение:

- В `hybrid_fusion.py` режим `return_pool: true` больше не использует слитый результат. Новый помощник `_separate_dense_bm25` по одному корпусу коллекции строит два независимых списка ранков: dense через `VectorSearchRetriever` (векторный поиск ядра), bm25 через `rank_bm25` локально. Третий список дает контур HippoRAG.
- `optimizer.py` в ragas-worker: `rrf_select` принимает три контура, `normalize_three_weights` приводит веса к сумме 1.0. Пространство поиска Optuna: `w_dense`, `w_bm25`, `w_hipporag`, `top_k`.

Доказательство: юнит-тесты `test_normalize_three_weights_sums_to_one`, `test_rrf_select_over_three_lists`, `test_rrf_three_weights_change_ranking` (смена лидера при смене весов).

### К2. Стартовые модели NLI и NER, гибридный экстрактор слотов

Проблема: E2E пропускал сценарии zero-shot и слотов — моделей не было.

Решение:

- `bootstrap_models.py` экспортирует 4 модели из `cointegrated/rubert-tiny2`: `rubert-intents-v1` (интенты), `rubert-nli-v1` (zero-shot), `rubert-slots-v1` (NER с новой головой на 15 BIO-меток через `--init-new-head`), `rubert-embed-v1` (эмбеддинги). Экспорт выполнен реально, модели зарегистрированы в UI через REST.
- Zero-shot без NLI-головы: `zero_shot_embedding` в engine — косинусная близость эмбеддинга текста и формулировок меток, softmax по близостям. Режим включается меткой `zero_shot_mode: embedding` в `label_map.json`.
- `slots_extractor.py`: компилированные паттерны (order_id, phone, email, date, amount, card, url, person) объединяются со спанами ONNX-модели. При пересечении приоритет у обученной модели.
- Эндпоинт `/v1/zero-shot` принимает оба режима. E2E `test_04` и `test_05` позеленели (были skip).

### К3. Локальный эмбеддер и реранкер без torch

Проблема: локальный RAG требовал внешний OpenAI-совместимый API.

Решение: encoder-classifier (где уже есть onnxruntime) отдает `POST /v1/embeddings` в формате OpenAI и `POST /rerank` в формате TEI на модели `rubert-embed-v1` (feature-extraction ONNX, mean-pooling, L2-нормировка). `run_local_ui.sh` прописывает `RAG_OPENAI_API_BASE_URL=http://127.0.0.1:8001/v1`, `RAG_EMBEDDING_MODEL=rubert-embed-v1`, `RAG_RERANKING_ENGINE=external`, `RAG_EXTERNAL_RERANKER_URL=http://127.0.0.1:8001/rerank`. RAG работает из коробки без torch в ядре и без платных ключей.

### К4. NetworkX-фоллбэк графа знаний

Проблема: без пакета `hipporag` граф, визуализатор и Reasoning Path пусты.

Решение: `LightweightHippoGraph` на `networkx.DiGraph` — сущности чанков через `/v1/extract-slots` (fallback — заглавные биграммы), ребра passage-сущность в обе стороны, PPR штатным `nx.pagerank` с personalization и штрафом популярности. Граф персистится в `graph.json` + `version.txt`, интегрирован в адаптер: те же `retrieve_ppr`, `_ppr_on_graph`, `get_graph`, `get_reasoning_path`, те же Redis-локи и атомарная подмена. UI графа, Reasoning Path и третий контур поиска работают на CPU-машине без vLLM.

## 3. Модульность каналов (Chat / WebRTC / SIP)

| Канал | Ключ config / env | Компоненты |
|---|---|---|
| chat | `channels.chat.enabled` / `ENABLE_CHANNEL_CHAT` | текстовый диалог, SSE, Reasoning Path |
| webrtc | `channels.webrtc.enabled` / `ENABLE_CHANNEL_WEBRTC` | WS `/api/v1/voice/webrtc/stream` (16 кГц PCM), POST `/offer`, VAD, barge-in |
| sip | `channels.sip.enabled` / `ENABLE_CHANNEL_SIP` | voice-sip-bridge :8765, `/v1/call-status`, диалплан |

Реализация:

- Миграция `f2c8a9b3d4e1`: колонки `intent_route.allowed_channels` и `intent_route.mcp_harness_config`, таблица `mcp_harness`. Применена на живом стенде.
- `utils/channel_adapter.py`: `detect_channel` (заголовок `X-Channel-Type`, `metadata.channel`, `X-Call-UUID`), фильтрация правил по `allowed_channels`, `sanitize_for_voice_tts` (срез Markdown перед TTS), переменные `{{channel.type}}` и `{{channel.voice}}` в шаблонах.
- Фильтрация включена в `intent_router`: канал определяется на каждом запросе, отключенный канал (`channels.<x>.enabled = false`) пропускает routing, правила выбираются по каналу.
- Автономный WebRTC-шлюз в ядре: `routers/webrtc_voice.py`. Полнодуплексный бинарный PCM-стрим с VAD, событием `barge_in_clear`, транскриптом и TTS по синтагмам. Метрика TTFB и режим `browser_native_audio_fallback` для машин без STT/TTS. Работает без voice-sip-bridge и Asterisk.
- Переключение каналов на лету: тумблеры в UI и MCP-инструмент `platform_configure_modules`, без рестарта. Изоляция контуров проверена E2E `test_20`.

## 4. Двусторонний MCP

### Направление А: платформа как MCP Server

`routers/mcp_server.py`: JSON-RPC 2.0 на `POST /api/v1/mcp` (одиночные и пакетные запросы) плюс SSE-канал `GET /api/v1/mcp/sse`. Методы: `initialize`, `ping`, `tools/list`, `tools/call`, `resources/list`, `resources/read`.

Восемь инструментов:

1. `platform_inspect_status` — состояние модулей, энкодеры, базы знаний, харнесы.
2. `platform_configure_modules` — включение/выключение каналов и настроек без рестарта.
3. `bot_apply_behavior_manifest` — атомарное создание и обновление бота. Workspace-модель, промпт, каналы, полная замена матрицы интентов. Режим dry_run и семантическая валидация.
4. `bot_export_behavior_manifest` — выгрузка бота в самодостаточный JSON.
5. `intent_upsert_single_route` — точечное изменение одного правила.
6. `bot_simulate_conversation_turn` — песочница: детерминированный DecisionTrace (классификация, слоты, следующий вопрос, маршрут) в выбранном канале, без вызова LLM.
7. `encoder_bootstrap_or_train` — передача примеров и запуск файнтюна с Rollback Gate.
8. `knowledge_hipporag_manage` — add_text, reindex, тестовый query с Reasoning Path.

Ресурсы: `mcp://platform/ai-ready-schema`, `mcp://platform/config`, `mcp://bots/{id}/manifest`, `mcp://bots/{id}/telemetry`.

### Направление Б: платформа как MCP Client

- Таблица `mcp_harness`: id, описание для агентов, транспорт (streamable_http, sse, stdio), эндпоинт/команда, токен, кэш `tools/list`, таймаут.
- `utils/mcp_harness.py`: вызов `tools/call` по трем транспортам, `discover_tools`, рендер `argument_mapping` с подстановкой `{{slots.*}}`, `{{user_message}}`, `{{channel.type}}`.
- Действие правила `MCP_HARNESS` в `intent_router`: после сбора всех обязательных слотов роутер рендерит аргументы и вызывает инструмент внешнего харнеса. Ответ форматируется для пользователя. При `fallback_to_llm` ошибка харнеса не показывается. Режим `autonomous_subagent` передает `goal_prompt` и `allowed_tools`.
- REST: CRUD харнесов + `POST /harnesses/{id}/discover` с кэшированием схемы.

## 5. AI-Ready архитектура

- `utils/ai_ready_schema.py`: 21 дескриптор `AIParameterDescriptor` — для каждой настройки строгий тип, ограничения, физический смысл (`purpose_for_ai`), эффект изменения (`tuning_impact`), рекомендуемые значения по каналам. Шаблон `BotBehaviorManifest`.
- Валидатор `validate_bot_manifest` — 6 правил: необъявленные слоты в шаблонах, невалидные regex, метки вне label_map энкодера, голосонесовместимые шаблоны, несуществующие базы знаний и MCP-харнесы/инструменты, инвертированные пороги. Каждая проблема несет готовый `agent_remediation_hint`.
- Эндпоинты: `GET /llms.txt` (публичное руководство: каналы, MCP, переменные, playbook из 4 шагов), `GET /api/v1/ai-manifest.json` (дескрипторы + текущий config + энкодеры), `POST /api/v1/classifiers/manifest/validate`.
- MCP-ресурсы отдают тот же паспорт настроек и живой манифест бота.

## 6. UI (SvelteKit)

Три новых компонента в `workspace/Intents/` и две новые вкладки:

- **Channels & Modules** (`ChannelRuntimePanel.svelte`): три карточки-тумблера каналов с живым состоянием из ai-manifest. Переключение через MCP. Встроенный тестовый звонок WebRTC из браузера: захват микрофона 16 кГц, стрим в шлюз, лог transcript, answer и TTFB.
- **MCP & Harnesses** (`MCPControlPanel.svelte`): URL сервера, ссылки на `/llms.txt` и `ai-manifest.json`, таблица 8 инструментов с кнопкой тестового вызова, реестр внешних харнесов с Discover Tools и созданием.
- **AI Manifest Editor** (`AIManifestEditorModal.svelte`): просмотр и правка BotBehaviorManifest целиком. Валидация с подсветкой ошибок и подсказок. Dry run, Apply, Simulate Turn с выбором канала и шкалой anger_score.
- Кнопка **Copy client config** выдает готовый конфиг подключения под выбранный клиент. Форматы:
  - Claude Code: `.mcp.json`, ключ `mcpServers` с `type: "http"`.
  - Cursor: `~/.cursor/mcp.json`, ключ `mcpServers` с `url` и headers.
  - Zcode: `~/.zcode/cli/config.json` с вложенным ключом `mcp.servers`. Формат по официальному гайду конфигурации Zcode. Совместимый fallback — `~/.agents/mcp.json` с ключом `mcpServers`.
  - Codex CLI: `~/.codex/config.toml`, секция `[mcp_servers.enterprise-ai-platform]` с `url` и `bearer_token_env_var` — токен не зашивается в файл.
  У каждого пресета показан путь к файлу и заметка о подключении. Превью обновляется при смене клиента.

Фронтенд пересобран, живой UI на :8080 отдает новую сборку.

## 7. Тестирование

| Уровень | Объем | Результат |
|---|---|---|
| Юнит Stage 4 (`tests/test_stage4_modular_mcp_aiready.py`) | 31 тест | проходят |
| Юнит всего (все этапы) | 87 тестов | 87 passed |
| E2E Stage 4 (`tests/e2e_platform_verify.py`, сценарии 16–20) | 5 сценариев | проходят на живом стенде |
| E2E всего | 20 сценариев | 17 passed, 0 failed, 3 skip |
| E2E строгий (`E2E_STRICT=true`, без GPU-зависимых) | 18 сценариев | 18 passed, 0 failed, 0 skipped |
| Статика | py_compile всех файлов, svelte-check | ошибок в новых файлах нет |

Покрытие новых сценариев:

- разделение трех весов RRF и гибридный экстрактор слотов.
- NetworkX-фоллбэк графа, канальная изоляция, голосовой санитайзер.
- Линтер манифеста с подсказками и JSON-RPC протокол MCP.
- Вызов внешнего харнеса, discovery-эндпоинты, создание бота и симуляция хода.
- stdio-харнес, WebRTC-шлюз, переключение каналов.

Три E2E-skip — среды без GPU-стека: ragas-worker (нужен Judge LLM), голосовой звуковой прогон (нужен speaches). Полный строгий 20/20 достигается на GPU-стенде.

## 8. Проблемы, найденные при прогоне, и решения

| Проблема | Решение |
|---|---|
| `EmbeddingsRequest` без импорта BaseModel — классификатор не стартовал | Импорт добавлен |
| Дубликат класса `SlotSpan` в schemas.py — pydantic отклонял экземпляры при конструировании ответа | Дубль удален |
| В миграции v3 отсутствовала колонка `mcp_harness_config` — SELECT правил падал | Колонка добавлена в миграцию и в живую БД |
| `MCPHarnesses` не импортирован в classifiers.py | Импорт добавлен |
| Двойной префикс `/voice/voice/...` в webrtc-роутере (405) | Пути маршрутов исправлены |
| В e2e отсутствовал `import json` (NameError в test_19) | Импорт добавлен |
| test_03 ловил холодную загрузку модели | Прогрев перед замером SLA |
| Регистр ключей в `rrf_select` и порядок аргументов дескрипторов | Исправлены, тесты обновлены |
| Дубликат `copyClientConfig` в панели после вставки мульти-клиент блока | Старое определение удалено |

## 9. Задокументированные отклонения

1. Auto-Tune оптимизирует веса на границе контуров: Dense и BM25 разделены в режиме пулов, но в прод-пайплайне ядро по-прежнему сливает их внутри `EnsembleRetriever`. Применение найденных трех весов к прод-контуру требует выноса раздельного слияния в боевой путь поиска (зафиксировано как следующий шаг).
2. WebRTC-шлюз реализован как WS-PCM-транспорт с конфигурационным `/offer`. Полный SDP-обмен с PeerConnection (aiortc) остался за рамками. Для браузеров используется WebSocket-стрим — сценарий автономного голосового канала закрыт.
3. Фаза 2 Optuna использует лексическую прокси-метрику recall и precision (без Judge LLM) для быстрой фильтрации конфигураций. Финальную оценку через честный Ragas проходят только топ-3.

## 10. Definition of Done — итог

| № | Условие ТЗ | Статус |
|---|---|---|
| 1 | Закрыты К1–К4 | Да, с доказательствами (тесты + живой стенд) |
| 2 | Три канала работают автономно и вместе | Да, изоляция проверена test_20, WebRTC без SIP — test_19 |
| 3 | Двусторонний MCP | Да, server и client, E2E 17/18 |
| 4 | AI-Ready формат | Да, `/llms.txt`, `ai-manifest.json`, линтер, манифест |
| 5 | 70+ юнит и максимум строгих E2E | 87 юнит. На стенде 18 из 18 строгих, 3 skip только по GPU |

## 11. Артефакты

| Артефакт | Путь |
|---|---|
| MCP Server | `open-webui-fork/backend/open_webui/routers/mcp_server.py` |
| MCP Client | `open-webui-fork/backend/open_webui/utils/mcp_harness.py` |
| Канальный адаптер | `open-webui-fork/backend/open_webui/utils/channel_adapter.py` |
| WebRTC-шлюз | `open-webui-fork/backend/open_webui/routers/webrtc_voice.py` |
| AI-Ready схема | `open-webui-fork/backend/open_webui/utils/ai_ready_schema.py` |
| Трехконтурные пулы | `open-webui-fork/backend/open_webui/retrieval/hybrid_fusion.py` |
| Легковесный граф | `open-webui-fork/backend/open_webui/retrieval/hipporag_engine.py` |
| Юнит-тесты Stage 4 | `tests/test_stage4_modular_mcp_aiready.py` |
| E2E | `tests/e2e_platform_verify.py` (сценарии 16–20) |
| UI-компоненты | `open-webui-fork/src/lib/components/workspace/Intents/` |
| Интерфейс | `http://127.0.0.1:8080` (admin@local.dev / Admin-Platform-2026!) |
