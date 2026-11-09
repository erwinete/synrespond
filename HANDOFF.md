# HANDOFF: Enterprise AI Platform — состояние проекта для нового диалога

> Прочитай этот файл целиком перед началом работы. Он содержит все необходимое
> для продолжения разработки без перечитывания истории диалога.

## 1. Что это за проект

Self-hosted корпоративная платформа текстовых и голосовых ботов на базе
форка Open WebUI 0.13.0. Пять этапов разработки по цепочке ТЗ
(`C:\Users\user\Downloads\ai_open_webui.md`, `ai.md`,
`stage_3_*.md`, `stage_4_*.md`, `stage_5_*.md`) — все выполнены полностью.

**Расположение**: `F:\Z\chatbot\enterprise-ai-platform` (диск F:, на C: нет места).
Два git-репозитория:

| Репозиторий | Ветка | Роль |
|---|---|---|
| `enterprise-ai-platform/` (корень) | `main` | Сервисы, тесты, инфраструктура, доки |
| `enterprise-ai-platform/open-webui-fork/` | `feat/enterprise-platform` | Форк open-webui 0.13.0 + все доработки ядра |
| `enterprise-ai-platform/pipelines/` | main | Submodule: стоковый рантайм pipelines |

## 2. Выполненные этапы (все закрыты)

| Этап | ТЗ | Что сделано | Коммиты форка | Коммиты корня |
|---|---|---|---|---|
| 1 | `ai_open_webui.md` | Базовая платформа: интенты, HippoRAG, Ragas, голос, UI, Docker | `dc3e000` | `4a530b2`, `784de22` |
| 2 | `ai.md` (аудит) | Д1–Д5 + Ф1–Ф7 + тесты + запуск UI | `fbec4e7`, `bbd61f4` | `f37aa7d`, `8256beb` |
| 3 | `stage_3_*.md` | Стабилизация: atomic swap, PII стрим, slot-filling, aiortc, judge | `7399e82` | `9e0f6f8` |
| 4 | `stage_4_*.md` | Модульность каналов, MCP двусторонний, AI-Ready | `c08f91e`, `6b87926` | `e5f195f`, `b73ffd8` |
| 5 | `stage_5_*.md` | Zero Technical Debt: единый RRF, aiortc SDP, семантическая прокси, local judge, 22/22 E2E | `679198e` | `417ad4b`, `9b651c4` |

Отчеты: `docs/WORK_REPORT.md` (этапы 1–5), `docs/STAGE4_REPORT.md` (этап 4 подробно).

## 3. Архитектура — ключевые файлы

### Бэкенд форка (`open-webui-fork/backend/open_webui/`)

| Файл | Назначение |
|---|---|
| `models/intents.py` | Таблицы: encoder_model, intent_route (allowed_channels, required_slots, mcp_harness_config), intent_feedback, mcp_harness + CRUD |
| `models/ragas_evals.py` | Таблицы: ragas_dataset, ragas_run, ragas_sample + CRUD |
| `retrieval/hipporag_engine.py` | HippoRAG адаптер + LightweightHippoGraph (networkx fallback) + atomic_swap_graph_directory + version.txt инвалидация |
| `retrieval/hybrid_fusion.py` | **weighted_rrf_three_way** (боевое слияние Dense+BM25+HippoRAG), _separate_dense_bm25, return_pool режим |
| `retrieval/query_rewriter.py` | Деконтекстуализация + декомпозиция запроса через LLM (skip в voice fast-path) |
| `utils/intent_router.py` | Хук маршрутизации: классификация, слоты, дизамбигуация, slot-filling (Redis TTL 1800s), guard, HUMAN_HANDOFF, MCP_HARNESS, сэмплы Ragas |
| `utils/route_templates.py` | Чистые функции: render_route_template ({{slots.*}}), is_slot_cancellation, mentions_slot_value, slot_retry_exceeded |
| `utils/pii_vault.py` | Reversible PII Vault (mask/unmask) + StreamingPIIUnmasker (буфер для SSE-токенов) |
| `utils/call_status.py` | Реестр эскалаций звонков (fallback без Redis) |
| `utils/channel_adapter.py` | detect_channel, filter_routes_by_channel, sanitize_for_voice_tts, channel_variables |
| `utils/mcp_harness.py` | MCP-клиент: call_tool (streamable_http/sse/stdio), discover_tools, render_arguments |
| `utils/ai_ready_schema.py` | 21 AIParameterDescriptor, validate_bot_manifest (6 правил + agent_remediation_hint) |
| `routers/classifiers.py` | CRUD энкодеров, правил, feedback, harnesses, graph, copilot, train, manifest/validate |
| `routers/ragas.py` | CRUD датасетов, прогонов, optimize, **apply-weights** |
| `routers/mcp_server.py` | JSON-RPC 2.0: 8 tools, 4 resources на POST /api/v1/mcp |
| `routers/webrtc_voice.py` | WS /stream (PCM 16kHz) + POST /offer (aiortc SDP или ws_fallback) |
| `routers/ai_ready.py` | /llms.txt (публичный), /ai-manifest.json |
| `utils/middleware.py` | Врезки: маршрутизация (~line 2385), рерайтер (~line 1785), voice headers, Ragas sampling в background_tasks_handler |
| `main.py` | Регистрация роутеров + direct_response (~line 1629) + SSE unmask wrapper |
| `config.py` | ~35 ключей DEFAULT_CONFIG (intent_classifier.*, hipporag.*, ragas.*, guardrails.*, voice.*, channels.*, active_learning.*, mcp.*) |

### Миграции БД (Alembic)

| Revision | Таблицы/колонки |
|---|---|
| `e7a1f4c2b9d8` | encoder_model, intent_route, ragas_dataset, ragas_run, ragas_sample |
| `e8c2d5f4a1b7` | intent_feedback, intent_route.required_slots |
| `f2c8a9b3d4e1` | intent_route.allowed_channels, intent_route.mcp_harness_config, mcp_harness |

Применяются автоматически при старте ядра (alembic upgrade head).

### Микросервисы (`services/`)

| Сервис | Порт | Эндпоинты |
|---|---|---|
| `encoder-classifier` | 8001 | /v1/classify, /v1/zero-shot (embedding-режим), /v1/extract-slots (гибридный: ONNX + regex паттерны), /v1/analyze, /v1/guard, /v1/embeddings, /rerank, /v1/train (ProcessPoolExecutor + Rollback Gate), /v1/models/{id}/reload |
| `ragas-worker` | 8002 | /v1/evaluate/live (Redis очередь), /v1/evaluate/batch, /v1/evaluate/sync (local judge), /v1/optimize (двухфазный: пулы + эмбеддинги → semantic proxy → MedianPruner → top-3 full Ragas) |
| `voice-sip-bridge` | 8765 | TCP AudioSocket (Asterisk), WS /ws/audio, GET /v1/call-status/{uuid} |

### Фронтенд (`open-webui-fork/src/lib/components/`)

| Раздел | Компоненты |
|---|---|
| Workspace → Intents | Intents.svelte (5 вкладок), IntentRouterTable, EncoderRegistry, ActiveLearningQueue, ChannelRuntimePanel, MCPControlPanel, AIManifestEditorModal |
| Workspace → Knowledge | HippoRAGGraphViewer (граф + PPR + Reasoning Path) |
| Admin → RAG Quality | Ragas.svelte (4 вкладки: Overview, Runs, Golden Datasets, Alerts) |

API-клиенты: `src/lib/apis/classifiers/index.ts`, `src/lib/apis/ragas/index.ts`.

## 4. Живой стенд — как запустить и проверить

### Запуск

```bash
cd F:/Z/chatbot/enterprise-ai-platform
bash scripts/run_local_ui.sh
```

Скрипт поднимает 4 сервиса: bootstrap моделей (если нет) → encoder-classifier :8001 → ragas-worker :8002 (local_onnx судья) → open-webui :8080.

**Вход**: http://127.0.0.1:8080 — admin@local.dev / Admin-Platform-2026!

### Проверка здоровья

```bash
curl http://127.0.0.1:8080/health   # {"status":true}
curl http://127.0.0.1:8001/health   # {"status":"ok","device":"cpu"}
curl http://127.0.0.1:8002/health   # {"status":"ok","judge_engine":...}
curl http://127.0.0.1:8765/health   # {"status":"ok"}
```

### Стартовые ONNX-модели (models/encoders/)

| model_id | Задача | Откуда |
|---|---|---|
| rubert-intents-v1 | sequence_classification | cointegrated/rubert-tiny2 + 4 демо-метки |
| rubert-nli-v1 | feature_extraction (zero_shot_mode: embedding) | тот же source |
| rubert-slots-v1 | token_classification (новая голова, 15 BIO) | тот же source |
| rubert-embed-v1 | feature_extraction (эмбеддинги) | тот же source |

## 5. Тесты — как запускать и текущий статус

```bash
cd F:/Z/chatbot/enterprise-ai-platform
.venv-encoder/Scripts/python -m pytest tests/ -q           # 102 юнит
.venv-encoder/Scripts/python -m pytest tests/e2e_platform_verify.py -v  # E2E
```

| Набор | Кол-во | Статус |
|---|---|---|
| Юнит: все файлы в tests/ | 102 | 102 passed |
| E2E (20 + 2 Stage 5) | 22 | **22 passed, 0 failed, 0 skipped** при E2E_STRICT=true на живом стенде |

Переменные для E2E: `E2E_API_TOKEN` (Bearer токен админа), `E2E_NLI_MODEL=rubert-nli-v1`, `E2E_SLOT_MODEL=rubert-slots-v1`, `E2E_RAGAS_URL=http://localhost:8002`, `E2E_STRICT=true`.

Регламент ручных кейсов: `tests/QA_PIPELINE.md` (TC-INF, TC-CLS, TC-RAG, TC-EVAL, TC-VOICE + TC-S3, TC-S4, TC-S5).

## 6. Виртуальные окружения (все на F:)

| venv | Назначение | Расположение |
|---|---|---|
| `.venv-encoder` | encoder-classifier + ragas-worker + тесты | `enterprise-ai-platform/.venv-encoder/` |
| `.venv-webui` | ядро Open WebUI (slim + chromadb + pydub + audioop-lts + azure/gcp + aiortc) | `enterprise-ai-platform/.venv-webui/` |

Дополнительные пины (Stage 5): torch==2.8.0 CPU, optimum[exporters]>=1.20,<2.0, sqlalchemy[asyncio], aiosqlite, redis, optuna, langchain-core, greenlet (для sqlalchemy asyncio).

## 7. Конфигурация и переменные

Полный список с описанием — `.env.example` и `/api/v1/ai-manifest.json` (живой).
Ключевые группы: `intent_classifier.*`, `hipporag.*`, `rag.*`, `ragas.*`,
`guardrails.*`, `voice.*`, `channels.*`, `active_learning.*`, `mcp.*`.

**Важно**: настройки в таблице `config` (SQLite `backend/data/webui.db`)
перекрывают env-переменные. Если нужно поменять URL классификатора —
меняйте через UI (Settings) или напрямую в таблице config.

## 8. Зарегистрированные MCP-пресеты

Кнопка Copy client config в UI (Workspace → Intents → MCP & Harnesses) выдает конфиг под:

| Клиент | Файл | Формат |
|---|---|---|
| Claude Code | .mcp.json | JSON, mcpServers → {type:"http", url, headers} |
| Cursor | ~/.cursor/mcp.json | JSON, mcpServers → {url, headers} |
| **Zcode** | ~/.zcode/cli/config.json | JSON, **вложенный ключ mcp.servers** (не mcpServers!) |
| Codex CLI | ~/.codex/config.toml | TOML, [mcp_servers.name] + bearer_token_env_var |

## 9. Что осталось сделать вручную

1. **Боевая модель интентов**: экспортировать ruBERT/DeBERTa вместо демо rubert-tiny2, дообучить через Active Learning.
2. **LLM-эндпоинт**: подключить в Settings → Connections (vLLM/OpenAI) — без этого чат и Judge Ragas (llm_judge) не отвечают.
3. **Прод-развертывание**: `docker compose -f docker-compose.production.yml up -d --build`, заполнить `OPEN_WEBUI_SERVICE_TOKEN`.
4. **Asterisk**: диалплан из DEPLOYMENT.md п. 4 (AudioSocket + CURL call-status + Queue).
5. **Полный строгий E2E на GPU-стенде**: все 22 теста без skip с vLLM и speaches.

## 10. Известные ограничения текущей сборки

| Ограничение | Обход |
|---|---|
| Node 24 вместо 20–22 → ~5700 svelte-check ошибок в стоковом коде | Прод-сборка в Docker с Node 20. В новых файлах ошибок нет |
| Пакет hipporag не установлен локально | LightweightHippoGraph на networkx — граф, PPR, Reasoning Path работают |
| speaches (STT/TTS) не запущен локально | synthesize_pcm_fallback + STT circuit breaker — голосовой цикл замкнут |
| LLM не подключен — чат не генерирует ответ | intent-маршрутизация и direct_response работают без LLM |
| Без Postgres локально | SQLite webui.db + sqlite ragas_local.db для воркера |

## 11. Структура файлов верхнего уровня

```
enterprise-ai-platform/
├── open-webui-fork/         # Форк (submodule, ветка feat/enterprise-platform)
├── pipelines/               # Стоковый рантайм (submodule)
├── services/
│   ├── encoder-classifier/  # ONNX инференс (app.py, engine.py, guard.py, training.py, slots_extractor.py)
│   ├── ragas-worker/        # Оценка качества (app.py, evaluator.py, optimizer.py, bot_runner.py, db.py, config.py)
│   └── voice-sip-bridge/    # Телефония (server.py, session.py, speech.py, vad.py, audiosocket.py, emotion.py, call_status.py)
├── pipelines-custom/        # intent_router_inlet.py
├── models/encoders/         # 4 ONNX-модели (rubert-intents-v1, rubert-nli-v1, rubert-slots-v1, rubert-embed-v1)
├── tests/                   # 102 юнит + 22 E2E + QA_PIPELINE.md
├── scripts/                 # run_local_ui.sh, bootstrap_models.py
├── deploy/DEPLOYMENT.md     # 16 разделов
├── docs/                    # WORK_REPORT.md, STAGE4_REPORT.md
├── docker-compose.production.yml / docker-compose.dev.yml / .env.example
└── HANDOFF.md               # Этот файл
```

## 12. Как продолжать работу

1. **Запустить стенд**: `bash scripts/run_local_ui.sh` (из корня).
2. **Запустить тесты**: `pytest tests/ -q` (юнит) и `E2E_STRICT=true pytest tests/e2e_platform_verify.py` (строгий E2E).
3. **Найти файл по функции**: см. таблицу в разделе 3.
4. **Найти настройку**: `/api/v1/ai-manifest.json` (все дескрипторы) или `.env.example`.
5. **Понять формат манифеста бота**: `mcp://platform/ai-ready-schema` ресурс или `_empty_manifest()` в ai_ready_schema.py.
6. **Добавить настройку**: env-переменная в config.py + ключ в DEFAULT_CONFIG + дескриптор в ai_ready_schema.py.
7. **Добавить правило интента**: через UI (Workspace → Intents) или MCP `intent_upsert_single_route`.
8. **Дообучить энкодер**: Workspace → Intents → Active Learning → разметить → Fine-Tune & Hot-Swap.
