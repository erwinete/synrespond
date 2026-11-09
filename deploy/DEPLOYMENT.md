# Развертывание и эксплуатация

## 1. Требования

- Docker 24+ с NVIDIA Container Toolkit (для продакшн-профиля с GPU).
- 1 GPU не ниже 24 ГБ VRAM для стека vLLM + Whisper + TEI (либо распределите
  сервисы по узлам, задав `deploy.resources` и `NVIDIA_VISIBLE_DEVICES`).
- Свободные порты: 8080 (UI), 8000 (vLLM), 8001 (классификатор), 8002 (ragas-worker),
  8003 (speaches), 8765 (AudioSocket), 9099 (pipelines).

## 2. Первый запуск

```bash
cp .env.example .env
# Заполните OPEN_WEBUI_SERVICE_TOKEN (API-ключ администратора Open WebUI)
docker compose -f docker-compose.production.yml up -d --build
```

Миграции БД применяются автоматически при старте open-webui
(`ENABLE_DB_MIGRATIONS=true` по умолчанию). Создаются таблицы:
`encoder_model`, `intent_route`, `ragas_dataset`, `ragas_run`, `ragas_sample`.

## 3. Настройка маршрутизации интентов

1. Экспортируйте ONNX-модель (см. README, шаг 3) в `models/encoders/<model_id>`.
2. Workspace → Intents → Encoder Models: зарегистрируйте энкодер
   (путь каталога = `<model_id>`), нажмите Test Inference и проверьте top-K и задержку.
3. Вкладка Intent Matrix: создайте правила.
   Поля правила:

   | Поле | Назначение |
   |---|---|
   | Intent name | Метка класса энкодера (например, `billing_refund`) |
   | Bot model | Workspace-модель бота, для которой действует правило |
   | Confidence | Порог `tau_high` прямой маршрутизации |
   | Action | `RAG_COLLECTION`, `DIRECT_RESPONSE`, `WEBHOOK_TOOL`, `SYSTEM_PROMPT_OVERRIDE` |
   | Knowledge | Привязанные базы знаний (фильтр RAG-коллекций) |
   | HippoRAG | Включить графовый контур для этого интента |

4. Логика порогов: `p >= tau_high` → правило; `tau_low <= p < tau_high` → обычный
   сценарий с сохранением кандидатов в `metadata['intent_candidates']`;
   `p < tau_low` → fallback на общий RAG.

Шаблоны поддерживают подстановку `{{user_message}}` и `{{intent}}`.

## 4. Голосовая телефония (Asterisk)

`extensions.conf` (Д5, Stage 3: перевод звонка управляется статусом моста):

```ini
[voice-bots]
exten => _X.,1,Answer()
 same => n,Set(CALL_UUID=${SHELL(uuidgen):0:-1})
 same => n,AudioSocket(${CALL_UUID},${BRIDGE_HOST}:8765)
 same => n,Set(TRANSFER_TARGET=${CURL(http://${BRIDGE_HOST}:8765/v1/call-status/${CALL_UUID})})
 same => n,GotoIf($["${TRANSFER_TARGET}" != "NONE"]?handoff,1)
 same => n,Hangup()

exten => handoff,1,NoOp(Escalating call ${CALL_UUID} to queue ${TRANSFER_TARGET})
 same => n,Queue(${TRANSFER_TARGET},tT,,,300)
 same => n,Hangup()
```

Мост отдает имя очереди через `/v1/call-status/{uuid}`: бэкенд пишет
`call_transfer:{uuid}` в Redis при срабатывании правила `HUMAN_HANDOFF`
(передает заголовок `X-Call-UUID`), а сам мост — при `anger_score >= VOICE_HANDOFF_ANGER`.
TTFB в логах моста раздельный: `TTFB_filler_ms` (акустический филлер, ~250 мс)
и `TTFB_llm_answer_ms` (первый речевой чанк ответа) — филлер не маскирует
деградацию бэкенда в мониторинге.

Транспорт AudioSocket: PCM 16-bit mono, 8 кГц, вход и выход.
Альтернативы: WebSocket `ws://<host>:8765/ws/audio?sample_rate=8000`
(FreeSWITCH mod_audio_fork или собственный шлюз).

Переменные шлюза: `VOICE_CHAT_MODEL` (Workspace-модель бота),
`VOICE_STT_MODEL`, `VOICE_TTS_MODEL`, `VOICE_TTS_VOICE`, `VOICE_BARGE_IN`
(по умолчанию включен: речь абонента отменяет воспроизведение).

Задержка: лог `voice-sip-bridge` печатает TTFB первого аудио-чанка
цель — ≤ 650 мс от конца реплики абонента.

## 5. Контроль качества (Ragas)

**Batch Benchmark:** Admin → RAG Quality → Golden Datasets → загрузите CSV
(`question,ground_truth`) или JSONL → Runs → New Evaluation Run (датасет +
Workspace-модель бота). Воркер прогоняет вопросы через полный пайплайн бота и
считает Faithfulness, Answer Relevancy, Context Precision, Context Recall.

**Live Sampling:** доля трафика `RAGAS_LIVE_SAMPLE_RATE` (по умолчанию 0.15)
отправляется в очередь Redis; воркер считает Faithfulness и Answer Relevancy.
Ответы с Faithfulness ниже `RAGAS_HALLUCINATION_THRESHOLD` получают флаг
`hallucination` и появляются во вкладке Hallucination Alerts.

Для оценки нужен судья: `JUDGE_LLM_URL` / `RAGAS_JUDGE_MODEL` (vLLM) и
эмбеддинги `EMBEDDINGS_URL` (TEI).

## 6. Два способа маршрутизации

| Способ | Когда выбрать |
|---|---|
| Нативные хуки форка (`utils/intent_router.py`) | Основной путь: полный контроль — подмена промпта, фильтрация коллекций, direct-response без LLM |
| Inlet-пайплайн (`pipelines-custom/intent_router_inlet.py`) | Бонус: апгрейд стокового Open WebUI без форка; подменяет промпт и коллекции через `features` |

## 7. Обслуживание

- **Бэкап:** volume `pg_data` (Postgres), `openwebui_data` (файлы), `qdrant_data` (векторы).
- **Графы HippoRAG:** каталог `${DATA_DIR}/hipporag_storage/<knowledge_id>` внутри
  контейнера open-webui. Пересобирается автоматически при изменениях файлов базы;
  удаление базы удаляет граф.
- **Деградация:** пакет `hipporag` опционален
  (`backend/requirements-enterprise.txt`). Без него поиск возвращает Dense + BM25,
  в логах — предупреждение. Классификатор недоступен → чат работает без маршрутизации.
- **Логи:** `docker compose logs -f open-webui encoder-classifier ragas-worker voice-sip-bridge`.

## 8. Переменные окружения

Полный список с описанием — в `.env.example`. Ключевые:

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `ENABLE_ENCODER_CLASSIFIER` | `true` | Хук классификации в middleware |
| `ENCODER_CLASSIFIER_URL` | `http://encoder-classifier:8001` | Адрес сервиса инференса |
| `ENCODER_CLASSIFIER_SLOT_MODEL_ID` | — | NER-модель слотов (`{{slots.*}}`, Voice Fast-Path) |
| `INTENT_DISAMBIGUATION_DELTA` | `0.15` | Порог активной дизамбигуации серой зоны |
| `ENABLE_HIPPORAG` | `true` | Графовый контур поиска |
| `HIPPORAG_VOICE_FAST_NER` | `true` | В голосе NER для PPR через ONNX вместо LLM |
| `RAG_HYBRID_WEIGHT_DENSE/BM25/HIPPORAG` | `0.4/0.2/0.4` | Веса Weighted RRF |
| `ENABLE_RAG_QUERY_REWRITING` | `true` | Переписывание запросов (пропускается в голосе) |
| `ENABLE_RAGAS` | `true` | Контур оценки качества |
| `RAGAS_LIVE_SAMPLE_RATE` | `0.15` | Доля сэмплируемого продового трафика |
| `GUARDRAILS_PROMPT_GUARD` | `true` | Отсев джейлбрейков (`/v1/guard`) |
| `GUARDRAILS_PII_MASKING` | `true` | Маскирование ПДн в логах/Ragas/вебхуках |
| `VOICE_FILLER` / `VOICE_FILLER_DELAY_MS` | `true` / `250` | Акустический филлер в телефонии |
| `VOICE_EMOTION` | `true` | Оценка гнева (ONNX или акустический прокси) |
| `ACTIVE_LEARNING_ENABLE` | `true` | Очередь серой зоны для разметки |

Все настройки также читаются из таблицы `config` и управляются из UI без перезапуска.

## 9. Слоты и slot-filling (Д1/Ф2)

1. Экспортируйте NER-модель: `scripts/export_onnx.py --model Davlan/bert-base-cased-ner-hrl --task token_classification --out models/encoders/slots-ner-v1`.
2. Задайте `ENCODER_CLASSIFIER_SLOT_MODEL_ID=slots-ner-v1`.
3. В правиле Intent Matrix → Details → Required Slots добавьте строки:
   имя (`order_id`), regex-валидатор (`\d{5,8}`), уточняющий вопрос.
4. Поведение: пока слоты не собраны и не прошли regex — бот задает
   заготовленные вопросы без LLM; после заполнения выполняется вебхук
   с `{{slots.<name>}}` в теле. Состояние сессии живет 30 минут.

## 10. Guardrails и ПДн (Ф5)

- Джейлбрейк-проверка: `POST /v1/guard` в encoder-classifier. С моделью
  `models/encoders/prompt-guard` (метки INJECTION/SAFE) — скор модели,
  без неё — эвристика шаблонов. При срабатывании — отказной ответ без LLM.
- PII Vault: маскирование телефона/карты/e-mail/паспорта/ИНН/СНИЛС на
  токены `[CARD_1]` в sэмплах Ragas и телах вебхуков; восстановление
  перед отправкой клиенту. Маскирование входа LLM включайте осознанно —
  при стриминге токены не расшифровываются.

## 11. Active Learning (Ф1)

1. Серая зона (`tau_low ≤ p < tau_high`) автоматически попадает в
   Workspace → Intents → Active Learning с подсказкой LLM
   (`ACTIVE_LEARNING_HINT_MODEL`, например `Qwen/Qwen2.5-32B-Instruct-AWQ`).
2. Аналитик подтверждает метки (Enter), затем «Fine-Tune & Hot-Swap ONNX».
3. На узле encoder-classifier установите `requirements-train.txt`
   (torch, optimum, scikit-learn). Обучается классификационная голова
   (энкодер заморожен), считаются accuracy/macro-F1 на holdout 20%,
   новая ONNX-модель подменяется атомарно, статус — в UI.

## 12. Auto-Tune (Ф4)

Admin → RAG Quality → Runs → New Evaluation Run → режим Auto-Optimize.
Optuna (по умолчанию 10 trials × 10 вопросов) подбирает
`w_dense/w_bm25/w_hippo/threshold/top_k` через полный пайплайн бота
(`metadata.rag_overrides`), целевая функция — `0.4·Recall + 0.3·Precision + 0.3·Faithfulness`.
Лучший конфиг появляется в metrics запуска; примените значения в `.env`
или правилах интентов. Оптимизация промптов (DSPy MIPROv2) — второй этап.

## 13. Handoff на оператора (Ф7)

Действие правила `HUMAN_HANDOFF`: бот отвечает заглушкой из шаблона правила,
карточка (интент, уверенность, собранные слоты, саммари) уходит в
`webhook_config` (CRM/таск-трекер). Перевод линии — в диалплане Asterisk:

```ini
exten => _X.,n,Dial(SIP/2000,30,tT)   ; после ответа бота «Соединяю...»
```

Подсказки живому оператору: `POST /api/v1/classifiers/copilot`
`{"query": "<реплика клиента>", "collection_ids": ["kb_id"]}` — карточки
знаний из HippoRAG без вызова LLM.

## 14. Stage 3: промышленная стабилизация

| Доработка | Что изменилось |
|---|---|
| Д1: атомарный граф | Трехфазная подмена каталога (current → backup, tmp → current) + `version.txt`; воркеры сверяют версию при каждом обращении и подхватывают новый граф без рестарта |
| Д2: PII в стриминге | `StreamingPIIUnmasker` держит незакрытый `[` в буфере до `]` (лимит 24 символа); размаскирование включено для direct-ответов и SSE-стрима LLM; вход LLM маскируется (`GUARDRAILS_PII_MASK_LLM_INPUT`) |
| Д3: slot-filling | Состояние в Redis (`slot_filling_state:*`, TTL 1800 c, fallback — память); стоп-слова отмены; escape при уверенном новом интенте без признаков слотов; 3 неудачные валидации → `HUMAN_HANDOFF` с собранными полями |
| Д4: файнтюн | `/v1/train` — подпроцесс `ProcessPoolExecutor(max_workers=1)` с `os.nice(10)`, HTTP 202, прогресс (эпохи/loss) в UI; Rollback Gate: подмена блокируется при ΔF1 < −0.02 (без `force_swap=true`), чекпоинт в `candidates/`; Anchor Dataset (`anchor_examples`) против забывания |
| Д5: SIP handoff | `GET /v1/call-status/{uuid}` + диалплан выше; раздельные метрики TTFB |
| Д6: Auto-Tune 15x | Двухфазная схема Zero Re-Retrieval: пулы ранков кэшируются один раз (`return_pool: true`), Optuna с `MedianPruner` считает Weighted RRF и прокси-метрики в памяти, Judge LLM вызывается только для топ-3 финалистов |
| Д7: сборка | Пины в `requirements-enterprise.txt`, `.nvmrc` (20.18.0) + `engines` в `package.json`, `scripts/bootstrap_models.py` (Zero-Config First Run) |
| Д8: строгий E2E | `E2E_STRICT=true` превращает skip в fail; 4 новых сценария (стрим-PII, slot-filling, Rollback Gate, call-status) |

Zero-Config First Run: `bash scripts/run_local_ui.sh` сам экспортирует
дефолтную модель `cointegrated/rubert-tiny2` (110 МБ) при отсутствии
`models/encoders/rubert-intents-v1/model.onnx` и регистрирует энкодер
(`--open-webui-url`, `--api-token`). Для compose: `docker compose -f
docker-compose.dev.yml --profile bootstrap run --rm bootstrap-models`.

Веса Auto-Tune: базовый контур сливает Dense и BM25 внутри ядра, поэтому
оптимизатор подбирает `w_hybrid` (контур целиком) и `w_hipporag`; раздельные
`w_dense`/`w_bm25` задаются конфигом ядра.

## 15. Stage 4: модульность каналов, MCP и AI-Ready

### Три автономных канала

| Канал | Ключ config / env | Компоненты |
|---|---|---|
| chat | channels.chat.enabled / ENABLE_CHANNEL_CHAT | /api/chat/completions, SSE, Reasoning Path |
| webrtc | channels.webrtc.enabled / ENABLE_CHANNEL_WEBRTC | WS /api/v1/voice/webrtc/stream (16 кГц PCM), VAD, barge-in, POST /offer |
| sip | channels.sip.enabled / ENABLE_CHANNEL_SIP | voice-sip-bridge :8765, /v1/call-status, диалплан |

Переключение на лету — из UI (Workspace → Intents → Channels & Modules)
или через MCP-инструмент platform_configure_modules.

### Канальная адаптация

Заголовки `X-Channel-Type: chat|webrtc|sip` (или metadata.channel) выбирают
канал; правила Intent Matrix фильтруются по колонке allowed_channels;
шаблоны получают `{{channel.type}}`; для голосовых каналов ответ
очищается от Markdown перед TTS (sanitize_for_voice_tts).

### Двусторонний MCP

- **Платформа = MCP Server**: POST /api/v1/mcp (JSON-RPC 2.0), GET /api/v1/mcp/sse.
  8 инструментов управления, 4 ресурса (ai-ready-schema, config, manifest, telemetry).
- **Платформа = MCP Client**: реестр внешних харнесов (таблица mcp_harness),
  действие правила `MCP_HARNESS` — после сбора слотов рендерит
  argument_mapping и вызывает tools/call внешнего сервера; режим
  autonomous_subagent передает goal_prompt и allowed_tools.

### AI-Ready

- `GET /llms.txt` — руководство для агентов (каналы, переменные, playbook).
- `GET /api/v1/ai-manifest.json` — дескрипторы всех настроек
  (purpose_for_ai, tuning_impact, recommended_by_channel) + текущий config.
- `POST /api/v1/classifiers/manifest/validate` — семантический линтер
  BotBehaviorManifest: 6 правил, ошибки с agent_remediation_hint.
- UI: вкладки Channels & Modules, MCP & Harnesses, редактор манифеста
  с Dry run и Simulate Turn. Кнопка Copy client config выдает готовый
  конфиг под выбранный клиент: Claude Code (.mcp.json, ключ mcpServers),
  Cursor (~/.cursor/mcp.json), Zcode (~/.zcode/cli/config.json, вложенный
  ключ mcp.servers; совместимый fallback ~/.agents/mcp.json -> mcpServers),
  Codex CLI (~/.codex/config.toml, секция [mcp_servers.*] с url и
  bearer_token_env_var).

### Локальный RAG без torch и внешних ключей (К3)

encoder-classifier отдает OpenAI-совместимые `/v1/embeddings` и
TEI-совместимый `/rerank` на rubert-embed-v1. В run_local_ui.sh
прописаны RAG_OPENAI_API_BASE_URL/RAG_EMBEDDING_MODEL/RAG_RERANKING_ENGINE —
векторный поиск работает из коробки.

### Стартовые модели (К2) и граф без hipporag (К4)

bootstrap_models.py экспортирует 4 модели: rubert-intents-v1,
rubert-nli-v1 (zero-shot через эмбеддинговую близость),
rubert-slots-v1 (NER + гибридный экстрактор паттернов: order_id, phone,
email, date, amount, card), rubert-embed-v1.
Без пакета hipporag графовый контур работает на встроенном
LightweightHippoGraph (networkx.DiGraph + nx.pagerank, сущности через
/v1/extract-slots) — UI графа, Reasoning Path и третий контур поиска
доступны на CPU-машине.

## 16. Stage 5: Zero Technical Debt

### Единый трехконтурный поиск

`weighted_rrf_three_way(dense, bm25, hipporag, w_dense, w_bm25, w_hipporag)` —
одна формула в боевом поиске и Auto-Tune. Телеметрия в metadata каждого чанка:
`fusion_score`, `rank_breakdown` (dense_rank/bm25_rank/hipporag_rank),
`applied_weights`. Применение весов: POST /ragas/runs/{id}/apply-weights
или кнопка Apply Best Weights в UI.

### Гибридный WebRTC offer

POST /api/v1/voice/webrtc/offer: с aiortc — честный SDP Answer + DataChannel
"events"; без — `{"type": "ws_fallback", "ws_url": ".../stream"}`.

### Локальный ONNX-судья

`RAGAS_JUDGE_ENGINE=local_onnx` в ragas-worker: Faithfulness (NLI + числовые
расхождения), Relevancy/Precision/Recall (эмбеддинги rubert-embed-v1).
Авто-режим: пробует Judge LLM, при недоступности — local_onnx.
Эндпоинт `POST /v1/evaluate/sync` — синхронная оценка одной пары.
sqlite-фоллбэк для БД воркера без Postgres.

### Локальный голос без speaches

`synthesize_pcm_fallback` (слогово-модулированный тональный PCM) + STT
circuit breaker (после ошибки соединения STT пропускается 60 с) + fallback
ответ при 401 от LLM. Голосовой цикл работает на CPU-машине.
