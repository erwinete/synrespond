# Enterprise AI Platform

Self-hosted платформа текстовых и голосовых ботов на базе Open WebUI: маршрутизация
интентов на энкодерах за 5–20 мс, трехконтурный гибридный RAG (Dense + BM25 +
HippoRAG) и непрерывный контроль качества по фреймворку Ragas.

Собрано по ТЗ (`ai_open_webui.md`, разделы 1–10).

## Состав монорепозитория

```text
enterprise-ai-platform/
├── open-webui-fork/            # Форк github.com/open-webui/open-webui (ветка feat/enterprise-platform)
│   ├── backend/open_webui/
│   │   ├── models/intents.py           # [NEW] encoder_model, intent_route
│   │   ├── models/ragas_evals.py       # [NEW] ragas_dataset, ragas_run, ragas_sample
│   │   ├── retrieval/hipporag_engine.py# [NEW] OpenIE + Personalized PageRank
│   │   ├── retrieval/query_rewriter.py # [NEW] деконтекстуализация + декомпозиция
│   │   ├── retrieval/hybrid_fusion.py  # [NEW] Weighted RRF трех контуров
│   │   ├── utils/intent_router.py      # [NEW] клиент классификатора, direct-response, Ragas-сэмплы
│   │   ├── routers/classifiers.py      # [NEW] /api/v1/classifiers (+ инспектор графа)
│   │   ├── routers/ragas.py            # [NEW] /api/v1/ragas
│   │   └── migrations/versions/e7a1f4c2b9d8_*.py  # [NEW] 5 таблиц
│   └── src/                            # SvelteKit: Intents, HippoRAGGraphViewer, Ragas дашборд
├── services/
│   ├── encoder-classifier/     # FastAPI + ONNX Runtime: /v1/classify, /v1/zero-shot, /v1/extract-slots
│   ├── ragas-worker/           # Оценка метрик Ragas, очередь Redis, запись в общую БД
│   └── voice-sip-bridge/       # Asterisk AudioSocket / WebSocket -> Open WebUI (VAD + barge-in)
├── pipelines-custom/           # Inlet-фильтр маршрутизации для рантайма open-webui/pipelines
├── models/encoders/            # Каталог ONNX-моделей (монтируется в /models)
├── deploy/                     # Документы развертывания и эксплуатации
├── docker-compose.production.yml  # Полный стек на GPU-сервере
├── docker-compose.dev.yml         # Локальная разработка без GPU
└── .env.example
```

## Схема запроса

```text
[Чат / Браузер WebRTC / SIP-звонок]
   → [encoder-classifier: интент за 5–20 мс]
       ├─ direct + DIRECT_RESPONSE/WEBHOOK_TOOL → ответ без вызова LLM
       └─ RAG → [Query Rewriter]
             → Dense + BM25 + HippoRAG PPR → Weighted RRF (0.4/0.2/0.4)
             → BGE-Reranker-v2-m3 → LLM (vLLM)
   → [ragas-worker: Faithfulness / Relevancy; флаг Hallucination Alert при < 0.7]
```

## Быстрый старт (продакшн, GPU-сервер)

1. Скопируйте переменные окружения и заполните секреты:

   ```bash
   cp .env.example .env
   ```

2. Запустите стек:

   ```bash
   docker compose -f docker-compose.production.yml up -d --build
   ```

3. Экспортируйте модель интентов и положите в `models/encoders/`:

   ```bash
   python services/encoder-classifier/scripts/export_onnx.py \
     --model ai-forever/ruBert-base --task sequence_classification \
     --labels "billing_refund,order_status,greeting" --out models/encoders/rubert-intents-v1
   ```

4. Откройте `http://<host>:8080`. Разделы:
   - **Workspace → Intents** — энкодеры, тест инференса, матрица маршрутизации;
   - **Knowledge → карточка базы → меню → Knowledge Graph (HippoRAG)** — граф и тестовый PPR;
   - **Admin → RAG Quality** — метрики Ragas, Golden Datasets, прогоны, алерты.

Подробные шаги, настройка Asterisk/FreeSWITCH и эксплуатация — в [deploy/DEPLOYMENT.md](deploy/DEPLOYMENT.md).

## Локальная разработка (Windows, без GPU)

```bash
# Инфраструктура и сервисы
docker compose -f docker-compose.dev.yml up -d

# Ядро Open WebUI (форк) — backend
cd open-webui-fork/backend
python -m venv ../.venv-webui && ../.venv-webui/Scripts/pip install -r requirements.txt
uvicorn open_webui.main:app --port 8080

# Фронтенд
cd ../
npm install
npm run dev
```

Тесты сервисов:

```bash
.venv-encoder/Scripts/python -m pytest services/encoder-classifier/tests -q
```

## Критерии приемки по этапам ТЗ

| Этап | Критерий | Где проверить |
|---|---|---|
| 1. Классификатор | Классификация 100+ интентов ≤ 15 мс без LLM; детерминированный интент отвечает напрямую | `encoder-classifier` + `utils/intent_router.py`; UI Intents → Test Inference |
| 2. HippoRAG | Прирост Context Recall на multi-hop вопросах 15–25% | Admin → RAG Quality: прогоны с `use_hipporag` вкл/выкл |
| 3. Ragas | Автоматические метрики и подсветка галлюцинаций | Admin → RAG Quality: Overview / Alerts |
| 4. Голос | TTFB ≤ 650 мс, barge-in | `voice-sip-bridge`; логи TTFB по каждой реплике |

## Доработки по аудиту (Часть 1)

| № | Доработка | Реализация |
|---|---|---|
| Д1 | Слоты в шаблонах и вебхуках | `/v1/analyze` (интент+NER одним вызовом), `{{slots.<key>}}` в шаблонах, `utils/route_templates.py` |
| Д2 | Активная дизамбигуация серой зоны | Δp < 0.15 → уточняющая инструкция в system prompt (`metadata.disambiguation`) |
| Д3 | Voice Fast-Path ≤ 650 мс | `X-Voice-Mode` → скип LLM-рерайтера, NER для PPR через ONNX (8 мс), филлер через 250 мс |
| Д4 | Акустический ресемплинг | soxr HQ (анти-алиасинг) с fallback на линейную интерполяцию |
| Д5 | Гонки при перестроении графа | Redis-лок `hipporag_lock:<id>` + запись в `<id>_tmp` + атомарный `os.rename` swap |

## Киллер-фичи (Часть 3 аудита)

| № | Фича | Где смотреть |
|---|---|---|
| Ф1 | Active Learning Flywheel | Workspace → Intents → Active Learning: серая зона с LLM-подсказками, «Fine-Tune & Hot-Swap ONNX» (torch-файнтюн + оценка F1 + атомарная подмена) |
| Ф2 | Декларативный slot-filling | Колонка Required Slots в правиле: имя / regex / вопрос; детерминированный сбор до выстрела вебхука |
| Ф3 | Детектор эмоций голоса | `voice-sip-bridge/emotion.py` (ONNX или акустический прокси) → `X-Voice-Anger` → `{{voice.anger_score}}`, авто-деэскалация после 2 ходов |
| Ф4 | Auto-Tune RAG | Admin → RAG Quality → Runs → режим «Auto-Optimize» (Optuna: w_dense/w_bm25/w_hippo/threshold/top_k → J = 0.4·Recall + 0.3·Precision + 0.3·Faithfulness); переопределение на запрос — `metadata.rag_overrides` |
| Ф5 | Guardrails + Reversible PII Vault | `/v1/guard` (ONNX или эвристика) — отказ на джейлбрейк без LLM; маскирование ПДн в логах/Ragas/вебхуках с восстановлением (`utils/pii_vault.py`) |
| Ф6 | Graph-Grounded Reasoning Trace | Knowledge Graph → «Show Reasoning Path»: цепочка `[сущность] —ребро→ [узел]` + текст доказательства |
| Ф7 | Warm Handoff + Agent Copilot | Действие `HUMAN_HANDOFF` (карточка: интент, слоты, саммари → CRM-вебхук), `POST /api/v1/classifiers/copilot` — карточки знаний оператору без LLM |

## Тестирование

```bash
.venv-encoder/Scripts/python -m pytest tests/ -q                       # юнит-тесты (30+)
.venv-encoder/Scripts/python -m pytest tests/e2e_platform_verify.py -v # e2e по живому стенду
```

Регламент приемки по 5 контурам — [tests/QA_PIPELINE.md](tests/QA_PIPELINE.md).

## Ветки и коммиты

Изменения форка живут в ветке `feat/enterprise-platform` (`open-webui-fork/`).
Инфраструктура и сервисы коммитятся в корневой репозиторий монорепозитория.
