# QA Pipeline: регламент приемочного тестирования

Часть 2 аудита, адаптирована под фактический API платформы. Пять контуров;
автоматизируемая часть — `tests/e2e_platform_verify.py`, остальное —
ручные процедуры по шагам.

## Запуск автотестов

```bash
# Стенд: docker compose -f docker-compose.dev.yml up -d + ядро Open WebUI на :8080
.venv-encoder/Scripts/python -m pytest tests/ -q            # юнит-тесты
.venv-encoder/Scripts/python -m pytest tests/e2e_platform_verify.py -v
```

Переменные окружения автотестов: `E2E_WEBUI_URL`, `E2E_CLASSIFIER_URL`,
`E2E_RAGAS_URL`, `E2E_VOICE_WS`, `E2E_API_TOKEN`, `E2E_INTENT_MODEL`,
`E2E_NLI_MODEL`, `E2E_SLOT_MODEL`. Тесты без стенда пропускаются (skip), не падают.

---

## Контур 1. Инфраструктура, БД и отказоустойчивость

### TC-INF-01. Старт контейнеров и миграции
1. `docker compose -f docker-compose.production.yml up -d --build`.
2. `docker compose exec postgres psql -U postgres -d openwebui -c "\dt"`.
**Ожидается:** контейнеры `Up (healthy)`; таблицы `encoder_model`,
`intent_route`, `ragas_dataset`, `ragas_run`, `ragas_sample`,
`intent_feedback` (7 таблиц с учётом двух миграций).

### TC-INF-02. Плавная деградация
1. `docker compose stop encoder-classifier` → отправить сообщение в чат.
   **Ожидается:** `WARNING Encoder classifier unavailable` в логах open-webui,
   штатный ответ LLM (HTTP 200).
2. Вернуть классификатор, выключить `ENABLE_HIPPORAG` (или `hipporag.enable` в UI-конфиге) →
   поисковый запрос. **Ожидается:** поиск по Dense + BM25, перерасчет весов RRF,
   ответ корректен.

---

## Контур 2. Классификатор и маршрутизация

### TC-CLS-01. Экспорт модели и latency
1. `python services/encoder-classifier/scripts/export_onnx.py --model ai-forever/ruBert-base --task sequence_classification --labels "billing_refund,order_status,tech_support,greeting" --quantize --out models/encoders/rubert-intents-v1`
2. `curl -X POST http://localhost:8001/v1/classify -H "Content-Type: application/json" -d "{\"text\":\"Верните деньги за списание по подписке!\",\"model_id\":\"rubert-intents-v1\"}"`.
**Ожидается:** 200; `intent_name == "billing_refund"`; `latency_ms <= 15` (GPU)
или `<= 35` (CPU); сумма вероятностей top-K <= 1.0.

### TC-CLS-02. Четыре типа действий Intent Matrix
Предусловие: правила для бота с порогом 0.80:
`greeting → DIRECT_RESPONSE` (шаблон `«Здравствуйте! Вы обратились по теме {{intent}}.»`),
`order_status → WEBHOOK_TOOL` (`http://httpbin.org/post`),
`billing_refund → RAG_COLLECTION` (только `kb_billing`, HippoRAG вкл.),
`tech_support → SYSTEM_PROMPT_OVERRIDE`.
**Ожидается:**
1. `greeting` — ответ мгновенно, 0 токенов LLM, текст из шаблона;
2. `order_status` — POST во внешний вебхук, ответ без галлюцинаций;
3. `billing_refund` — в metadata трейс только чанки `kb_billing`;
4. `p < tau_low` — fallback на все коллекции бота.

### TC-CLS-03. Zero-shot и слоты
1. `/v1/zero-shot` с метками `["жалоба на курьера", "смена тарифа", "закрытие счета"]` — ранжирование по NLI.
2. `/v1/extract-slots` с `"Проверьте заказ № 94821 от 15 мая на имя Иванов Иван"` — спаны `text/label/start/end`.

### TC-CLS-04 (Д1/Ф2). Слоты в шаблонах и slot-filling
1. Правило `return_request → WEBHOOK_TOOL` с `required_slots`:
   `order_id` (regex `\d{5,8}`), `reason` (string).
2. «Хочу вернуть товар» → **Ожидается:** детерминированный вопрос
   «Подскажите номер заказа...» без LLM; `metadata.slot_filling.missing = [order_id, ...]`.
3. «Заказ 94821, причина — брак» → **Ожидается:** вебхук получил
   `slots.order_id = "94821"`, ответ из CRM.

### TC-CLS-05 (Д2). Активная дизамбигуация
Фраза в серой зоне с Δp < 0.15 → **Ожидается:** в системном промпте появилась
инструкция уточнения (`metadata.disambiguation` заполнен), бот задал один
уточняющий вопрос.

---

## Контур 3. Трехконтурный RAG и HippoRAG

### TC-RAG-01. Построение графа
1. База `MultiHop_Test_KB`, три документа (Орион-7 / Смирнов / лаборатория-регламент).
2. Knowledge → карточка → «Knowledge Graph (HippoRAG)».
**Ожидается:** файлы графа в `${DATA_DIR}/hipporag_storage/<id>`; в UI узлы
`Орион-7`, `Алексей Смирнов`, `Лаборатория...`, `Регламент № 404-ФЗ`.

### TC-RAG-02. Multi-hop сравнение
Вопрос «Какому регламенту подчиняется лаборатория, которую возглавлял гендиректор Орион-7?»
**Ожидается:** без HippoRAG — только Документ А; с HippoRAG — все 3 документа,
у чанков метка `retrieval_source: hipporag_ppr`.

### TC-RAG-03 (Ф6). Reasoning Path
В графе: тестовый запрос → «Show Reasoning Path».
**Ожидается:** цепочка `[Орион-7] → [Алексей Смирнов] → [Лаборатория...] → [Регламент № 404-ФЗ]`
с текстом доказательства.

### TC-RAG-04 (Ф4). Auto-Tune
Admin → RAG Quality → Runs → New Evaluation Run → режим «Auto-Optimize».
**Ожидается:** run `optimize` со `metrics.best_params` (веса, threshold, top_k)
и `baseline_score → best_score`; применённые веса видны в чанках (`fusion_score`).

---

## Контур 4. Ragas

### TC-EVAL-01. Batch Benchmark
Golden Datasets → CSV 20 пар → New Evaluation Run (Benchmark).
**Ожидается:** статус `completed`; 4 метрики в [0,1]; детализация по вопросам.

### TC-EVAL-02. Live Sampling и Hallucination Alert
1. `RAGAS_LIVE_SAMPLE_RATE=1.0`, порог 0.7.
2. Вызвать галлюцинацию (промптом заставить бота выдумать факт).
3. Через 5–10 с: Admin → RAG Quality → Hallucination Alerts.
**Ожидается:** диалог в списке с `faithfulness < 0.7`.
Примечание: ПДн в сэмплах замаскированы (`[PHONE_1]`, `[CARD_1]`).

---

## Контур 5. Голос

### TC-VOICE-01. AudioSocket и TTFB
1. Asterisk: `AudioSocket(${UUID4},host:8765)`; тестовым клиентом — PCM 16-bit 8 кГц.
2. Реплика 2.5 с → замер от конца речи до первого аудио-кадра ответа.
**Ожидается:** `TTFB_ms <= 650` в логах bridge; звук без дребезга
(soxr-ресемплинг; проверяется и тестом `test_tone_frequency_preserved_after_resample`).
При задержке: через 250 мс слышен филлер «Секунду, проверяю информацию.»

### TC-VOICE-02. Barge-in
Реплика «Стоп, переключи на другой вопрос!» на 3-й секунде ответа.
**Ожидается:** ≤ 150 мс — старый аудиопоток прекращен (`Task.cancel()`),
новая реплика распознана, ответ по ней.

### TC-VOICE-03 (Ф3). Эмоции и деэскалация
Раздраженная громкая речь 2 хода подряд (`anger_score >= 0.75`).
**Ожидается:** в системный промпт добавлена деэскалация
(`metadata.voice_deescalated = true`), переменная `{{voice.anger_score}}`
доступна в шаблонах правил.

### TC-VOICE-04 (Ф7). Handoff
Правило `HUMAN_HANDOFF` (или срабатывание по гневу) → **Ожидается:** ответ-заглушка
«Соединяю с оператором», карточка (интент, слоты, саммари) ушла в CRM-вебхук;
SIP-перевод — Dial() в диалплане Asterisk (см. DEPLOYMENT.md).

---

## Stage 3: кейсы промышленной стабилизации

### TC-S3-01. Zero-Config First Run (Д7, DoD 1)
1. Удалить `models/encoders/rubert-intents-v1/`, запустить `bash scripts/run_local_ui.sh`.
2. Открыть Workspace → Intents → Encoder Models → Test Inference.
**Ожидается:** HTTP 200 с распределением вероятностей и `latency_ms` (bootstrap экспортировал rubert-tiny2 и зарегистрировал энкодер), а не 404.

### TC-S3-02. Атомарность графа на Windows и Linux (Д1, DoD 2)
1. Загрузить файлы в базу с включенным HippoRAG, не дожидаясь переиндексации — загрузить еще раз.
**Ожидается:** нет `FileExistsError`/`OSError` в логах; в каталоге графа появился `version.txt`; воркеры отдают чанки новой версии без рестарта.

### TC-S3-03. PII в потоковой генерации (Д2, DoD 3)
1. Включить `GUARDRAILS_PII_MASK_LLM_INPUT=true`, задать вопрос с номером карты; бот отвечает со стримингом.
**Ожидается:** в UI ни одного сырого тега `[CARD_1]`/`[PHONE_1]` даже при посимвольной нарезке; юнит-тест `test_split_token_restored_across_three_chunks` зеленый.

### TC-S3-04. Slot-Filling и файнтюн (Д3+Д4, DoD 4)
1. Начать сбор слотов, сказать «Отмена, сколько стоит доставка?» — форма сброшена, бот отвечает на новый вопрос.
2. Трижды ввести номер заказа без цифр — бот переводит на оператора с собранными полями.
3. Запустить файнтюн и параллельно гонять Test Inference — латентность классификации не деградирует (обучение в подпроцессе).
4. Сминтить падение F1 (8 шумных примеров) — подмена блокируется, чекпоинт в `candidates/`, warning в UI.

### TC-S3-05. SIP-эскалация (Д5, DoD 5)
1. Позвонить, разозлить бота (или сработать правилу HUMAN_HANDOFF).
2. После выхода из AudioSocket диалплан запрашивает `GET /v1/call-status/{uuid}`.
**Ожидается:** ответ — имя очереди (`support_queue`/`vip_support`); свободный звонок — `NONE` + Hangup.

### TC-S3-06. Строгий E2E (Д8, DoD 6)
`E2E_STRICT=true pytest tests/e2e_platform_verify.py -v` на полном стенде: **0 failed, 0 skipped**. Любой упавший контейнер красит пайплайн.

---

## Stage 4: каналы, MCP, AI-Ready

### TC-S4-01. Автономность каналов (DoD 2)
1. Отключить channels.chat.enabled через MCP platform_configure_modules.
2. Проверить ai-manifest.json: chat=false, webrtc/sip без изменений.
3. Включить обратно.
**Ожидается:** тумблер в UI Channels & Modules меняет состояние без рестарта; E2E test_20 зеленый.

### TC-S4-02. MCP Control Plane (DoD 3)
1. Подключить Claude Code/Cursor: конфиг из кнопки Copy client config.
2. tools/list — 8 инструментов; bot_apply_behavior_manifest с dry_run=true на манифесте с необъявленным слотом.
**Ожидается:** initialize/tools/list/tools/call работают; линтер возвращает UNDECLARED_SLOT_IN_TEMPLATE + agent_remediation_hint. E2E test_17.

### TC-S4-03. Внешний MCP-харнес (DoD 3)
1. MCP & Harnesses → Add Harness (stdio или HTTP).
2. Discover Tools → привязать инструмент к правилу с action_type=MCP_HARNESS.
3. Прогнать диалог с заполнением required_slots.
**Ожидается:** tools/list кэшируется; после сбора слотов вызывается инструмент внешнего сервера; ответ форматируется пользователю. E2E test_18.

### TC-S4-04. WebRTC без SIP (DoD 2)
1. channels.sip.enabled=false, webrtc=true.
2. Test WebRTC Call in Browser (кнопка в Channels & Modules).
**Ожидается:** голосовой диалог из браузера работает без моста и Asterisk; лог TTFB. E2E test_19.

### TC-S4-05. AI-Ready обнаружение (DoD 4)
1. GET /llms.txt (публично), GET /api/v1/ai-manifest.json (по токену).
**Ожидается:** руководство содержит каналы, MCP, playbook; манифест — дескрипторы всех настроек с purpose_for_ai и tuning_impact. E2E test_16.

### TC-S4-06. Три веса Auto-Tune (К1)
1. Прогнать Auto-Optimize на датасете.
2. Проверить metrics.best_params.
**Ожидается:** w_dense, w_bm25, w_hipporag — три независимых значения с суммой 1.0; пулы ранков кэшируются один раз (return_pool). Юнит test_k1_three_weight_rrf_separation.

---

## Stage 5: Zero Technical Debt

### TC-S5-01. Единый RRF в бою и оптимизаторе (Д1)
1. Прогнать Auto-Optimize, применить веса кнопкой Apply Best Weights.
2. Выполнить поисковый запрос, проверить metadata чанков.
**Ожидается:** `applied_weights` в чанках совпадает с best_params; `rank_breakdown` заполнен; юнит test_dedup_across_contours.

### TC-S5-02. Гибридный WebRTC (Д2)
1. POST /api/v1/voice/webrtc/offer с SDP.
**Ожидается:** transport=webrtc_sdp с aiortc, ws_fallback без. Юнит test_webrtc_sdp_offer_and_fallback_contract.

### TC-S5-03. Семантическая прокси (Д3)
**Ожидается:** синонимичные чанки с нулевым лексическим пересечением получают высокий Semantic Recall. Юнит test_synonyms_rank_high_without_lexical_overlap.

### TC-S5-04. Локальный судья без GPU (Д4)
1. POST ragas-worker /v1/evaluate/sync с галлюцинирующим ответом.
**Ожидается:** faithfulness < 0.3; на верном ответе > 0.75; judge_engine = LocalONNXRagasJudge. E2E test_22.

### TC-S5-05. Zero-skip E2E (DoD)
**Ожидается:** E2E_STRICT=true pytest tests/e2e_platform_verify.py — **22 passed, 0 failed, 0 skipped** на 4 живых сервисах.
