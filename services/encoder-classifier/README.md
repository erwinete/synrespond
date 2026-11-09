# encoder-classifier — классификация интентов и слотов на энкодерах

Микросервис инференса ONNX-энкодеров для маршрутизации без вызовов LLM (ТЗ, раздел 4.1).
Целевая задержка классификации — 5–20 мс.

## Задачи сервиса

| Эндпоинт | Задача | Модель |
|---|---|---|
| `POST /v1/classify` | Multi-class / multi-label классификация интента | `AutoModelForSequenceClassification` (ruBERT, DeBERTa, XLM-R) |
| `POST /v1/classify/batch` | Пакетная классификация | то же |
| `POST /v1/zero-shot` | NLI-классификация по меткам из запроса (новые интенты без переобучения) | NLI-модель (mDeBERTa-v3-XNLI) |
| `POST /v1/extract-slots` | NER / заполнение слотов (номер заказа, дата, сумма) | `AutoModelForTokenClassification` |

Правило порогов в ответе (`action`): `direct` при `p >= tau_high`, `disambiguate` в зоне
неопределенности, `fallback` ниже `tau_low`. Пороги берутся из `label_map.json`
модели или из параметров запроса.

## Формат каталога модели

`MODELS_DIR/<model_id>/`:

```text
model.onnx          # экспорт: services/encoder-classifier/scripts/export_onnx.py
tokenizer.json      # файлы токенизатора AutoTokenizer
label_map.json      # метаданные (см. ниже)
```

`label_map.json`:

```json
{
  "task": "sequence_classification",
  "labels": {"0": "billing_refund", "1": "order_status"},
  "max_length": 512,
  "thresholds": {"high": 0.85, "low": 0.50}
}
```

Для `zero_shot_nli` добавляются `hypothesis_template` и `entailment_id`;
для `token_classification` — `aggregation_threshold`.

## Локальный запуск

```bash
python -m venv .venv-encoder && .venv-encoder/Scripts/pip install -r services/encoder-classifier/requirements.txt
set MODELS_DIR=F:\Z\chatbot\enterprise-ai-platform\models\encoders
.venv-encoder/Scripts/uvicorn app:app --port 8001   # из каталога сервиса
```

Тесты: `.venv-encoder/Scripts/python -m pytest services/encoder-classifier/tests -q`

## Экспорт модели

```bash
python scripts/export_onnx.py --model ai-forever/ruBert-base \
  --task sequence_classification --labels "billing_refund,order_status,greeting" \
  --quantize --out ../../models/encoders/rubert-intents-v1
```

Переменные окружения: `MODELS_DIR`, `ONNX_PROVIDERS` (`auto`, `CUDAExecutionProvider,CPUExecutionProvider`), `MAX_LENGTH`, `ORT_INTRA_OP_THREADS`, `WARMUP_ON_STARTUP`.

## Роли в платформе

- Форк Open WebUI вызывает `/v1/classify` из `utils/intent_router.py` на каждом сообщении.
- Рантайм `pipelines` использует `pipelines-custom/intent_router_inlet.py` (альтернатива форку).
