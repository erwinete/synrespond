#!/usr/bin/env bash
# Единый кроссплатформенный запуск всех 4 контуров synrespond
# (Windows Git Bash / MSYS2 / WSL / Linux). PERF-OPS-01.
#
# Требует: .venv-webui (backend/requirements-slim.txt + chromadb, pydub,
# audioop-lts, azure-storage-blob, google-cloud-storage) и .venv-encoder
# (services/encoder-classifier + ragas-worker). Фронтенд должен быть собран:
#   cd open-webui-fork && npm run build  ->  build/
set -e

RAW_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# PERF-OPS-01: в Git Bash / MSYS2 приводим /f/... к F:/..., чтобы Windows
# Python и SQLite открывали пути без ошибок вида sqlite:////f/...
if command -v cygpath >/dev/null 2>&1; then
  ROOT_PATH="$(cygpath -m "$RAW_ROOT")"
elif (cd "$RAW_ROOT" && pwd -W) >/dev/null 2>&1; then
  ROOT_PATH="$(cd "$RAW_ROOT" && pwd -W | tr '\\' '/')"
else
  ROOT_PATH="$RAW_ROOT"
fi

echo "[synrespond] Рабочий корень: $ROOT_PATH"

mkdir -p "$RAW_ROOT/.cache"
mkdir -p "$RAW_ROOT/open-webui-fork/backend/data"

# Пути к интерпретаторам: Windows Scripts/python.exe или POSIX bin/python
if [ -x "$RAW_ROOT/.venv-encoder/Scripts/python.exe" ]; then
  PY_ENC="$RAW_ROOT/.venv-encoder/Scripts/python.exe"
elif [ -x "$RAW_ROOT/.venv-encoder/bin/python" ]; then
  PY_ENC="$RAW_ROOT/.venv-encoder/bin/python"
else
  PY_ENC="python3"
fi

if [ -x "$RAW_ROOT/.venv-webui/Scripts/python.exe" ]; then
  PY_WEBUI="$RAW_ROOT/.venv-webui/Scripts/python.exe"
elif [ -x "$RAW_ROOT/.venv-webui/bin/python" ]; then
  PY_WEBUI="$RAW_ROOT/.venv-webui/bin/python"
else
  PY_WEBUI="python3"
fi

# Чистое гашение всех фоновых сервисов по EXIT/INT/TERM
PIDS=()
cleanup() {
  echo "[synrespond] Остановка фоновых сервисов..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# 0) Bootstrap стартовых ONNX-моделей (если еще не экспортированы)
"$PY_ENC" "$RAW_ROOT/scripts/bootstrap_models.py" || echo "[bootstrap] пропуск экспорта (модели уже на диске)"

# 1) Микросервис encoder-classifier (:8001)
cd "$RAW_ROOT/services/encoder-classifier"
MODELS_DIR="$ROOT_PATH/models/encoders" WARMUP_ON_STARTUP=true \
  "$PY_ENC" -m uvicorn app:app --host 127.0.0.1 --port 8001 &
PIDS+=($!)

# 2) Микросервис ragas-worker (:8002) с кроссплатформенным путем SQLite
cd "$RAW_ROOT/services/ragas-worker"
DATABASE_URL="sqlite:///$ROOT_PATH/.cache/ragas_local.db" \
REDIS_URL="" \
RAGAS_JUDGE_ENGINE=local_onnx \
CLASSIFIER_URL=http://127.0.0.1:8001 \
EMBEDDINGS_URL=http://127.0.0.1:8001/v1 \
  "$PY_ENC" -m uvicorn app:app --host 127.0.0.1 --port 8002 &
PIDS+=($!)

# 3) Голосовой шлюз телефонии voice-sip-bridge (:8765)
cd "$RAW_ROOT/services/voice-sip-bridge"
OPEN_WEBUI_URL=http://127.0.0.1:8080 \
SPEACHES_URL=http://127.0.0.1:8003/v1 \
AUDIOSOCKET_PORT=8765 \
  "$PY_ENC" -m uvicorn server:app --host 127.0.0.1 --port 8765 &
PIDS+=($!)

# 4) Фоновый сидер платформы: дождется :8080 и идемпотентно создаст
# админа, зарегистрирует 4 ONNX-энкодера и демо-датасет (PERF-OPS-02)
"$PY_ENC" "$RAW_ROOT/scripts/seed_platform.py" \
  --webui-url "http://127.0.0.1:8080" \
  --models-dir "$RAW_ROOT/models/encoders" &
PIDS+=($!)

# 5) Ядро Open WebUI с собранным UI (:8080)
cd "$RAW_ROOT/open-webui-fork/backend"
export DATA_DIR="$ROOT_PATH/open-webui-fork/backend/data"
export WEBUI_SECRET_KEY="${WEBUI_SECRET_KEY:-enterprise-local-dev-key}"
export OPEN_WEBUI_URL="${OPEN_WEBUI_URL:-http://127.0.0.1:8080}"
export SPEACHES_URL="${SPEACHES_URL:-http://127.0.0.1:8003/v1}"
export RAG_EMBEDDING_ENGINE=openai
# К3 (Stage 4): локальный эмбеддер и реранкер на encoder-classifier —
# RAG работает из коробки без torch в ядре и без внешних API-ключей
export RAG_OPENAI_API_BASE_URL=http://127.0.0.1:8001/v1
export RAG_EMBEDDING_MODEL=rubert-embed-v1
export RAG_RERANKING_ENGINE=external
export RAG_EXTERNAL_RERANKER_URL=http://127.0.0.1:8001/rerank
export ENCODER_CLASSIFIER_URL=http://127.0.0.1:8001
export ENCODER_CLASSIFIER_MODEL_ID=rubert-intents-v1
export ENCODER_CLASSIFIER_SLOT_MODEL_ID=rubert-slots-v1
export RAGAS_WORKER_URL=http://127.0.0.1:8002
# PERF-CORE-01/SEC-02: dev-режим допускает вебхуки на localhost
export ALLOW_LOCALHOST_WEBHOOKS=true
export ENABLE_OLLAMA_API=false
export ENABLE_OPENAI_API=false

"$PY_WEBUI" -m uvicorn open_webui.main:app --host 127.0.0.1 --port 8080
