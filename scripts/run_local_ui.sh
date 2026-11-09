#!/usr/bin/env bash
# Локальный запуск интерфейса платформы на Windows/WSL без Docker.
# Требует: .venv-webui (backend/requirements-slim.txt + chromadb, pydub, audioop-lts,
# azure-storage-blob, google-cloud-storage) и .venv-encoder (services/encoder-classifier).
# Фронтенд должен быть собран: cd open-webui-fork && npm run build  ->  build/
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 0) Zero-Config First Run (Д7, Stage 3): стартовая ONNX-модель, если её нет
python "$ROOT/scripts/bootstrap_models.py" || echo "[bootstrap] пропущен (см. сообщение выше)"

# 1) Классификатор интентов (:8001)
(
  cd "$ROOT/services/encoder-classifier"
  MODELS_DIR="$ROOT/models/encoders" WARMUP_ON_STARTUP=false \
    "$ROOT/.venv-encoder/Scripts/python" -m uvicorn app:app --host 127.0.0.1 --port 8001 &
)

# 2) ragas-worker (:8002) — локальный ONNX-судья без GPU (Д4, Stage 5)
(
  cd "$ROOT/services/ragas-worker"
  DATABASE_URL="sqlite:///$ROOT/.cache/ragas_local.db"   REDIS_URL=""   RAGAS_JUDGE_ENGINE=local_onnx   CLASSIFIER_URL=http://127.0.0.1:8001   EMBEDDINGS_URL=http://127.0.0.1:8001/v1     "$ROOT/.venv-encoder/Scripts/python" -m uvicorn app:app --host 127.0.0.1 --port 8002 &
)

# 3) Ядро Open WebUI с собранным UI (:8080)
cd "$ROOT/open-webui-fork/backend"
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
export ENCODER_CLASSIFIER_SLOT_MODEL_ID=rubert-slots-v1
export ENABLE_OLLAMA_API=false
export ENABLE_OPENAI_API=false
export ENCODER_CLASSIFIER_URL=http://127.0.0.1:8001
exec "$ROOT/.venv-webui/Scripts/python" -m uvicorn open_webui.main:app --host 127.0.0.1 --port 8080
