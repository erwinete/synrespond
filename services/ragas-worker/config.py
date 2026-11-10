"""Конфигурация ragas-worker из переменных окружения."""

import os
import re
from pathlib import Path


def _db_url() -> str:
    url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@postgres:5432/openwebui').strip()
    # Д4 (Stage 5): sqlite-фоллбэк для локального стенда без Postgres
    if url.startswith('sqlite'):
        # PERF-OPS-01: авто-исправление путей Git Bash вида
        # sqlite:////f/Z/... -> sqlite+aiosqlite:///F:/Z/...
        url = re.sub(
            r'^sqlite(\+aiosqlite)?:////([a-zA-Z])/',
            lambda m: f"sqlite+aiosqlite:///{m.group(2).upper()}:/",
            url,
        )
        if url.startswith('sqlite://') and not url.startswith('sqlite+aiosqlite://'):
            url = url.replace('sqlite://', 'sqlite+aiosqlite://', 1)
        raw_path = url.split('sqlite+aiosqlite:///', 1)[-1]
        if raw_path and raw_path != ':memory:':
            Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
        return url
    # SQLAlchemy async-драйверу нужен явный dialect
    if url.startswith('postgresql://'):
        url = url.replace('postgresql://', 'postgresql+asyncpg://', 1)
    return url


DATABASE_URL = _db_url()

REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/1')
LIVE_QUEUE_KEY = os.getenv('RAGAS_LIVE_QUEUE_KEY', 'ragas:live_samples')

JUDGE_LLM_URL = os.getenv('JUDGE_LLM_URL', 'http://vllm-server:8000/v1').rstrip('/')
JUDGE_LLM_MODEL = os.getenv('JUDGE_LLM_MODEL', 'Qwen/Qwen2.5-32B-Instruct-AWQ')
JUDGE_API_KEY = os.getenv('JUDGE_API_KEY', 'none')

EMBEDDINGS_URL = os.getenv('EMBEDDINGS_URL', 'http://tei-embeddings:80/v1').rstrip('/')
EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'BAAI/bge-m3')

OPEN_WEBUI_URL = os.getenv('OPEN_WEBUI_URL', 'http://open-webui:8080').rstrip('/')
OPEN_WEBUI_TOKEN = os.getenv('OPEN_WEBUI_TOKEN', '')

RAGAS_HALLUCINATION_THRESHOLD = float(os.getenv('RAGAS_HALLUCINATION_THRESHOLD', '0.7'))
BATCH_MAX_ITEMS = int(os.getenv('RAGAS_BATCH_MAX_ITEMS', '200'))

CONSUMER_ENABLED = os.getenv('RAGAS_CONSUMER_ENABLED', 'True').lower() == 'true'

# Д4 (Stage 5): движок судьи — llm_judge | local_onnx | auto (llm, при недоступности local_onnx)
RAGAS_JUDGE_ENGINE = os.getenv('RAGAS_JUDGE_ENGINE', 'auto')

# Д3 (Stage 5): encoder-classifier для семантической прокси и local-судьи
CLASSIFIER_URL = os.getenv('CLASSIFIER_URL', 'http://encoder-classifier:8001').rstrip('/')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'rubert-embed-v1')
