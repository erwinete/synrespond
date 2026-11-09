"""Bootstrap стартовой ONNX-модели интентов (Д7, Stage 3).

Zero-Config First Run: проверяет models/encoders/rubert-intents-v1/model.onnx.
Если каталога нет — экспортирует легковесную дефолтную модель
cointegrated/rubert-tiny2 (~110 МБ) с демо-набором интентов и
(опционально) регистрирует энкодер через REST API Open WebUI, чтобы
кнопка Test Inference работала сразу после первого запуска.

Использование:
  python scripts/bootstrap_models.py --open-webui-url http://127.0.0.1:8080 \
      --api-token <OPEN_WEBUI_SERVICE_TOKEN>

Зависимости экспорта: torch (CPU-сборка), optimum[exporters], onnx,
transformers. При отсутствии скрипт предложит команду установки.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / 'models' / 'encoders'
DEFAULT_MODEL_ID = 'rubert-intents-v1'
DEFAULT_SOURCE = 'cointegrated/rubert-tiny2'
DEFAULT_LABELS = 'greeting,order_status,billing_refund,tech_support'
TAU_HIGH, TAU_LOW = 0.85, 0.50


def ensure_train_deps() -> bool:
    """Проверяет зависимости экспорта; при необходимости предлагает установку."""
    try:
        import optimum  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        print(
            'Нет зависимостей экспорта. Установите (CPU-сборка torch, ~200 МБ):\n'
            '  pip install torch --index-url https://download.pytorch.org/whl/cpu\n'
            '  pip install "optimum[exporters]>=1.20" onnx scikit-learn'
        )
        return False


def model_exists(model_id: str) -> bool:
    return (MODELS_DIR / model_id / 'model.onnx').exists()


def export_default(model_id: str, source: str, labels: str) -> None:
    script = Path(__file__).resolve().parents[1] / 'services' / 'encoder-classifier' / 'scripts' / 'export_onnx.py'
    out = MODELS_DIR / model_id
    cmd = [
        sys.executable,
        str(script),
        '--model', source,
        '--task', 'sequence_classification',
        '--labels', labels,
        '--out', str(out),
        '--tau-high', str(TAU_HIGH),
        '--tau-low', str(TAU_LOW),
    ]
    print('Экспорт дефолтной модели:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def register_encoder(model_id: str, base_url: str, token: str) -> None:
    import httpx

    resp = httpx.post(
        f'{base_url.rstrip("/")}/api/v1/classifiers/encoders/create',
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
        json={
            'name': 'Intents v1 (bootstrap)',
            'task': 'sequence_classification',
            'model_path': model_id,
            'endpoint_url': None,
            'is_active': True,
        },
        timeout=15,
    )
    if resp.status_code in (200, 201):
        print(f'Энкодер {model_id} зарегистрирован в Open WebUI ({base_url})')
    else:
        print(f'Регистрация не удалась: HTTP {resp.status_code} {resp.text[:200]}')
        print('Зарегистрируйте вручную: Workspace -> Intents -> Encoder Models')


# К2 (Stage 4): полный набор стартовых моделей — интенты, zero-shot, слоты, эмбеддинги
BOOTSTRAP_MODELS = [
    {
        'model_id': 'rubert-intents-v1',
        'task': 'sequence_classification',
        'labels': DEFAULT_LABELS,
        'extra': [],
    },
    {
        'model_id': 'rubert-nli-v1',
        'task': 'feature_extraction',
        'labels': '',
        'extra': ['--zero-shot-mode', 'embedding'],
    },
    {
        'model_id': 'rubert-slots-v1',
        'task': 'token_classification',
        'labels': ','.join([
            'O', 'B-PERSON', 'I-PERSON', 'B-ORDER_ID', 'I-ORDER_ID', 'B-DATE', 'I-DATE',
            'B-EMAIL', 'I-EMAIL', 'B-PHONE', 'I-PHONE', 'B-AMOUNT', 'I-AMOUNT',
            'B-CARD', 'I-CARD',
        ]),
        'extra': ['--init-new-head'],
    },
    {
        'model_id': 'rubert-embed-v1',
        'task': 'feature_extraction',
        'labels': '',
        'extra': [],
    },
]


def export_model(spec: dict) -> None:
    script = Path(__file__).resolve().parents[1] / 'services' / 'encoder-classifier' / 'scripts' / 'export_onnx.py'
    out = MODELS_DIR / spec['model_id']
    cmd = [
        sys.executable, str(script),
        '--model', DEFAULT_SOURCE,
        '--task', spec['task'],
        '--out', str(out),
        '--tau-high', str(TAU_HIGH),
        '--tau-low', str(TAU_LOW),
    ]
    if spec['labels']:
        cmd += ['--labels', spec['labels']]
    cmd += spec['extra']
    print('Экспорт:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description='Bootstrap стартовых ONNX-моделей платформы')
    parser.add_argument('--model-id', default=None, help='Экспорт только одной модели (совместимость)')
    parser.add_argument('--source', default=DEFAULT_SOURCE)
    parser.add_argument('--labels', default=DEFAULT_LABELS)
    parser.add_argument('--open-webui-url', default='')
    parser.add_argument('--api-token', default='')
    parser.add_argument('--force', action='store_true', help='Переэкспортировать даже при наличии моделей')
    args = parser.parse_args()

    if not ensure_train_deps():
        return 1

    if args.model_id:
        # Совместимый одиночный режим
        specs = [s for s in BOOTSTRAP_MODELS if s['model_id'] == args.model_id] or [
            {'model_id': args.model_id, 'task': 'sequence_classification', 'labels': args.labels, 'extra': []}
        ]
    else:
        specs = BOOTSTRAP_MODELS

    exported: list[str] = []
    for spec in specs:
        model_id = spec['model_id']
        if model_exists(model_id) and not args.force:
            print(f'Модель {model_id} уже установлена: {MODELS_DIR / model_id}')
            continue
        print(f'Модель {model_id} не найдена, экспортируем {DEFAULT_SOURCE}...')
        export_model(spec)
        if not model_exists(model_id):
            print(f'Экспорт {model_id} не создал model.onnx — проверьте вывод выше')
            return 1
        exported.append(model_id)
        print(f'Готово: {MODELS_DIR / model_id}')

    if args.open_webui_url and args.api_token:
        for model_id in exported:
            register_encoder(model_id, args.open_webui_url, args.api_token)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
