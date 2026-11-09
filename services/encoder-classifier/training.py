"""Дообучение энкодера и горячая подмена ONNX (Active Learning, Ф1).

Д4 (Stage 3): обучение выполняется в изолированном подпроцессе
(ProcessPoolExecutor, max_workers=1, приоритет понижен через os.nice),
чтобы градиенты PyTorch не блокировали event loop и боевые запросы
/v1/classify сохраняли латентность. Прогресс пишется в файл job-а,
GET /v1/train/{job_id} отдает его для прогресс-бара в UI.

Rollback Gate: macro-F1 новой модели сравнивается с F1 текущей базовой
ONNX-модели на одном holdout; при падении больше чем на 0.02 (без
force_swap=true) подмена блокируется, артефакт уходит в candidates/.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import Any

logger = logging.getLogger('encoder-classifier.train')

_jobs: dict[str, dict[str, Any]] = {}
_futures: dict[str, Any] = {}
_jobs_lock = threading.Lock()
_executor: ProcessPoolExecutor | None = None
_jobs_dir = os.path.join(os.getenv('MODELS_DIR', '/models'), '.train_jobs')

ROLLBACK_F1_DELTA = -0.02


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=1)
    return _executor


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        result = dict(job)
    # Прогресс из файла подпроцесса
    progress_path = result.get('progress_path')
    if progress_path and result.get('status') == 'running':
        try:
            with open(progress_path, encoding='utf-8') as f:
                result['progress'] = json.load(f)
        except OSError:
            pass
    if result.get('status') == 'running':
        future = _futures.get(job_id)
        if future is not None and future.done():
            exc = future.exception()
            if exc is not None:
                result['status'] = 'failed'
                result['error'] = str(exc)
            else:
                result['status'] = 'completed'
                result['result'] = future.result()
    return result


def start_train_job(registry, models_dir: str, params: dict[str, Any]) -> str:
    """Ставит задачу в ProcessPoolExecutor и мгновенно возвращает job_id."""
    job_id = f'train-{uuid.uuid4().hex[:8]}'
    os.makedirs(_jobs_dir, exist_ok=True)
    progress_path = os.path.join(_jobs_dir, f'{job_id}.progress.json')

    with _jobs_lock:
        _jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'created_at': int(time.time()),
            'params': params,
            'result': None,
            'error': None,
            'progress_path': progress_path,
        }

    future = _get_executor().submit(
        _job_entry,
        params,
        models_dir,
        progress_path,
    )

    def _done(fut):
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                return
            exc = fut.exception()
            if exc is not None:
                job['status'] = 'failed'
                job['error'] = str(exc)
            else:
                job['status'] = 'completed'
                job['result'] = fut.result()
            _futures.pop(job_id, None)

    future.add_done_callback(_done)
    with _jobs_lock:
        _futures[job_id] = future
    return job_id


def _job_entry(params: dict[str, Any], models_dir: str, progress_path: str) -> dict[str, Any]:
    """Точка входа подпроцесса: обучение, экспорт, Rollback Gate, подмена."""
    # Д4: понижаем приоритет процесса, чтобы не голодать инференс
    if hasattr(os, 'nice'):
        try:
            os.nice(10)
        except OSError:
            pass

    def report_progress(**fields: Any) -> None:
        try:
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(fields, f, ensure_ascii=False)
        except OSError:
            pass

    report_progress(stage='starting')
    return _train_and_export(params, models_dir, report_progress)


def _collect_examples(params: dict[str, Any]) -> list[dict[str, str]]:
    """Примеры из запроса плюс Anchor Dataset против забывания старых классов."""
    examples = [
        {'text': str(item['text']), 'label': str(item['label'])}
        for item in (params.get('examples') or [])
        if item.get('text') and item.get('label')
    ]
    # Д4: эталонная базовая выборка (anchor) мешается с новыми фразами
    anchors = [
        {'text': str(item['text']), 'label': str(item['label'])}
        for item in (params.get('anchor_examples') or [])
        if item.get('text') and item.get('label')
    ]
    return anchors + examples


def _train_and_export(
    params: dict[str, Any], models_dir: str, report_progress
) -> dict[str, Any]:
    examples = _collect_examples(params)
    if len(examples) < 8:
        raise ValueError('Need at least 8 labeled examples to fine-tune')

    base_model_id = params.get('base_model_id')
    if not base_model_id:
        raise ValueError('base_model_id is required (existing ONNX model in catalog)')

    base_path = os.path.join(models_dir, base_model_id)
    meta_path = os.path.join(base_path, 'label_map.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'Base model {base_model_id} not found')

    with open(meta_path, encoding='utf-8') as f:
        base_meta = json.load(f)
    source_model = base_meta.get('source_model')
    if not source_model:
        raise ValueError('label_map.json has no source_model — re-export the base with scripts/export_onnx.py')

    out_model_id = params.get('out_model_id') or f'{base_model_id}-v{int(time.time())}'
    epochs = int(params.get('epochs') or 4)
    lr = float(params.get('lr') or 5e-5)
    batch_size = int(params.get('batch_size') or 8)
    activate = bool(params.get('activate', True))
    force_swap = bool(params.get('force_swap', False))

    report_progress(stage='training', epochs=epochs)
    label_map, metrics = _train_sync(
        source_model, base_path, examples, epochs, lr, batch_size, report_progress
    )

    report_progress(stage='exporting')
    from optimum.exporters.onnx import main_export

    final_dir = os.path.join(models_dir, out_model_id)
    tmp_dir = f'{final_dir}_tmp'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    main_export(source_model, output=tmp_dir, task='text-classification', opset=17, device='cpu')

    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(base_path).save_pretrained(tmp_dir)

    merged_meta = dict(base_meta)
    merged_meta['labels'] = label_map
    merged_meta['source_model'] = source_model
    merged_meta['fine_tuned_from'] = base_model_id
    merged_meta['trained_at'] = int(time.time())
    with open(os.path.join(tmp_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(merged_meta, f, ensure_ascii=False, indent=2)

    if params.get('quantize'):
        try:
            from onnxruntime.quantization import quantize_dynamic

            model_path = os.path.join(tmp_dir, 'model.onnx')
            quantize_dynamic(model_path, model_path + '.int8')
            os.replace(model_path + '.int8', model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning('Quantization skipped: %s', exc)

    # Д4: Rollback Gate — сравнение с базовой моделью на том же holdout
    baseline_f1 = metrics.get('baseline_macro_f1')
    new_f1 = metrics.get('macro_f1')
    swapped = False
    warning = None
    if baseline_f1 is not None and new_f1 is not None:
        delta_f1 = new_f1 - baseline_f1
        metrics['delta_f1'] = round(delta_f1, 4)
        if delta_f1 < ROLLBACK_F1_DELTA and not force_swap:
            candidates_dir = os.path.join(models_dir, 'candidates', out_model_id)
            os.makedirs(os.path.dirname(candidates_dir), exist_ok=True)
            shutil.move(tmp_dir, candidates_dir)
            warning = (
                f'Rollback Gate: new macro_f1 {new_f1:.3f} < baseline {baseline_f1:.3f} '
                f'(delta {delta_f1:.3f}). Swap blocked, checkpoint saved to candidates/{out_model_id}. '
                'Improve labels or pass force_swap=true.'
            )
            logger.warning(warning)
            return {
                'out_model_id': out_model_id,
                'swapped': False,
                'blocked_by_gate': True,
                'metrics': metrics,
                'examples_used': len(examples),
                'warning': warning,
            }

    if activate:
        # Атомарная подмена каталога модели на диске
        if os.path.exists(final_dir):
            old_dir = f'{final_dir}_old_{uuid.uuid4().hex[:6]}'
            os.replace(final_dir, old_dir)
            shutil.rmtree(old_dir, ignore_errors=True)
        os.replace(tmp_dir, final_dir)
        swapped = True
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        'out_model_id': out_model_id,
        'swapped': swapped,
        'metrics': metrics,
        'examples_used': len(examples),
        'note': f'Model {out_model_id} ready; parent process will reload it into the registry',
        'warning': warning,
    }


def _train_sync(
    source_model: str,
    tokenizer_path: str,
    examples: list[dict[str, str]],
    epochs: int,
    lr: float,
    batch_size: int,
    report_progress,
):
    """Минимальный torch-цикл: дообучение классификационной головы модели."""
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    labels = sorted({item['label'] for item in examples})
    label2id = {label: idx for idx, label in enumerate(labels)}

    texts = [item['text'] for item in examples]
    targets = [label2id[item['label']] for item in examples]

    if len(set(targets)) > 1 and len(texts) >= 10:
        train_texts, eval_texts, train_targets, eval_targets = train_test_split(
            texts, targets, test_size=0.2, random_state=42, stratify=targets
        )
    else:
        train_texts, eval_texts, train_targets, eval_targets = texts, texts, targets, targets

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(
        source_model, num_labels=len(labels), ignore_mismatched_sizes=True
    ).to(device)
    # Энкодер заморожен — обучается только голова (быстро и стабильно)
    for param in model.base_model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def encode(batch_texts, batch_targets):
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        return enc, torch.tensor(batch_targets, device=device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for epoch in range(epochs):
        order = np.random.permutation(len(train_texts))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            enc, targets_tensor = encode(
                [train_texts[i] for i in batch_idx], [train_targets[i] for i in batch_idx]
            )
            outputs = model(**enc, labels=targets_tensor)
            optimizer.zero_grad()
            outputs.loss.backward()
            optimizer.step()
            total_loss += float(outputs.loss)
        avg_loss = total_loss / max(1, (len(order) + batch_size - 1) // batch_size)
        logger.info('Train epoch %s/%s loss=%.4f', epoch + 1, epochs, avg_loss)
        report_progress(stage='training', epoch=epoch + 1, epochs=epochs, loss=round(avg_loss, 4))

    # Оценка новой модели на отложенной выборке
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(eval_texts), batch_size):
            enc, _ = encode(eval_texts[start : start + batch_size], eval_targets[start : start + batch_size])
            logits = model(**enc).logits
            predictions.extend(torch.argmax(logits, dim=-1).tolist())

    accuracy = float(np.mean([p == t for p, t in zip(predictions, eval_targets)])) if eval_targets else 0.0
    macro_f1 = float(f1_score(eval_targets, predictions, average='macro', zero_division=0)) if eval_targets else 0.0

    # Д4: baseline — текущая базовая ONNX-модель на том же holdout
    baseline_macro_f1 = _eval_baseline_onnx(
        tokenizer_path, eval_texts, eval_targets, label2id
    )

    id2label = {str(idx): label for label, idx in label2id.items()}
    return id2label, {
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'baseline_macro_f1': baseline_macro_f1,
        'device': device,
        'epochs': epochs,
    }


def _eval_baseline_onnx(tokenizer_path: str, eval_texts: list[str], eval_targets: list[int], label2id: dict) -> float | None:
    """F1 базовой ONNX-модели на том же holdout (для Rollback Gate)."""
    try:
        import numpy as np
        import onnxruntime as ort
        from sklearn.metrics import f1_score
        from transformers import AutoTokenizer

        base_dir = os.path.dirname(os.path.abspath(tokenizer_path))
        model_path = os.path.join(base_dir, 'model.onnx')
        if not os.path.exists(model_path) or not eval_targets:
            return None

        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        input_names = [i.name for i in session.get_inputs()]

        predictions: list[int] = []
        for start in range(0, len(eval_texts), 16):
            batch_texts = eval_texts[start : start + 16]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='np')
            inputs = {k: v for k, v in enc.items() if k in input_names}
            logits = session.run(None, inputs)[0]
            predictions.extend(int(idx) for idx in np.argmax(logits, axis=-1))

        return float(f1_score(eval_targets, predictions, average='macro', zero_division=0))
    except Exception as exc:  # noqa: BLE001 — нет baseline, гейт не применяется
        logger.warning('Baseline ONNX evaluation skipped: %s', exc)
        return None
