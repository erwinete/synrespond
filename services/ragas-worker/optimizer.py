"""Auto-Tune RAG: двухфазная оптимизация Zero Re-Retrieval (Д6, Stage 3).

Проблема старой версии: каждая итерация Optuna заново гоняла ретриверы
и Ragas — 20 итераций x 30 вопросов x 4 метрики = тысячи вызовов Judge LLM.

Архитектура:
  Фаза 1 (один раз): для каждого вопроса один прогон пайплайна бота с
    rag_overrides={'return_pool': True, 'top_k': 50} — кэшируются сырые
    ранковые списки двух контуров (гибрид Dense+BM25 и HippoRAG PPR).
  Фаза 2 (быстрый цикл Optuna, MedianPruner): Weighted RRF и отсечение
    top_k/threshold пересчитываются чисто в памяти (~2 мс на конфиг),
    качество оценивается лексической прокси-метрикой против ground_truth
    (без Judge LLM). Промежуточные значения репортятся для MedianPruner.
  Финал: для топ-3 конфигураций — реальный прогон через полный пайплайн
    бота и честные метрики Ragas (Recall/Precision на выбранных чанках,
    Faithfulness/Relevancy на сгенерированных ответах).

Примечание о семантике весов: базовый контур уже сливает Dense и BM25
внутри EnsembleRetriever ядра, поэтому Auto-Tune подбирает связку
w_hybrid (весь Dense+BM25 контур) и w_hipporag. Раздельные w_dense/w_bm25
задаются конфигом ядра и не участвуют в границе слияния контуров.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import optuna

import httpx
import numpy as np

from bot_runner import run_bot_pipeline
from config import CLASSIFIER_URL, EMBED_MODEL

log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or '').lower()))


def lexical_recall_precision(
    ground_truth: str, selected_chunks: list[str]
) -> tuple[float, float]:
    """Быстрая прокси-метрика фазы 2: пересечение токенов без Judge LLM."""
    gt_tokens = _tokens(ground_truth)
    if not gt_tokens:
        return 0.0, 0.0
    selected_tokens: set[str] = set()
    for chunk in selected_chunks:
        selected_tokens |= _tokens(chunk)
    if not selected_tokens:
        return 0.0, 0.0
    overlap = gt_tokens & selected_tokens
    recall = len(overlap) / len(gt_tokens)
    precision = len(overlap) / len(selected_tokens)
    return round(recall, 4), round(precision, 4)


def rrf_select(
    pool: list[dict[str, Any]],
    weights: dict[str, float],
    top_k: int,
    rrf_k: int = 60,
) -> list[str]:
    """К1 (Stage 4): Weighted RRF над тремя раздельными пулами (dense, bm25, hipporag)."""
    scores: dict[str, float] = {}
    display: dict[str, str] = {}
    for list_name in ('dense', 'bm25', 'hipporag'):
        weight = float(weights.get(list_name) or 0.0)
        if weight <= 0:
            continue
        for entry in pool:
            if entry.get('pool_list') != list_name:
                continue
            rank = entry.get('pool_rank') or 1
            text = entry.get('text') or ''
            key = text.strip().lower()
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rrf_k + rank))
            display.setdefault(key, text)
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [display[key] for key, _ in ranked[:top_k]]


def normalize_three_weights(raw_dense: float, raw_bm25: float, raw_hippo: float) -> dict[str, float]:
    """Нормализация трех весов к сумме 1.0 (ТЗ К1)."""
    total = raw_dense + raw_bm25 + raw_hippo
    if total <= 0:
        return {'dense': 1 / 3, 'bm25': 1 / 3, 'hipporag': 1 / 3}
    return {
        'dense': round(raw_dense / total, 4),
        'bm25': round(raw_bm25 / total, 4),
        'hipporag': round(raw_hippo / total, 4),
    }


def semantic_recall_precision(
    gt_sent_embs: "np.ndarray",
    selected_chunk_embs: "np.ndarray",
    gt_emb: "np.ndarray",
    tau_rel: float = 0.55,
) -> tuple[float, float]:
    """Д3 (Stage 5): семантическая прокси-метрика фазы 2 (чистый numpy).

    Recall: доля предложений эталона, покрытых выбранными чанками
    (max cosine). Precision: взвешенная по рангу 1/k доля чанков с
    cosine >= tau_rel к эталону. Ноль лексического пересечения не мешает —
    синонимы и морфология ловятся векторно.
    """
    if gt_sent_embs.size == 0 or selected_chunk_embs.size == 0:
        return 0.0, 0.0
    # Semantic Recall
    sim_matrix = gt_sent_embs @ selected_chunk_embs.T
    recall = float(np.mean(np.max(sim_matrix, axis=1)))
    # Semantic Precision (rank-weighted indicator)
    chunk_sims = selected_chunk_embs @ gt_emb
    weights = np.array([1.0 / (k + 1) for k in range(len(chunk_sims))])
    precision = float(np.sum((chunk_sims >= tau_rel) * weights) / np.sum(weights))
    return round(recall, 4), round(precision, 4)


async def embed_texts(texts: list[str]) -> "np.ndarray | None":
    """Батч-эмбеддинги через encoder-classifier /v1/embeddings. None при недоступности."""
    if not texts:
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f'{CLASSIFIER_URL}/v1/embeddings',
                json={'input': texts, 'model': EMBED_MODEL},
            )
            response.raise_for_status()
            data = response.json()['data']
        vecs = np.array([item['embedding'] for item in data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs / norms
    except Exception as exc:  # noqa: BLE001
        log.warning('Semantic embeddings unavailable, lexical proxy fallback: %s', exc)
        return None


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r'[.!?;\n]+', text or '') if len(s.strip()) > 3]
    return parts or [text.strip()]


def _build_pool_cache(
    pool_payload: dict[str, Any], ground_truth: str
) -> dict[str, Any]:
    """Извлекает пулы из ответа бота: чанки с pool_list/pool_rank."""
    entries: list[dict[str, Any]] = []
    metadatas = (pool_payload.get('metadatas') or [[]])[0]
    documents = (pool_payload.get('documents') or [[]])[0]
    for texts, metas in ((documents, metadatas),):
        for idx, text in enumerate(texts or []):
            meta = metas[idx] if metas and idx < len(metas) and metas[idx] else {}
            entries.append(
                {
                    'text': str(text),
                    'pool_list': str(meta.get('pool_list') or 'base'),
                    'pool_rank': int(meta.get('pool_rank') or idx + 1),
                }
            )
    return {'entries': entries, 'ground_truth': ground_truth}


async def run_optimization(
    run_id: str,
    items: list[dict[str, Any]],
    bot_model_id: Optional[str],
    n_trials: int,
    max_items: int,
    engine,  # RagasEvaluationEngine
    finalize_run,
) -> dict[str, Any]:
    """Двухфазный Auto-Tune. Возвращает лучший конфиг для записи в run."""
    subset = [item for item in items if str(item.get('question') or '').strip()][:max_items]
    questions = [str(item['question']).strip() for item in subset]
    ground_truths = [str(item.get('ground_truth') or '') for item in subset]
    if not questions:
        return {'error': 'No valid questions'}

    pool_k = 50

    # ---- Фаза 1: кэш пулов + эмбеддингов (по одному прогону на вопрос) ----
    caches: list[dict[str, Any]] = []
    for question, ground_truth in zip(questions, ground_truths):
        bot_result = await run_bot_pipeline(
            question,
            bot_model_id=bot_model_id,
            rag_overrides={'top_k': pool_k, 'relevance_threshold': 0.0, 'return_pool': True},
        )
        raw = bot_result.get('pool') or {}
        if not raw and bot_result.get('contexts'):
            # Резерв: бэкенд без return_pool — трактуем контексты как dense-пул
            raw = {
                'documents': [bot_result['contexts']],
                'metadatas': [
                    [{'pool_list': 'dense', 'pool_rank': i + 1} for i in range(len(bot_result['contexts']))]
                ],
            }
        cache = _build_pool_cache(raw, ground_truth)

        # Д3: эмбеддинги эталона (целиком + по предложениям) и уникальных чанков
        gt_sents = _split_sentences(ground_truth)
        unique_chunks = list({entry['text'] for entry in cache['entries'] if entry['text']})
        embs = await embed_texts([ground_truth, *gt_sents, *unique_chunks])
        if embs is not None:
            cache['gt_emb'] = embs[0]
            cache['gt_sent_embs'] = embs[1 : 1 + len(gt_sents)]
            chunk_index = {text: i for i, text in enumerate(unique_chunks)}
            cache['chunk_embs'] = {t: embs[1 + len(gt_sents) + chunk_index[t]] for t in unique_chunks}
        else:
            cache['gt_emb'] = None
            cache['gt_sent_embs'] = None
            cache['chunk_embs'] = None
        caches.append(cache)

    # ---- Фаза 2: быстрый цикл Optuna (MedianPruner, без Judge LLM) ----
    def objective(trial: optuna.Trial) -> float:
        # К1: три независимых веса с нормализацией суммы к 1.0
        weights = normalize_three_weights(
            trial.suggest_float('w_dense', 0.05, 1.0),
            trial.suggest_float('w_bm25', 0.05, 1.0),
            trial.suggest_float('w_hipporag', 0.0, 1.0),
        )
        top_k = trial.suggest_int('top_k', 2, 12)
        params = {**weights, 'top_k': top_k}

        if sum(weights.values()) < 0.05:
            return 0.0

        proxy_scores: list[float] = []
        completed = 0
        for cache in caches:
            keys = rrf_select(cache['entries'], weights, top_k)
            key_set = set(keys)
            selected_entries = [
                entry for entry in cache['entries'] if entry['text'].strip().lower() in key_set
            ]

            # Д3: семантическая прокси за <1 мс (эмбеддинги предрасчитаны в фазе 1)
            if cache.get('gt_sent_embs') is not None:
                sel_embs = np.array(
                    [cache['chunk_embs'][e['text']] for e in selected_entries if e['text'] in cache['chunk_embs']]
                )
                recall, precision = semantic_recall_precision(
                    cache['gt_sent_embs'], sel_embs, cache['gt_emb']
                )
            else:
                chunks = [entry['text'] for entry in selected_entries]
                recall, precision = lexical_recall_precision(cache['ground_truth'], chunks)

            proxy_scores.append(0.4 * recall + 0.3 * precision)
            completed += 1
            # MedianPruner: отсев слабых конфигураций по промежуточному значению
            trial.report(sum(proxy_scores) / completed, completed)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return sum(proxy_scores) / max(1, len(proxy_scores))

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, interval_trials=2),
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    # ---- Финал: топ-3 конфигурации через полный пайплайн + честный Ragas ----
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    finalists = completed_trials[:3]

    best_record: dict[str, Any] | None = None
    for trial in finalists:
        params = {
            'hybrid_weights': {
                'dense': trial.params['w_dense'],
                'bm25': trial.params['w_bm25'],
                'hipporag': trial.params['w_hipporag'],
            },
            'top_k': trial.params['top_k'],
            'relevance_threshold': 0.0,
        }
        answers: list[str] = []
        contexts_batch: list[list[str]] = []
        for question in questions:
            bot_result = await run_bot_pipeline(
                question, bot_model_id=bot_model_id, rag_overrides=params
            )
            answers.append(bot_result.get('answer') or '')
            contexts_batch.append(bot_result.get('contexts') or [])

        result = await engine.evaluate_samples(
            questions=questions,
            answers=answers,
            contexts=contexts_batch,
            ground_truths=ground_truths,
        )
        s = result['summary_scores']
        score = (
            0.4 * (s.get('context_recall') or 0.0)
            + 0.3 * (s.get('context_precision') or 0.0)
            + 0.3 * (s.get('faithfulness') or 0.0)
        )
        record = {
            'score': round(score, 4),
            'params': params,
            'summary': s,
            'proxy_value': round(trial.value, 4),
        }
        if best_record is None or record['score'] > best_record['score']:
            best_record = record

    if best_record is None:
        return {'error': 'No completed trials'}

    return {
        'baseline_score': None,
        'best_score': best_record['score'],
        'best_params': best_record['params'],
        'finalists': len(finalists),
        'trials': n_trials,
        'questions_per_trial': len(questions),
        'pruned_trials': sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ),
    }
