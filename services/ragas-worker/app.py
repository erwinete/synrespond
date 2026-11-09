"""ragas-worker: асинхронная оценка качества RAG (ТЗ, раздел 6).

Маршруты:
  GET  /health              — состояние
  POST /v1/evaluate/live    — сэмпл живого диалога: очередь Redis -> фоновый расчет
  POST /v1/evaluate/batch   — прогон эталонного датасета через бота и Ragas

Live-режим считает Faithfulness и Answer Relevancy (без ground_truth),
помечает ответ флагом hallucination при Faithfulness ниже порога.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bot_runner import run_bot_pipeline
from config import (
    BATCH_MAX_ITEMS,
    CONSUMER_ENABLED,
    JUDGE_API_KEY,
    JUDGE_LLM_MODEL,
    JUDGE_LLM_URL,
    LIVE_QUEUE_KEY,
    RAGAS_HALLUCINATION_THRESHOLD,
    REDIS_URL,
)
from db import ensure_live_run, finalize_run, init_db_if_sqlite, insert_sample, set_run_status
from evaluator import LocalONNXRagasJudge, RagasEvaluationEngine

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
log = logging.getLogger('ragas-worker')

VERSION = '1.0.0'

app = FastAPI(title='Ragas Evaluation Worker', version=VERSION)

engine = RagasEvaluationEngine(
    judge_llm_url=JUDGE_LLM_URL,
    judge_llm_model=JUDGE_LLM_MODEL,
    judge_api_key=JUDGE_API_KEY,
    embeddings_url=os.getenv('EMBEDDINGS_URL', 'http://tei-embeddings:80/v1'),
    embeddings_model=os.getenv('EMBEDDINGS_MODEL', 'BAAI/bge-m3'),
)

# Д4 (Stage 5): локальный ONNX-судья — Faithfulness через NLI и числовые
# расхождения, Relevancy/Precision/Recall через эмбеддинги. Без GPU и vLLM.
local_judge = LocalONNXRagasJudge(classifier_url=os.getenv('CLASSIFIER_URL', 'http://127.0.0.1:8001'))
_judge_engine = os.getenv('RAGAS_JUDGE_ENGINE', 'auto')


async def get_evaluator():
    """Выбор судьи: local_onnx | llm_judge | auto (llm, при недоступности local_onnx)."""
    if _judge_engine == 'local_onnx':
        return local_judge
    if _judge_engine == 'llm_judge':
        return engine
    # auto: если judge LLM сконфигурирован и доступен — ragas, иначе local_onnx
    if JUDGE_LLM_URL:
        try:
            import aiohttp

            probe_url = JUDGE_LLM_URL.rstrip('/')
            if probe_url.endswith('/v1'):
                probe_url = probe_url[:-3]
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                async with session.get(f'{probe_url}/models') as response:
                    if response.status < 400:
                        return engine
        except Exception:  # noqa: BLE001
            pass
    return local_judge

_redis = None


async def get_redis():
    global _redis
    if _redis is None:
        import redis.asyncio as aioredis

        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


class LiveSampleRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    bot_model_id: str | None = None
    chat_id: str | None = None
    message_id: str | None = None
    rewritten_query: str | None = None
    retrieval_trace: list[Any] | None = None


class BatchRunRequest(BaseModel):
    run_id: str
    items: list[dict[str, Any]]
    bot_model_id: str | None = None


@app.get('/health')
async def health():
    return {'status': 'ok', 'version': VERSION, 'judge_model': JUDGE_LLM_MODEL}


@app.post('/v1/evaluate/live')
async def evaluate_live(sample: LiveSampleRequest):
    """Кладет сэмпл в очередь Redis; считает воркер в фоне."""
    try:
        redis = await get_redis()
        await redis.lpush(LIVE_QUEUE_KEY, sample.model_dump_json())
        return {'status': 'queued'}
    except Exception as exc:  # noqa: BLE001
        # Redis недоступен — считаем синхронно, чтобы не терять сэмплы
        log.warning('Redis unavailable, evaluating inline: %s', exc)
        await evaluate_live_sample(sample.model_dump())
        return {'status': 'evaluated_inline'}


async def evaluate_live_sample(payload: dict[str, Any]) -> None:
    """Faithfulness + Answer Relevancy для одного живого диалога."""
    run_id = f'live-{datetime.now(timezone.utc).strftime("%Y%m%d")}'
    question = payload.get('question') or ''
    answer = payload.get('answer') or ''
    contexts = [str(c) for c in (payload.get('contexts') or []) if c]
    if not question or not answer:
        return

    try:
        evaluator = await get_evaluator()
        result = await evaluator.evaluate_samples(
            questions=[question],
            answers=[answer],
            contexts=[contexts] if contexts else [['']],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning('Live evaluation failed: %s', exc)
        return

    scores = result['summary_scores']
    faithfulness = scores.get('faithfulness')
    flag = None
    if faithfulness is not None and faithfulness < RAGAS_HALLUCINATION_THRESHOLD:
        flag = 'hallucination'
        log.warning('Hallucination Alert: faithfulness=%.2f chat=%s', faithfulness, payload.get('chat_id'))

    await insert_sample(
        {
            'run_id': run_id,
            'question': question,
            'rewritten_query': payload.get('rewritten_query'),
            'answer': answer,
            'contexts': contexts,
            'scores': scores,
            'flag': flag,
            'bot_model_id': payload.get('bot_model_id'),
            'chat_id': payload.get('chat_id'),
            'message_id': payload.get('message_id'),
            'retrieval_trace': payload.get('retrieval_trace'),
        }
    )


async def live_queue_consumer() -> None:
    """Фоновый потребитель очереди live-сэмплов."""
    while True:
        try:
            redis = await get_redis()
            if not redis:
                await asyncio.sleep(1)
                continue
            _, raw = await redis.brpop(LIVE_QUEUE_KEY, timeout=5)
            if raw is None:
                await asyncio.sleep(0.2)
                continue
            payload = json.loads(raw)
            await evaluate_live_sample(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning('Live consumer error: %s', exc)
            await asyncio.sleep(1)


@app.post('/v1/evaluate/batch')
async def evaluate_batch(form: BatchRunRequest):
    """Прогоняет эталонные вопросы через бота и оценивает все 4 метрики Ragas."""
    items = form.items[:BATCH_MAX_ITEMS]
    if not items:
        raise HTTPException(status_code=400, detail='Dataset is empty')

    await set_run_status(form.run_id, 'running')
    asyncio.create_task(_run_batch(form.run_id, items, form.bot_model_id))
    return {'status': 'accepted', 'run_id': form.run_id, 'items': len(items)}


class SyncEvalRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str | None = None


@app.post('/v1/evaluate/sync')
async def evaluate_sync(form: SyncEvalRequest):
    """Д4 (Stage 5): синхронная оценка одной пары выбранным судьей.

    Возвращает summary_scores без записи в БД — для детерминированных E2E.
    """
    evaluator = await get_evaluator()
    contexts = [form.contexts] if form.contexts else [['']]
    ground_truths = [form.ground_truth] if form.ground_truth else None
    result = await evaluator.evaluate_samples(
        questions=[form.question],
        answers=[form.answer],
        contexts=contexts,
        ground_truths=ground_truths,
    )
    result['judge_engine'] = getattr(evaluator, '__class__', {}).__name__
    return result


class OptimizeRequest(BaseModel):
    run_id: str
    items: list[dict[str, Any]]
    bot_model_id: str | None = None
    n_trials: int = Field(default=10, ge=2, le=60)
    max_items: int = Field(default=10, ge=2, le=50)


@app.post('/v1/optimize')
async def optimize(form: OptimizeRequest):
    """Ф4: Auto-Tune — Optuna подбирает веса поиска по метрикам Ragas."""
    if not form.items:
        raise HTTPException(status_code=400, detail='Dataset is empty')

    await set_run_status(form.run_id, 'running')
    asyncio.create_task(_run_optimize(form))
    return {'status': 'accepted', 'run_id': form.run_id, 'n_trials': form.n_trials}


async def _run_optimize(form: OptimizeRequest) -> None:
    from optimizer import run_optimization

    try:
        evaluator = await get_evaluator()
        result = await run_optimization(
            run_id=form.run_id,
            items=form.items,
            bot_model_id=form.bot_model_id,
            n_trials=form.n_trials,
            max_items=form.max_items,
            engine=evaluator,
            finalize_run=finalize_run,
        )
        if result.get('error'):
            await finalize_run(form.run_id, {}, 0, error=result['error'])
            return
        await finalize_run(
            form.run_id,
            {
                'baseline_score': result['baseline_score'],
                'best_score': result['best_score'],
                'best_params': result['best_params'],
                'objective': '0.4*recall + 0.3*precision + 0.3*faithfulness',
                'trials': result['trials'],
                'questions_per_trial': result['questions_per_trial'],
            },
            result['questions_per_trial'],
        )
        log.info('Auto-Tune %s finished: %.3f -> %.3f', form.run_id, result['baseline_score'], result['best_score'])
    except Exception as exc:  # noqa: BLE001
        log.exception('Optimization failed')
        await finalize_run(form.run_id, {}, 0, error=str(exc))


async def _run_batch(run_id: str, items: list[dict[str, Any]], bot_model_id: str | None) -> None:
    questions: list[str] = []
    answers: list[str] = []
    contexts_batch: list[list[str]] = []
    ground_truths: list[str] = []

    for item in items:
        question = str(item.get('question') or '').strip()
        if not question:
            continue
        ground_truths.append(str(item.get('ground_truth') or ''))
        # Этап 1: прогон через полный пайплайн бота (Rewriter -> Hybrid+Hippo -> LLM)
        bot_result = await run_bot_pipeline(question, bot_model_id)
        answer = bot_result.get('answer') or ''
        if not answer:
            answer = item.get('expected_answer') or ''
        questions.append(question)
        answers.append(answer)
        contexts_batch.append(bot_result.get('contexts') or [])

    if not questions:
        await finalize_run(run_id, {}, 0, error='No valid questions in dataset')
        return

    try:
        evaluator = await get_evaluator()
        result = await evaluator.evaluate_samples(
            questions=questions,
            answers=answers,
            contexts=contexts_batch,
            ground_truths=ground_truths,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception('Batch evaluation failed')
        await finalize_run(run_id, {}, 0, error=str(exc))
        return

    summary = result['summary_scores']
    rows = result['detailed_rows']
    for index, row in enumerate(rows):
        row_scores = {
            name: row.get(name) for name in result['metric_names'] if row.get(name) is not None
        }
        faithfulness = row_scores.get('faithfulness')
        await insert_sample(
            {
                'run_id': run_id,
                'question': questions[index],
                'answer': answers[index],
                'contexts': contexts_batch[index],
                'ground_truth': ground_truths[index] or None,
                'scores': row_scores,
                'flag': 'hallucination'
                if faithfulness is not None and faithfulness < RAGAS_HALLUCINATION_THRESHOLD
                else None,
                'bot_model_id': bot_model_id,
            }
        )
    await finalize_run(run_id, summary, len(rows))


@app.on_event('startup')
async def startup():
    await init_db_if_sqlite()
    if CONSUMER_ENABLED:
        task = asyncio.create_task(live_queue_consumer())
        app.state.consumer_task = task


@app.on_event('shutdown')
async def shutdown():
    task = getattr(app.state, 'consumer_task', None)
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
