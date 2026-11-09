"""REST API оценки качества RAG по фреймворку Ragas (ТЗ, раздел 6.3).

Batch-прогоны эталонных датасетов делегируются сервису ragas-worker:
бэкенд создает запись запуска, воркер выполняет пайплайн бота и оценку
метрик, затем пишет сэмплы и сводные метрики в общую БД.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from open_webui.internal.db import get_async_session
from open_webui.models.config import Config
from open_webui.models.ragas_evals import (
    RagasDatasetModel,
    RagasDatasets,
    RagasRunModel,
    RagasRuns,
    RagasSampleModel,
    RagasSamples,
)
from open_webui.models.models import Models
from open_webui.utils.auth import get_admin_user, get_verified_user

router = APIRouter()


class RagasConfigForm(BaseModel):
    enable: bool = True
    worker_url: str = ''
    live_sample_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    hallucination_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class RagasDatasetForm(BaseModel):
    name: str = Field(min_length=1)
    description: Optional[str] = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    meta: Optional[dict[str, Any]] = None


class RagasRunForm(BaseModel):
    name: Optional[str] = None
    dataset_id: str = Field(min_length=1)
    bot_model_id: Optional[str] = None
    # Ограничение размера прогона для защиты от непредвиденных расходов LLM
    max_items: int = Field(default=50, ge=1, le=1000)


@router.get('/config')
async def get_ragas_config(user=Depends(get_verified_user)):
    values = await Config.get_many(
        'ragas.enable',
        'ragas.worker_url',
        'ragas.live_sample_rate',
        'ragas.hallucination_threshold',
    )
    return {
        'enable': bool(values.get('ragas.enable')),
        'worker_url': values.get('ragas.worker_url') or '',
        'live_sample_rate': float(values.get('ragas.live_sample_rate') or 0),
        'hallucination_threshold': float(values.get('ragas.hallucination_threshold') or 0.7),
    }


@router.post('/config')
async def update_ragas_config(form: RagasConfigForm, user=Depends(get_admin_user)):
    await Config.upsert(
        {
            'ragas.enable': form.enable,
            'ragas.worker_url': form.worker_url,
            'ragas.live_sample_rate': form.live_sample_rate,
            'ragas.hallucination_threshold': form.hallucination_threshold,
        }
    )
    return await get_ragas_config(user)


# Эталонные датасеты (Golden Datasets)


@router.get('/datasets', response_model=list[RagasDatasetModel])
async def list_datasets(
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    return await RagasDatasets.get_datasets(db=db)


@router.post('/datasets/create', response_model=RagasDatasetModel)
async def create_dataset(
    form: RagasDatasetForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    data = form.model_dump()
    data['user_id'] = user.id
    return await RagasDatasets.insert_new_dataset(data, db=db)


@router.post('/datasets/{id}/update', response_model=RagasDatasetModel)
async def update_dataset(
    id: str,
    form: RagasDatasetForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    dataset = await RagasDatasets.update_dataset_by_id(id, form.model_dump(), db=db)
    if not dataset:
        raise HTTPException(status_code=404, detail='Dataset not found')
    return dataset


@router.delete('/datasets/{id}/delete')
async def delete_dataset(
    id: str,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    deleted = await RagasDatasets.delete_dataset_by_id(id, db=db)
    if not deleted:
        raise HTTPException(status_code=404, detail='Dataset not found')
    return {'status': True}


# Запуски оценки


@router.get('/runs', response_model=list[RagasRunModel])
async def list_runs(
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    return await RagasRuns.get_runs(db=db)


@router.get('/runs/{id}', response_model=RagasRunModel)
async def get_run(
    id: str,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    run = await RagasRuns.get_run_by_id(id, db=db)
    if not run:
        raise HTTPException(status_code=404, detail='Run not found')
    return run


@router.get('/runs/{id}/samples', response_model=list[RagasSampleModel])
async def get_run_samples(
    id: str,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    return await RagasSamples.get_samples_by_run_id(id, db=db)


@router.post('/runs/create', response_model=RagasRunModel)
async def create_run(
    form: RagasRunForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Создает прогон и передает эталон в ragas-worker на исполнение."""
    dataset = await RagasDatasets.get_dataset_by_id(form.dataset_id, db=db)
    if not dataset:
        raise HTTPException(status_code=404, detail='Dataset not found')

    worker_url = (await Config.get('ragas.worker_url') or '').rstrip('/')
    if not worker_url:
        raise HTTPException(status_code=400, detail='Ragas worker URL is not configured')

    run = await RagasRuns.insert_new_run(
        {
            'name': form.name or f'{dataset.name} ({form.bot_model_id or "default"})',
            'mode': 'batch',
            'dataset_id': dataset.id,
            'bot_model_id': form.bot_model_id,
            'status': 'pending',
            'sample_count': min(len(dataset.items), form.max_items),
            'config': {'max_items': form.max_items},
        },
        db=db,
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f'{worker_url}/v1/evaluate/batch',
                json={
                    'run_id': run.id,
                    'items': dataset.items[: form.max_items],
                    'bot_model_id': form.bot_model_id,
                },
            )
            response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        await RagasRuns.update_run_by_id(run.id, {'status': 'failed', 'error': str(exc)}, db=db)
        raise HTTPException(status_code=502, detail=f'Ragas worker unreachable: {exc}')

    updated = await RagasRuns.get_run_by_id(run.id, db=db)
    return updated or run


@router.delete('/runs/{id}/delete')
async def delete_run(
    id: str,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    await RagasSamples.delete_samples_by_run_id(id, db=db)
    deleted = await RagasRuns.delete_run_by_id(id, db=db)
    if not deleted:
        raise HTTPException(status_code=404, detail='Run not found')
    return {'status': True}


# Auto-Tune (киллер-фича №4)


class RagasOptimizeForm(BaseModel):
    dataset_id: str = Field(min_length=1)
    bot_model_id: Optional[str] = None
    n_trials: int = Field(default=10, ge=2, le=60)
    max_items: int = Field(default=10, ge=2, le=50)


@router.post('/optimize/create', response_model=RagasRunModel)
async def create_optimization(
    form: RagasOptimizeForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Подбор весов поиска (w_dense, w_bm25, w_hippo, threshold, top_k) через Optuna."""
    dataset = await RagasDatasets.get_dataset_by_id(form.dataset_id, db=db)
    if not dataset:
        raise HTTPException(status_code=404, detail='Dataset not found')

    worker_url = (await Config.get('ragas.worker_url') or '').rstrip('/')
    if not worker_url:
        raise HTTPException(status_code=400, detail='Ragas worker URL is not configured')

    run = await RagasRuns.insert_new_run(
        {
            'name': f'Auto-Tune: {dataset.name}',
            'mode': 'optimize',
            'dataset_id': dataset.id,
            'bot_model_id': form.bot_model_id,
            'status': 'pending',
            'sample_count': min(len(dataset.items), form.max_items),
            'config': {'n_trials': form.n_trials, 'max_items': form.max_items},
        },
        db=db,
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f'{worker_url}/v1/optimize',
                json={
                    'run_id': run.id,
                    'items': dataset.items,
                    'bot_model_id': form.bot_model_id,
                    'n_trials': form.n_trials,
                    'max_items': form.max_items,
                },
            )
            response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        await RagasRuns.update_run_by_id(run.id, {'status': 'failed', 'error': str(exc)}, db=db)
        raise HTTPException(status_code=502, detail=f'Ragas worker unreachable: {exc}')

    updated = await RagasRuns.get_run_by_id(run.id, db=db)
    return updated or run


@router.get('/alerts', response_model=list[RagasSampleModel])
async def list_hallucination_alerts(
    limit: int = 100,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Диалоги с Faithfulness ниже порога — флаг Hallucination Alert."""
    return await RagasSamples.get_flagged_samples('hallucination', limit=limit, db=db)


# Д1 (Stage 5): применение найденных весов Auto-Tune к боевому поиску


def extract_weights_from_run(run_metrics: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Достает best_params из metrics завершенного optimize-прогона.

    Поддерживает формат Stage 4 (w_dense/w_bm25/w_hipporag внутри
    hybrid_weights) и валидирует границы. None — веса применить нельзя.
    """
    if not run_metrics:
        return None
    params = run_metrics.get('best_params') or {}
    weights = params.get('hybrid_weights') or {
        'dense': params.get('w_dense'),
        'bm25': params.get('w_bm25'),
        'hipporag': params.get('w_hipporag'),
    }
    result: dict[str, Any] = {}
    for key in ('dense', 'bm25', 'hipporag'):
        value = weights.get(key)
        if value is None:
            return None
        value = max(0.0, min(1.0, float(value)))
        result[f'rag.hybrid_weight_{key}'] = value
    if params.get('relevance_threshold') is not None:
        result['rag.relevance_threshold'] = max(0.0, min(1.0, float(params['relevance_threshold'])))
    if params.get('top_k') is not None:
        result['rag.top_k'] = max(1, min(20, int(params['top_k'])))
    return result or None


class ApplyWeightsForm(BaseModel):
    bot_model_id: Optional[str] = None


@router.post('/runs/{id}/apply-weights')
async def apply_best_weights(
    id: str,
    form: ApplyWeightsForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Кнопка Apply Best Weights to Production: веса optimize-прогона
    записываются в глобальный config и (опционально) в overrides бота."""
    run = await RagasRuns.get_run_by_id(id, db=db)
    if not run:
        raise HTTPException(status_code=404, detail='Run not found')
    if run.mode != 'optimize':
        raise HTTPException(status_code=400, detail='Only optimize runs carry tunable weights')
    weights = extract_weights_from_run(run.metrics)
    if not weights:
        raise HTTPException(status_code=400, detail='Run has no best_params to apply')

    applied_to_config = dict(weights)
    applied_to_config['rag.weights_source'] = f'run:{id}'
    await Config.upsert(applied_to_config)

    if form.bot_model_id:
        model = await Models.get_model_by_id(form.bot_model_id)
        if not model:
            raise HTTPException(status_code=404, detail='Bot model not found')
        meta = model.meta or {}
        meta['rag_overrides'] = {'hybrid_weights': run.metrics['best_params']['hybrid_weights']}
        model.meta = meta
        await Models.update_model_by_id(form.bot_model_id, model)

    return {'status': True, 'applied': weights, 'bot_model_id': form.bot_model_id}
