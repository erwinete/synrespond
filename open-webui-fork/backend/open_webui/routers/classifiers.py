"""REST API управления энкодерами, правилами интентов и инспектором графа HippoRAG.

ТЗ, раздел 4.3 и 8.1. Формат ответов следует паттерну routers/memories.py.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from open_webui.internal.db import get_async_session
from open_webui.models.config import Config
from open_webui.models.intents import (
    EncoderModelModel,
    Encoders,
    IntentFeedbacks,
    IntentRouteModel,
    IntentRoutes,
    MCPHarnesses,
)
from open_webui.utils.auth import get_admin_user, get_verified_user

router = APIRouter()


class EncoderForm(BaseModel):
    name: str = Field(min_length=1)
    task: str = 'sequence_classification'
    endpoint_url: Optional[str] = None
    model_path: Optional[str] = None
    labels_json: Optional[dict[str, Any]] = None
    thresholds: Optional[dict[str, float]] = None
    is_active: bool = True
    meta: Optional[dict[str, Any]] = None


class EncoderTestForm(BaseModel):
    text: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    endpoint_url: Optional[str] = None
    tau_high: Optional[float] = None
    tau_low: Optional[float] = None


class IntentRouteForm(BaseModel):
    bot_model_id: str = Field(min_length=1)
    intent_name: str = Field(min_length=1)
    examples: Optional[list[str]] = None
    min_confidence: float = 0.80
    tau_low: float = 0.50
    action_type: str = 'RAG_COLLECTION'
    target_collection_ids: Optional[list[str]] = None
    use_hipporag: bool = False
    hybrid_weights: Optional[dict[str, float]] = None
    system_prompt_template: Optional[str] = None
    direct_response: Optional[str] = None
    webhook_config: Optional[dict[str, Any]] = None
    required_slots: Optional[list[dict[str, Any]]] = None
    priority: int = 0
    is_active: bool = True


# Энкодеры


@router.get('/encoders', response_model=list[EncoderModelModel])
async def list_encoders(
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    return await Encoders.get_available_encoders(db=db)


@router.post('/encoders/create', response_model=EncoderModelModel)
async def create_encoder(
    form: EncoderForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    data = form.model_dump()
    data['user_id'] = user.id
    return await Encoders.insert_new_encoder(data, db=db)


@router.post('/encoders/{id}/update', response_model=EncoderModelModel)
async def update_encoder(
    id: str,
    form: EncoderForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    encoder = await Encoders.update_encoder_by_id(id, form.model_dump(), db=db)
    if not encoder:
        raise HTTPException(status_code=404, detail='Encoder not found')
    return encoder


@router.delete('/encoders/{id}/delete')
async def delete_encoder(
    id: str,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    deleted = await Encoders.delete_encoder_by_id(id, db=db)
    if not deleted:
        raise HTTPException(status_code=404, detail='Encoder not found')
    return {'status': True}


@router.post('/encoders/test')
async def test_encoder(
    form: EncoderTestForm,
    user=Depends(get_admin_user),
):
    """Прокси-тест инференса: гистограмма top-k и время ответа (ТЗ, раздел 4.3)."""
    base_url = (form.endpoint_url or await Config.get('intent_classifier.url') or '').rstrip('/')
    if not base_url:
        raise HTTPException(status_code=400, detail='Encoder service URL is not configured')
    payload: dict[str, Any] = {'text': form.text, 'model_id': form.model_path}
    if form.tau_high is not None:
        payload['tau_high'] = form.tau_high
    if form.tau_low is not None:
        payload['tau_low'] = form.tau_low
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f'{base_url}/v1/classify', json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f'Encoder service unreachable: {exc}')


# Правила маршрутизации


@router.get('/routes', response_model=list[IntentRouteModel])
async def list_routes(
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    return await IntentRoutes.get_routes(db=db)


@router.get('/routes/model/{model_id}', response_model=list[IntentRouteModel])
async def list_routes_for_model(
    model_id: str,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    return await IntentRoutes.get_active_routes_by_model(model_id, db=db)


@router.post('/routes/create', response_model=IntentRouteModel)
async def create_route(
    form: IntentRouteForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    return await IntentRoutes.insert_new_route(form.model_dump(), db=db)


@router.post('/routes/{id}/update', response_model=IntentRouteModel)
async def update_route(
    id: str,
    form: IntentRouteForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    route = await IntentRoutes.update_route_by_id(id, form.model_dump(), db=db)
    if not route:
        raise HTTPException(status_code=404, detail='Intent route not found')
    return route


@router.delete('/routes/{id}/delete')
async def delete_route(
    id: str,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    deleted = await IntentRoutes.delete_route_by_id(id, db=db)
    if not deleted:
        raise HTTPException(status_code=404, detail='Intent route not found')
    return {'status': True}


# Инспектор графа HippoRAG (ТЗ, раздел 8.2)


@router.get('/graph/{collection_id}')
async def get_knowledge_graph(
    collection_id: str,
    request: Request,
    user=Depends(get_admin_user),
):
    """Узлы и ребра графа коллекции для визуализатора."""
    from open_webui.retrieval.hipporag_engine import get_hipporag_adapter

    graph = await get_hipporag_adapter().get_graph(request, collection_id)
    return graph


@router.post('/graph/{collection_id}/test')
async def test_graph_query(
    collection_id: str,
    form: dict[str, Any],
    request: Request,
    user=Depends(get_admin_user),
):
    """Тестовый прогон Personalized PageRank: подсветка пройденных узлов."""
    from open_webui.retrieval.hipporag_engine import get_hipporag_adapter

    query = str(form.get('query') or '').strip()
    top_k = int(form.get('top_k') or 10)
    if not query:
        raise HTTPException(status_code=400, detail='Query is required')
    docs = await get_hipporag_adapter().retrieve_ppr(request, collection_id, [query], top_k=top_k)
    return {'query': query, 'results': docs}


# Active Learning Flywheel (киллер-фича №1)


class FeedbackLabelForm(BaseModel):
    intent_name: str = Field(min_length=1)


class FeedbackTrainForm(BaseModel):
    base_model_id: str = Field(min_length=1)
    bot_model_id: Optional[str] = None
    out_model_id: Optional[str] = None
    epochs: int = 4
    activate: bool = True
    quantize: bool = False


@router.get('/feedback')
async def list_feedback(
    status: Optional[str] = None,
    bot_model_id: Optional[str] = None,
    limit: int = 200,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Очередь серой зоны: фразы энкодера с низкой уверенностью."""
    return await IntentFeedbacks.get_feedback(
        status=status, bot_model_id=bot_model_id, limit=limit, db=db
    )


@router.post('/feedback/{id}/label')
async def label_feedback(
    id: str,
    form: FeedbackLabelForm,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Подтверждение метки аналитиком: Enter в UI Active Learning Queue."""
    updated = await IntentFeedbacks.update_feedback_by_id(
        id, {'labeled_intent': form.intent_name, 'status': 'labeled'}, db=db
    )
    if not updated:
        raise HTTPException(status_code=404, detail='Feedback not found')
    return updated


@router.post('/feedback/{id}/dismiss')
async def dismiss_feedback(
    id: str,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    updated = await IntentFeedbacks.update_feedback_by_id(id, {'status': 'dismissed'}, db=db)
    if not updated:
        raise HTTPException(status_code=404, detail='Feedback not found')
    return updated


@router.post('/feedback/train')
async def train_from_feedback(
    form: FeedbackTrainForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Fine-Tune & Hot-Swap: дообучение энкодера на размеченных фразах.

    Примеры уходят в encoder-classifier /v1/train; задача там выполняется
    в фоне, статус — через GET /classifiers/train/{job_id}.
    """
    examples_rows = await IntentFeedbacks.get_labeled_feedback(form.bot_model_id, db=db)
    examples = [
        {'text': row.text, 'label': row.labeled_intent}
        for row in examples_rows
        if row.labeled_intent
    ]
    if len(examples) < 8:
        raise HTTPException(
            status_code=400,
            detail=f'Need at least 8 labeled examples, have {len(examples)}',
        )

    base_url = (await Config.get('intent_classifier.url') or '').rstrip('/')
    if not base_url:
        raise HTTPException(status_code=400, detail='Encoder service URL is not configured')

    payload = {
        'base_model_id': form.base_model_id,
        'out_model_id': form.out_model_id,
        'epochs': form.epochs,
        'activate': form.activate,
        'quantize': form.quantize,
        'examples': examples,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f'{base_url}/v1/train', json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f'Encoder service unreachable: {exc}')


@router.get('/train/{job_id}')
async def train_job_status(
    job_id: str,
    user=Depends(get_admin_user),
):
    """Статус задачи дообучения (прокси в encoder-classifier)."""
    base_url = (await Config.get('intent_classifier.url') or '').rstrip('/')
    if not base_url:
        raise HTTPException(status_code=400, detail='Encoder service URL is not configured')
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f'{base_url}/v1/train/{job_id}')
            response.raise_for_status()
            return response.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f'Encoder service unreachable: {exc}')


# Reasoning Path по графу (киллер-фича №6)


@router.post('/graph/{collection_id}/path')
async def graph_reasoning_path(
    collection_id: str,
    form: dict[str, Any],
    request: Request,
    user=Depends(get_verified_user),
):
    """Цепочка логического вывода: сущность запроса -> ... -> чанк-доказательство."""
    from open_webui.retrieval.hipporag_engine import get_hipporag_adapter

    query = str(form.get('query') or '').strip()
    if not query:
        raise HTTPException(status_code=400, detail='Query is required')
    path = await get_hipporag_adapter().get_reasoning_path(request, collection_id, query)
    return path


# Agent Copilot (киллер-фича №7): подсказки оператору без генерации


class CopilotForm(BaseModel):
    query: str = Field(min_length=1)
    collection_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)


@router.post('/copilot')
async def agent_copilot(
    form: CopilotForm,
    request: Request,
    user=Depends(get_verified_user),
):
    """Real-Time Agent Assist: карточки знаний по текущей реплике разговора.

    Только ретривер (HippoRAG + гибрид), без вызова LLM — ответ за десятки мс.
    """
    from open_webui.retrieval.hybrid_fusion import _result_to_docs
    from open_webui.retrieval.utils import query_collection

    collection_names = form.collection_ids
    if not collection_names:
        raise HTTPException(status_code=400, detail='collection_ids is required')

    embedding_function = request.app.state.EMBEDDING_FUNCTION
    results = await query_collection(
        request,
        collection_names=collection_names,
        queries=[form.query],
        embedding_function=embedding_function,
        k=form.top_k,
        user=user,
    )
    docs = _result_to_docs(results, 'dense+bm25')[: form.top_k]
    return {
        'query': form.query,
        'cards': [
            {
                'text': doc['text'],
                'metadata': doc['metadata'],
            }
            for doc in docs
        ],
    }


# AI-Ready: семантический линтер манифеста (Stage 4, раздел 4.3)


@router.post('/manifest/validate')
async def validate_manifest(
    manifest: dict[str, Any],
    user=Depends(get_verified_user),
):
    """Валидация BotBehaviorManifest: 6 правил консистентности + подсказки агенту."""
    from open_webui.models.knowledge import Knowledges
    from open_webui.utils.ai_ready_schema import validate_bot_manifest

    harnesses = await MCPHarnesses.get_harnesses()
    harness_tools = {
        harness.id: [
            tool.get('name')
            for tool in ((harness.discovered_tools_schema or {}).get('tools') or [])
            if isinstance(tool, dict) and tool.get('name')
        ]
        for harness in harnesses
    }
    knowledges = await Knowledges.get_knowledge_bases()
    encoder_model_id = manifest.get('encoder_model_id') or 'rubert-intents-v1'
    encoder = await Encoders.get_encoder_by_id(encoder_model_id)
    labels = (encoder.labels_json or {}) if encoder else {}
    context = {
        'encoder_labels': {str(v): True for v in labels.values()} if labels else {},
        'knowledge_ids': [kb.id for kb in knowledges],
        'harness_tools': harness_tools,
    }
    return validate_bot_manifest(manifest, context)


# Внешние MCP-харнесы (Stage 4, направление Б)


class HarnessForm(BaseModel):
    name: str = Field(min_length=1)
    description_for_agents: Optional[str] = None
    transport: str = 'streamable_http'
    endpoint_or_cmd: str = Field(min_length=1)
    auth_token: Optional[str] = None
    timeout_ms: int = 5000
    is_active: bool = True


@router.get('/harnesses')
async def list_harnesses(
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    return await MCPHarnesses.get_harnesses(db=db)


@router.post('/harnesses/create')
async def create_harness(
    form: HarnessForm,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    data = form.model_dump()
    data['discovered_tools_schema'] = {}
    return await MCPHarnesses.insert_new_harness(data, db=db)


@router.delete('/harnesses/{id}/delete')
async def delete_harness(
    id: str,
    user=Depends(get_admin_user),
    db: AsyncSession = Depends(get_async_session),
):
    deleted = await MCPHarnesses.delete_harness_by_id(id, db=db)
    if not deleted:
        raise HTTPException(status_code=404, detail='Harness not found')
    return {'status': True}


@router.post('/harnesses/{id}/discover')
async def discover_harness_tools(
    id: str,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    """tools/list у внешнего харнеса + кэш схемы в discovered_tools_schema."""
    from open_webui.utils.mcp_harness import discover_tools

    harness = await MCPHarnesses.get_harness_by_id(id, db=db)
    if not harness:
        raise HTTPException(status_code=404, detail='Harness not found')
    result = await discover_tools(harness)
    if result.get('ok'):
        await MCPHarnesses.update_harness_by_id(
            id, {'discovered_tools_schema': {'tools': result.get('tools') or []}}, db=db
        )
    else:
        raise HTTPException(status_code=502, detail=str(result.get('error')))
    return {'tools': result.get('tools') or []}
