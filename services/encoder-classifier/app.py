"""FastAPI-сервис классификации интентов и извлечения слотов (ТЗ, раздел 4.1).

Маршруты:
  GET  /health                 — состояние сервиса
  GET  /v1/models              — каталог доступных моделей
  POST /v1/classify            — multi-class/multi-label классификация интента
  POST /v1/classify/batch      — пакетная классификация
  POST /v1/zero-shot           — NLI-классификация по списку меток из запроса
  POST /v1/extract-slots       — Token Classification (NER/заполнение слотов)
"""

import logging
import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

import slots_extractor
from decision import DIRECT, decide
from engine import (
    TASK_EMBEDDING,
    TASK_SEQUENCE,
    TASK_TOKEN,
    TASK_ZERO_SHOT,
    InvalidModelId,
    ModelRegistry,
    resolve_providers,
)
from guard import detect_injection
from schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    Candidate,
    ClassifyRequest,
    ClassifyResponse,
    ExtractSlotsRequest,
    ExtractSlotsResponse,
    GuardRequest,
    GuardResponse,
    HealthResponse,
    ModelsListResponse,
    ModelInfo,
    PipelineAnalyzeRequest,
    PipelineAnalyzeResponse,
    ReloadResponse,
    SlotSpan,
    TrainJobResponse,
    TrainJobStatus,
    TrainRequest,
    ZeroShotRequest,
)
import training

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("encoder-classifier")

VERSION = "1.0.0"

app = FastAPI(
    title="Encoder Intent & Slot Classifier",
    description="Классификация интентов, zero-shot NLI и NER на ONNX-энкодерах. Целевая задержка 5-20 мс.",
    version=VERSION,
)

registry = ModelRegistry()
_providers = resolve_providers()


@app.exception_handler(InvalidModelId)
async def invalid_model_id_handler(request, exc: InvalidModelId):
    # SEC-03 / DoD: traversal или недопустимые символы в model_id -> HTTP 400
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, detail=str(exc))


def _device_label() -> str:
    return "cuda" if "CUDAExecutionProvider" in _providers else "cpu"


@app.on_event("startup")
def startup() -> None:
    if os.getenv("WARMUP_ON_STARTUP", "true").lower() in ("1", "true", "yes"):
        errors = registry.warmup()
        if errors:
            logger.warning("Warmup errors for: %s", ", ".join(errors))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=_device_label(),
        providers=_providers,
        models_loaded=len(registry._artifacts),
        version=VERSION,
    )


@app.get("/v1/models", response_model=ModelsListResponse)
def list_models() -> ModelsListResponse:
    items: list[ModelInfo] = []
    for model_id in registry.available():
        try:
            artifact = registry._artifacts.get(model_id)
            task = artifact.task if artifact else None
            if task is None:
                import json

                with open(os.path.join(registry.models_dir, model_id, "label_map.json"), encoding="utf-8") as f:
                    meta = json.load(f)
                task = meta.get("task", TASK_SEQUENCE)
                thresholds = meta.get("thresholds", {})
            else:
                thresholds = {"high": artifact.tau_high, "low": artifact.tau_low}
            items.append(
                ModelInfo(
                    model_id=model_id,
                    task=task,
                    labels_count=len(artifact.labels) if artifact else len(meta.get("labels", {})),
                    thresholds=thresholds,
                    loaded=artifact is not None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot read meta for %s: %s", model_id, exc)
    return ModelsListResponse(models=items)


def _classify_one(req: ClassifyRequest) -> ClassifyResponse:
    t0 = time.perf_counter()
    try:
        artifact = registry.get(req.model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}")

    if artifact.task not in (TASK_SEQUENCE, TASK_ZERO_SHOT):
        raise HTTPException(status_code=400, detail=f"Model {req.model_id} is {artifact.task}, use /v1/extract-slots")

    try:
        if artifact.task == TASK_ZERO_SHOT:
            labels_meta = artifact.meta.get("zero_shot_labels") or []
            labels = [str(v) for v in labels_meta] or [str(v) for v in artifact.labels.values()]
            template = artifact.meta.get("hypothesis_template", "This example is {}.")
            probs = artifact.zero_shot(req.text, labels, template)
        else:
            import numpy as np

            logits = artifact.classify(req.text)
            scaled = logits / max(req.temperature, 1e-5)
            exp = np.exp(scaled - scaled.max())
            probs = exp / exp.sum()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    order = probs.argsort()[::-1][: req.top_k]
    labels_map = artifact.labels
    candidates = [
        Candidate(
            intent_id=str(int(i)),
            intent_name=labels_map.get(int(i), f"class_{int(i)}"),
            confidence=float(probs[i]),
        )
        for i in order
    ]
    best = candidates[0]
    tau_high = req.tau_high if req.tau_high is not None else artifact.tau_high
    tau_low = req.tau_low if req.tau_low is not None else artifact.tau_low
    verdict = decide(best.intent_id, best.intent_name, best.confidence, tau_high, tau_low)
    latency = (time.perf_counter() - t0) * 1000

    return ClassifyResponse(
        intent_id=verdict.intent_id,
        intent_name=verdict.intent_name,
        confidence=verdict.confidence,
        action=DIRECT if verdict.action == DIRECT else verdict.action,
        top_k_candidates=candidates,
        latency_ms=round(latency, 2),
    )


@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest) -> ClassifyResponse:
    # Инференс в threadpool: не блокируем event loop
    return await run_in_threadpool(_classify_one, req)


@app.post("/v1/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(req: BatchClassifyRequest) -> BatchClassifyResponse:
    single = ClassifyRequest(text="", model_id=req.model_id, temperature=req.temperature, top_k=req.top_k)
    results: list[ClassifyResponse] = []
    for text in req.texts:
        single.text = text
        results.append(await run_in_threadpool(_classify_one, single))
    return BatchClassifyResponse(results=results)


@app.post("/v1/zero-shot", response_model=ClassifyResponse)
async def zero_shot(req: ZeroShotRequest) -> ClassifyResponse:
    t0 = time.perf_counter()
    try:
        artifact = registry.get(req.model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if artifact.task not in (TASK_ZERO_SHOT, TASK_EMBEDDING):
        raise HTTPException(status_code=400, detail=f"Model {req.model_id} is {artifact.task}, не NLI/zero-shot")

    try:
        if artifact.meta.get('zero_shot_mode') == 'embedding' or artifact.task == TASK_EMBEDDING:
            # К2: zero-shot без NLI-головы — эмбеддинговая близость меток
            probs = artifact.zero_shot_embedding(req.text, req.labels, req.hypothesis_template)
        else:
            probs = artifact.zero_shot(req.text, req.labels, req.hypothesis_template)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    import numpy as np

    order = probs.argsort()[::-1][: req.top_k]
    candidates = [
        Candidate(intent_id="-", intent_name=req.labels[int(i)], confidence=float(probs[i]))
        for i in order
    ]
    best = candidates[0]
    latency = (time.perf_counter() - t0) * 1000
    return ClassifyResponse(
        intent_id="-",
        intent_name=best.intent_name,
        confidence=best.confidence,
        action=decide(best.intent_name, best.intent_name, best.confidence).action,
        top_k_candidates=candidates,
        latency_ms=round(latency, 2),
    )


@app.post("/v1/extract-slots", response_model=ExtractSlotsResponse)
async def extract_slots(req: ExtractSlotsRequest) -> ExtractSlotsResponse:
    t0 = time.perf_counter()
    if not req.text and not req.texts:
        raise HTTPException(status_code=400, detail="Field 'text' or 'texts' is required")
    try:
        artifact = registry.get(req.model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if artifact.task != TASK_TOKEN:
        raise HTTPException(status_code=400, detail=f"Model {req.model_id} is {artifact.task}, не token_classification")

    threshold = float(artifact.meta.get("aggregation_threshold", 0.6))

    if req.texts:
        # BUG-06: батч-режим — один прогон ONNX на весь список текстов
        try:
            batch = await run_in_threadpool(artifact.extract_spans_batch, req.texts, threshold)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
        merged_batch = []
        for text, spans in zip(req.texts, batch):
            regex_spans = slots_extractor.extract_pattern_spans(text)
            merged_batch.append(slots_extractor.merge_spans(spans, regex_spans))
        latency = (time.perf_counter() - t0) * 1000
        return ExtractSlotsResponse(
            spans=[],
            batch_spans=[[SlotSpan(**s) for s in spans] for spans in merged_batch],
            latency_ms=round(latency, 2),
        )

    try:
        spans = await run_in_threadpool(artifact.extract_spans, req.text, threshold)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # К2 (Stage 4): гибридный экстрактор — паттерны заполняют пробелы ONNX
    regex_spans = slots_extractor.extract_pattern_spans(req.text)
    spans = slots_extractor.merge_spans(spans, regex_spans)

    latency = (time.perf_counter() - t0) * 1000
    return ExtractSlotsResponse(
        spans=[SlotSpan(**s) for s in spans],
        latency_ms=round(latency, 2),
    )


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Д1: интент + слоты одним батч-вызовом (~12 мс)."""
    t0 = time.perf_counter()
    classify_req = ClassifyRequest(
        text=req.text,
        model_id=req.model_id,
        temperature=req.temperature,
        tau_high=req.tau_high,
        tau_low=req.tau_low,
    )
    classification = await run_in_threadpool(_classify_one, classify_req)
    spans: list[SlotSpan] = []
    if req.slot_model_id:
        try:
            slots_req = ExtractSlotsRequest(text=req.text, model_id=req.slot_model_id)
            slots_result = await extract_slots(slots_req)
            spans = slots_result.spans
        except HTTPException:
            pass  # слотовая модель не настроена — отдаем только классификацию
    return AnalyzeResponse(
        classification=classification,
        slots=spans,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/v1/guard", response_model=GuardResponse)
async def guard(req: GuardRequest) -> GuardResponse:
    """Ф5: детекция промпт-инъекций и джейлбрейков."""
    result = await detect_injection(registry, req.text, req.threshold)
    return GuardResponse(**result)


@app.post("/v1/pipeline-analyze", response_model=PipelineAnalyzeResponse)
async def pipeline_analyze(req: PipelineAnalyzeRequest) -> PipelineAnalyzeResponse:
    """ARCH-01: guard + classify + extract-slots за один HTTP-вызов (8-15 мс).

    Единый вход для ядра вместо трех последовательных запросов. Ошибка
    любого контура не валит весь пайплайн: контур возвращается как None/[].
    """
    import asyncio

    t0 = time.perf_counter()

    async def _guard() -> Optional[GuardResponse]:
        if not req.check_guard:
            return None
        try:
            result = await detect_injection(registry, req.text, req.guard_threshold)
            return GuardResponse(**result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pipeline guard stage failed: %s", exc)
            return None

    async def _classify() -> Optional[ClassifyResponse]:
        try:
            return await run_in_threadpool(
                _classify_one,
                ClassifyRequest(
                    text=req.text,
                    model_id=req.model_id,
                    temperature=req.temperature,
                    tau_high=req.tau_high,
                    tau_low=req.tau_low,
                ),
            )
        except HTTPException as exc:
            if exc.status_code == 404:
                return None  # модель не найдена — ядро уйдет в fallback
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pipeline classify stage failed: %s", exc)
            return None

    async def _slots() -> list[SlotSpan]:
        if not req.slot_model_id:
            return []
        try:
            result = await extract_slots(
                ExtractSlotsRequest(text=req.text, model_id=req.slot_model_id)
            )
            return result.spans
        except Exception:  # noqa: BLE001
            return []

    guard_result, classification, spans = await asyncio.gather(_guard(), _classify(), _slots())
    return PipelineAnalyzeResponse(
        guard=guard_result,
        classification=classification,
        slots=spans,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/v1/train", response_model=TrainJobResponse, status_code=202)
async def start_training(req: TrainRequest) -> TrainJobResponse:
    """Ф1/Д4: файнтюн в изолированном подпроцессе. 202 Accepted + job_id.

    Обучение не блокирует event loop и не деградирует /v1/classify;
    прогресс — GET /v1/train/{job_id} (эпохи, loss, F1, Rollback Gate).
    """
    job_id = training.start_train_job(
        registry, registry.models_dir, req.model_dump()
    )
    return TrainJobResponse(job_id=job_id, status="queued")


@app.get("/v1/train/{job_id}", response_model=TrainJobStatus)
async def training_status(job_id: str) -> TrainJobStatus:
    job = training.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return TrainJobStatus(**job)


@app.post("/v1/models/{model_id}/reload", response_model=ReloadResponse)
async def reload_model(model_id: str) -> ReloadResponse:
    """Горячая перезагрузка ONNX-модели без даунтайма."""
    try:
        registry.reload(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ReloadResponse(model_id=model_id, reloaded=True)


@app.post("/v1/models/{model_id}/rollback", response_model=ReloadResponse)
async def rollback_model(model_id: str) -> ReloadResponse:
    """PERF-CORE-09: 1-клик откат модели к предыдущей стабильной версии.

    Подменяет каталог модели на {model_id}_prev, прогревает реестр весов.
    """
    import shutil

    if not model_id or ".." in model_id or "/" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail=f"Invalid model id: {model_id!r}")
    final_dir = os.path.join(registry.models_dir, model_id)
    prev_dir = f"{final_dir}_prev"
    if not os.path.isdir(prev_dir):
        raise HTTPException(status_code=404, detail=f"No previous stable version for {model_id}")
    trash_dir = f"{final_dir}_rollback_{uuid.uuid4().hex[:6]}"
    # Откатная подмена: prev -> current, а откатываемая текущая -> новый prev
    os.replace(final_dir, trash_dir)
    try:
        os.replace(prev_dir, final_dir)
    except Exception:
        os.replace(trash_dir, final_dir)  # восстановление при ошибке
        raise
    os.replace(trash_dir, prev_dir)
    try:
        registry.reload(model_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Rollback reload failed: {exc}")
    return ReloadResponse(model_id=model_id, reloaded=True)

from pydantic import BaseModel as _BaseModel


class EmbeddingsRequest(_BaseModel):
    input: list[str] | str
    model: str = "rubert-embed-v1"


class RerankRequest(_BaseModel):
    model: str = "rubert-embed-v1"
    query: str
    documents: list[str]
    top_n: int | None = None
    raw_scores: bool = False
    return_text: bool = False


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest):
    """К3 (Stage 4): OpenAI-совместимый локальный эмбеддер без torch в ядре."""
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(status_code=400, detail="input is empty")
    try:
        artifact = registry.get(req.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if artifact.task != TASK_EMBEDDING:
        raise HTTPException(status_code=400, detail=f"Model {req.model} is {artifact.task}, нужна feature_extraction")
    try:
        vectors = await run_in_threadpool(artifact.embed, texts)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    return {
        "object": "list",
        "model": req.model,
        "data": [
            {"object": "embedding", "index": idx, "embedding": [round(float(v), 6) for v in vector]}
            for idx, vector in enumerate(vectors)
        ],
    }


@app.post("/rerank")
async def rerank(req: RerankRequest):
    """К3: TEI-совместимый реранкер (bi-encoder: косинус запроса и документов)."""
    try:
        artifact = registry.get(req.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if artifact.task != TASK_EMBEDDING:
        raise HTTPException(status_code=400, detail=f"Model {req.model} is {artifact.task}, нужна feature_extraction")
    if not req.documents:
        raise HTTPException(status_code=400, detail="documents is empty")
    try:
        vectors = await run_in_threadpool(artifact.embed, [req.query, *req.documents])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    query_vec, doc_vecs = vectors[0], vectors[1:]
    scores = doc_vecs @ query_vec
    order = scores.argsort()[::-1]
    if req.top_n is not None:
        order = order[: req.top_n]
    results = [
        {"index": int(idx), "score": round(float(scores[idx]), 6),
         **({"text": req.documents[idx]} if req.return_text else {})}
        for idx in order
    ]
    return {"results": results}
