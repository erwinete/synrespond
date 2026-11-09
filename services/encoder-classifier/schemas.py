"""Pydantic-схемы REST API классификатора интентов и слотов."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Запрос на классификацию интента."""

    text: str = Field(min_length=1, max_length=8000)
    model_id: str = "rubert-intents-v1"
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int = Field(default=5, ge=1, le=50)
    # Переопределение порогов из label_map.json
    tau_high: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tau_low: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Candidate(BaseModel):
    intent_id: str
    intent_name: str
    confidence: float


class ClassifyResponse(BaseModel):
    intent_id: str
    intent_name: str
    confidence: float
    # Действие по порогам: direct | disambiguate | fallback
    action: str
    top_k_candidates: list[Candidate]
    latency_ms: float


class BatchClassifyRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)
    model_id: str = "rubert-intents-v1"
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int = Field(default=5, ge=1, le=50)


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


class ZeroShotRequest(BaseModel):
    """Zero-shot классификация через NLI-модель. Метки задаются в запросе."""

    text: str = Field(min_length=1, max_length=8000)
    labels: list[str] = Field(min_length=2, max_length=200)
    model_id: str = "deberta-v3-nli"
    hypothesis_template: str = "This example is {}."
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int = Field(default=5, ge=1, le=50)


class SlotSpan(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float
    source: Optional[str] = None


class ExtractSlotsRequest(BaseModel):
    text: str = Field(default="", max_length=8000)
    # BUG-06: пакетный режим — один прогон ONNX на список текстов (до 32)
    texts: Optional[list[str]] = Field(default=None, max_length=32)
    model_id: str = "slots-ner-v1"


class ExtractSlotsResponse(BaseModel):
    spans: list[SlotSpan]
    # Спаны по каждому тексту из texts (порядок совпадает с запросом)
    batch_spans: Optional[list[list[SlotSpan]]] = None
    latency_ms: float


class ModelInfo(BaseModel):
    model_id: str
    task: str
    labels_count: int
    thresholds: dict[str, float]
    loaded: bool


class ModelsListResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    device: str
    providers: list[str]
    models_loaded: int
    version: str


class ErrorResponse(BaseModel):
    detail: Any


class AnalyzeRequest(BaseModel):
    """Комбинированный вызов: классификация интента + извлечение слотов."""

    text: str = Field(min_length=1, max_length=8000)
    model_id: str = 'rubert-intents-v1'
    slot_model_id: str = ''
    temperature: float = Field(default=1.0, gt=0.0)
    tau_high: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tau_low: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    classification: ClassifyResponse
    slots: list[SlotSpan]
    latency_ms: float


class GuardRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class GuardResponse(BaseModel):
    injection: bool
    score: float
    labels: dict[str, Any]
    engine: str


class PipelineAnalyzeRequest(BaseModel):
    """ARCH-01: guard + классификация + слоты за один сетевой RTT."""

    text: str = Field(min_length=1, max_length=8000)
    model_id: str = "rubert-intents-v1"
    slot_model_id: str = ""
    guard_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, gt=0.0)
    tau_high: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tau_low: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PipelineAnalyzeResponse(BaseModel):
    guard: Optional[GuardResponse] = None
    classification: Optional[ClassifyResponse] = None
    slots: list[SlotSpan] = []
    latency_ms: float


class TrainRequest(BaseModel):
    base_model_id: str = Field(min_length=1)
    examples: list[dict[str, str]] = Field(min_length=8)
    # Д4: эталонная базовая выборка против катастрофического забывания
    anchor_examples: list[dict[str, str]] = Field(default_factory=list)
    out_model_id: Optional[str] = None
    epochs: int = Field(default=4, ge=1, le=50)
    lr: float = Field(default=5e-5, gt=0.0)
    batch_size: int = Field(default=8, ge=1, le=64)
    activate: bool = True
    quantize: bool = False
    # Д4: форс-подмена несмотря на Rollback Gate
    force_swap: bool = False


class TrainJobResponse(BaseModel):
    job_id: str
    status: str


class TrainJobStatus(BaseModel):
    job_id: str
    status: str
    params: dict[str, Any]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[dict[str, Any]] = None
    blocked_by_gate: Optional[bool] = None


class ReloadResponse(BaseModel):
    model_id: str
    reloaded: bool
