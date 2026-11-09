"""Реестр ONNX-моделей и инференс для классификации интентов, NLI и NER.

Формат каталога модели: MODELS_DIR/<model_id>/
  model.onnx       — экспортированная сеть (см. scripts/export_onnx.py)
  tokenizer.json и прочие файлы токенизатора (AutoTokenizer.from_pretrained)
  label_map.json   — метаданные задачи, метки и пороги (см. README)
"""

import json
import logging
import os
import threading
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from decision import (
    DEFAULT_TAU_HIGH,
    DEFAULT_TAU_LOW,
    decide,
)

logger = logging.getLogger("encoder-classifier")

TASK_SEQUENCE = "sequence_classification"
TASK_ZERO_SHOT = "zero_shot_nli"
TASK_TOKEN = "token_classification"
TASK_EMBEDDING = "feature_extraction"

_HYPOTHESIS_LABEL = "entailment"


class InvalidModelId(ValueError):
    """SEC-03: недопустимый id модели (traversal или недопустимые символы) — HTTP 400."""


def resolve_providers(setting: str | None = None) -> list[str]:
    """Список провайдеров ONNX Runtime из переменной окружения или auto."""
    setting = (setting or os.getenv("ONNX_PROVIDERS", "auto")).strip().lower()
    if setting not in ("", "auto"):
        return [p.strip() for p in setting.split(",") if p.strip()]
    available = ort.get_available_providers()
    ordered = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    return ordered or ["CPUExecutionProvider"]


class ModelArtifact:
    """Загруженная модель: сессия ONNX, токенизатор и метаданные label_map.json."""

    def __init__(self, model_id: str, models_dir: str):
        # SEC-03: id модели проверяется до конкатенации путей (Path Traversal)
        if not model_id or ".." in model_id or "/" in model_id or "\\" in model_id:
            raise InvalidModelId(f"Invalid model id: {model_id!r}")
        self.model_id = model_id
        self.path = os.path.join(models_dir, model_id)
        model_path = os.path.join(self.path, "model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_id} not found at {model_path}")

        self.meta = self._load_meta()
        self.task = self.meta.get("task", TASK_SEQUENCE)
        thresholds = self.meta.get("thresholds", {})
        self.tau_high = float(thresholds.get("high", DEFAULT_TAU_HIGH))
        self.tau_low = float(thresholds.get("low", DEFAULT_TAU_LOW))
        self.max_length = int(self.meta.get("max_length", os.getenv("MAX_LENGTH", 512)))
        self.entailment_id = int(self.meta.get("entailment_id", 2))

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = os.getenv("ORT_INTRA_OP_THREADS")
        if threads:
            options.intra_op_num_threads = int(threads)

        t0 = time.perf_counter()
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=resolve_providers()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.input_names = [i.name for i in self.session.get_inputs()]
        load_ms = (time.perf_counter() - t0) * 1000
        logger.info("Loaded model %s (%s) in %.0f ms", model_id, self.task, load_ms)

    def _load_meta(self) -> dict[str, Any]:
        meta_path = os.path.join(self.path, "label_map.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Приводим ключи-индексы к int
        if "labels" in raw and isinstance(raw["labels"], dict):
            raw["labels"] = {int(k): v for k, v in raw["labels"].items()}
        return raw

    @property
    def labels(self) -> dict[int, str]:
        return self.meta.get("labels", {})

    def _run(self, features: dict[str, np.ndarray]) -> np.ndarray:
        ort_inputs = {k: v for k, v in features.items() if k in self.input_names}
        return self.session.run(None, ort_inputs)[0]

    def classify(self, text: str) -> np.ndarray:
        """Логиты multi-class / multi-label классификатора. Форма: [num_classes]."""
        enc = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        logits = self._run({k: v for k, v in enc.items()})
        return np.asarray(logits[0])

    def zero_shot(self, text: str, labels: list[str], hypothesis_template: str) -> np.ndarray:
        """Вероятности меток через NLI: P(entailment) для каждой гипотезы."""
        pairs = [(text, hypothesis_template.format(label)) for label in labels]
        enc = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="np",
        )
        logits = self._run({k: v for k, v in enc.items()})
        # logits: [num_pairs, 3] — entailment-колонка задаётся в label_map.json
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        entail = probs[:, self.entailment_id]
        # Нормировка по множеству гипотез (Yin et al., 2019)
        total = entail.sum()
        return entail / total if total > 0 else entail

    def extract_spans(self, text: str, aggregation_threshold: float) -> list[dict[str, Any]]:
        """Token Classification: слияние BIO-меток в спаны по offset_mapping."""
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        offsets = enc.pop("offset_mapping")[0]
        logits = self._run({k: v for k, v in enc.items()})  # [seq, num_labels]
        probs = softmax_batch(logits)
        pred_ids = probs.argmax(axis=-1)

        id2label = {int(k): v for k, v in self.labels.items()}
        spans: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        special = self.tokenizer.all_special_ids
        for idx, pred_id in enumerate(pred_ids):
            start, end = (int(offsets[idx][0]), int(offsets[idx][1]))
            if idx in special or end <= start:
                if current:
                    spans.append(current)
                    current = None
                continue
            label = id2label.get(int(pred_id), "O")
            conf = float(probs[idx][pred_id])
            if label == "O" or conf < aggregation_threshold:
                if current:
                    spans.append(current)
                    current = None
                continue
            if label.startswith("B-") or current is None or current["label"] != label[2:]:
                if current:
                    spans.append(current)
                current = {
                    "label": label[2:] if label.startswith(("B-", "I-")) else label,
                    "start": start,
                    "end": end,
                    "confidence": conf,
                }
            else:
                current["end"] = end
                current["confidence"] = min(current["confidence"], conf)
        if current:
            spans.append(current)
        for span in spans:
            span["text"] = text[span["start"] : span["end"]]
        return spans

    def extract_spans_batch(self, texts: list[str], aggregation_threshold: float) -> list[list[dict[str, Any]]]:
        """Пакетный Token Classification: один прогон ONNX на список текстов (BUG-06)."""
        if not texts:
            return []
        enc = self.tokenizer(
            list(texts),
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        offsets = enc.pop("offset_mapping")  # [batch, seq, 2]
        logits = self._run({k: v for k, v in enc.items()})  # [batch, seq, num_labels]
        probs = softmax_batch(logits)
        pred_ids = probs.argmax(axis=-1)

        id2label = {int(k): v for k, v in self.labels.items()}
        special = set(self.tokenizer.all_special_ids)
        result: list[list[dict[str, Any]]] = []
        for b, text in enumerate(texts):
            spans: list[dict[str, Any]] = []
            current: dict[str, Any] | None = None
            for idx in range(pred_ids.shape[1]):
                start, end = int(offsets[b][idx][0]), int(offsets[b][idx][1])
                pred_id = int(pred_ids[b][idx])
                if pred_id in special or end <= start:
                    if current:
                        spans.append(current)
                        current = None
                    continue
                label = id2label.get(pred_id, "O")
                conf = float(probs[b][idx][pred_id])
                if label == "O" or conf < aggregation_threshold:
                    if current:
                        spans.append(current)
                        current = None
                    continue
                if label.startswith("B-") or current is None or current["label"] != label[2:]:
                    if current:
                        spans.append(current)
                    current = {
                        "label": label[2:] if label.startswith(("B-", "I-")) else label,
                        "start": start,
                        "end": end,
                        "confidence": conf,
                    }
                else:
                    current["end"] = end
                    current["confidence"] = min(current["confidence"], conf)
            if current:
                spans.append(current)
            for span in spans:
                span["text"] = text[span["start"] : span["end"]]
            result.append(spans)
        return result


    def embed(self, texts: list[str]) -> "np.ndarray":
        """К3 (Stage 4): mean-pooled эмбеддинги из feature-extraction ONNX."""
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="np"
        )
        ort_inputs = {k: v for k, v in enc.items() if k in self.input_names}
        outputs = self.session.run(None, ort_inputs)
        hidden = outputs[0]  # [batch, seq, dim]
        mask = ort_inputs.get("attention_mask")
        if mask is not None:
            mask = mask.astype(hidden.dtype)[..., None]
            summed = (hidden * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1e-9, None)
            pooled = summed / counts
        else:
            pooled = hidden.mean(axis=1)
        # L2-нормировка
        norms = np.clip(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-9, None)
        return pooled / norms

    def zero_shot_embedding(self, text: str, labels: list[str], hypothesis_template: str) -> "np.ndarray":
        """К2 (Stage 4): zero-shot без NLI-головы — косинусная близость
        эмбеддинга текста и формулировок меток, softmax по близостям."""
        hypotheses = [hypothesis_template.format(label) for label in labels]
        embeddings = self.embed([text, *hypotheses])
        text_vec, label_vecs = embeddings[0], embeddings[1:]
        sims = label_vecs @ text_vec
        # Масштаб: softmax по (sim / T), T подобрана для распределения близостей
        scaled = sims / 0.05
        exp = np.exp(scaled - scaled.max())
        return exp / exp.sum()


def softmax_batch(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class ModelRegistry:
    """Ленивый реестр моделей с прогревом и защитой от гонок загрузки."""

    def __init__(self, models_dir: str | None = None):
        self.models_dir = models_dir or os.getenv("MODELS_DIR", "/models")
        self._artifacts: dict[str, ModelArtifact] = {}
        self._lock = threading.Lock()

    def available(self) -> list[str]:
        """ID моделей, у которых есть model.onnx и label_map.json."""
        result: list[str] = []
        if not os.path.isdir(self.models_dir):
            return result
        for name in sorted(os.listdir(self.models_dir)):
            path = os.path.join(self.models_dir, name)
            if os.path.exists(os.path.join(path, "model.onnx")) and os.path.exists(
                os.path.join(path, "label_map.json")
            ):
                result.append(name)
        return result

    def get(self, model_id: str) -> ModelArtifact:
        """Возвращает модель, загружая при первом обращении (потокобезопасно)."""
        # SEC-03: защита от traversal в id модели до обращения к файловой системе
        if not model_id or ".." in model_id or "/" in model_id or "\\" in model_id:
            raise InvalidModelId(f"Invalid model id: {model_id!r}")
        artifact = self._artifacts.get(model_id)
        if artifact is not None:
            return artifact
        with self._lock:
            artifact = self._artifacts.get(model_id)
            if artifact is None:
                artifact = ModelArtifact(model_id, self.models_dir)
                self._artifacts[model_id] = artifact
            return artifact

    def reload(self, model_id: str) -> ModelArtifact:
        """Сбрасывает кэш артефакта и загружает модель с диска заново (BUG-05)."""
        if not model_id or ".." in model_id or "/" in model_id or "\\" in model_id:
            raise InvalidModelId(f"Invalid model id: {model_id!r}")
        with self._lock:
            self._artifacts.pop(model_id, None)
        return self.get(model_id)

    def warmup(self) -> list[str]:
        """Загружает все доступные модели. Возвращает список ошибок."""
        errors: list[str] = []
        for model_id in self.available():
            try:
                self.get(model_id)
            except Exception as exc:  # noqa: BLE001 — реестр не должен падать целиком
                logger.error("Warmup failed for %s: %s", model_id, exc)
                errors.append(model_id)
        return errors
