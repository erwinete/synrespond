"""Акустическая оценка эмоционального состояния абонента (киллер-фича №3).

Основной путь — ONNX-аудио-энкодер (например, wav2vec2-эмоции, каталог
VOICE_EMOTION_MODEL_PATH с model.onnx + label_map.json, где есть метка
anger/urgency). Без модели — прокси-скор по акустике: энергия (громкость),
перепад энергии и доля фрикаций через zero-crossing rate. Прокси не
семантичен, но устойчиво ловит крик и раздражение; честно помечен engine.
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Optional

import numpy as np

from config import EMOTION_MODEL_PATH, EMOTION_SAMPLE_WINDOW_S

log = logging.getLogger(__name__)

_session = {'onnx': None, 'warned': False}


def _load_onnx():
    if _session['onnx'] is not None or _session['warned']:
        return _session['onnx']
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            f'{EMOTION_MODEL_PATH}/model.onnx', providers=['CPUExecutionProvider']
        )
        with open(f'{EMOTION_MODEL_PATH}/label_map.json', encoding='utf-8') as f:
            meta = json.load(f)
        _session['onnx'] = (session, meta)
    except Exception as exc:  # noqa: BLE001
        if not _session['warned']:
            log.info('Emotion ONNX model unavailable, using acoustic proxy: %s', exc)
            _session['warned'] = True
    return _session['onnx']


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _zcr(samples: np.ndarray) -> float:
    if samples.size < 2:
        return 0.0
    return float(np.mean(np.diff(np.signbit(samples.astype(np.int32)))))


def _acoustic_proxy(samples: np.ndarray) -> float:
    """Прокси-скор гнева: громкость + резкие перепады + высокая частота переходов."""
    if samples.size == 0:
        return 0.0
    int16 = samples.astype(np.int16)
    rms = _rms(int16)
    # Нормировка громкости: 500 — спокойная речь, 8000+ — крик
    loudness = min(1.0, max(0.0, (rms - 500) / 7500))
    # Динамический диапазон внутри реплики (резкие всплески = раздражение)
    window = max(len(int16) // 8, 80)
    peaks = [ _rms(int16[i : i + window]) for i in range(0, len(int16) - window, window) ] or [0.0]
    variability = min(1.0, (max(peaks) - min(peaks)) / 6000) if len(peaks) > 1 else 0.0
    # Высокий ZCR указывает на напряженные фрикации/шипение
    tension = min(1.0, max(0.0, (_zcr(int16) - 0.25) * 2.0))
    return round(min(1.0, 0.55 * loudness + 0.25 * variability + 0.20 * tension), 3)


def estimate(pcm: bytes, sample_rate: int) -> Optional[float]:
    """Скор гнева/срочности 0..1 для реплики абонента. None — оценка отключена."""
    if not pcm:
        return None
    loaded = _load_onnx()
    if loaded is not None:
        session, meta = loaded
        try:
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            max_len = int(sample_rate * EMOTION_SAMPLE_WINDOW_S)
            samples = samples[:max_len]
            # ONNX-модели эмоций принимают float32 моно 16 кГц
            if sample_rate != 16000:
                samples = _linear_to_16k(samples, sample_rate)
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: samples[np.newaxis, :]})
            logits = np.asarray(outputs[0][0])
            labels = meta.get('labels', {})
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            anger_idx = next(
                (int(k) for k, v in labels.items() if str(v).lower() in ('anger', 'angry', 'urgency')),
                int(probs.argmax()),
            )
            return round(float(probs[anger_idx]), 3)
        except Exception as exc:  # noqa: BLE001
            log.warning('Emotion ONNX inference failed, fallback to proxy: %s', exc)

    samples = np.frombuffer(pcm, dtype=np.int16)
    return _acoustic_proxy(samples)


def _linear_to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16000 or samples.size == 0:
        return samples
    duration = samples.size / sample_rate
    out_count = max(1, int(duration * 16000))
    in_idx = np.linspace(0, samples.size - 1, out_count)
    return np.interp(in_idx, np.arange(samples.size), samples).astype(np.float32)
