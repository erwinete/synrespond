"""Детектор голосовой активности.

Основной путь — Silero VAD из pipecat (ТЗ, раздел 7.2). Если pipecat не
установлен, работает энергетический VAD с гистерезисом — достаточен для
отладки и тихих линий.
"""

from __future__ import annotations

import logging
import struct
import time

from config import (
    ENERGY_THRESHOLD,
    INTERNAL_SAMPLE_RATE,
    VAD_END_MS,
    VAD_MAX_SEGMENT_MS,
    VAD_START_MS,
)

log = logging.getLogger(__name__)


def rms(payload: bytes) -> float:
    """Среднеквадратичный уровень PCM 16-bit."""
    count = len(payload) // 2
    if not count:
        return 0.0
    samples = struct.unpack(f'<{count}h', payload[: count * 2])
    return (sum(s * s for s in samples) / count) ** 0.5


def _make_silero():
    """Возвращает Silero VAD (pipecat) или None, если библиотека недоступна."""
    try:
        import numpy as np
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADAnalyzer

        analyzer = SileroVADAnalyzer(sample_rate=INTERNAL_SAMPLE_RATE)
        return analyzer, VADAnalyzer, np
    except Exception as exc:  # noqa: BLE001
        log.info('pipecat Silero VAD недоступен, используется энергетический VAD: %s', exc)
        return None


class UtteranceSegmenter:
    """Накапливает PCM и выделяет реплики по VAD-решениям.

    feed(pcm_bytes, sample_rate) -> bytes | None: сегмент реплики или None.
    """

    def __init__(self) -> None:
        self.buffer = b''
        self.speaking = False
        self.last_voice_ts = 0.0
        self.speech_started_ts = 0.0
        self._silero = _make_silero()
        self._residual = b''

    def feed(self, pcm: bytes, sample_rate: int) -> bytes | None:
        now = time.monotonic()
        self.buffer += pcm
        if self._has_voice(pcm, sample_rate):
            self.last_voice_ts = now
            if not self.speaking:
                self.speaking = True
                self.speech_started_ts = now
        elif self.speaking:
            silence_ms = (now - self.last_voice_ts) * 1000
            if silence_ms >= VAD_END_MS:
                self.speaking = False
                duration_ms = (now - self.speech_started_ts) * 1000
                segment = self._flush()
                if duration_ms >= VAD_START_MS:
                    return segment
                return None

        if self.speaking:
            duration_ms = (now - self.speech_started_ts) * 1000
            if duration_ms >= VAD_MAX_SEGMENT_MS:
                self.speaking = False
                return self._flush()
        return None

    def force_flush(self) -> bytes | None:
        """Возвращает накопленное как реплику (конец звонка/закрытие)."""
        if not self.buffer:
            return None
        return self._flush()

    def has_active_speech_onset(self) -> bool:
        """True — речь активна дольше VAD_START_MS (120 мс): триггер мгновенного barge-in.

        BUG-07: прерывание не ждет конца всей фразы абонента и тишины VAD_END_MS.
        """
        return self.speaking and (time.monotonic() - self.speech_started_ts) * 1000 >= VAD_START_MS

    def _flush(self) -> bytes:
        segment, self.buffer = self.buffer, b''
        return segment

    def _has_voice(self, pcm: bytes, sample_rate: int) -> bool:
        if self._silero is None:
            return rms(pcm) >= ENERGY_THRESHOLD

        analyzer, vad_state, np = self._silero
        # Silero требует 16 кГц и чанк 512 сэмплов (1024 байта)
        self._residual += pcm
        if sample_rate != INTERNAL_SAMPLE_RATE:
            samples = np.frombuffer(self._residual, dtype=np.int16).astype(np.float32)
            self._residual = b''
            if samples.size == 0:
                return False
            duration = samples.size / sample_rate
            out_count = max(1, int(duration * INTERNAL_SAMPLE_RATE))
            in_idx = np.linspace(0, samples.size - 1, out_count)
            upsampled = np.interp(in_idx, np.arange(samples.size), samples)
            self._residual = np.clip(upsampled, -32768, 32767).astype(np.int16).tobytes()

        chunk_bytes = 1024  # 512 сэмплов * 2 байта
        while len(self._residual) >= chunk_bytes:
            chunk = self._residual[:chunk_bytes]
            self._residual = self._residual[chunk_bytes:]
            audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            try:
                state, _ = analyzer.analyze_audio(audio)
                if state == vad_state.VADState.SPEAKING:
                    return True
            except Exception:  # noqa: BLE001
                self._silero = None
                return rms(pcm) >= ENERGY_THRESHOLD
        return False
