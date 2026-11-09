"""Диалоговая сессия звонка: VAD -> STT -> бот -> TTS с поддержкой barge-in."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

import call_status
import emotion as emotion_io
import speech as speech_io
from config import (
    BARGE_IN_ENABLED,
    FALLBACK_ON_EMPTY_STT,
    FALLBACK_TEXT,
    FILLER_DELAY_MS,
    FILLER_ENABLED,
    FILLER_TEXT,
    HANDOFF_ANGER_THRESHOLD,
    HANDOFF_QUEUE,
    INTERNAL_SAMPLE_RATE,
)
from vad import UtteranceSegmenter

log = logging.getLogger(__name__)

SENTENCE_TTS_RATE = 24000  # Kokoro/Piper отдают 24 кГц


class CallSession:
    """Одна телефонная/браузерная сессия.

    Пайплайн (ТЗ, раздел 7.2):
      входящий PCM -> VAD -> реплика -> STT (speaches)
        -> /api/chat/completions stream (Open WebUI, полный RAG)
        -> агрегация по синтагмам -> TTS -> исходящий PCM.
    Barge-in: речь абонента отменяет текущую генерацию/воспроизведение.
    """

    def __init__(
        self,
        session_id: str,
        in_sample_rate: int,
        out_sample_rate: int,
        send_audio: Callable[[bytes], Any],
        on_status: Optional[Callable[[str], Any]] = None,
        call_uuid: Optional[str] = None,
    ):
        self.session_id = session_id
        self.call_uuid = call_uuid
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self._send_audio = send_audio
        self._on_status = on_status
        self.segmenter = UtteranceSegmenter()
        self.history: list[dict[str, str]] = []
        self._tts_task: asyncio.Task | None = None
        self._barge_in = False
        self._closed = False
        self._processing_lock = asyncio.Lock()
        self._first_audio_sent = False
        self._filler_cache: dict[int, bytes] = {}
        self.last_anger: float | None = None
        self._t_speech_end: float = 0.0

    async def feed_audio(self, pcm: bytes) -> None:
        """Входящий фрагмент PCM 16-bit mono."""
        if self._closed:
            return
        if self._barge_in:
            # Новая речь абонента прерывает текущее воспроизведение
            self._cancel_tts()
        segment = self.segmenter.feed(pcm, self.in_sample_rate)
        if segment:
            await self._handle_utterance(segment)

    async def finish(self) -> None:
        """Завершение: дорабатываем накопленную реплику."""
        segment = self.segmenter.force_flush()
        if segment:
            await self._handle_utterance(segment)

    async def close(self) -> None:
        self._closed = True
        self._cancel_tts()

    def _cancel_tts(self) -> None:
        if BARGE_IN_ENABLED and self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            log.info('Barge-in: отмена воспроизведения (%s)', self.session_id)
        self._barge_in = False

    def _status(self, message: str) -> None:
        if self._on_status:
            try:
                result = self._on_status(message)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:  # noqa: BLE001
                pass

    async def _handle_utterance(self, segment: bytes) -> None:
        if self._processing_lock.locked():
            # Предыдущая реплика еще обрабатывается; не теряем новую
            self._barge_in = True
        async with self._processing_lock:
            try:
                t0 = time.perf_counter()
                self._status('stt')
                pcm_16k = speech_io.resample(segment, self.in_sample_rate, INTERNAL_SAMPLE_RATE)
                text = await speech_io.transcribe(pcm_16k, INTERNAL_SAMPLE_RATE)
                if not text:
                    if not FALLBACK_ON_EMPTY_STT:
                        return
                    # Д4: без STT-стенда реплика не распознана — отдаем
                    # детерминированный fallback-ответ с локальным синтезом
                    text = FALLBACK_TEXT
                log.info('[%s] STT (%.0f мс): %s', self.session_id, (time.perf_counter() - t0) * 1000, text)

                # Ф3: акустическая эмоция реплики (гнев/срочность)
                self.last_anger = emotion_io.estimate(segment, self.in_sample_rate)

                # Д5: эскалация по порогу гнева -> статус перевода для диалплана
                if self.call_uuid and self.last_anger is not None and self.last_anger >= HANDOFF_ANGER_THRESHOLD:
                    await call_status.mark_transfer(self.call_uuid, HANDOFF_QUEUE)
                    log.warning(
                        '[%s] Anger %.2f >= %s: звонок помечен для перевода в очередь %s',
                        self.session_id, self.last_anger, HANDOFF_ANGER_THRESHOLD, HANDOFF_QUEUE,
                    )

                self._status('chat')
                self._t_speech_end = time.perf_counter()
                await self._respond(text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception('[%s] Utterance handling failed: %s', self.session_id, exc)

    async def _respond(self, text: str) -> None:
        """Стримим ответ бота и синтезируем аудио по предложениям."""
        self._tts_task = asyncio.current_task()
        self._first_audio_sent = False
        answer_parts: list[str] = []
        filler_task = asyncio.create_task(self._play_filler_if_silent())

        async def stream_tts() -> None:
            first_audio = True
            async for sentence in speech_io.chat_stream(
                text, self.history, anger_score=self.last_anger, call_uuid=self.call_uuid
            ):
                answer_parts.append(sentence)
                try:
                    pcm = await speech_io.synthesize_with_fallback(
                        sentence, SENTENCE_TTS_RATE, self.out_sample_rate
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning('[%s] TTS failed: %s', self.session_id, exc)
                    continue
                if first_audio:
                    ttfb_ms = (time.perf_counter() - self._t_speech_end) * 1000
                    log.info('[%s] TTFB_llm_answer_ms=%.0f (первый речевой чанк ответа)', self.session_id, ttfb_ms)
                    first_audio = False
                    self._first_audio_sent = True
                self._send_audio(pcm)
                # Мелкая нарезка повышает отзывчивость barge-in
                await asyncio.sleep(0)

        self._t0 = time.perf_counter()
        try:
            try:
                await stream_tts()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — Д4: без LLM отдаем локальный ответ
                log.warning('[%s] Bot backend unavailable (%s), local fallback answer', self.session_id, exc)
                pcm = await speech_io.synthesize_with_fallback(FALLBACK_TEXT, SENTENCE_TTS_RATE, self.out_sample_rate)
                self._send_audio(pcm)
                answer_parts.append(FALLBACK_TEXT)
        except asyncio.CancelledError:
            log.info('[%s] Ответ прерван абонентом', self.session_id)
            raise
        else:
            self.history.append({'role': 'user', 'content': text})
            self.history.append({'role': 'assistant', 'content': ''.join(answer_parts)})
            if len(self.history) > 12:
                self.history = self.history[-12:]
        finally:
            filler_task.cancel()

    async def _play_filler_if_silent(self) -> None:
        """Д3: через FILLER_DELAY_MS маскируем задержку коротким филлером.

        Филлер синтезируется один раз и кэшируется в PCM транспортного
        формата; barge-in отменяет его вместе с основным ответом.
        """
        if not FILLER_ENABLED:
            return
        try:
            await asyncio.sleep(FILLER_DELAY_MS / 1000)
            if self._first_audio_sent or self._closed:
                return
            log.info(
                '[%s] TTFB_filler_ms=%.0f (филлер маскирует задержку)',
                self.session_id,
                (time.perf_counter() - self._t_speech_end) * 1000,
            )
            pcm = self._filler_cache.get(self.out_sample_rate)
            if pcm is None:
                pcm = await speech_io.synthesize_with_fallback(
                    FILLER_TEXT, SENTENCE_TTS_RATE, self.out_sample_rate
                )
                self._filler_cache[self.out_sample_rate] = pcm
            self._send_audio(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning('[%s] Filler playback failed: %s', self.session_id, exc)
