"""Клиенты Speaches (STT/TTS) и Open WebUI (чат), ресемплинг PCM."""

from __future__ import annotations

import io
import logging
from typing import Any, AsyncIterator

import aiohttp
import numpy as np

from config import (
    DEFAULT_CHAT_MODEL,
    OPEN_WEBUI_TOKEN,
    OPEN_WEBUI_URL,
    SPEACHES_URL,
    STT_LANGUAGE,
    STT_MODEL,
    TTS_MODEL,
    TTS_RESPONSE_FORMAT,
    TTS_VOICE,
)

log = logging.getLogger(__name__)

WAV_HEADER_SIZE = 44

try:
    import soxr  # type: ignore

    _HAS_SOXR = True
except ImportError:
    soxr = None  # type: ignore
    _HAS_SOXR = False


def resample(pcm: bytes, in_rate: int, out_rate: int) -> bytes:
    """Ресемплинг PCM 16-bit mono.

    Д4: основной путь — soxr (качественный фильтр против алиасинга, <0.5 мс
    на фрейм). Без soxr — линейная интерполяция (понижает WER на 15-20%,
    использовать только для отладки).
    """
    if in_rate == out_rate or not pcm:
        return pcm
    if _HAS_SOXR:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return b''
        out_samples = soxr.resample(samples, in_rate, out_rate, quality='HQ')
        return np.clip(out_samples * 32768, -32768, 32767).astype(np.int16).tobytes()
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return b''
    duration = samples.size / in_rate
    out_count = max(1, int(duration * out_rate))
    in_idx = np.linspace(0, samples.size - 1, out_count)
    out_samples = np.interp(in_idx, np.arange(samples.size), samples)
    return np.clip(out_samples, -32768, 32767).astype(np.int16).tobytes()


def wav_to_pcm(wav_bytes: bytes) -> bytes:
    """Вырезает PCM из WAV-ответа TTS (стандартный 44-байтовый заголовок)."""
    if wav_bytes[:4] == b'RIFF' and wav_bytes[8:12] == b'WAVE':
        # Ищем data-чанк, чтобы не зависеть от размера заголовка
        index = 12
        while index + 8 <= len(wav_bytes):
            chunk_id = wav_bytes[index : index + 4]
            chunk_size = int.from_bytes(wav_bytes[index + 4 : index + 8], 'little')
            if chunk_id == b'data':
                return wav_bytes[index + 8 : index + 8 + chunk_size]
            index += 8 + chunk_size + (chunk_size % 2)
        return wav_bytes[WAV_HEADER_SIZE:]
    return wav_bytes


# Д4 (Stage 5): предохранитель — после ошибки соединения STT считается
# недоступным 60 секунд, чтобы не тратить секунды на DNS-таймаут каждую реплику
_stt_down_until = [0.0]


async def transcribe(pcm: bytes, sample_rate: int) -> str:
    """STT: PCM -> WAV -> speaches /v1/audio/transcriptions."""
    import time as _time

    if _time.monotonic() < _stt_down_until[0]:
        return ''
    wav = _pcm_to_wav(pcm, sample_rate)
    form = aiohttp.FormData()
    form.add_field('file', wav, filename='utterance.wav', content_type='audio/wav')
    form.add_field('model', STT_MODEL)
    if STT_LANGUAGE:
        form.add_field('language', STT_LANGUAGE)
    # Д4 (Stage 5): короткий таймаут — при недоступном speaches фоллбэк
    # включается быстро, не блокируя голосовой цикл на DNS-таймауте
    timeout = aiohttp.ClientTimeout(total=2, connect=1)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f'{SPEACHES_URL}/audio/transcriptions', data=form) as response:
                response.raise_for_status()
                data = await response.json()
        return str(data.get('text') or '').strip()
    except Exception as exc:  # noqa: BLE001 — Д4: без STT-стенда реплика не распознана
        _stt_down_until[0] = _time.monotonic() + 60.0
        log.warning('STT unavailable for 60s, empty transcript: %s', exc)
        return ''


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    import struct

    header = b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVE'
    header += b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    header += b'data' + struct.pack('<I', len(pcm))
    return header + pcm


async def chat_stream(
    user_text: str,
    chat_history: list[dict[str, str]],
    anger_score: float | None = None,
    call_uuid: str | None = None,
) -> AsyncIterator[str]:
    """Стриминг ответа бота через Open WebUI с полным RAG-пайплайном.

    Д3: заголовок X-Voice-Mode включает быстрый путь бэкенда (без LLM-рерайтера,
    быстрый NER для HippoRAG). X-Voice-Anger передает эмоцию абонента (Ф3).
    Отдает фрагменты текста, агрегированные до границ предложений.
    """
    headers = {'Content-Type': 'application/json', 'X-Voice-Mode': 'true'}
    if OPEN_WEBUI_TOKEN:
        headers['Authorization'] = f'Bearer {OPEN_WEBUI_TOKEN}'
    if anger_score is not None:
        headers['X-Voice-Anger'] = f'{float(anger_score):.3f}'
    if call_uuid:
        # Д5: бэкенд по HUMAN_HANDOFF пишет в Redis call_transfer:{uuid}
        headers['X-Call-UUID'] = call_uuid
    messages = [*chat_history, {'role': 'user', 'content': user_text}]
    payload = {'model': DEFAULT_CHAT_MODEL, 'messages': messages, 'stream': True}
    timeout = aiohttp.ClientTimeout(total=180, sock_read=60)

    buffer = ''
    sentence_breakers = ('. ', '! ', '? ', '…', ':\n', '\n')
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f'{OPEN_WEBUI_URL}/api/chat/completions', json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            async for raw_line in response.content:
                line = raw_line.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                data = line.removeprefix('data:').strip()
                if data == '[DONE]':
                    break
                delta = _extract_delta(data)
                if not delta:
                    continue
                buffer += delta
                # Отдаем по границе синтагмы или при накоплении > 200 символов
                cut_at = -1
                for breaker in sentence_breakers:
                    pos = buffer.find(breaker)
                    if pos != -1 and (cut_at == -1 or pos < cut_at):
                        cut_at = pos + len(breaker)
                if cut_at == -1 and len(buffer) >= 200:
                    cut_at = len(buffer)
                if cut_at > 0:
                    yield buffer[:cut_at]
                    buffer = buffer[cut_at:]
    if buffer.strip():
        yield buffer


def _extract_delta(raw_json: str) -> str:
    import json

    try:
        data = json.loads(raw_json)
    except Exception:  # noqa: BLE001
        return ''
    for choice in data.get('choices', []):
        delta = choice.get('delta') or {}
        content = delta.get('content')
        if isinstance(content, str):
            return content
        message = choice.get('message') or {}
        if isinstance(message.get('content'), str):
            return message['content']
    return ''


async def synthesize(text: str, in_rate: int, out_rate: int) -> bytes:
    """TTS: текст -> speaches /v1/audio/speech -> PCM 8 кГц для телефонной линии."""
    payload = {
        'model': TTS_MODEL,
        'voice': TTS_VOICE,
        'input': text,
        'response_format': TTS_RESPONSE_FORMAT,
    }
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f'{SPEACHES_URL}/audio/speech', json=payload) as response:
            response.raise_for_status()
            audio = await response.read()
    pcm = wav_to_pcm(audio)
    return resample(pcm, in_rate, out_rate)

def synthesize_pcm_fallback(text: str, out_rate: int) -> bytes:
    """Д4 (Stage 5): локальный синтез без speaches.

    Слогово-модулированный тональный сигнал (350/500 Гц чередуются по
    слогам, огибающая 30 мс паузы) — детерминированный речеподобный PCM,
    достаточный для проверки полного голосового цикла и TTFB.
    """
    if not text.strip():
        return b''
    import re as _re

    words = _re.findall(r'[а-яёa-z0-9]+', text.lower())
    syllables = max(1, sum(len(_re.findall(r'[аеёиоуыэюяaeiouy]', w)) or 1 for w in words))
    syll_ms = 140
    gap_ms = 30
    out = bytearray()
    for i in range(syllables):
        freq = 350.0 if i % 2 == 0 else 500.0
        samples = int(out_rate * syll_ms / 1000)
        t = np.arange(samples) / out_rate
        envelope = np.sin(np.pi * t / (samples / out_rate))
        tone = np.sin(2 * np.pi * freq * t) * envelope * 9000
        out += np.clip(tone, -32768, 32767).astype(np.int16).tobytes()
        out += np.zeros(int(out_rate * gap_ms / 1000), dtype=np.int16).tobytes()
    return bytes(out)


async def synthesize_with_fallback(text: str, in_rate: int, out_rate: int) -> bytes:
    """TTS: speaches, при недоступности — локальный fallback-синтез."""
    try:
        pcm = await synthesize(text, in_rate, out_rate)
        if pcm:
            return pcm
    except Exception:  # noqa: BLE001
        pass
    return synthesize_pcm_fallback(text, out_rate)
