"""Автономный WebRTC голосовой шлюз в ядре (Stage 4, раздел 2.3).

Контур 2 работает независимо от SIP-моста:
  WS /api/v1/voice/webrtc/stream — полнодуплексный бинарный PCM-стрим
  (16 кГц mono int16) для браузеров и встраиваемых виджетов;
  POST /api/v1/voice/webrtc/offer — конфигурация сессии (ice/mode).

Пайплайн реплики: VAD -> STT (speaches) -> /api/chat/completions ядра
(заголовки X-Channel-Type: webrtc, X-Voice-Mode: true включают Voice
Fast-Path) -> TTS. Без внешнего STT/TTS поддерживается режим
browser_native_audio_fallback: браузер сам распознает/озвучивает
(Web Speech API), шлюз отдает только текст бота.

Отключение канала: channels.webrtc.enabled=false закрывает соединение.

SEC-04: WS-сессия требует JWT-токен из query-параметра ?token=, cookie
token или заголовка Authorization. Проверенный токен прокидывается в
внутренний вызов /api/chat/completions (OPEN_WEBUI_INTERNAL_URL).

BUG-07: barge-in срабатывает по onset речи (VAD_ONSET_MS = 120 мс),
не дожидаясь конца фразы: активный ход ответа отменяется, исходящая
очередь аудио сбрасывается, клиенту уходит barge_in_clear.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import aiohttp
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from open_webui.models.config import Config
from open_webui.utils.auth import create_token, get_verified_user, get_verified_user_by_token

log = logging.getLogger(__name__)

router = APIRouter()

SAMPLE_RATE = 16000
# VAD (мс): onset речи (barge-in), конец реплики, минимум речи, максимум сегмента
VAD_ONSET_MS = 120
VAD_START_MS = 100
VAD_END_MS = 500
VAD_MAX_MS = 15000
ENERGY_THRESHOLD = 500.0
# PERF-CORE-07 (Duplex Echo Guard): пока играет TTS, порог энергии повышается,
# а barge-in требует устойчивой речи, чтобы эхо бота не вызывало самопрерывание
VAD_PLAYBACK_MULTIPLIER = float(os.getenv('VAD_PLAYBACK_MULTIPLIER', '1.8'))
VAD_BARGE_IN_MIN_MS = int(os.getenv('VAD_BARGE_IN_MIN_MS', '180'))


def _rms(payload: bytes) -> float:
    count = len(payload) // 2
    if not count:
        return 0.0
    samples = np.frombuffer(payload[: count * 2], dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


class _Segmenter:
    """Энергетический VAD с гистерезисом (Silero подключается в браузерном канале опционально)."""

    def __init__(self):
        self.buffer = b''
        self.speaking = False
        self.last_voice = 0.0
        self.started = 0.0
        # PERF-CORE-07: флаг активного воспроизведения TTS (эхо-защита)
        self.playback_active = False

    def feed(self, pcm: bytes) -> bytes | None:
        now = time.monotonic()
        self.buffer += pcm
        # Эхо-защита: во время озвучки порог энергии повышается
        effective_threshold = (
            ENERGY_THRESHOLD * VAD_PLAYBACK_MULTIPLIER if self.playback_active else ENERGY_THRESHOLD
        )
        if _rms(pcm) >= effective_threshold:
            self.last_voice = now
            if not self.speaking:
                self.speaking = True
                self.started = now
        elif self.speaking and (now - self.last_voice) * 1000 >= VAD_END_MS:
            self.speaking = False
            segment = self.buffer
            self.buffer = b''
            if (now - self.started) * 1000 >= VAD_START_MS:
                return segment
        return None

    def has_active_speech_onset(self) -> bool:
        """True — речь активна дольше порога onset (триггер мгновенного barge-in).

        При активной озвучке требуется VAD_BARGE_IN_MIN_MS устойчивой речи."""
        if not self.speaking:
            return False
        min_ms = VAD_BARGE_IN_MIN_MS if self.playback_active else VAD_ONSET_MS
        return (time.monotonic() - self.started) * 1000 >= min_ms

    def flush(self) -> bytes | None:
        segment, self.buffer = self.buffer, b''
        return segment or None


class OfferRequest(BaseModel):
    sdp: str | None = None
    type: str = 'offer'
    bot_model: str | None = None
    mode: str = 'server_audio'  # server_audio | browser_native_audio_fallback
    sample_rate: int = SAMPLE_RATE
    transport_preference: str = 'auto'  # auto | webrtc_sdp | ws_pcm


def _aiortc_available() -> bool:
    try:
        import aiortc  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# Активные PeerConnection удерживаются от сборки мусора
_active_pcs: set = set()


def _internal_base_url() -> str:
    return os.getenv('OPEN_WEBUI_INTERNAL_URL', 'http://127.0.0.1:8080').rstrip('/')


async def _bot_stream(token: str, history: list[dict[str, str]], text: str):
    """Ответ ядра с канальными заголовками (Voice Fast-Path включен).

    Функция на уровне модуля (BUG-07): доступна и aiortc-шлюзу, и WS-стриму.
    """
    import json as jsonlib

    headers = {'Content-Type': 'application/json', 'X-Channel-Type': 'webrtc', 'X-Voice-Mode': 'true'}
    if token:
        headers['Authorization'] = token if token.lower().startswith('bearer ') else f'Bearer {token}'
    payload = {'model': '', 'messages': [*history, {'role': 'user', 'content': text}], 'stream': True}
    timeout = aiohttp.ClientTimeout(total=180, sock_read=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f'{_internal_base_url()}/api/chat/completions', json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            async for raw in response.content:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                data = line.removeprefix('data:').strip()
                if data == '[DONE]':
                    break
                try:
                    chunk = jsonlib.loads(data)
                except jsonlib.JSONDecodeError:
                    continue
                for choice in chunk.get('choices', []):
                    delta = (choice.get('delta') or {}).get('content') or ''
                    if delta:
                        yield delta


async def _tts_sentence(tts_url: str, tts_voice: str, sentence: str) -> bytes:
    """Синтез одной синтагмы через speaches (модульный уровень — BUG-07)."""
    if not tts_url:
        return b''
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        async with session.post(
            f'{tts_url}/v1/audio/speech',
            json={'model': 'kokoro', 'voice': tts_voice, 'input': sentence, 'response_format': 'wav'},
        ) as response:
            response.raise_for_status()
            audio = await response.read()
    # WAV -> PCM (поиск data-чанка)
    if audio[:4] == b'RIFF':
        index = 12
        while index + 8 <= len(audio):
            chunk_id = audio[index : index + 4]
            chunk_size = int.from_bytes(audio[index + 4 : index + 8], 'little')
            if chunk_id == b'data':
                return audio[index + 8 : index + 8 + chunk_size]
            index += 8 + chunk_size + (chunk_size % 2)
    return audio


async def tts_sentence_with_fallback(sentence: str, tts_url: str = '', tts_voice: str = '') -> bytes:
    """TTS с локальным фоллбэком, когда speaches недоступен."""
    if not tts_url:
        tts_url = (await Config.get('audio.tts.openai.api_base_url') or '').rstrip('/').removesuffix('/v1')
    if not tts_voice:
        tts_voice = await Config.get('audio.tts.voice') or 'af_heart'
    pcm = await _tts_sentence(tts_url, tts_voice, sentence)
    if pcm:
        return pcm
    return synthesize_pcm_fallback(sentence, SAMPLE_RATE)


async def _stt_bytes(pcm: bytes) -> str:
    """STT PCM-сегмента через speaches (пусто, если недоступен)."""
    import struct

    stt_url = (await Config.get('audio.stt.openai.api_base_url') or '').rstrip('/').removesuffix('/v1')
    if not stt_url:
        return ''
    header = b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVE'
    header += b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
    header += b'data' + struct.pack('<I', len(pcm))
    form = aiohttp.FormData()
    form.add_field('file', header + pcm, filename='utterance.wav', content_type='audio/wav')
    form.add_field('model', 'faster-whisper-large-v3-turbo')
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(f'{stt_url}/v1/audio/transcriptions', data=form) as response:
                response.raise_for_status()
                return str((await response.json()).get('text') or '').strip()
    except Exception:  # noqa: BLE001
        return ''


@router.post('/webrtc/offer')
async def webrtc_offer(form: OfferRequest, user=Depends(get_verified_user)):
    """Д2 (Stage 5): гибридный обмен голосовой сессией.

    С валидным SDP Offer и установленным aiortc возвращает честный
    SDP Answer (transport webrtc_sdp) c DataChannel events. Без aiortc
    или по preference ws_pcm — фоллбэк на WebSocket PCM-стрим.
    """
    enabled = await Config.get('channels.webrtc.enabled')
    if enabled is False:
        raise HTTPException(status_code=403, detail='WebRTC channel is disabled')
    stt_url = await Config.get('audio.stt.openai.api_base_url')
    tts_url = await Config.get('audio.tts.openai.api_base_url')

    use_sdp = (
        _aiortc_available()
        and form.sdp
        and form.sdp.strip().startswith('v=0')
        and form.transport_preference != 'ws_pcm'
    )
    if use_sdp:
        try:
            # SEC-04: проверенный пользователь — выпускаем краткоживущий токен
            # для внутреннего вызова чат-пайплайна
            internal_token = create_token({'id': user.id})
            answer_sdp = await _handle_aiortc_offer(form.sdp, internal_token)
            return {
                'sdp': answer_sdp,
                'type': 'answer',
                'transport': 'webrtc_sdp',
                'channel': 'webrtc',
                'datachannel': 'events',
            }
        except Exception as exc:  # noqa: BLE001
            log.warning('aiortc offer failed, falling back to WS PCM: %s', exc)

    return {
        'type': 'ws_fallback',
        'transport': 'ws_pcm',
        'ws_url': '/api/v1/voice/webrtc/stream',
        'stream_endpoint': '/api/v1/voice/webrtc/stream',
        'sample_rate': form.sample_rate,
        'encoding': 'pcm_s16le',
        'channel': 'webrtc',
        'voice_mode_headers': {'X-Channel-Type': 'webrtc', 'X-Voice-Mode': 'true'},
        'speaches_available': bool(stt_url and tts_url),
        'browser_native_audio_fallback': form.mode == 'browser_native_audio_fallback',
        'sdp': form.sdp,
    }


async def _handle_aiortc_offer(sdp: str, internal_token: str = '') -> str:
    """Д2 (Stage 5): честный SDP Offer/Answer через aiortc.

    Входящий аудиотрек браузера проходит VAD + STT + ядро бота + TTS,
    ответные фреймы уходят в исходящий MediaStreamTrack, события
    (barge_in_clear / transcript / ttfb) — через DataChannel "events".
    """
    import numpy as np
    from aiortc import RTCSessionDescription, RTCPeerConnection
    from aiortc.mediastreams import MediaStreamTrack

    class BotAudioStreamTrack(MediaStreamTrack):
        kind = 'audio'

        def __init__(self):
            super().__init__()
            self._queue: asyncio.Queue = asyncio.Queue(maxsize=200)

        def clear(self):
            """Сброс исходящей очереди аудио при barge-in."""
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:  # noqa: BLE001
                    break

        async def recv(self):
            pcm = await self._queue.get()  # int16 mono 16 kHz
            import av

            samples_48k = int(len(pcm) / 2 / SAMPLE_RATE * 48000)
            samples_48k = max(samples_48k // 2 * 2, 960)
            data16 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            if len(data16) == 0:
                data16 = np.zeros(samples_48k, dtype=np.float32)
            idx = np.linspace(0, len(data16) - 1, samples_48k)
            up = np.interp(idx, np.arange(len(data16)), data16)
            frame = av.AudioFrame.from_ndarray(
                np.repeat(up[np.newaxis, :], 2, axis=0), format='s16', layout='stereo'
            )
            frame.sample_rate = 48000
            frame.pts = None
            return frame

    pc = RTCPeerConnection()
    _active_pcs.add(pc)
    outgoing = BotAudioStreamTrack()
    events_channel: list = []
    turn_task: asyncio.Task | None = None

    @pc.on('datachannel')
    def on_datachannel(channel):
        events_channel.append(channel)

    def send_event(payload: dict) -> None:
        channel = events_channel[0] if events_channel else None
        if channel is not None and channel.readyState == 'open':
            try:
                channel.send(json.dumps(payload))
            except Exception:  # noqa: BLE001
                pass

    # BUG-07.2: закрытые PeerConnection удаляются из реестра (утечка памяти)
    @pc.on('connectionstatechange')
    async def on_connectionstatechange():
        if pc.connectionState in ('failed', 'closed', 'disconnected'):
            if turn_task is not None and not turn_task.done():
                turn_task.cancel()
            outgoing.clear()
            try:
                await pc.close()
            except Exception:  # noqa: BLE001
                pass
            _active_pcs.discard(pc)

    @pc.on('track')
    def on_track(track):
        if track.kind != 'audio':
            return

        async def run_turn(segment: bytes, segmenter: '_Segmenter') -> None:
            tts_url = (await Config.get('audio.tts.openai.api_base_url') or '').rstrip('/').removesuffix('/v1')
            tts_voice = await Config.get('audio.tts.voice') or 'af_heart'
            try:
                text = await _stt_bytes(segment)
                send_event({'type': 'transcript', 'text': text})
                if not text:
                    text = 'Здравствуйте! Я вас слышу. Задайте ваш вопрос.'
                t0 = time.perf_counter()
                answer_parts = []
                async for delta in _bot_stream(internal_token, [], text):
                    answer_parts.append(delta)
                    send_event({'type': 'delta', 'text': delta})
                answer = ''.join(answer_parts)
                if answer:
                    first = True
                    for sentence in _split_sentences(answer):
                        pcm = await _tts_sentence(tts_url, tts_voice, sentence) or synthesize_pcm_fallback(
                            sentence, SAMPLE_RATE
                        )
                        if pcm and first:
                            send_event({'type': 'ttfb', 'ms': round((time.perf_counter() - t0) * 1000)})
                            first = False
                        if pcm:
                            await outgoing._queue.put(pcm)
                send_event({'type': 'answer', 'text': answer})
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning('aiortc turn failed: %s', exc)

        async def consume_incoming():
            nonlocal turn_task
            segmenter = _Segmenter()
            barge_in_fired = False
            while True:
                try:
                    frame = await track.recv()
                except Exception:  # noqa: BLE001
                    return
                arr = frame.to_ndarray()
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                data = np.clip(arr, -32768, 32767).astype(np.int16).tobytes()
                segmenter.feed(data)
                # BUG-07.3: мгновенный barge-in по onset речи — активный ход
                # отменяется, исходящая очередь аудио сбрасывается;
                # PERF-CORE-07: эхо-защита при активной озвучке
                segmenter.playback_active = turn_task is not None and not turn_task.done()
                if segmenter.has_active_speech_onset() and turn_task is not None and not turn_task.done():
                    if not barge_in_fired:
                        send_event({'type': 'barge_in_clear'})
                        barge_in_fired = True
                    turn_task.cancel()
                    outgoing.clear()
                    turn_task = None
                if not segmenter.speaking:
                    barge_in_fired = False
                segment = None
                if not segmenter.speaking and segmenter.buffer:
                    segment = segmenter.flush()
                if segment:
                    if turn_task is not None and not turn_task.done():
                        turn_task.cancel()
                        outgoing.clear()
                    turn_task = asyncio.create_task(run_turn(segment, segmenter))

        asyncio.create_task(consume_incoming())

    pc.addTrack(outgoing)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type='offer'))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return pc.localDescription.sdp


def _ws_token(websocket: WebSocket) -> str:
    """Извлечение JWT из query ?token=, cookie token или заголовка Authorization (SEC-04)."""
    token = websocket.query_params.get('token') or ''
    if token:
        return token
    token = websocket.cookies.get('token') or ''
    if token:
        return token
    auth = websocket.headers.get('authorization') or ''
    return auth[7:].strip() if auth.lower().startswith('bearer ') else auth.strip()


@router.websocket('/webrtc/stream')
async def webrtc_stream(websocket: WebSocket):
    """Полнодуплексный аудиострим канала WebRTC.

    Клиент шлет PCM int16 mono 16 кГц кадрами; сервер возвращает бинарные
    PCM-кадры ответа и JSON-события: {"type": "barge_in_clear"},
    {"type": "transcript", ...}, {"type": "anger_score", ...}.

    SEC-04: соединение без валидного JWT закрывается кодом 4401.
    """
    await websocket.accept()
    enabled = await Config.get('channels.webrtc.enabled')
    if enabled is False:
        await websocket.send_json({'type': 'error', 'detail': 'WebRTC channel is disabled'})
        await websocket.close()
        return

    user = await get_verified_user_by_token(_ws_token(websocket))
    if user is None:
        await websocket.send_json({'type': 'error', 'detail': 'Unauthorized'})
        await websocket.close(code=4401)
        return
    token = _ws_token(websocket)

    speaches_url = (await Config.get('audio.stt.openai.api_base_url') or '').rstrip('/').removesuffix('/v1')
    tts_url = (await Config.get('audio.tts.openai.api_base_url') or '').rstrip('/').removesuffix('/v1')
    tts_voice = await Config.get('audio.tts.voice') or 'af_heart'
    session_key = f'webrtc-{id(websocket)}'

    await websocket.send_json(
        {
            'type': 'session',
            'mode': 'server_audio' if speaches_url else 'browser_native_audio_fallback',
            'sample_rate': SAMPLE_RATE,
        }
    )

    segmenter = _Segmenter()
    history: list[dict[str, str]] = []
    response_task: asyncio.Task | None = None

    async def stt(pcm: bytes) -> str:
        if not speaches_url:
            return ''
        import struct

        header = b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVE'
        header += b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
        header += b'data' + struct.pack('<I', len(pcm))
        form = aiohttp.FormData()
        form.add_field('file', header + pcm, filename='utterance.wav', content_type='audio/wav')
        form.add_field('model', 'faster-whisper-large-v3-turbo')
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(f'{speaches_url}/v1/audio/transcriptions', data=form) as response:
                response.raise_for_status()
                return str((await response.json()).get('text') or '').strip()

    async def run_voice_turn(segment: bytes) -> None:
        """STT -> ядро -> TTS по синтагмам. Отменяется при barge-in."""
        nonlocal history
        text = await stt(segment)
        if not text:
            # Д4: без STT-стенда — детерминированный fallback-ответ
            text = 'Здравствуйте! Я вас слышу. Задайте ваш вопрос.'
        await websocket.send_json({'type': 'transcript', 'text': text})

        t0 = time.perf_counter()
        answer_parts = []
        async for delta in _bot_stream(token, history, text):
            answer_parts.append(delta)
        answer = ''.join(answer_parts)
        if not answer:
            return

        # Озвучка по синтагмам
        sentences = [s for s in _split_sentences(answer) if s.strip()]
        first_sent = True
        for sentence in sentences:
            pcm = await _tts_sentence(tts_url, tts_voice, sentence)
            if pcm and first_sent:
                await websocket.send_json(
                    {'type': 'ttfb', 'ms': round((time.perf_counter() - t0) * 1000)}
                )
                first_sent = False
            if pcm:
                await websocket.send_bytes(pcm)
        await websocket.send_json({'type': 'answer', 'text': answer})
        history = [*history, {'role': 'user', 'content': text}, {'role': 'assistant', 'content': answer}][-12:]

    try:
        while True:
            frame = await websocket.receive()
            if frame.get('type') == 'websocket.disconnect':
                break
            data = frame.get('bytes')
            text_message = frame.get('text')
            if text_message:
                # browser_native_audio_fallback: браузер шлет распознанный текст сам
                if response_task is not None and not response_task.done():
                    response_task.cancel()
                    response_task = None
                    await websocket.send_json({'type': 'barge_in_clear'})
                await websocket.send_json({'type': 'thinking'})
                t0 = time.perf_counter()
                answer_parts: list[str] = []
                async for delta in _bot_stream(token, history, text_message):
                    answer_parts.append(delta)
                    await websocket.send_json({'type': 'delta', 'text': delta})
                answer = ''.join(answer_parts)
                await websocket.send_json(
                    {'type': 'answer', 'text': answer, 'ttfb_ms': round((time.perf_counter() - t0) * 1000)}
                )
                history = [
                    *history,
                    {'role': 'user', 'content': text_message},
                    {'role': 'assistant', 'content': answer},
                ][-12:]
                continue
            if not data:
                continue

            # BUG-07.3: onset речи мгновенно прерывает озвучку бота;
            # PERF-CORE-07: пока идет озвучка — эхо-защита VAD
            segmenter.playback_active = response_task is not None and not response_task.done()
            segmenter.feed(data)
            if (
                segmenter.has_active_speech_onset()
                and response_task is not None
                and not response_task.done()
            ):
                response_task.cancel()
                response_task = None
                await websocket.send_json({'type': 'barge_in_clear'})

            if segmenter.speaking:
                continue
            segment = segmenter.flush()
            if not segment:
                continue
            if response_task is not None and not response_task.done():
                response_task.cancel()
                await websocket.send_json({'type': 'barge_in_clear'})
            response_task = asyncio.create_task(run_voice_turn(segment))
    except WebSocketDisconnect:
        log.info('WebRTC WS-сессия закрыта')
    except Exception as exc:  # noqa: BLE001
        log.warning('WebRTC WS error: %s', exc)
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
    finally:
        if response_task is not None and not response_task.done():
            response_task.cancel()


def synthesize_pcm_fallback(text: str, out_rate: int) -> bytes:
    """Д4 (Stage 5): локальный синтез без speaches (слоговый тональный PCM)."""
    import re as _re

    if not text.strip():
        return b''
    words = _re.findall(r'[а-яёa-z0-9]+', text.lower())
    syllables = max(1, sum(len(_re.findall(r'[аеёиоуыэюяaeiouy]', w)) or 1 for w in words))
    out = bytearray()
    for i in range(syllables):
        freq = 350.0 if i % 2 == 0 else 500.0
        samples = int(out_rate * 140 / 1000)
        t = np.arange(samples) / out_rate
        envelope = np.sin(np.pi * t / (samples / out_rate))
        tone = np.sin(2 * np.pi * freq * t) * envelope * 9000
        out += np.clip(tone, -32768, 32767).astype(np.int16).tobytes()
        out += np.zeros(int(out_rate * 30 / 1000), dtype=np.int16).tobytes()
    return bytes(out)


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r'(?<=[.!?…])\s+|\n', text)
    buffer = ''
    chunks: list[str] = []
    for part in parts:
        buffer += (' ' if buffer else '') + part
        if len(buffer) >= 120:
            chunks.append(buffer)
            buffer = ''
    if buffer.strip():
        chunks.append(buffer)
    return chunks
