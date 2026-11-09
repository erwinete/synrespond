"""voice-sip-bridge: шлюз телефонии в Open WebUI (ТЗ, раздел 7.2).

Транспорты:
  TCP  :8765 — Asterisk AudioSocket (slin 8 кГц);
  WS   /ws/audio?sample_rate=8000|16000 — FreeSWITCH и кастомные клиенты.

Исходящий и входящий звук — PCM 16-bit mono, sample rate транспорта.
"""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from audiosocket import start_audiosocket_server
from config import AUDIOSOCKET_HOST, AUDIOSOCKET_PORT
from session import CallSession

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
log = logging.getLogger('voice-sip-bridge')

VERSION = '1.0.0'

app = FastAPI(title='Voice SIP Bridge (Pipecat-based)', version=VERSION)


def make_session(
    session_id: str, in_rate: int, out_rate: int, send_audio, call_uuid: str | None = None
) -> CallSession:
    return CallSession(
        session_id=session_id,
        in_sample_rate=in_rate,
        out_sample_rate=out_rate,
        send_audio=send_audio,
        call_uuid=call_uuid,
    )


@app.get('/v1/call-status/{call_uuid}')
async def call_status(call_uuid: str):
    """Д5: цель перевода для диалплана Asterisk (CURL после AudioSocket).

    Ответ 'NONE' — звонок решен ботом, Hangup; имя очереди — Goto handoff.
    """
    from call_status import get_transfer

    queue = await get_transfer(call_uuid)
    return {'uuid': call_uuid, 'transfer': queue or 'NONE'}


@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'version': VERSION,
        'audiosocket': f'{AUDIOSOCKET_HOST}:{AUDIOSOCKET_PORT}',
    }


@app.websocket('/ws/audio')
async def ws_audio(websocket: WebSocket, sample_rate: int = Query(default=8000)):
    """Двунаправленный аудиопоток: PCM 16-bit mono кадрами по WebSocket."""
    await websocket.accept()
    session_id = f'ws-{id(websocket)}'
    log.info('WS-сессия %s, sample_rate=%s', session_id, sample_rate)

    def send_audio(pcm: bytes) -> None:
        if websocket.client_state.name == 'CONNECTED':
            asyncio.create_task(websocket.send_bytes(pcm))

    session = make_session(session_id, sample_rate, sample_rate, send_audio)
    try:
        while True:
            frame = await websocket.receive_bytes()
            await session.feed_audio(frame)
    except WebSocketDisconnect:
        log.info('WS-сессия закрыта: %s', session_id)
    except Exception as exc:  # noqa: BLE001
        log.warning('WS-сессия %s оборвалась: %s', session_id, exc)
    finally:
        await session.finish()
        await session.close()


@app.on_event('startup')
async def startup():
    async def audiosocket_session_factory(session_id: str, send_audio, call_uuid: str | None = None) -> CallSession:
        return make_session(session_id, 8000, 8000, send_audio, call_uuid=call_uuid)

    server = await start_audiosocket_server(audiosocket_session_factory)
    app.state.audiosocket_server = server


@app.on_event('shutdown')
async def shutdown():
    server = getattr(app.state, 'audiosocket_server', None)
    if server:
        server.close()
        await server.wait_closed()
