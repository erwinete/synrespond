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

from fastapi import FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect

from audiosocket import start_audiosocket_server
from config import AUDIOSOCKET_HOST, AUDIOSOCKET_PORT
from session import CallSession

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
log = logging.getLogger('voice-sip-bridge')

VERSION = '1.0.0'

# SEC-04: если секрет задан, /ws/audio и /v1/call-status требуют его
# через заголовок X-Bridge-Secret или query-параметр ?token=
VOICE_BRIDGE_SECRET = os.getenv('VOICE_BRIDGE_SECRET', '')


def _bridge_authorized(websocket: WebSocket) -> bool:
    if not VOICE_BRIDGE_SECRET:
        return True
    provided = websocket.query_params.get('token') or websocket.headers.get('x-bridge-secret') or ''
    return provided == VOICE_BRIDGE_SECRET


app = FastAPI(title='Voice SIP Bridge (Pipecat-based)', version=VERSION)


def make_session(
    session_id: str,
    in_rate: int,
    out_rate: int,
    send_audio,
    call_uuid: str | None = None,
    clear_outbound=None,
) -> CallSession:
    return CallSession(
        session_id=session_id,
        in_sample_rate=in_rate,
        out_sample_rate=out_rate,
        send_audio=send_audio,
        call_uuid=call_uuid,
        clear_outbound=clear_outbound,
    )


@app.get('/v1/call-status/{call_uuid}')
async def call_status(
    call_uuid: str,
    token: str = Query(default=''),
    x_bridge_secret: str = Header(default='', alias='X-Bridge-Secret'),
):
    """Д5: цель перевода для диалплана Asterisk (CURL после AudioSocket).

    Ответ 'NONE' — звонок решен ботом, Hangup; имя очереди — Goto handoff.
    SEC-04: при заданном VOICE_BRIDGE_SECRET требуется ?token= или
    заголовок X-Bridge-Secret.
    """
    if VOICE_BRIDGE_SECRET and token != VOICE_BRIDGE_SECRET and x_bridge_secret != VOICE_BRIDGE_SECRET:
        raise HTTPException(status_code=403, detail='Unauthorized bridge secret')
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
    if not _bridge_authorized(websocket):
        await websocket.send_json({'type': 'error', 'detail': 'Unauthorized bridge secret'})
        await websocket.close(code=4401)
        return
    session_id = f'ws-{id(websocket)}'
    log.info('WS-сессия %s, sample_rate=%s', session_id, sample_rate)

    pending_sends: set[asyncio.Task] = set()

    def send_audio(pcm: bytes) -> None:
        if websocket.client_state.name == 'CONNECTED':
            task = asyncio.create_task(websocket.send_bytes(pcm))
            pending_sends.add(task)
            task.add_done_callback(pending_sends.discard)

    def clear_outbound() -> None:
        """PERF-CORE-07: сброс еще не отправленных WS-кадров при barge-in."""
        for task in list(pending_sends):
            task.cancel()

    session = make_session(session_id, sample_rate, sample_rate, send_audio, clear_outbound=clear_outbound)
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
    async def audiosocket_session_factory(session_id: str, send_audio, call_uuid: str | None = None, clear_outbound=None) -> CallSession:
        return make_session(session_id, 8000, 8000, send_audio, call_uuid=call_uuid, clear_outbound=clear_outbound)

    server = await start_audiosocket_server(audiosocket_session_factory)
    app.state.audiosocket_server = server


@app.on_event('shutdown')
async def shutdown():
    server = getattr(app.state, 'audiosocket_server', None)
    if server:
        server.close()
        await server.wait_closed()
