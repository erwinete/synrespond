"""Транспорт Asterisk AudioSocket: TLV-протокол поверх TCP.

Формат кадра: type (1 байт) + length (2 байта, big-endian) + payload.
  0x01 — аудио (PCM 16-bit mono 8 кГц slin);
  0x00 — завершение звонка (от Asterisk);
  0xFF — ошибка.
Первое сообщение входящего соединения — UUID звонка (16 байт).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Awaitable, Callable

from config import AUDIOSOCKET_HOST, AUDIOSOCKET_PORT

log = logging.getLogger(__name__)

TYPE_AUDIO = 0x01
TYPE_HANGUP = 0x00
TYPE_ERROR = 0xFF

MAX_FRAME = 320 * 4  # 640 байт аудио на кадр — норма для slin 20 мс


async def _read_exact(reader: asyncio.StreamReader, n: int) -> bytes | None:
    try:
        data = await reader.readexactly(n)
        return data
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return None


async def handle_audiosocket_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    session_factory: Callable[[str, Callable[[bytes], Any]], Any],
) -> None:
    """Принимает звонок AudioSocket и связывает его с CallSession."""
    peer = writer.get_extra_info('peername')
    header = await _read_exact(reader, 16)
    if header is None:
        writer.close()
        return
    call_uuid = str(uuid.UUID(bytes=header))
    log.info('AudioSocket звонок %s от %s', call_uuid, peer)

    loop = asyncio.get_running_loop()

    def send_audio(pcm: bytes) -> None:
        if writer.is_closing():
            return
        for offset in range(0, len(pcm), MAX_FRAME):
            chunk = pcm[offset : offset + MAX_FRAME]
            frame = bytes([TYPE_AUDIO]) + len(chunk).to_bytes(2, 'big') + chunk
            loop.create_task(_write(writer, frame))

    session = session_factory(call_uuid, send_audio, call_uuid=call_uuid)

    hangup = False
    try:
        while not hangup:
            frame_type_raw = await _read_exact(reader, 1)
            if frame_type_raw is None:
                break
            frame_type = frame_type_raw[0]
            length_raw = await _read_exact(reader, 2)
            if length_raw is None:
                break
            length = int.from_bytes(length_raw, 'big')
            payload = await _read_exact(reader, length) if length else b''
            if payload is None:
                break

            if frame_type == TYPE_AUDIO and payload:
                await session.feed_audio(payload)
            elif frame_type == TYPE_HANGUP:
                log.info('AudioSocket hangup: %s', call_uuid)
                hangup = True
            elif frame_type == TYPE_ERROR:
                log.warning('AudioSocket ошибка от Asterisk: %s', call_uuid)
                hangup = True
    except Exception as exc:  # noqa: BLE001
        log.warning('AudioSocket соединение оборвалось (%s): %s', call_uuid, exc)
    finally:
        try:
            await session.finish()
        except Exception:  # noqa: BLE001
            pass
        await session.close()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


async def _write(writer: asyncio.StreamWriter, frame: bytes) -> None:
    try:
        writer.write(frame)
        await writer.drain()
    except Exception:  # noqa: BLE001
        pass


async def start_audiosocket_server(
    session_factory: Callable[[str, Callable[[bytes], Any]], Any],
) -> asyncio.AbstractServer:
    """Запускает TCP-сервер AudioSocket для Asterisk (ТЗ, раздел 7.2)."""

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await handle_audiosocket_connection(reader, writer, session_factory)

    server = await asyncio.start_server(handler, AUDIOSOCKET_HOST, AUDIOSOCKET_PORT)
    log.info('AudioSocket TCP-сервер слушает %s:%s', AUDIOSOCKET_HOST, AUDIOSOCKET_PORT)
    return server
