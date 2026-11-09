"""Канальная адаптация (Stage 4, раздел 2.2).

Платформа — модульный конструктор из трех канальных контуров
(chat / webrtc / sip) с единым ядром поведения. Модуль:

1. detect_channel — определение канала запроса;
2. filter_routes_by_channel — селективная маршрутизация Intent Matrix;
3. sanitize_for_voice_tts — очистка ответа перед озвучкой;
4. is_channel_enabled — матрица отключения модулей из config.
"""

from __future__ import annotations

import re
from typing import Any, Literal

Channel = Literal['chat', 'webrtc', 'sip']

VOICE_CHANNELS = ('webrtc', 'sip')

_MARKDOWN_NOISE = [
    (re.compile(r'\*\*\*(.+?)\*\*\*', re.S), r'\1'),
    (re.compile(r'\*\*(.+?)\*\*', re.S), r'\1'),
    (re.compile(r'__(.+?)__', re.S), r'\1'),
    (re.compile(r'_(.+?)_', re.S), r'\1'),
    (re.compile(r'~~(.+?)~~', re.S), r'\1'),
    (re.compile(r'`{3}.*?`{3}', re.S), ' '),
    (re.compile(r'`([^`]+)`'), r'\1'),
    (re.compile(r'#{1,6}\s*'), ''),
    (re.compile(r'!?\[([^\]]*)\]\(([^)]+)\)'), r'\1'),
    (re.compile(r'^\s*[-*+]\s+', re.M), ', '),
    (re.compile(r'^\s*\|\s*', re.M), ' '),
    (re.compile(r'\|\s*$', re.M), ' '),
    (re.compile(r'\s*\|\s*'), ', '),
    (re.compile(r'^\s*>\s?', re.M), ''),
    (re.compile(r'\n{3,}'), '\n'),
]


def detect_channel(request, payload: dict | None = None) -> Channel:
    """Канал запроса: заголовок X-Channel-Type, metadata.channel, X-Call-UUID."""
    header = None
    try:
        header = request.headers.get('x-channel-type')
    except Exception:  # noqa: BLE001
        header = None
    if header in ('chat', 'webrtc', 'sip'):
        return header
    try:
        if request.headers.get('x-call-uuid'):
            return 'sip'
    except Exception:  # noqa: BLE001
        pass
    metadata = (payload or {}).get('metadata') or {}
    channel = metadata.get('channel')
    if channel in ('chat', 'webrtc', 'sip'):
        return channel
    return 'chat'


async def is_channel_enabled(channel: Channel, config_get) -> bool:
    """Матрица отключения: асинхронно проверяет channels.<channel>.enabled из config (BUG-02)."""
    key = {
        'chat': 'channels.chat.enabled',
        'webrtc': 'channels.webrtc.enabled',
        'sip': 'channels.sip.enabled',
    }.get(channel, 'channels.chat.enabled')
    value = await config_get(key)
    return value is not False and bool(value if value is not None else True)


def filter_routes_by_channel(routes: list[Any], channel: Channel) -> list[Any]:
    """Селективная маршрутизация: правило активно в канале по allowed_channels."""
    result = []
    for route in routes:
        allowed = getattr(route, 'allowed_channels', None) or ['chat', 'webrtc', 'sip']
        if channel in allowed:
            result.append(route)
    return result


def sanitize_for_voice_tts(text: str, channel: Channel) -> str:
    """Очистка ответа от нечитаемой разметки перед TTS (webrtc / sip)."""
    if channel not in VOICE_CHANNELS or not text:
        return text
    result = text
    for pattern, replacement in _MARKDOWN_NOISE:
        result = pattern.sub(replacement, result)
    return result.strip()


def channel_variables(channel: Channel) -> dict[str, str]:
    """Шаблонные переменные канала: {{channel.type}}, {{channel.voice}}."""
    return {
        'channel.type': channel,
        'channel.voice': 'true' if channel in VOICE_CHANNELS else 'false',
    }
