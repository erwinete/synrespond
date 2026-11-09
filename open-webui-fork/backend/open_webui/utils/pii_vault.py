"""Обратимое маскирование ПДн (Reversible PII Vault).

ТЗ-доработка №5 / киллер-фича №5. Регулярные выражения заменяют телефон,
карту, e-mail, паспорт, ИНН и СНИЛС на токены вида [CARD_1]. Вольт хранит
соответствия в памяти с TTL; unmask восстанавливает значения перед отправкой
клиенту или в CRM.

Модуль не тянет зависимости open_webui — покрывается юнит-тестами напрямую.
"""

from __future__ import annotations

import re
import time
from typing import Any

_PATTERNS: dict[str, re.Pattern] = {
    'CARD': re.compile(r'\b(?:\d[ -]?){15,16}\d\b'),
    'PHONE': re.compile(r'(?:\+7|8)[\s\-(]?\d{3}[\s\-)]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b'),
    'EMAIL': re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
    'PASSPORT': re.compile(r'\b\d{4}\s?\d{6}\b'),
    'INN': re.compile(r'\b(?:\d{10}|\d{12})\b'),
    'SNILS': re.compile(r'\b\d{3}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{2}\b'),
}

# Порядок важен: сначала самые длинные/структурированные (карта, паспорт, СНИЛС), затем ИНН
_MASK_ORDER = ('CARD', 'PASSPORT', 'SNILS', 'PHONE', 'EMAIL', 'INN')

DEFAULT_TTL_SECONDS = 3600


class PiiVault:
    """Вольт соответствий токен -> значение для одного чата/сессии."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._vault: dict[str, tuple[str, float]] = {}
        self._counters: dict[str, int] = {}

    def mask(self, text: str) -> str:
        """Заменяет ПДн на токены и запоминает соответствия."""
        if not text:
            return text
        masked = text
        for kind in _MASK_ORDER:
            pattern = _PATTERNS[kind]

            def replace(match: re.Match, kind: str = kind) -> str:
                value = match.group(0)
                # Защита от ложных срабатываний на коротких числах
                digits = re.sub(r'\D', '', value)
                min_digits = {'CARD': 16, 'PHONE': 11, 'PASSPORT': 10, 'SNILS': 11, 'INN': 10, 'EMAIL': 5}[kind]
                if len(digits) < min_digits and kind != 'EMAIL':
                    return value
                token = f'[{kind}_{self._next_index(kind)}]'
                self._vault[token] = (value, time.monotonic() + self.ttl_seconds)
                return token

            masked = pattern.sub(replace, masked)
        return masked

    def unmask(self, text: str) -> str:
        """Восстанавливает значения по токенам. Истекшие токены остаются как есть."""
        if not text:
            return text
        now = time.monotonic()
        for token, (value, expires_at) in list(self._vault.items()):
            if expires_at < now:
                self._vault.pop(token, None)
                continue
            text = text.replace(token, value)
        return text

    def _next_index(self, kind: str) -> int:
        self._counters[kind] = self._counters.get(kind, 0) + 1
        return self._counters[kind]

    def mapping(self) -> dict[str, str]:
        """Снимок соответствий токен -> значение для стримингового демаскировщика."""
        now = time.monotonic()
        return {
            token: value
            for token, (value, expires_at) in self._vault.items()
            if expires_at >= now
        }

    def clear(self) -> None:
        self._vault.clear()
        self._counters.clear()


class PiiVaultRegistry:
    """Вольты по ключу сессии с очисткой просроченных записей."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, max_sessions: int = 10000):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._vaults: dict[str, PiiVault] = {}
        self._created: dict[str, float] = {}

    def get(self, session_key: str) -> PiiVault:
        now = time.monotonic()
        if session_key not in self._vaults or now - self._created.get(session_key, 0) > self.ttl_seconds:
            if len(self._vaults) >= self.max_sessions:
                self._evict_oldest()
            self._vaults[session_key] = PiiVault(self.ttl_seconds)
            self._created[session_key] = now
        return self._vaults[session_key]

    def _evict_oldest(self) -> None:
        oldest = min(self._created, key=self._created.get, default=None)
        if oldest:
            self._vaults.pop(oldest, None)
            self._created.pop(oldest, None)


_vaults = PiiVaultRegistry()


def get_vault(session_key: str) -> PiiVault:
    """Вольт для сессии чата (ключ: model_id + chat_id/user_id)."""
    return _vaults.get(session_key)


class StreamingPIIUnmasker:
    """Д2 (Stage 3): потоковый демаскировщик для SSE-стримов и TTS-пайплайна.

    Токен вида [CARD_1] может прийти за 3-4 чанка ("[", "CARD_", "1]"). Буфер
    удерживает незакрытый префикс '[' до прихода ']' или превышения
    max_tag_len, затем подставляет реальное значение из mapping.
    """

    def __init__(self, mapping: dict[str, str], max_tag_len: int = 24):
        self.mapping = mapping or {}
        self.max_tag_len = max_tag_len
        self.buffer = ''

    def feed(self, chunk: str) -> str:
        if not self.mapping:
            return chunk

        self.buffer += chunk
        for token, real_value in self.mapping.items():
            if token in self.buffer:
                self.buffer = self.buffer.replace(token, real_value)

        last_open = self.buffer.rfind('[')
        if last_open != -1:
            last_close = self.buffer.rfind(']', last_open)
            tail_len = len(self.buffer) - last_open
            if last_close == -1 and tail_len <= self.max_tag_len:
                # Удерживаем хвост до следующего чанка
                emit_part = self.buffer[:last_open]
                self.buffer = self.buffer[last_open:]
                return emit_part

        emit_part = self.buffer
        self.buffer = ''
        return emit_part

    def flush(self) -> str:
        for token, real_value in self.mapping.items():
            if token in self.buffer:
                self.buffer = self.buffer.replace(token, real_value)
        rem = self.buffer
        self.buffer = ''
        return rem


def unmask_stream_lines(lines, mapping: dict[str, str]):
    """Обертка итератора строк SSE: размаскирование токенов через буфер."""
    unmasker = StreamingPIIUnmasker(mapping)
    for line in lines:
        yield unmasker.feed(line)
    yield unmasker.flush()


def unmask_sse_stream_chunk(raw_chunk: str, unmasker: "StreamingPIIUnmasker") -> str:
    """Безопасное размаскирование ПДн внутри SSE-стрима OpenAI-формата (исправление BUG-08).

    Парсит строки `data: {...}`, пропускает через StreamingPIIUnmasker только
    текстовое содержимое delta.content / message.content и собирает обратно
    валидный SSE-пакет. Структурные скобки JSON ("choices":[{) не попадают
    в буфер демаскировщика, поэтому SSE-фреймы не задерживаются.
    """
    import json

    if not unmasker.mapping or not raw_chunk:
        return raw_chunk

    out_lines: list[str] = []
    for line in raw_chunk.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped.startswith("data:"):
            out_lines.append(line)
            continue
        payload_str = stripped.removeprefix("data:").strip()
        if payload_str == "[DONE]" or not payload_str.startswith("{"):
            if payload_str == "[DONE]":
                tail = unmasker.flush()
                if tail:
                    flush_pkt = {
                        "id": "pii-flush",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}],
                    }
                    out_lines.append(f"data: {json.dumps(flush_pkt, ensure_ascii=False)}\n\n")
            out_lines.append(line)
            continue
        try:
            packet = json.loads(payload_str)
            changed = False
            for choice in packet.get("choices") or []:
                delta = choice.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    delta["content"] = unmasker.feed(delta["content"])
                    changed = True
                msg = choice.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    msg["content"] = unmasker.feed(msg["content"]) + unmasker.flush()
                    changed = True
            if changed:
                ending = "\n\n" if line.endswith("\n\n") else ("\n" if line.endswith("\n") else "")
                out_lines.append(f"data: {json.dumps(packet, ensure_ascii=False)}{ending}")
            else:
                out_lines.append(line)
        except json.JSONDecodeError:
            out_lines.append(line)
    return "".join(out_lines)
