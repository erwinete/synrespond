"""MCP-клиент для вызова внешних агентных харнесов (Stage 4, направление Б).

Реестр харнесов хранится в таблице `mcp_harness`. Поддерживаемые транспорты:
  - streamable_http: JSON-RPC 2.0 поверх HTTP POST;
  - sse: JSON-RPC поверх HTTP POST (SSE-ответ сворачивается в JSON);
  - stdio: процесс, JSON-RPC по stdin/stdout.

Вызовы из диалога: intent_router при action_type == 'MCP_HARNESS' собирает
слоты, рендерит argument_mapping и вызывает perform_tool_call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

from open_webui.utils.security_guard import decrypt_secret, parse_safe_stdio_command, validate_outbound_url

log = logging.getLogger(__name__)

_JSONRPC_HEADERS = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}


def _harness_auth_header(harness) -> dict[str, str]:
    """Bearer-заголовок из расшифрованного auth_token харнеса (8.1)."""
    token = decrypt_secret(getattr(harness, 'auth_token', None))
    if token:
        return {'Authorization': f'Bearer {token}'}
    return {}


def _validated_harness_url(harness) -> str:
    """URL харнеса после SSRF-проверки (SEC-02)."""
    return validate_outbound_url(harness.endpoint_or_cmd, allow_internal_services=True)


def render_arguments(argument_mapping: dict[str, str], variables: dict[str, Any]) -> dict[str, Any]:
    """Подстановка {{slots.*}}, {{user_message}}, {{channel.type}} в аргументы инструмента."""
    from open_webui.utils.route_templates import render_route_template

    slots = {k[len('slots.'):]: v for k, v in variables.items() if k.startswith('slots.')}
    rendered: dict[str, Any] = {}
    for key, template in (argument_mapping or {}).items():
        if isinstance(template, str):
            value: Any = render_route_template(
                template,
                user_message=str(variables.get('user_message') or ''),
                intent=str(variables.get('intent') or ''),
                slots=slots,
                extra=variables,
            )
            # Числовые значения не тащим строками
            if isinstance(value, str) and value.isdigit():
                value = int(value)
        else:
            value = template
        rendered[key] = value
    return rendered


async def call_tool(
    harness,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_ms: int | None = None,
) -> dict[str, Any]:
    """JSON-RPC tools/call к внешнему MCP-харнесу.

    Возвращает {'ok': bool, 'content': ..., 'is_error': ..., 'error': ...}.
    """
    transport = (harness.transport or 'streamable_http').lower()
    timeout_s = (timeout_ms or harness.timeout_ms or 5000) / 1000
    payload = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'tools/call',
        'params': {'name': tool_name, 'arguments': arguments},
    }
    headers = dict(_JSONRPC_HEADERS)
    headers.update(_harness_auth_header(harness))

    try:
        if transport in ('streamable_http', 'sse'):
            url = _validated_harness_url(harness)
            from open_webui.utils.http_pool import get_http_client

            client = get_http_client()
            response = await client.post(url, json=payload, headers=headers, timeout=timeout_s)
            response.raise_for_status()
            data = _parse_rpc_response(response.text)
        elif transport == 'stdio':
            data = await _call_stdio_async(harness.endpoint_or_cmd, payload, timeout_s)
        else:
            return {'ok': False, 'error': f'Unsupported transport: {transport}'}

        if 'error' in data:
            return {'ok': False, 'error': data['error']}
        result = data.get('result') or {}
        return {
            'ok': not result.get('isError', False),
            'content': result.get('content'),
            'structured': result.get('structuredContent'),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning('MCP harness call failed (%s.%s): %s', getattr(harness, 'id', '?'), tool_name, exc)
        return {'ok': False, 'error': str(exc)}


def _parse_rpc_response(raw: str) -> dict[str, Any]:
    """JSON или SSE (data: строки) -> JSON-RPC ответ."""
    text = raw.strip()
    if text.startswith('{'):
        return json.loads(text)
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('data:'):
            payload = line.removeprefix('data:').strip()
            if payload and payload != '[DONE]':
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    continue
    raise ValueError('No JSON-RPC payload in response')


async def _call_stdio_async(cmd: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Неблокирующий запуск stdio-харнеса через exec без shell и с очищенным окружением (SEC-01, 8.2)."""
    argv = parse_safe_stdio_command(cmd)
    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
    }
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=safe_env,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=(json.dumps(payload) + "\n").encode("utf-8")),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Stdio harness timed out after {timeout_s}s") from exc

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    if proc.returncode != 0 and not stdout_text.strip():
        raise RuntimeError(f"Stdio harness failed ({proc.returncode}): {stderr_bytes.decode('utf-8', 'replace')[:300]}")
    return _parse_rpc_response(stdout_text)


async def discover_tools(harness) -> dict[str, Any]:
    """tools/list у внешнего харнеса — для кнопки Discover Tools и кэша схемы."""
    payload = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list', 'params': {}}
    headers = dict(_JSONRPC_HEADERS)
    headers.update(_harness_auth_header(harness))
    timeout_s = (harness.timeout_ms or 5000) / 1000 + 5
    try:
        if (harness.transport or '').lower() == 'stdio':
            data = await _call_stdio_async(harness.endpoint_or_cmd, payload, timeout_s)
        else:
            url = _validated_harness_url(harness)
            from open_webui.utils.http_pool import get_http_client

            client = get_http_client()
            response = await client.post(url, json=payload, headers=headers, timeout=timeout_s)
            response.raise_for_status()
            data = _parse_rpc_response(response.text)
        if 'error' in data:
            return {'ok': False, 'error': data['error']}
        return {'ok': True, 'tools': (data.get('result') or {}).get('tools') or []}
    except Exception as exc:  # noqa: BLE001
        return {'ok': False, 'error': str(exc)}


def content_to_text(content: Any) -> str:
    """MCP content -> читаемый текст для пользователя."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    parts.append(str(item.get('text') or ''))
                elif 'text' in item:
                    parts.append(str(item['text']))
    return '\n'.join(p for p in parts if p)
