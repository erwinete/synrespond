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
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

_JSONRPC_HEADERS = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}


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
    if harness.auth_token:
        headers['Authorization'] = f'Bearer {harness.auth_token}'

    try:
        if transport in ('streamable_http', 'sse'):
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(
                    harness.endpoint_or_cmd, json=payload, headers=headers
                )
                response.raise_for_status()
                data = _parse_rpc_response(response.text)
        elif transport == 'stdio':
            data = await asyncio.to_thread(
                _call_stdio, harness.endpoint_or_cmd, payload, timeout_s
            )
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
        log.warning('MCP harness call failed (%s.%s): %s', harness.id, tool_name, exc)
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


def _call_stdio(cmd: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Один shot stdin/stdout вызов stdio-процесса."""
    import subprocess

    proc = subprocess.run(
        cmd, input=json.dumps(payload), capture_output=True, text=True, timeout=timeout_s, shell=True
    )
    return _parse_rpc_response(proc.stdout)


async def discover_tools(harness) -> dict[str, Any]:
    """tools/list у внешнего харнеса — для кнопки Discover Tools и кэша схемы."""
    payload = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list', 'params': {}}
    headers = dict(_JSONRPC_HEADERS)
    if harness.auth_token:
        headers['Authorization'] = f'Bearer {harness.auth_token}'
    timeout_s = (harness.timeout_ms or 5000) / 1000 + 5
    try:
        if (harness.transport or '').lower() == 'stdio':
            data = await asyncio.to_thread(_call_stdio, harness.endpoint_or_cmd, payload, timeout_s)
        else:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(
                    harness.endpoint_or_cmd, json=payload, headers=headers
                )
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
