"""Юнит-тесты мастер-ТЗ: SEC-01..06, BUG-01..09, рефакторинг гибридного поиска.

Проверяются: SSRF-фильтр, Path Traversal, ReDoS-защита, безопасный stdio,
Fernet-шифрование секретов, SSE-размаскировщик (BUG-08), async is_channel_enabled
(BUG-02), SLOT_CAPABLE_ACTIONS (BUG-04), live-раны Ragas (BUG-03), onset barge-in.
"""

import asyncio
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / 'open-webui-fork' / 'backend' / 'open_webui'
WORKER = ROOT / 'services' / 'ragas-worker'
BRIDGE = ROOT / 'services' / 'voice-sip-bridge'


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Namespace-пакеты open_webui — чтобы вложенные импорты модулей ядра
# резолвились без полного поднятия приложения
_ow = sys.modules.get('open_webui') or types.ModuleType('open_webui')
_ow.__path__ = [str(FORK)]
sys.modules['open_webui'] = _ow
_ow_utils = sys.modules.get('open_webui.utils') or types.ModuleType('open_webui.utils')
_ow_utils.__path__ = [str(FORK / 'utils')]
sys.modules['open_webui.utils'] = _ow_utils


# --- security_guard (SEC-01/02/03/06, 8.1) ---

security_guard = _load('tz_security_guard', FORK / 'utils' / 'security_guard.py')


class TestSecurityGuardSSRF:
    def test_loopback_blocked(self):
        with pytest.raises(ValueError):
            security_guard.validate_outbound_url('http://127.0.0.1:8080/api', allow_internal_services=False)

    def test_link_local_blocked(self):
        with pytest.raises(ValueError):
            security_guard.validate_outbound_url('http://169.254.169.254/latest/meta-data', allow_internal_services=True)

    def test_private_ip_blocked(self):
        with pytest.raises(ValueError):
            security_guard.validate_outbound_url('http://192.168.1.10/report', allow_internal_services=True)

    def test_internal_service_host_allowed(self):
        url = 'http://encoder-classifier:8001/v1/classify'
        assert security_guard.validate_outbound_url(url, allow_internal_services=True) == url

    def test_non_http_scheme_blocked(self):
        with pytest.raises(ValueError):
            security_guard.validate_outbound_url('file:///etc/passwd', allow_internal_services=True)

    def test_localhost_allowed_with_env(self, monkeypatch):
        monkeypatch.setenv('ALLOW_LOCALHOST_WEBHOOKS', 'true')
        url = 'http://localhost:9999/hook'
        assert security_guard.validate_outbound_url(url, allow_internal_services=False) == url


class TestSecurityGuardPath:
    def test_traversal_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            security_guard.safe_subpath(str(tmp_path), '../../etc', field_name='collection_id')

    def test_invalid_chars_rejected(self):
        with pytest.raises(ValueError):
            security_guard.validate_safe_identifier('col\\..\\x', field_name='collection_id')

    def test_safe_subpath_ok(self, tmp_path):
        target = security_guard.safe_subpath(str(tmp_path), 'kb-123', field_name='collection_id')
        assert str(tmp_path) in target and 'kb-123' in target

    def test_safe_id_regex(self):
        assert security_guard.validate_safe_identifier('rubert-intents-v1') == 'rubert-intents-v1'
        with pytest.raises(ValueError):
            security_guard.validate_safe_identifier('a/b')


class TestSecurityGuardReDoS:
    def test_nested_quantifier_rejected(self):
        with pytest.raises(ValueError):
            security_guard.validate_safe_regex('(a+)+b')

    def test_too_long_pattern_rejected(self):
        with pytest.raises(ValueError):
            security_guard.validate_safe_regex('a' * 300)

    def test_safe_regex_fullmatch(self):
        assert security_guard.safe_regex_fullmatch(r'\d{4}', '1234') is True
        assert security_guard.safe_regex_fullmatch(r'\d{4}', '12345') is False
        # Длинный вход не проходит сопоставление по лимиту длины
        assert security_guard.safe_regex_fullmatch(r'\w+', 'x' * 600) is False


class TestSecurityGuardStdio:
    def test_shell_metacharacters_rejected(self):
        for cmd in ('npx -y foo; rm -rf /', 'python x.py && echo pwned', 'uvx mcp-server $(whoami)'):
            with pytest.raises(ValueError):
                security_guard.parse_safe_stdio_command(cmd)

    def test_binary_outside_allowlist_rejected(self):
        with pytest.raises(ValueError):
            security_guard.parse_safe_stdio_command('powershell -Command x')

    def test_clean_command_parsed(self):
        argv = security_guard.parse_safe_stdio_command('python mcp_server.py --port 9')
        assert argv[0] == 'python' and '--port' in argv


class TestSecrets:
    def test_mask_secret(self):
        assert security_guard.mask_secret(None) is None
        assert security_guard.mask_secret('short') == '••••••••'
        masked = security_guard.mask_secret('sk-1234567890abcdef')
        assert masked.startswith('sk-') and masked.endswith('def') and '••••' in masked

    def test_fernet_roundtrip(self):
        pytest.importorskip('cryptography')
        token = 'super-secret-token'
        encrypted = security_guard.encrypt_secret(token)
        assert encrypted.startswith('enc::')
        assert security_guard.decrypt_secret(encrypted) == token
        # Повторное шифрование не оборачивает дважды
        assert security_guard.encrypt_secret(encrypted) == encrypted

    def test_decrypt_plain_passthrough(self):
        assert security_guard.decrypt_secret('plain-token') == 'plain-token'


# --- http_pool ---

http_pool = _load('tz_http_pool', FORK / 'utils' / 'http_pool.py')


class TestHttpPool:
    def test_singleton_client(self):
        client1 = http_pool.get_http_client()
        client2 = http_pool.get_http_client()
        assert client1 is client2

    def test_close_client(self):
        client = http_pool.get_http_client()
        asyncio.run(http_pool.close_http_client())
        assert client.is_closed


# --- pii_vault BUG-08 ---

pii_vault = _load('tz_pii_vault', FORK / 'utils' / 'pii_vault.py')


class TestBug08SseUnmask:
    def _sse(self, content: str) -> str:
        packet = {'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]}
        return f"data: {json.dumps(packet, ensure_ascii=False)}\n\n"

    def test_structural_brackets_not_delayed(self):
        unmasker = pii_vault.StreamingPIIUnmasker({'[CARD_1]': '4111 1111 1111 1111'})
        raw = self._sse('Карта [CARD_1] списана')
        out = pii_vault.unmask_sse_stream_chunk(raw, unmasker)
        for line in out.splitlines():
            if line.startswith('data: {'):
                packet = json.loads(line.removeprefix('data:').strip())
                content = packet['choices'][0]['delta']['content']
                assert '[CARD_1]' not in content
        assert '4111 1111 1111 1111' in out
        # Структурные скобки JSON не залипают в буфере демаскировщика
        assert unmasker.buffer == ''

    def test_token_split_across_chunks_flushed_on_done(self):
        unmasker = pii_vault.StreamingPIIUnmasker({'[CARD_1]': '4111 1111 1111 1111'})
        out1 = pii_vault.unmask_sse_stream_chunk(self._sse('Платеж [CARD_'), unmasker)
        json.loads(out1.splitlines()[0].removeprefix('data:').strip())
        # Хвост токена приходит следующим чанком и восстанавливается
        out2 = pii_vault.unmask_sse_stream_chunk(self._sse('1] ок'), unmasker)
        assert '4111 1111 1111 1111' in out2
        # [DONE] порождает flush-пакет, если буфер еще держал хвост
        out3 = pii_vault.unmask_sse_stream_chunk('data: [DONE]\n\n', unmasker)
        assert 'data: [DONE]' in out3

    def test_no_mapping_passthrough(self):
        unmasker = pii_vault.StreamingPIIUnmasker({})
        raw = 'data: {"choices":[{"delta":{"content":"choices":[{"}}]}\n\n'
        assert pii_vault.unmask_sse_stream_chunk(raw, unmasker) == raw


# --- channel_adapter BUG-02 ---

channel_adapter = _load('tz_channel_adapter', FORK / 'utils' / 'channel_adapter.py')


class TestBug02ChannelEnabled:
    def test_is_coroutine_function(self):
        assert inspect.iscoroutinefunction(channel_adapter.is_channel_enabled)

    def _run(self, value):
        async def config_get(key):
            return value

        return asyncio.run(channel_adapter.is_channel_enabled('chat', config_get))

    def test_disabled_channel_returns_false(self):
        assert self._run(False) is False

    def test_none_defaults_to_enabled(self):
        assert self._run(None) is True

    def test_explicit_true(self):
        assert self._run(True) is True


# --- intent_router BUG-04/SEC-02/SEC-06 ---

_config_stub = types.ModuleType('open_webui.models.config')


class _ConfigStub:
    _store = {}

    @classmethod
    async def get(cls, key, default=None):
        return cls._store.get(key, default)

    @classmethod
    async def get_many(cls, *keys):
        return {k: cls._store.get(k) for k in keys}


_config_stub.Config = _ConfigStub
sys.modules.setdefault('open_webui.models', types.ModuleType('open_webui.models'))
sys.modules['open_webui.models.config'] = _config_stub

_intents_stub = types.ModuleType('open_webui.models.intents')
_intents_stub.IntentRoutes = type('IntentRoutes', (), {})
sys.modules['open_webui.models.intents'] = _intents_stub

_misc_stub = types.ModuleType('open_webui.utils.misc')


def _add_or_update_system_message(content: str, messages: list, append: bool = False):
    return messages


def _get_last_user_message(messages: list):
    for message in reversed(messages):
        if message.get('role') == 'user':
            content = message.get('content')
            if isinstance(content, str):
                return content
    return None


_misc_stub.add_or_update_system_message = _add_or_update_system_message
_misc_stub.get_last_user_message = _get_last_user_message
sys.modules['open_webui.utils.misc'] = _misc_stub

_load('tz_route_templates', FORK / 'utils' / 'route_templates.py')
_load('tz_channel_adapter_ns', FORK / 'utils' / 'channel_adapter.py')
_load('tz_security_guard_ns', FORK / 'utils' / 'security_guard.py')
_load('tz_http_pool_ns', FORK / 'utils' / 'http_pool.py')
_load('tz_mcp_harness_ns', FORK / 'utils' / 'mcp_harness.py')

intent_router = _load('tz_intent_router', FORK / 'utils' / 'intent_router.py')


class TestBug04SlotCapableActions:
    def test_constant_includes_mcp_and_rag(self):
        assert set(intent_router.SLOT_CAPABLE_ACTIONS) == {
            'WEBHOOK_TOOL',
            'DIRECT_RESPONSE',
            'MCP_HARNESS',
            'RAG_COLLECTION',
            'SYSTEM_PROMPT_OVERRIDE',
        }

    def test_both_branches_use_constant(self):
        source = (FORK / 'utils' / 'intent_router.py').read_text(encoding='utf-8')
        assert source.count('rule.action_type in SLOT_CAPABLE_ACTIONS') >= 2
        assert "'WEBHOOK_TOOL', 'DIRECT_RESPONSE'" not in source.replace(
            "SLOT_CAPABLE_ACTIONS = (", ""
        ).replace("'WEBHOOK_TOOL',\n    'DIRECT_RESPONSE',", "")


class TestMissingSlotsSafeRegex:
    def _rule(self, pattern):
        class Rule:
            required_slots = [{'name': 'order', 'pattern': pattern, 'question': 'номер?'}]

        return Rule()

    def test_redos_pattern_counts_as_missing(self):
        missing = intent_router._missing_required_slots(self._rule('(a+)+'), {'order': 'aaaa'})
        assert len(missing) == 1

    def test_valid_pattern_passes(self):
        missing = intent_router._missing_required_slots(self._rule(r'\d{5}'), {'order': '12345'})
        assert missing == []

    def test_value_over_length_limit(self):
        missing = intent_router._missing_required_slots(self._rule(r'\w+'), {'order': 'x' * 600})
        assert len(missing) == 1


class TestWebhookSsrf:
    def test_internal_metadata_ip_blocked(self):
        result = asyncio.run(
            intent_router._call_webhook({'url': 'http://169.254.169.254/latest/meta-data/'}, {'user_message': 'x'})
        )
        assert result is None

    def test_loopback_blocked(self):
        result = asyncio.run(
            intent_router._call_webhook({'url': 'http://127.0.0.1:5432/'}, {'user_message': 'x'})
        )
        assert result is None

    def test_json_injection_safe_rendering(self, monkeypatch):
        # Кавычки и переводы строк в переменных больше не ломают JSON тела
        import socket as socket_mod

        def fake_getaddrinfo(host, port, *args, **kwargs):
            return [(socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, '', ('93.184.216.34', port or 80))]

        monkeypatch.setattr(socket_mod, 'getaddrinfo', fake_getaddrinfo)

        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {'answer': 'ok'}

        class FakeClient:
            async def post(self, url, json=None, headers=None, timeout=None):
                captured['url'] = url
                captured['json'] = json

                async def _noop():
                    return None

                return FakeResponse()

        import open_webui.utils.http_pool as pool_mod

        original = pool_mod.get_http_client
        pool_mod.get_http_client = lambda: FakeClient()
        try:
            result = asyncio.run(
                intent_router._call_webhook(
                    {'url': 'https://crm.example.com/api', 'body': {'message': '{{user_message}}'}},
                    {'user_message': 'текст "с кавычками"\nи переносом'},
                )
            )
        finally:
            pool_mod.get_http_client = original
        assert result == 'ok'
        assert captured['json']['message'] == 'текст "с кавычками"\nи переносом'


# --- hybrid_fusion BUG-06 ---

hybrid_fusion = _load('tz_hybrid_fusion', FORK / 'retrieval' / 'hybrid_fusion.py')


class TestBug06Bm25Cache:
    def test_cache_registry_exists(self):
        assert isinstance(hybrid_fusion._BM25_CACHE, dict)

    def test_invalidate_bm25_cache(self):
        hybrid_fusion._BM25_CACHE['col-x'] = ('v1', [], None)
        hybrid_fusion.invalidate_bm25_cache('col-x')
        assert 'col-x' not in hybrid_fusion._BM25_CACHE

    def test_weights_normalized_with_inactive_hippo(self):
        wd, wb, wh = hybrid_fusion.normalize_three_weights(0.4, 0.2, 0.4, hipporag_active=False)
        assert wh == 0.0 and abs(wd + wb - 1.0) < 1e-6


# --- ragas-worker BUG-03 ---

class TestBug03LiveRun:
    @pytest.fixture()
    def worker_db(self, tmp_path, monkeypatch):
        db_path = tmp_path / 'ragas_test.db'
        monkeypatch.setenv('DATABASE_URL', f'sqlite+aiosqlite:///{db_path}')
        for mod_name in ('tz_worker_config', 'tz_worker_db'):
            sys.modules.pop(mod_name, None)
        # db.py импортирует 'from config import DATABASE_URL' — регистрируем
        # конфиг воркера под каноническим именем 'config'
        config_mod = _load('tz_worker_config', WORKER / 'config.py')
        assert config_mod.DATABASE_URL.startswith('sqlite')
        sys.modules['config'] = config_mod
        db = _load('tz_worker_db', WORKER / 'db.py')
        asyncio.run(db.init_db_if_sqlite())
        return db

    def test_insert_sample_creates_live_run(self, worker_db):
        db = worker_db
        asyncio.run(
            db.insert_sample(
                {
                    'run_id': 'live-20990101',
                    'question': 'Где заказ?',
                    'answer': 'Заказ в пути',
                    'bot_model_id': 'support-bot',
                }
            )
        )

        async def _check():
            from sqlalchemy import select

            async with db.SessionFactory() as session:
                row = (await session.execute(select(db.ragas_run))).mappings().all()
                samples = (await session.execute(select(db.ragas_sample))).mappings().all()
                return row, samples

        runs, samples = asyncio.run(_check())
        assert any(r['id'] == 'live-20990101' and r['mode'] == 'live' for r in runs)
        assert len(samples) == 1

    def test_existing_batch_run_not_duplicated(self, worker_db):
        db = worker_db
        asyncio.run(
            db.insert_sample({'run_id': 'batch-42', 'question': 'q', 'answer': 'a'})
        )
        asyncio.run(
            db.insert_sample({'run_id': 'batch-42', 'question': 'q2', 'answer': 'a2'})
        )

        async def _check():
            from sqlalchemy import select

            async with db.SessionFactory() as session:
                return (await session.execute(select(db.ragas_run))).mappings().all()

        runs = asyncio.run(_check())
        # Пакетный прогон без ensure не создается, сэмплы пишутся без висячих ошибок
        assert all(r['id'] != 'batch-42' for r in runs)


# --- voice bridge: onset barge-in ---

_bridge_config = sys.modules.get('config')
if _bridge_config is None:
    _bridge_config = types.ModuleType('config')
    sys.modules['config'] = _bridge_config
exec(
    compile((BRIDGE / 'config.py').read_text(encoding='utf-8'), str(BRIDGE / 'config.py'), 'exec'),
    _bridge_config.__dict__,
)
bridge_vad = _load('tz_bridge_vad', BRIDGE / 'vad.py')


class TestVadSpeechOnset:
    def _loud(self, ms: int) -> bytes:
        rate = _bridge_config.INTERNAL_SAMPLE_RATE
        count = int(rate * ms / 1000)
        return b'\x4d\x2a' * count  # постоянный ненулевой уровень

    def test_onset_false_in_silence(self):
        segmenter = bridge_vad.UtteranceSegmenter()
        segmenter.feed(b'\x00\x00' * 512, _bridge_config.INTERNAL_SAMPLE_RATE)
        assert segmenter.has_active_speech_onset() is False

    def test_onset_after_start_ms(self):
        import time as _time

        segmenter = bridge_vad.UtteranceSegmenter()
        segmenter.feed(self._loud(100), _bridge_config.INTERNAL_SAMPLE_RATE)
        assert segmenter.speaking is True
        # Меньше VAD_START_MS — onset еще нет
        if (_time.monotonic() - segmenter.speech_started_ts) * 1000 < _bridge_config.VAD_START_MS:
            assert segmenter.has_active_speech_onset() is False
        _time.sleep(0.13)
        assert segmenter.has_active_speech_onset() is True
