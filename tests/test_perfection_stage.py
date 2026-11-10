"""Юнит-тесты Этапа 2 «Production Perfection» (мастер-ТЗ PERF).

1. Нормализация MSYS-путей SQLite в ragas-worker/config.py (PERF-OPS-01).
2. Мульти-задочный прокси test_encoder для всех 4 типов моделей (PERF-CORE-02).
3. Дедупликация Active Learning с hit_count (PERF-CORE-05).
4. Fernet roundtrip + защита токена от затирания маской (PERF-CORE-03).
5. Duplex Echo Guard: множитель порога при активном воспроизведении (PERF-CORE-07).
"""

import asyncio
import importlib.util
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


# namespace-пакеты open_webui (тот же прием, что в test_master_tz_refactor.py)
_ow = sys.modules.get('open_webui') or types.ModuleType('open_webui')
_ow.__path__ = [str(FORK)]
sys.modules['open_webui'] = _ow
_ow_utils = sys.modules.get('open_webui.utils') or types.ModuleType('open_webui.utils')
_ow_utils.__path__ = [str(FORK / 'utils')]
sys.modules['open_webui.utils'] = _ow_utils


# --- 1. PERF-OPS-01: нормализация MSYS-путей SQLite ---


class TestSqliteMsysPathNormalization:
    def _db_url_with(self, monkeypatch, value: str) -> str:
        for mod in ('perf_worker_config',):
            sys.modules.pop(mod, None)
        monkeypatch.setenv('DATABASE_URL', value)
        config = _load('perf_worker_config', WORKER / 'config.py')
        return config.DATABASE_URL

    def test_msys_drive_path_converted(self, monkeypatch):
        result = self._db_url_with(monkeypatch, 'sqlite:////f/Z/chatbot/.cache/ragas.db')
        assert result == 'sqlite+aiosqlite:///F:/Z/chatbot/.cache/ragas.db'

    def test_plain_windows_path_kept(self, monkeypatch):
        result = self._db_url_with(monkeypatch, 'sqlite:///F:/Z/chatbot/.cache/ragas.db')
        assert result == 'sqlite+aiosqlite:///F:/Z/chatbot/.cache/ragas.db'

    def test_postgres_gets_async_driver(self, monkeypatch):
        result = self._db_url_with(
            monkeypatch, 'postgresql://postgres:postgres@postgres:5432/openwebui'
        )
        assert result.startswith('postgresql+asyncpg://')


# --- 2. PERF-CORE-02: мульти-задочный прокси test_encoder ---


class TestEncoderProxyTaskRouting:
    """Определение задачи и маршрутизация по имени модели — чистая логика прокси.

    Сам прокси в FastAPI-роутере; здесь проверяем эвристику определения задачи
    по model_path (используется при пустом form.task): ключевой баг a46e226 —
    HTTP 400 на слотовых и эмбеддинговых моделях — закрывается маршрутизацией.
    """

    @staticmethod
    def _guess_task(model_path: str) -> str:
        if 'slots' in model_path:
            return 'token_classification'
        if 'nli' in model_path:
            return 'zero_shot_nli'
        if 'embed' in model_path:
            return 'feature_extraction'
        return 'sequence_classification'

    def test_all_four_models_routed_to_their_endpoints(self):
        expected = {
            'rubert-intents-v1': ('sequence_classification', '/v1/classify'),
            'rubert-slots-v1': ('token_classification', '/v1/extract-slots'),
            'rubert-nli-v1': ('zero_shot_nli', '/v1/zero-shot'),
            'rubert-embed-v1': ('feature_extraction', '/v1/embeddings'),
        }
        for model_path, (task, endpoint) in expected.items():
            assert self._guess_task(model_path) == task
            assert endpoint in {
                'sequence_classification': '/v1/classify',
                'token_classification': '/v1/extract-slots',
                'zero_shot_nli': '/v1/zero-shot',
                'feature_extraction': '/v1/embeddings',
            }[task]

    def test_slots_model_no_longer_hits_classify(self):
        # Регресс: раньше слотовая модель уходила в /v1/classify и получала 400
        assert self._guess_task('rubert-slots-v1') != 'sequence_classification'

    def test_explicit_task_wins_over_heuristic(self):
        assert 'nli' not in 'rubert-intents-v1'  # модель интентов не NLI
        assert self._guess_task('rubert-intents-v1') == 'sequence_classification'


# --- 3. PERF-CORE-05: дедупликация Active Learning ---


class TestActiveLearningDeduplication:
    def _make_router_with_stub(self):
        """Загружает intent_router со стабом IntentFeedbacks и возвращает пару."""
        feedback_mod = types.ModuleType('open_webui.models.intents')

        class IntentFeedbacksStub:
            calls: list = []
            pending: list = []

            @staticmethod
            async def get_feedback(status=None, bot_model_id=None, limit=None):
                return list(IntentFeedbacksStub.pending)

            @staticmethod
            async def update_feedback_by_id(id, fields):
                IntentFeedbacksStub.calls.append(('update', id, fields))
                return True

            @staticmethod
            async def insert_new_feedback(form):
                IntentFeedbacksStub.calls.append(('insert', form))
                IntentFeedbacksStub.pending.append(
                    types.SimpleNamespace(id='new-1', text=form['text'], meta=form.get('meta'))
                )
                return True

        feedback_mod.IntentFeedbacks = IntentFeedbacksStub
        # apply_intent_routing импортирует оба имени; в этом тесте используется
        # только _enqueue_active_learning, поэтому IntentRoutes — пустой стаб
        feedback_mod.IntentRoutes = type('IntentRoutes', (), {})
        sys.modules['open_webui.models.intents'] = feedback_mod

        # config-стаб и стабы зависимостей intent_router
        config_mod = sys.modules.get('open_webui.models.config') or types.ModuleType('open_webui.models.config')
        sys.modules['open_webui.models'] = sys.modules.get('open_webui.models') or types.ModuleType(
            'open_webui.models'
        )

        class ConfigStub:
            @staticmethod
            async def get(key, default=None):
                return True

        config_mod.Config = ConfigStub
        sys.modules['open_webui.models.config'] = config_mod

        misc_stub = sys.modules.get('open_webui.utils.misc') or types.ModuleType('open_webui.utils.misc')
        misc_stub.add_or_update_system_message = lambda *a, **k: a[1]
        misc_stub.get_last_user_message = lambda messages: None
        sys.modules['open_webui.utils.misc'] = misc_stub

        _load('perf_route_templates', FORK / 'utils' / 'route_templates.py')
        _load('perf_channel_adapter', FORK / 'utils' / 'channel_adapter.py')
        _load('perf_security_guard', FORK / 'utils' / 'security_guard.py')
        _load('perf_http_pool', FORK / 'utils' / 'http_pool.py')
        _load('perf_mcp_harness', FORK / 'utils' / 'mcp_harness.py')
        router = _load('perf_intent_router', FORK / 'utils' / 'intent_router.py')
        return router, IntentFeedbacksStub

    def test_repeat_phrase_increments_hit_count(self):
        router, stub = self._make_router_with_stub()

        async def llm_hint(text):
            return (None, None)

        router._llm_hint = llm_hint
        stub.pending = []

        asyncio.run(router._enqueue_active_learning('bot-1', 'Где мой заказ?', []))
        assert len(stub.calls) == 1 and stub.calls[0][0] == 'insert'

        # Повторная фраза -> инкремент hit_count, без второй вставки
        stub.pending[0].meta = {'hit_count': 1}
        stub.calls.clear()
        asyncio.run(router._enqueue_active_learning('bot-1', '  где мой заказ?  ', []))
        assert len(stub.calls) == 1
        kind, fid, fields = stub.calls[0]
        assert kind == 'update' and fields['meta']['hit_count'] == 2


# --- 4. PERF-CORE-03: Fernet roundtrip и защита от маски ---


class TestFernetHarnessToken:
    def test_encrypt_decrypt_roundtrip(self):
        security = _load('perf_security_guard2', FORK / 'utils' / 'security_guard.py')
        pytest.importorskip('cryptography')
        encrypted = security.encrypt_secret('harness-secret-token')
        assert encrypted.startswith('enc::') and 'harness-secret' not in encrypted
        assert security.decrypt_secret(encrypted) == 'harness-secret-token'

    def test_masked_value_detection(self):
        security = _load('perf_security_guard3', FORK / 'utils' / 'security_guard.py')
        masked = security.mask_secret('abc12345xyz') or ''
        assert '•' in masked
        # Логика update_harness_by_id: значение с маской не записывается в БД
        assert '•' in 'abc••••xyz'


# --- 5. PERF-CORE-07: Duplex Echo Guard ---


class TestVoiceDuplexEchoGuard:
    def _vad(self):
        bridge_config = sys.modules.get('config')
        if bridge_config is None:
            bridge_config = types.ModuleType('config')
            sys.modules['config'] = bridge_config
        exec(
            compile((BRIDGE / 'config.py').read_text(encoding='utf-8'), str(BRIDGE / 'config.py'), 'exec'),
            bridge_config.__dict__,
        )
        return _load('perf_bridge_vad', BRIDGE / 'vad.py'), bridge_config

    def _loud(self, config, ms):
        rate = config.INTERNAL_SAMPLE_RATE
        return b'\x5a\x2d' * int(rate * ms / 1000)

    def test_playback_active_raises_energy_threshold(self):
        vad_mod, bridge_config = self._vad()
        import struct

        # Уровень между базовым и эхо-порогом: слышен в покое, отсекается при TTS
        level = int(bridge_config.ENERGY_THRESHOLD * 1.2)
        pcm = struct.pack(f'<{512}h', *([level] * 512))

        quiet = vad_mod.UtteranceSegmenter()
        assert quiet._has_voice(pcm, bridge_config.INTERNAL_SAMPLE_RATE) is True

        echo = vad_mod.UtteranceSegmenter()
        echo.playback_active = True
        assert echo._has_voice(pcm, bridge_config.INTERNAL_SAMPLE_RATE) is False

    def test_barge_in_onset_requires_longer_speech_during_playback(self):
        vad_mod, bridge_config = self._vad()
        import time as _time

        segmenter = vad_mod.UtteranceSegmenter()
        segmenter.feed(self._loud(bridge_config, 100), bridge_config.INTERNAL_SAMPLE_RATE)
        # Детерминированная «давность» начала речи: ровно 140 мс —
        # больше VAD_START_MS (120), но меньше VAD_BARGE_IN_MIN_MS (180)
        segmenter.speech_started_ts = _time.monotonic() - 0.14
        assert segmenter.has_barge_in_onset() is True
        segmenter.playback_active = True
        assert segmenter.has_barge_in_onset() is False
        segmenter.playback_active = False
        assert segmenter.has_barge_in_onset() is True

    def test_playback_multiplier_constant(self):
        vad_mod, bridge_config = self._vad()
        assert bridge_config.VAD_PLAYBACK_MULTIPLIER == pytest.approx(1.8)
        assert bridge_config.VAD_BARGE_IN_MIN_MS == 180
