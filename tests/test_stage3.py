"""Юнит-тесты Stage 3: стриминговое размаскирование, slot-filling хардненинг,
call-status реестр, атомарная ротация графа HippoRAG."""

import importlib.util
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORK_UTILS = ROOT / 'open-webui-fork' / 'backend' / 'open_webui' / 'utils'
BRIDGE_DIR = ROOT / 'services' / 'voice-sip-bridge'


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pii_vault = _load('pii_vault_s3', FORK_UTILS / 'pii_vault.py')
route_templates = _load('route_templates_s3', FORK_UTILS / 'route_templates.py')
call_status = _load('call_status_s3', FORK_UTILS / 'call_status.py')
hipporag_engine = _load('hipporag_engine_s3', ROOT / 'open-webui-fork' / 'backend' / 'open_webui' / 'retrieval' / 'hipporag_engine.py')

_load('config', BRIDGE_DIR / 'config.py')
bridge_call_status = _load('bridge_call_status', BRIDGE_DIR / 'call_status.py')


class TestStreamingPIIUnmasker:
    def test_split_token_restored_across_three_chunks(self):
        unmasker = pii_vault.StreamingPIIUnmasker({'[CARD_1]': '4276 1111 2222 3333'})
        out = unmasker.feed('Перевод на карту [')
        out += unmasker.feed('CARD_')
        out += unmasker.feed('1] выполнен')
        out += unmasker.flush()
        assert out == 'Перевод на карту 4276 1111 2222 3333 выполнен'
        assert '[CARD_1]' not in out

    def test_two_tokens_in_one_chunk(self):
        unmasker = pii_vault.StreamingPIIUnmasker({'[PHONE_1]': '+7 913', '[PHONE_2]': '+7 914'})
        out = unmasker.feed('[PHONE_1] или [PHONE_2]?') + unmasker.flush()
        assert out == '+7 913 или +7 914?'

    def test_open_bracket_beyond_max_len_is_emitted(self):
        unmasker = pii_vault.StreamingPIIUnmasker({'[CARD_1]': 'x'}, max_tag_len=6)
        out = unmasker.feed('[очень длинный текст без тега')
        assert '[очень' in out  # буфер не удерживает вечно

    def test_no_mapping_passthrough(self):
        unmasker = pii_vault.StreamingPIIUnmasker({})
        assert unmasker.feed('как есть [CARD_1]') == 'как есть [CARD_1]'

    def test_regular_brackets_not_held_forever(self):
        unmasker = pii_vault.StreamingPIIUnmasker({'[CARD_1]': 'x'})
        out = unmasker.feed('см. [сноску 1] в тексте')
        assert out == 'см. [сноску 1] в тексте'


class TestSlotFillingHardening:
    def test_stop_words_cancel_form(self):
        for phrase in ('Отмена', 'стоп', 'забудь про возврат', 'соедини с оператором', 'Передумал'):
            assert route_templates.is_slot_cancellation(phrase), phrase

    def test_normal_input_is_not_cancellation(self):
        assert not route_templates.is_slot_cancellation('номер заказа 94821')
        assert not route_templates.is_slot_cancellation('верну товар в пункт выдачи')

    def test_slot_value_detection(self):
        assert route_templates.mentions_slot_value('заказ 94821', [r'\d{5,8}'])
        assert not route_templates.mentions_slot_value('хочу вернуть', [r'\d{5,8}'])

    def test_retry_limit(self):
        assert route_templates.slot_retry_exceeded(3)
        assert route_templates.slot_retry_exceeded(4)
        assert not route_templates.slot_retry_exceeded(2)


class TestCallStatus:
    def test_mark_and_get(self):
        call_status.mark_transfer('uuid-1', 'vip_support')
        assert call_status.get_transfer('uuid-1') == 'vip_support'

    def test_unknown_uuid(self):
        assert call_status.get_transfer('no-such') is None

    def test_expired_entry_cleared(self):
        call_status.mark_transfer('uuid-exp', 'q')
        # Искусственно истекаем запись
        token, _ = call_status._transfers['uuid-exp']
        call_status._transfers['uuid-exp'] = (token, time.monotonic() - 1)
        assert call_status.get_transfer('uuid-exp') is None
        assert 'uuid-exp' not in call_status._transfers


class TestBridgeCallStatus:
    @pytest.mark.asyncio
    async def test_memory_fallback_without_redis(self):
        await bridge_call_status.mark_transfer('call-9', 'vip_support')
        assert await bridge_call_status.get_transfer('call-9') == 'vip_support'
        assert await bridge_call_status.get_transfer('call-404') is None


class TestAtomicGraphSwap:
    def test_swap_over_existing_directory(self, tmp_path):
        base = tmp_path / 'hipporag_storage'
        target = base / 'kb-1'
        target.mkdir(parents=True)
        (target / 'old_graph.json').write_text('old', encoding='utf-8')

        tmp_dir = base / 'kb-1_tmp'
        tmp_dir.mkdir()
        (tmp_dir / 'new_graph.json').write_text('new', encoding='utf-8')

        version = hipporag_engine.atomic_swap_graph_directory(str(base), 'kb-1', str(tmp_dir))

        assert version
        assert (target / 'version.txt').read_text(encoding='utf-8') == version
        assert (target / 'new_graph.json').read_text(encoding='utf-8') == 'new'
        # Старая версия не осталась на диске
        assert not list(base.glob('kb-1_old_*'))
        assert not tmp_dir.exists()

    def test_swap_when_target_missing(self, tmp_path):
        base = tmp_path / 'storage2'
        base.mkdir()
        tmp_dir = base / 'kb-2_tmp'
        tmp_dir.mkdir()
        (tmp_dir / 'graph.json').write_text('g', encoding='utf-8')

        version = hipporag_engine.atomic_swap_graph_directory(str(base), 'kb-2', str(tmp_dir))
        assert version
        assert (base / 'kb-2' / 'graph.json').exists()

    def test_graph_version_missing_file(self, tmp_path):
        assert hipporag_engine._graph_version(str(tmp_path / 'absent')) is None

    def test_graph_version_rotation_changes_key(self, tmp_path):
        base = tmp_path / 'st3'
        base.mkdir()
        tmp1 = base / 'kb_tmp1'
        tmp1.mkdir()
        (tmp1 / 'a.json').write_text('a', encoding='utf-8')
        v1 = hipporag_engine.atomic_swap_graph_directory(str(base), 'kb', str(tmp1))

        tmp2 = base / 'kb_tmp2'
        tmp2.mkdir()
        (tmp2 / 'b.json').write_text('b', encoding='utf-8')
        v2 = hipporag_engine.atomic_swap_graph_directory(str(base), 'kb', str(tmp2))

        assert v1 != v2
