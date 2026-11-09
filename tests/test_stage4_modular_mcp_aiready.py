"""Юнит-тесты Stage 4: К1-К4, канальная модульность, MCP, AI-Ready линтер."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / 'open-webui-fork' / 'backend' / 'open_webui'
BRIDGE = ROOT / 'services' / 'voice-sip-bridge'
WORKER = ROOT / 'services' / 'ragas-worker'


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# optimizer импортирует bot_runner -> config из каталога воркера
sys.path.insert(0, str(WORKER))
_load('config', WORKER / 'config.py')

channel_adapter = _load('channel_adapter', FORK / 'utils' / 'channel_adapter.py')
slots_extractor = _load('slots_extractor', ROOT / 'services' / 'encoder-classifier' / 'slots_extractor.py')
ai_ready = _load('ai_ready_schema', FORK / 'utils' / 'ai_ready_schema.py')
# mcp_harness импортирует open_webui.utils.route_templates — создаем легкие
# namespace-пакеты, чтобы не тянуть полный импорт open_webui
import types

_ow = types.ModuleType('open_webui')
_ow.__path__ = [str(FORK)]
_ow_utils = types.ModuleType('open_webui.utils')
_ow_utils.__path__ = [str(FORK / 'utils')]
sys.modules['open_webui'] = _ow
sys.modules['open_webui.utils'] = _ow_utils
mcp_harness = _load('mcp_harness', FORK / 'utils' / 'mcp_harness.py')
optimizer = _load('optimizer', WORKER / 'optimizer.py')
hipporag_engine = _load('hipporag_engine_s4', FORK / 'retrieval' / 'hipporag_engine.py')
pii_vault = _load('pii_vault_s4', FORK / 'utils' / 'pii_vault.py')


# --- К1: три веса и раздельные пулы ---


class TestK1ThreeWeights:
    def test_normalize_three_weights_sums_to_one(self):
        weights = optimizer.normalize_three_weights(0.4, 0.2, 0.4)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert weights == {'dense': 0.4, 'bm25': 0.2, 'hipporag': 0.4}

    def test_normalize_handles_zero_total(self):
        weights = optimizer.normalize_three_weights(0, 0, 0)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_rrf_select_over_three_lists(self):
        pool = []
        for rank, text in enumerate(['dense-a', 'dense-b'], start=1):
            pool.append({'text': text, 'pool_list': 'dense', 'pool_rank': rank})
        for rank, text in enumerate(['bm25-a', 'bm25-c'], start=1):
            pool.append({'text': text, 'pool_list': 'bm25', 'pool_rank': rank})
        for rank, text in enumerate(['hippo-a'], start=1):
            pool.append({'text': text, 'pool_list': 'hipporag', 'pool_rank': rank})
        selected = optimizer.rrf_select(
            pool, {'dense': 0.45, 'bm25': 0.15, 'hipporag': 0.4}, top_k=3
        )
        assert len(selected) == 3
        assert 'dense-a' in selected  # лидер dense-контура
        assert 'hippo-a' in selected  # единственный в своем контуре с весом 0.4

    def test_rrf_three_weights_change_ranking(self):
        pool = [
            {'text': 'A', 'pool_list': 'dense', 'pool_rank': 1},
            {'text': 'B', 'pool_list': 'bm25', 'pool_rank': 1},
            {'text': 'C', 'pool_list': 'hipporag', 'pool_rank': 1},
        ]
        dense_heavy = optimizer.rrf_select(pool, {'dense': 0.9, 'bm25': 0.05, 'hipporag': 0.05}, 3)
        hippo_heavy = optimizer.rrf_select(pool, {'dense': 0.05, 'bm25': 0.05, 'hipporag': 0.9}, 3)
        assert dense_heavy[0] == 'A'
        assert hippo_heavy[0] == 'C'


# --- К2: гибридный экстрактор слотов ---


class TestHybridSlotExtractor:
    def test_order_id_pattern(self):
        spans = slots_extractor.extract_pattern_spans('Проверьте заказ № 94821')
        labels = {s['label'] for s in spans}
        assert 'ORDER_ID' in labels
        assert any(s['text'] == '94821' for s in spans)

    def test_phone_and_email(self):
        text = 'Позвоните +7 913 123-45-67 или mail@example.com'
        spans = slots_extractor.extract_pattern_spans(text)
        labels = {s['label'] for s in spans}
        assert 'PHONE' in labels and 'EMAIL' in labels

    def test_amount_pattern(self):
        spans = slots_extractor.extract_pattern_spans('Сумма 4 500 руб.')
        assert any(s['label'] == 'AMOUNT' for s in spans)

    def test_merge_prefers_onnx_on_overlap(self):
        onnx = [{'text': '94821', 'label': 'ORDER_ID', 'start': 16, 'end': 21, 'confidence': 0.9}]
        regex = [{'text': '94821', 'label': 'ORDER_ID', 'start': 16, 'end': 21, 'confidence': 1.0, 'source': 'regex'}]
        merged = slots_extractor.merge_spans(onnx, regex)
        assert len(merged) == 1 and merged[0]['source'] == 'onnx'

    def test_merge_fills_gaps_with_regex(self):
        onnx = [{'text': 'Иванов Иван', 'label': 'PERSON', 'start': 25, 'end': 36, 'confidence': 0.9}]
        regex = slots_extractor.extract_pattern_spans('Заказ № 94821 на имя Иванов Иван')
        merged = slots_extractor.merge_spans(onnx, regex)
        labels = {s['label'] for s in merged}
        assert 'PERSON' in labels and 'ORDER_ID' in labels


# --- К4: NetworkX-фоллбэк графа ---


class TestLightweightGraph:
    def test_fallback_class_exists_and_indexes(self, tmp_path):
        graph = hipporag_engine.LightweightHippoGraph(str(tmp_path / 'kb'), 'kb')
        # Сервис недоступен на порту 1 -> fallback-сущности по заглавным словам
        hipporag_engine.os.environ['ENCODER_CLASSIFIER_URL'] = 'http://127.0.0.1:1'
        graph.index(['Генеральным директором проекта Орион назначен Алексей Смирнов.'])
        assert len(graph.graph.nodes) > 0

    def test_save_and_load_roundtrip(self, tmp_path):
        save_dir = str(tmp_path / 'kb2')
        graph = hipporag_engine.LightweightHippoGraph(save_dir, 'kb2')
        graph.graph.add_edge('passage text here', 'Entity', relation='mentions')
        graph.save()
        loaded = hipporag_engine.LightweightHippoGraph.load(save_dir, 'kb2')
        assert loaded is not None
        assert 'Entity' in loaded.graph.nodes

    def test_version_file_written(self, tmp_path):
        save_dir = str(tmp_path / 'kb3')
        graph = hipporag_engine.LightweightHippoGraph(save_dir, 'kb3')
        graph.save()
        assert hipporag_engine._graph_version(save_dir) is not None

    def test_ppr_on_graph_ranks_connected_passages(self, tmp_path):
        import networkx as nx

        passage_a = 'Документ про проект Орион и директора Алексея Смирнова целиком тут.'
        passage_b = 'Лаборатория квантовой криптографии подчиняется регламенту номер сорок.'
        graph_obj = nx.DiGraph()
        graph_obj.add_edge(passage_a, 'Орион', relation='mentions')
        graph_obj.add_edge('Орион', 'Алексей Смирнов', relation='related')
        graph_obj.add_edge('Алексей Смирнов', passage_b, relation='mentioned_in')

        docs = hipporag_engine.HippoRAGAdapter._ppr_on_graph(graph_obj, ['Орион'], 'kb', 5)
        texts = [d['text'] for d in docs]
        assert passage_a in texts  # стартовый чанк найден
        assert docs[0]['metadata']['retrieval_source'] == 'hipporag_ppr_fast'


# --- Stage 4: канальная модульность ---


class TestChannelAdapter:
    def test_detect_channel_from_header(self):
        class FakeRequest:
            headers = {'x-channel-type': 'sip'}

        assert channel_adapter.detect_channel(FakeRequest()) == 'sip'

    def test_detect_channel_from_call_uuid(self):
        class FakeRequest:
            headers = {'x-call-uuid': 'abc'}

        assert channel_adapter.detect_channel(FakeRequest()) == 'sip'

    def test_detect_channel_default_chat(self):
        class FakeRequest:
            headers = {}

        assert channel_adapter.detect_channel(FakeRequest()) == 'chat'

    def test_filter_routes_by_channel(self):
        class Route:
            def __init__(self, allowed):
                self.allowed_channels = allowed

        routes = [Route(['chat']), Route(['webrtc', 'sip']), Route(None)]
        chat_routes = channel_adapter.filter_routes_by_channel(routes, 'chat')
        sip_routes = channel_adapter.filter_routes_by_channel(routes, 'sip')
        assert len(chat_routes) == 2  # chat-правило + правило без ограничений
        assert len(sip_routes) == 2

    def test_voice_sanitizer_strips_markdown(self):
        markdown = '**Итого:**\n### Счет\n| A | B |\n|---|---|\n| 1 | 2 |\n```code```\n[Скачать](http://x)'
        plain = channel_adapter.sanitize_for_voice_tts(markdown, 'sip')
        assert '**' not in plain and '###' not in plain and '```' not in plain
        assert '](' not in plain
        # Для чата текст не трогаем
        assert channel_adapter.sanitize_for_voice_tts(markdown, 'chat') == markdown


# --- Stage 4: AI-Ready линтер ---


class TestManifestLinter:
    def _manifest(self, **route_overrides):
        route = {
            'intent_name': 'get_invoice',
            'action_type': 'DIRECT_RESPONSE',
            'min_confidence': 0.85,
            'tau_low': 0.5,
            'direct_response': 'Счет на {{slots.amount}} отправлен',
            'required_slots': [{'name': 'amount', 'pattern': r'\d+', 'question': 'Какая сумма?'}],
            **route_overrides,
        }
        return {'bot': {'model_id': 'support-bot', 'channels': ['chat']}, 'intent_routes': [route]}

    def test_valid_manifest_passes(self):
        result = ai_ready.validate_bot_manifest(self._manifest())
        assert result['valid'], result['errors']

    def test_undeclared_slot_detected(self):
        manifest = self._manifest(direct_response='Счет {{slots.total}} готов')
        result = ai_ready.validate_bot_manifest(manifest)
        codes = [e['code'] for e in result['errors']]
        assert 'UNDECLARED_SLOT_IN_TEMPLATE' in codes
        hint = next(e for e in result['errors'] if e['code'] == 'UNDECLARED_SLOT_IN_TEMPLATE')
        assert 'required_slots' in hint['agent_remediation_hint']

    def test_inverted_thresholds_detected(self):
        manifest = self._manifest(min_confidence=0.3, tau_low=0.9)
        result = ai_ready.validate_bot_manifest(manifest)
        assert any(e['code'] == 'INVALID_THRESHOLDS' for e in result['errors'])

    def test_invalid_regex_detected(self):
        manifest = self._manifest(
            required_slots=[{'name': 'amount', 'pattern': r'\d(+', 'question': 'q'}]
        )
        result = ai_ready.validate_bot_manifest(manifest)
        assert any(e['code'] == 'INVALID_SLOT_REGEX' for e in result['errors'])

    def test_intent_missing_in_encoder_warns(self):
        result = ai_ready.validate_bot_manifest(
            self._manifest(), {'encoder_labels': {'other_intent': True}}
        )
        codes = [w['code'] for w in result['warnings']]
        assert 'INTENT_MISSING_IN_ENCODER' in codes

    def test_voice_unfriendly_template_warns(self):
        route = {
            'intent_name': 'big_report',
            'action_type': 'DIRECT_RESPONSE',
            'allowed_channels': ['sip'],
            'direct_response': '| кол1 | кол2 |\n|---|---|\n| a | b |',
        }
        result = ai_ready.validate_bot_manifest(
            {'bot': {'model_id': 'b', 'channels': ['sip']}, 'intent_routes': [route]}
        )
        codes = [w['code'] for w in result['warnings']]
        assert 'VOICE_UNFRIENDLY_TEMPLATE' in codes

    def test_unknown_harness_detected(self):
        manifest = self._manifest(
            mcp_harness_config={'harness_id': 'no-such', 'tool_name': 'x'}
        )
        result = ai_ready.validate_bot_manifest(manifest)
        assert any(e['code'] == 'UNKNOWN_MCP_HARNESS' for e in result['errors'])

    def test_unknown_tool_in_known_harness(self):
        manifest = self._manifest(
            mcp_harness_config={'harness_id': 'erp', 'tool_name': 'missing_tool'}
        )
        context = {'harness_tools': {'erp': ['real_tool']}}
        result = ai_ready.validate_bot_manifest(manifest, context)
        assert any(e['code'] == 'UNKNOWN_MCP_TOOL' for e in result['errors'])

    def test_descriptors_cover_core_settings(self):
        keys = {d['key'] for d in ai_ready.AI_PARAMETER_DESCRIPTORS}
        assert {
            'intent_classifier.tau_high',
            'hipporag.enable',
            'ragas.live_sample_rate',
            'channels.sip.enabled',
            'guardrails.pii_masking',
        } <= keys
        # У каждого дескриптора есть смысл для агента и влияние настройки
        assert all(d['purpose_for_ai'] and d['tuning_impact'] for d in ai_ready.AI_PARAMETER_DESCRIPTORS)


# --- Stage 4: MCP-клиент харнесов ---


class TestMcpHarnessClient:
    def test_render_arguments_substitutes_slots_and_channel(self):
        rendered = mcp_harness.render_arguments(
            {
                'order_number': '{{slots.order_id}}',
                'reason': '{{slots.reason}}',
                'channel': '{{channel.type}}',
                'message': '{{user_message}}',
            },
            {
                'slots.order_id': '94821',
                'slots.reason': 'брак',
                'channel.type': 'sip',
                'user_message': 'вернуть товар',
            },
        )
        assert rendered == {
            'order_number': 94821,
            'reason': 'брак',
            'channel': 'sip',
            'message': 'вернуть товар',
        }

    def test_content_to_text_extracts_text_blocks(self):
        content = [
            {'type': 'text', 'text': 'Заявка создана'},
            {'type': 'image', 'url': 'x'},
            {'type': 'text', 'text': 'Номер 777'},
        ]
        assert mcp_harness.content_to_text(content) == 'Заявка создана\nНомер 777'

    @pytest.mark.asyncio
    async def test_call_tool_stdio_transport(self):
        import types

        import json as jsonlib

        harness = types.SimpleNamespace(
            transport='stdio',
            endpoint_or_cmd=(
                f'"{sys.executable}" -c "import json,sys;'
                "print(json.dumps({'jsonrpc':'2.0','id':1,'result':"
                "{'content':[{'type':'text','text':'order cancelled'}],'isError':False}}))\""
            ),
            auth_token=None,
            timeout_ms=10000,
        )
        result = await mcp_harness.call_tool(harness, 'cancel_order', {'id': 94821})
        assert result['ok'] is True
        assert 'order cancelled' in mcp_harness.content_to_text(result.get('content'))

    def test_streaming_pii_unmasker_is_reusable(self):
        # Регресс: Stage 3 модуль продолжает работать после изменений Stage 4
        unmasker = pii_vault.StreamingPIIUnmasker({'[CARD_1]': '4276'})
        out = unmasker.feed('[CAR') + unmasker.feed('D_1]') + unmasker.flush()
        assert out == '4276'
