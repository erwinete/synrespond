"""Юнит-тесты Stage 5: боевое трехконтурное слияние, веса Auto-Tune,
семантическая прокси, локальный ONNX-судья, sqlite-фоллбэк."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
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


# namespace-пакеты open_webui для импортов внутри модулей форка
_ow = types.ModuleType('open_webui')
_ow.__path__ = [str(FORK)]
_ow_utils = types.ModuleType('open_webui.utils')
_ow_utils.__path__ = [str(FORK / 'utils')]
_ow_retrieval = types.ModuleType('open_webui.retrieval')
_ow_retrieval.__path__ = [str(FORK / 'retrieval')]
sys.modules.setdefault('open_webui', _ow)
sys.modules.setdefault('open_webui.utils', _ow_utils)
sys.modules.setdefault('open_webui.retrieval', _ow_retrieval)

hybrid_fusion = _load('hybrid_fusion_s5', FORK / 'retrieval' / 'hybrid_fusion.py')

sys.path.insert(0, str(WORKER))
_load('config', WORKER / 'config.py')
optimizer = _load('optimizer_s5', WORKER / 'optimizer.py')
evaluator_mod = _load('evaluator_s5', WORKER / 'evaluator.py')
worker_config = sys.modules['config']

# bridge_speech импортирует свой config — мержим атрибуты двух конфигов
# в один модуль 'config' (воркер уже импортирован выше)
merged = types.ModuleType('config')
merged.__dict__.update(worker_config.__dict__)
merged.__file__ = str(BRIDGE / 'config.py')
exec(compile((BRIDGE / 'config.py').read_text(encoding='utf-8'), str(BRIDGE / 'config.py'), 'exec'), merged.__dict__)
sys.modules['config'] = merged
bridge_speech = _load('bridge_speech_s5', BRIDGE / 'speech.py')

import json  # noqa: E402

from langchain_core.documents import Document  # noqa: E402


def _doc(text: str, source: str = 'kb1.txt') -> Document:
    return Document(page_content=text, metadata={'source': source})


# --- Д1: боевое трехконтурное слияние ---


class TestWeightedRRFThreeWay:
    def test_rank_breakdown_present(self):
        fused = hybrid_fusion.weighted_rrf_three_way(
            [_doc('dense doc')], [_doc('bm25 doc')], [_doc('hippo doc')], 0.4, 0.2, 0.4, top_k=3
        )
        assert len(fused) == 3
        for doc, _score in fused:
            meta = doc.metadata
            assert set(meta['rank_breakdown']) == {'dense_rank', 'bm25_rank', 'hipporag_rank'}
            assert abs(sum(meta['applied_weights'].values()) - 1.0) < 1e-6
            assert meta['fusion_score'] > 0

    def test_weights_change_ranking(self):
        # A лидирует в dense, C — в hipporag
        dense_heavy = hybrid_fusion.weighted_rrf_three_way(
            [_doc('A'), _doc('X')], [_doc('B')], [_doc('C')], 0.9, 0.05, 0.05, top_k=3
        )
        hippo_heavy = hybrid_fusion.weighted_rrf_three_way(
            [_doc('A'), _doc('X')], [_doc('B')], [_doc('C')], 0.05, 0.05, 0.9, top_k=3
        )
        assert dense_heavy[0][0].page_content == 'A'
        assert hippo_heavy[0][0].page_content == 'C'

    def test_hippo_inactive_renormalizes(self):
        wd, wb, wh = hybrid_fusion.normalize_three_weights(0.4, 0.2, 0.4, hipporag_active=False)
        assert wh == 0.0
        assert abs(wd + wb - 1.0) < 1e-6

    def test_hippo_inactive_docs_not_merged(self):
        # Контур отключен: пустой список hippo-документов
        fused = hybrid_fusion.weighted_rrf_three_way(
            [_doc('a'), _doc('b')], [], [], 0.4, 0.2, 0.4, top_k=5
        )
        texts = [d.page_content for d, _ in fused]
        assert 'c' not in texts
        assert 'a' in texts and 'b' in texts

    def test_hippo_zero_weight_disables_contour(self):
        # Вес 0 при непустом списке тоже отключает контур
        fused = hybrid_fusion.weighted_rrf_three_way(
            [_doc('a')], [_doc('b')], [_doc('c')], 0.4, 0.2, 0.0, top_k=5
        )
        texts = [d.page_content for d, _ in fused]
        assert 'c' not in texts

    def test_doc_key_stable_by_content_and_source(self):
        assert hybrid_fusion._doc_key(_doc('same text')) == hybrid_fusion._doc_key(_doc('same text'))
        assert hybrid_fusion._doc_key(_doc('same text', 'a')) != hybrid_fusion._doc_key(_doc('same text', 'b'))

    def test_dedup_across_contours(self):
        # Один и тот же чанк найден всеми тремя контурами — в выдаче один экземпляр
        fused = hybrid_fusion.weighted_rrf_three_way(
            [_doc('common', 'kb1.txt')], [_doc('common', 'kb1.txt')], [_doc('common', 'kb1.txt')],
            0.4, 0.2, 0.4, top_k=5,
        )
        assert len(fused) == 1
        meta = fused[0][0].metadata
        assert meta['rank_breakdown'] == {'dense_rank': 1, 'bm25_rank': 1, 'hipporag_rank': 1}
        # Тройное попадание дает максимальный fusion_score
        assert meta['fusion_score'] == pytest.approx((0.4 + 0.2 + 0.4) / (60 + 1), abs=1e-4)

    def test_retrieval_source_dominance(self):
        # Чанк только в BM25-контуре -> метка bm25 (по формуле ТЗ)
        fused = hybrid_fusion.weighted_rrf_three_way(
            [_doc('dense hit', 's')], [_doc('only bm25 hit', 's')], [], 0.4, 0.2, 0.4, top_k=5
        )
        by_text = {d.page_content: d.metadata['retrieval_source'] for d, _ in fused}
        assert by_text['only bm25 hit'] == 'bm25'
        assert by_text['dense hit'] == 'dense'


# --- Д1: извлечение весов из прогона Auto-Tune ---


def _load_ragas_router():
    # routers/ragas.py тянет open_webui — загружаем только чистую функцию
    import re

    src = (FORK / 'routers' / 'ragas.py').read_text(encoding='utf-8')
    start = src.index('def extract_weights_from_run')
    end = src.index('class ApplyWeightsForm')
    code = 'from typing import Any, Optional\n\n' + src[start:end]
    module = types.ModuleType('ragas_weights')
    exec(compile(code, 'ragas_weights', 'exec'), module.__dict__)
    return module


class TestExtractWeightsFromRun:
    def test_stage4_format(self):
        module = _load_ragas_router()
        metrics = {'best_params': {'hybrid_weights': {'dense': 0.5, 'bm25': 0.3, 'hipporag': 0.2}}}
        weights = module.extract_weights_from_run(metrics)
        assert weights == {
            'rag.hybrid_weight_dense': 0.5,
            'rag.hybrid_weight_bm25': 0.3,
            'rag.hybrid_weight_hipporag': 0.2,
        }

    def test_clamps_and_includes_threshold_topk(self):
        module = _load_ragas_router()
        metrics = {
            'best_params': {
                'hybrid_weights': {'dense': 1.7, 'bm25': -0.1, 'hipporag': 0.2},
                'relevance_threshold': 5.0,
                'top_k': 99,
            }
        }
        weights = module.extract_weights_from_run(metrics)
        assert weights['rag.hybrid_weight_dense'] == 1.0
        assert weights['rag.hybrid_weight_bm25'] == 0.0
        assert weights['rag.relevance_threshold'] == 1.0
        assert weights['rag.top_k'] == 20

    def test_missing_params_returns_none(self):
        module = _load_ragas_router()
        assert module.extract_weights_from_run(None) is None
        assert module.extract_weights_from_run({}) is None


# --- Д3: семантическая прокси-метрика ---


class TestSemanticProxy:
    def test_synonyms_rank_high_without_lexical_overlap(self):
        # Эталон и чанк не имеют общих слов, но векторы близки (синонимия)
        base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        synonym = np.array([0.99, 0.05, 0.0], dtype=np.float32)
        synonym /= np.linalg.norm(synonym)
        noise = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        recall, precision = optimizer.semantic_recall_precision(
            base[np.newaxis, :], np.vstack([synonym, noise]), base, tau_rel=0.9
        )
        assert recall > 0.95
        assert precision > 0.4

    def test_unrelated_chunks_score_zero(self):
        gt = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        chunks = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        recall, precision = optimizer.semantic_recall_precision(gt, chunks, gt[0], tau_rel=0.9)
        assert recall == 0.0 and precision == 0.0

    @pytest.mark.asyncio
    async def test_embed_texts_unreachable_returns_none(self):
        worker_config.CLASSIFIER_URL = 'http://127.0.0.1:1'
        result = await optimizer.embed_texts(['текст'])
        assert result is None

    def test_lexical_fallback_still_works(self):
        recall, precision = optimizer.lexical_recall_precision('комиссия ноль процентов', ['комиссия ноль процентов на тарифе'])
        assert recall > 0.9


# --- Д4: локальный ONNX-судья ---


def make_judge(monkeypatch_vectors):
    judge = evaluator_mod.LocalONNXRagasJudge(classifier_url='http://127.0.0.1:1')

    async def fake_embed(texts):
        # Детерминированные векторы: близость определяется совпадением маркера
        vecs = []
        for text in texts:
            marker = 1.0 if 'тариф старт' in text.lower() or 'бесплатны' in text.lower() else 0.0
            vecs.append([marker, 0.2 if marker else 1.0, 0.0])
        arr = np.array(vecs, dtype=np.float32)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    import types

    judge._embed = fake_embed  # type: ignore[method-assign]
    return judge


class TestLocalOnnxJudge:
    @pytest.mark.asyncio
    async def test_numeric_hallucination_detected(self):
        judge = make_judge(None)
        result = await judge.evaluate_samples(
            questions=['Какая комиссия за перевод по СБП?'],
            answers=['Комиссия за перевод по СБП составляет 5.5% от суммы, минимум 500 рублей.'],
            contexts=[['По тарифу Старт переводы по СБП полностью бесплатны (комиссия 0%) до 100000 рублей.']],
        )
        faithfulness = result['summary_scores']['faithfulness']
        assert faithfulness < 0.3, result
        assert result['judge_engine'] == 'local_onnx'

    @pytest.mark.asyncio
    async def test_faithful_answer_scores_high(self):
        judge = make_judge(None)
        result = await judge.evaluate_samples(
            questions=['Какая комиссия за перевод по СБП?'],
            answers=['Переводы по СБП на тарифе Старт полностью бесплатны.'],
            contexts=[['По тарифу Старт переводы по СБП полностью бесплатны (комиссия 0%) до 100000 рублей.']],
        )
        assert result['summary_scores']['faithfulness'] > 0.75

    def test_extract_numbers(self):
        numbers = evaluator_mod.LocalONNXRagasJudge._extract_numbers('комиссия 5.5% и 500 рублей')
        assert {'5.5%', '500'} <= numbers

    def test_split_statements(self):
        judge = evaluator_mod.LocalONNXRagasJudge(classifier_url='http://127.0.0.1:1')
        parts = judge._split_statements('Первое предложение. Второе! Третье?')
        assert len(parts) == 3

    @pytest.mark.asyncio
    async def test_empty_contexts_zero_faithfulness(self):
        judge = evaluator_mod.LocalONNXRagasJudge(classifier_url='http://127.0.0.1:1')
        result = await judge.evaluate_samples(
            questions=['q'], answers=['ответ без контекста'], contexts=[[]]
        )
        assert result['summary_scores']['faithfulness'] == 0.0


# --- Д4: sqlite-фоллбэк и локальный синтез ---


class TestLocalInfra:
    def test_sqlite_url_translation(self, monkeypatch):
        monkeypatch.setenv('DATABASE_URL', 'sqlite:///./ragas_local.db')
        module = _load('worker_config_s5', WORKER / 'config.py')
        assert module.DATABASE_URL.startswith('sqlite+aiosqlite:///')

    def test_fallback_synth_nonempty_pcm(self):
        pcm = bridge_speech.synthesize_pcm_fallback('Здравствуйте, задайте вопрос', 8000)
        assert len(pcm) > 8000 * 2 * 0.2  # минимум 0.2 секунды аудио

    def test_fallback_synth_scales_with_text(self):
        short = bridge_speech.synthesize_pcm_fallback('да', 8000)
        long = bridge_speech.synthesize_pcm_fallback('это длинная фраза с несколькими словами для синтеза', 8000)
        assert len(long) > len(short)

    def test_json_codec_still_importable_for_direct_response(self):
        # Регресс: прямой ответ и стрим-размаскирование живы после изменений
        from pathlib import Path

        pii = _load('pii_vault_s5', FORK / 'utils' / 'pii_vault.py')
        unmasker = pii.StreamingPIIUnmasker({'[CARD_1]': '4276'})
        out = unmasker.feed('[CA') + unmasker.feed('RD_1]') + unmasker.flush()
        assert out == '4276'
