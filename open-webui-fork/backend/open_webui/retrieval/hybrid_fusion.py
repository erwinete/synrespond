"""Трехконтурный гибридный поиск: Dense + BM25 -> HippoRAG PPR -> Weighted RRF.

ТЗ, раздел 5.3. Базовый гибрид (Dense + BM25 + реранкер) уже реализован в
retrieval.utils.query_collection. Этот модуль добавляет третий контур
(HippoRAG Personalized PageRank) и сливает оба результата по формуле:

    Score(d) = w_vec * 1/(k + r_vec(d)) + w_bm25 * 1/(k + r_bm25(d)) + w_hippo * 1/(k + r_hippo(d))

где r_x(d) — ранг чанка в контуре x, k = 60 — константа сглаживания.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

RRF_K = 60


def weighted_reciprocal_rank_fusion(
    ranked_lists: list[tuple[float, list[Any]]],
    rrf_k: int = RRF_K,
) -> list[tuple[Any, float]]:
    """Взвешенный RRF.

    ranked_lists: список пар (вес контура, упорядоченный список элементов).
    Элементы хэшируются по своему содержимому — так дедуплицируются чанки,
    найденные несколькими контурами. Возвращает [(элемент, скор)], по убыванию.
    """
    scores: dict[Any, float] = {}
    for weight, ranked in ranked_lists:
        if not weight or not ranked:
            continue
        for rank, item in enumerate(ranked, start=1):
            key = _stable_key(item)
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rrf_k + rank))

    # Восстанавливаем элементы по ключам: ключ -> элемент из первого вхождения
    key_to_item: dict[Any, Any] = {}
    for weight, ranked in ranked_lists:
        for item in ranked:
            key = _stable_key(item)
            key_to_item.setdefault(key, item)

    fused = [(key_to_item[key], score) for key, score in scores.items()]
    fused.sort(key=lambda pair: pair[1], reverse=True)
    return fused


def _stable_key(item: Any) -> str:
    if isinstance(item, str):
        return f'text::{item.strip().lower()}'
    if isinstance(item, dict):
        text = item.get('text') or (item.get('metadata') or {}).get('source')
        meta_source = (item.get('metadata') or {}).get('source')
        return f"doc::{str(text or meta_source or item).strip().lower()}"
    return f'obj::{str(item)}'


def _result_to_docs(query_result: dict[str, Any], source_label: str) -> list[dict[str, Any]]:
    """Преобразует результат query_collection в список чанков с меткой источника."""
    docs: list[dict[str, Any]] = []
    if not query_result:
        return docs
    for collection_payload in query_result.values():
        if not isinstance(collection_payload, dict):
            continue
        documents = collection_payload.get('documents') or []
        metadatas = collection_payload.get('metadatas') or []
        distances = collection_payload.get('distances') or []
        for doc_list, meta_list, dist_list in zip(documents, metadatas, distances):
            for idx, text in enumerate(doc_list or []):
                meta = dict(meta_list[idx]) if meta_list and idx < len(meta_list) and meta_list[idx] else {}
                meta.setdefault('retrieval_source', source_label)
                docs.append(
                    {
                        'text': text,
                        'score': float(dist_list[idx]) if dist_list and idx < len(dist_list) else 0.0,
                        'metadata': meta,
                    }
                )
    return docs


def _docs_to_result(docs: list[dict[str, Any]], collection_name: str) -> dict[str, Any]:
    """Собирает результат в форме query_collection: {collection: {ids, documents, ...}}."""
    documents = [d['text'] for d in docs]
    metadatas = [d['metadata'] for d in docs]
    # fused score сохраняем в distances (ранжирование — по убыванию, как в merge_and_sort)
    distances = [d['score'] for d in docs]
    ids = [f'{collection_name}-{i}' for i in range(len(docs))]
    return {
        collection_name: {
            'ids': [ids],
            'documents': [documents],
            'metadatas': [metadatas],
            'distances': [distances],
        }
    }


async def _separate_dense_bm25(
    request,
    collection_name: str,
    queries: list[str],
    embedding_function,
    k: int,
    user=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """К1: два независимых ранжированных списка по одному корпусу.

    BM25 строится локально (rank_bm25), dense — через векторный поиск ядра.
    Дедупликация по тексту чанка; ранки считаются с 1.
    """
    import asyncio

    from rank_bm25 import BM25Okapi
    from langchain_core.documents import Document

    from open_webui.retrieval.utils import (
        BM25Retriever,
        VectorSearchRetriever,
        get_all_items_from_collections,
    )

    data = await asyncio.to_thread(get_all_items_from_collections, [collection_name])
    payload = (data or {}).get(collection_name) or {}
    documents = (payload.get('documents') or [[]])[0]
    metadatas = (payload.get('metadatas') or [[]])[0]
    if not documents:
        return [], []

    docs = [
        Document(page_content=str(text), metadata=metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {})
        for idx, text in enumerate(documents)
    ]
    bm25_retriever = BM25Retriever(
        docs=docs, vectorizer=BM25Okapi([d.page_content.split() for d in docs]), k=min(k, len(docs))
    )
    vector_retriever = VectorSearchRetriever(
        collection_name=collection_name, embedding_function=embedding_function, top_k=min(k, len(docs))
    )

    dense_merged: dict[str, Document] = {}
    bm25_merged: dict[str, Document] = {}
    for query in queries:
        try:
            dense_batch = await vector_retriever.ainvoke(query)
            bm25_batch = await bm25_retriever.ainvoke(query)
        except Exception as exc:  # noqa: BLE001
            log.warning('Separate retrieval failed for %s: %s', collection_name, exc)
            continue
        for doc in dense_batch:
            dense_merged.setdefault(doc.page_content.strip().lower(), doc)
        for doc in bm25_batch:
            bm25_merged.setdefault(doc.page_content.strip().lower(), doc)

    dense_docs = [
        {'text': d.page_content, 'score': 0.0, 'metadata': dict(d.metadata or {})}
        for d in dense_merged.values()
    ]
    bm25_docs = [
        {'text': d.page_content, 'score': 0.0, 'metadata': dict(d.metadata or {})}
        for d in bm25_merged.values()
    ]
    return dense_docs, bm25_docs


def _doc_key(doc) -> str:
    """Стабильный ключ дедупликации чанка по источнику и тексту."""
    import hashlib

    metadata = getattr(doc, 'metadata', None) or {}
    src = str(metadata.get('source') or metadata.get('file_id') or '')
    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
    content_hash = hashlib.md5(content.strip().encode('utf-8')).hexdigest()
    return f'{src}:{content_hash}'


def normalize_three_weights(
    w_dense: float,
    w_bm25: float,
    w_hipporag: float,
    hipporag_active: bool = True,
) -> tuple:
    """Нормализация трех весов RRF к сумме 1.0 с учетом активности контура HippoRAG."""
    wd = max(0.0, float(w_dense))
    wb = max(0.0, float(w_bm25))
    wh = max(0.0, float(w_hipporag)) if hipporag_active else 0.0
    total = wd + wb + wh
    if total <= 1e-9:
        return (0.5, 0.5, 0.0) if not hipporag_active else (0.4, 0.2, 0.4)
    return (round(wd / total, 4), round(wb / total, 4), round(wh / total, 4))


def weighted_rrf_three_way(
    dense_docs: list,
    bm25_docs: list,
    hipporag_docs: list,
    w_dense: float = 0.4,
    w_bm25: float = 0.2,
    w_hipporag: float = 0.4,
    rrf_k: int = 60,
    top_k: int = 10,
) -> list:
    """Д1 (Stage 5): боевое трехконтурное слияние (Dense + BM25 + HippoRAG).

    Единая формула для онлайн-поиска и оптимизатора Auto-Tune. Каждый
    возвращенный Document несет телеметрию: fusion_score, rank_breakdown,
    applied_weights и доминирующий retrieval_source.
    """
    from langchain_core.documents import Document

    hippo_active = len(hipporag_docs) > 0 and w_hipporag > 0.0
    wd, wb, wh = normalize_three_weights(w_dense, w_bm25, w_hipporag, hipporag_active=hippo_active)

    registry = {}
    ranks = {}
    scores = {}

    def _ingest_pool(pool, pool_name: str, weight: float):
        if weight <= 0.0:
            return
        seen_in_pool = set()
        rank_idx = 0
        for doc in pool:
            key = _doc_key(doc)
            if key in seen_in_pool:
                continue
            seen_in_pool.add(key)
            rank_idx += 1
            if key not in registry:
                registry[key] = doc
                ranks[key] = {'dense_rank': None, 'bm25_rank': None, 'hipporag_rank': None}
                scores[key] = 0.0
            ranks[key][f'{pool_name}_rank'] = rank_idx
            scores[key] += weight * (1.0 / (rrf_k + rank_idx))

    _ingest_pool(dense_docs, 'dense', wd)
    _ingest_pool(bm25_docs, 'bm25', wb)
    if hippo_active:
        _ingest_pool(hipporag_docs, 'hipporag', wh)

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
    merged_results = []
    for key in sorted_keys:
        base_doc = registry[key]
        meta = dict(getattr(base_doc, 'metadata', None) or {})
        meta['fusion_score'] = round(scores[key], 6)
        meta['rank_breakdown'] = ranks[key]
        meta['applied_weights'] = {'w_dense': wd, 'w_bm25': wb, 'w_hipporag': wh}
        r_info = ranks[key]
        if r_info['hipporag_rank'] is not None and (
            r_info['dense_rank'] is None or r_info['hipporag_rank'] <= r_info['dense_rank']
        ):
            meta.setdefault('retrieval_source', 'hipporag_ppr')
        elif r_info['bm25_rank'] is not None and r_info['dense_rank'] is None:
            meta.setdefault('retrieval_source', 'bm25')
        else:
            meta.setdefault('retrieval_source', 'dense')
        merged_results.append(
            (Document(page_content=base_doc.page_content, metadata=meta), scores[key])
        )

    return merged_results


def hybrid_weights_return_pool(hybrid_weights: Optional[dict[str, float]]) -> bool:
    return bool(hybrid_weights and hybrid_weights.get('return_pool'))


async def query_collection_with_graph_fusion(
    request,
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
    user=None,
    use_hipporag: bool = False,
    hybrid_weights: Optional[dict[str, float]] = None,
    retrieval_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Базовый гибридный поиск + контур HippoRAG + слияние Weighted RRF.

    При выключенном графе или ошибке HippoRAG возвращает результат
    базового query_collection без изменений — деградация прозрачна.
    """
    from open_webui.models.config import Config  # noqa: PLC0415
    from open_webui.retrieval.utils import query_collection  # noqa: PLC0415

    # Ф4: overrides могут нести и веса, и флаг return_pool одновременно
    if retrieval_overrides and retrieval_overrides.get('hybrid_weights'):
        merged = dict(retrieval_overrides['hybrid_weights'])
        merged['return_pool'] = retrieval_overrides.get('return_pool')
        hybrid_weights = merged

    base_result = await query_collection(
        request,
        collection_names=collection_names,
        queries=queries,
        embedding_function=embedding_function,
        k=k,
        user=user,
    )

    graph_enabled = bool(await Config.get('hipporag.enable')) and use_hipporag
    if not graph_enabled:
        return base_result

    # Д3: Voice Fast-Path — сущности для PPR из ONNX-энкодера вместо LLM
    voice_mode = bool(getattr(request.state, 'voice_mode', False))
    fast_ner = voice_mode and bool(await Config.get('hipporag.voice_fast_ner'))

    try:
        from open_webui.retrieval.hipporag_engine import get_hipporag_adapter  # noqa: PLC0415

        adapter = get_hipporag_adapter()
        weights = hybrid_weights or {}
        if sum(float(weights.get(name, 0.0) or 0.0) for name in ('dense', 'bm25', 'hipporag')) <= 0:
            weights = {
                'dense': await Config.get('rag.hybrid_weight_dense'),
                'bm25': await Config.get('rag.hybrid_weight_bm25'),
                'hipporag': await Config.get('rag.hybrid_weight_hipporag'),
            }

        # Ф4/Д6 (Stage 3): return_pool — сырые ранки двух списков для
        # Zero Re-Retrieval в Auto-Tune (фаза 1 кэширует пулы один раз).
        return_pool = bool(hybrid_weights_return_pool(retrieval_overrides))

        fused_result: dict[str, Any] = {}
        for collection_name in collection_names:
            base_docs = _result_to_docs(
                {collection_name: base_result.get(collection_name, {})}, 'dense+bm25'
            )
            hippo_docs = (
                await adapter.retrieve_ppr_fast(request, collection_name, queries, top_k=max(k, 15))
                if fast_ner
                else await adapter.retrieve_ppr(request, collection_name, queries, top_k=max(k, 15))
            )

            if return_pool:
                # К1 (Stage 4): честные три раздельных списка ранков
                # (dense, bm25, hipporag) — без слитого EnsembleRetriever.
                pool_docs = []
                dense_docs, bm25_docs = await _separate_dense_bm25(
                    request, collection_name, queries, embedding_function, k, user
                )
                for list_name, docs in (
                    ('dense', dense_docs),
                    ('bm25', bm25_docs),
                    ('hipporag', hippo_docs),
                ):
                    for rank, doc in enumerate(docs[:k]):
                        doc['metadata']['pool_list'] = list_name
                        doc['metadata']['pool_rank'] = rank + 1
                        pool_docs.append(doc)
                fused_result.update(_docs_to_result(pool_docs, collection_name))
                continue

            # Д1 (Stage 5): боевой путь — та же формула weighted_rrf_three_way,
            # что и в оптимизаторе Auto-Tune, с полной телеметрией ранков.
            dense_docs, bm25_docs = await _separate_dense_bm25(
                request, collection_name, queries, embedding_function, k, user
            )
            from langchain_core.documents import Document

            dense_lcdrs = [
                Document(page_content=d['text'], metadata=dict(d.get('metadata') or {}))
                for d in dense_docs
            ]
            bm25_lcdrs = [
                Document(page_content=d['text'], metadata=dict(d.get('metadata') or {}))
                for d in bm25_docs
            ]
            hippo_lcdrs = [
                Document(page_content=d['text'], metadata=dict(d.get('metadata') or {}))
                for d in hippo_docs
            ]

            weights = hybrid_weights or {
                'dense': await Config.get('rag.hybrid_weight_dense'),
                'bm25': await Config.get('rag.hybrid_weight_bm25'),
                'hipporag': await Config.get('rag.hybrid_weight_hipporag'),
            }
            fused_lcdrs = weighted_rrf_three_way(
                dense_lcdrs,
                bm25_lcdrs,
                hippo_lcdrs,
                w_dense=float(weights.get('dense', 0.4)),
                w_bm25=float(weights.get('bm25', 0.2)),
                w_hipporag=float(weights.get('hipporag', 0.4)),
                rrf_k=RRF_K,
                top_k=k,
            )
            top_docs = [(doc, doc.metadata.get('fusion_score', 0.0)) for doc, _ in fused_lcdrs]
            fused_result.update(_docs_to_result([d for d, _ in top_docs], collection_name))

        # Коллекции, которых нет в базовом результате, добавляем как есть
        for collection_name, payload in base_result.items():
            if collection_name not in fused_result:
                fused_result[collection_name] = payload
        return fused_result
    except Exception as exc:  # noqa: BLE001 — граф не должен ломать поиск
        log.warning('Graph fusion skipped, falling back to base hybrid: %s', exc)
        return base_result
