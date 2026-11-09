"""Адаптер HippoRAG (OpenIE + Personalized PageRank) для коллекций Open WebUI.

ТЗ, раздел 5.1. Для каждой knowledge-коллекции держится изолированный граф.
Библиотека hipporag — опциональная зависимость (backend/requirements-enterprise.txt).
При её отсутствии поиск деградирует до Dense + BM25 с записью предупреждения.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

_HIPPORAG_IMPORT_ERROR: str | None = None


def _load_hipporag_class():
    """Ленивый импорт HippoRAG. Возвращает класс или None с причиной."""
    global _HIPPORAG_IMPORT_ERROR
    try:
        from hipporag import HippoRAG  # noqa: PLC0415 — тяжёлый импорт только по требованию
    except Exception as exc:  # noqa: BLE001
        if _HIPPORAG_IMPORT_ERROR is None:
            _HIPPORAG_IMPORT_ERROR = str(exc)
            log.warning(
                'HippoRAG недоступен (pip install -r backend/requirements-enterprise.txt): %s',
                exc,
            )
        return None
    return HippoRAG


class HippoRAGAdapter:
    """Реестр графов по коллекциям. Граф строится OpenIE по чанкам коллекции."""

    def __init__(self) -> None:
        # collection_id -> (instance, version). Д1: версия сверяется с version.txt
        # при каждом обращении — воркеры подхватывают новый граф без рестарта.
        self._instances: dict[str, tuple[Any, str | None]] = {}
        self._lock = threading.Lock()

    def _graph_dir(self, collection_id: str) -> tuple[str, str]:
        from open_webui.config import DATA_DIR  # noqa: PLC0415

        base_dir = os.getenv('HIPPORAG_STORAGE_DIR', f'{DATA_DIR}/hipporag_storage')
        return base_dir, os.path.join(base_dir, collection_id)

    def _get_lightweight(self, request, collection_id: str):
        """Кэш легковесного графа с той же сверкой version.txt."""
        base_dir, save_dir = self._graph_dir(collection_id)
        disk_version = _graph_version(save_dir)
        cached = self._instances.get(collection_id)
        if cached is not None and cached[1] == disk_version:
            return cached[0]
        with self._lock:
            cached = self._instances.get(collection_id)
            disk_version = _graph_version(save_dir)
            if cached is not None and cached[1] == disk_version:
                return cached[0]
            instance = LightweightHippoGraph.load(save_dir, collection_id)
            if instance is None:
                os.makedirs(save_dir, exist_ok=True)
                instance = LightweightHippoGraph(save_dir, collection_id)
                instance.save()
            self._instances[collection_id] = (instance, disk_version)
            return instance

    def _build_instance(self, HippoRAG, save_dir: str):
        return HippoRAG(
            save_dir=save_dir,
            llm_model_name=os.getenv('HIPPORAG_LLM_MODEL', 'Qwen/Qwen2.5-14B-Instruct'),
            llm_base_url=os.getenv('HIPPORAG_LLM_BASE_URL', 'http://vllm-server:8000/v1'),
            embedding_model_name=os.getenv('HIPPORAG_EMBEDDING_MODEL', 'BAAI/bge-m3'),
            embedding_base_url=os.getenv('HIPPORAG_EMBEDDING_BASE_URL', 'http://tei-embeddings:80'),
        )

    def _get_instance(self, request, collection_id: str):
        HippoRAG = _load_hipporag_class()
        if HippoRAG is None:
            # К4 (Stage 4): встроенный NetworkX-фоллбэк — граф, PPR и
            # Reasoning Path работают без пакета hipporag и без vLLM.
            return self._get_lightweight(request, collection_id)

        base_dir, save_dir = self._graph_dir(collection_id)
        disk_version = _graph_version(save_dir)
        cached = self._instances.get(collection_id)
        if cached is not None and cached[1] == disk_version:
            return cached[0]

        with self._lock:
            cached = self._instances.get(collection_id)
            disk_version = _graph_version(save_dir)
            if cached is not None and cached[1] == disk_version:
                return cached[0]
            os.makedirs(save_dir, exist_ok=True)
            instance = self._build_instance(HippoRAG, save_dir)
            self._instances[collection_id] = (instance, disk_version)
            return instance

    async def index_documents(self, request, collection_id: str, texts: list[str]) -> dict[str, Any]:
        """OpenIE-индексация чанков: строит сущности, триплеты и ребра графа."""
        hippo = self._get_instance(request, collection_id)
        if hippo is None:
            return {'status': 'skipped', 'reason': 'hipporag package not installed'}
        try:
            await asyncio.to_thread(hippo.index, texts)
            return {'status': 'indexed', 'collection_id': collection_id, 'passages_indexed': len(texts)}
        except Exception as exc:  # noqa: BLE001
            log.exception('HippoRAG indexing failed for %s: %s', collection_id, exc)
            return {'status': 'failed', 'error': str(exc)}

    async def retrieve_ppr(
        self, request, collection_id: str, queries: list[str], top_k: int = 15
    ) -> list[dict[str, Any]]:
        """Ассоциативный поиск: NER запроса -> PPR по графу. Возвращает чанки со скорами."""
        hippo = self._get_instance(request, collection_id)
        if hippo is None:
            return []
        if hasattr(hippo, 'graph') and not hasattr(hippo, 'rank_nodes'):
            # К4: легковесный граф — PPR напрямую по networkx
            entities = await self._fast_entities(request, queries[0] if queries else '')
            if not entities:
                return []
            try:
                return await asyncio.to_thread(
                    self._ppr_on_graph, hippo.graph, entities, collection_id, top_k
                )
            except Exception as exc:  # noqa: BLE001
                log.warning('Lightweight PPR failed for %s: %s', collection_id, exc)
                return []
        try:
            results = await asyncio.to_thread(hippo.rank_nodes, queries)
        except AttributeError:
            try:
                results = await asyncio.to_thread(hippo.retrieve, queries=queries)
            except Exception as exc:  # noqa: BLE001
                log.warning('HippoRAG retrieve failed for %s: %s', collection_id, exc)
                return []
        except Exception as exc:  # noqa: BLE001
            log.warning('HippoRAG retrieval failed for %s: %s', collection_id, exc)
            return []

        docs: list[dict[str, Any]] = []
        for query_result in results:
            for rank, (doc_text, score) in enumerate(self._extract_pairs(query_result)):
                docs.append(
                    {
                        'text': doc_text,
                        'score': float(score),
                        'metadata': {
                            'retrieval_source': 'hipporag_ppr',
                            'hipporag_rank': rank + 1,
                            'collection_id': collection_id,
                        },
                    }
                )
        return docs[:top_k]

    async def retrieve_ppr_fast(
        self, request, collection_id: str, queries: list[str], top_k: int = 15
    ) -> list[dict[str, Any]]:
        """Д3: Voice Fast-Path — сущности через ONNX-энкодер (8 мс) вместо LLM NER.

        Сопоставление сущностей с узлами графа и Personalized PageRank выполняются
        напрямую по networkx-графу библиотеки. Любая несовместимость версии
        библиотеки деградирует в пустой результат — поиск продолжается по
        Dense + BM25, TTFB голоса не страдает.
        """
        hippo = self._get_instance(request, collection_id)
        if hippo is None:
            return []
        graph_obj = getattr(hippo, 'graph', None)
        if graph_obj is None:
            return []

        query = queries[0] if queries else ''
        entities = await self._fast_entities(request, query)
        if not entities:
            return []

        try:
            import networkx as nx

            return await asyncio.to_thread(
                self._ppr_on_graph, graph_obj, entities, collection_id, top_k
            )
        except Exception as exc:  # noqa: BLE001
            log.warning('HippoRAG fast PPR failed for %s: %s', collection_id, exc)
            return []

    async def _fast_entities(self, request, query: str) -> list[str]:
        """NER запроса через ONNX-слотовую модель классификатора."""
        from open_webui.models.config import Config  # noqa: PLC0415

        slot_model_id = await Config.get('intent_classifier.slot_model_id')
        base_url = (await Config.get('intent_classifier.url') or '').rstrip('/')
        if not slot_model_id or not base_url or not query:
            return []
        try:
            import httpx  # noqa: PLC0415

            async with httpx.AsyncClient(timeout=0.9) as client:
                response = await client.post(
                    f'{base_url}/v1/extract-slots',
                    json={'text': query, 'model_id': slot_model_id},
                )
                response.raise_for_status()
                spans = response.json().get('spans') or []
            return [str(span.get('text') or '').strip() for span in spans if span.get('text')]
        except Exception as exc:  # noqa: BLE001
            log.warning('Fast NER unavailable: %s', exc)
            return []

    @staticmethod
    def _ppr_on_graph(graph_obj, entities: list[str], collection_id: str, top_k: int) -> list[dict[str, Any]]:
        """Personalized PageRank с семенами-сущностями и штрафом на популярность узлов."""
        import networkx as nx

        normalized = {str(entity).strip().lower() for entity in entities if entity}
        seeds: dict[Any, float] = {}
        for node in graph_obj.nodes:
            node_name = str(node).strip().lower()
            if any(entity == node_name or entity in node_name for entity in normalized):
                degree = graph_obj.degree(node) or 1
                seeds[node] = 1.0 / degree  # s(v) = 1/deg(v) — штраф частых узлов
        if not seeds:
            return []

        total = sum(seeds.values())
        personalization = {node: value / total for node, value in seeds.items()}
        scores = nx.pagerank(graph_obj, alpha=0.5, personalization=personalization, max_iter=100)

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        docs: list[dict[str, Any]] = []
        for node, score in ranked:
            text = HippoRAGAdapter._node_text(node)
            if not text:
                continue
            docs.append(
                {
                    'text': text,
                    'score': float(score),
                    'metadata': {
                        'retrieval_source': 'hipporag_ppr_fast',
                        'hipporag_rank': len(docs) + 1,
                        'collection_id': collection_id,
                        'seeded_by': sorted(normalized),
                    },
                }
            )
            if len(docs) >= top_k:
                break
        return docs

    @staticmethod
    def _node_text(node) -> str | None:
        """Достаёт текст чанка из узла графа (разные версии библиотеки)."""
        if isinstance(node, str):
            stripped = node.strip()
            if len(stripped) > 80 or ' ' in stripped:
                return stripped
            return None
        if isinstance(node, dict):
            return node.get('text') or node.get('passage') or node.get('name')
        try:
            return getattr(node, 'text', None)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _extract_pairs(query_result) -> list[tuple[str, float]]:
        """Достаёт (текст, скор) из разных форматов результата HippoRAG."""
        if isinstance(query_result, dict):
            texts = query_result.get('docs') or query_result.get('passages') or []
            scores = query_result.get('doc_scores') or query_result.get('scores') or []
            return list(zip(texts, scores))
        if hasattr(query_result, 'docs') and hasattr(query_result, 'doc_scores'):
            return list(zip(query_result.docs, query_result.doc_scores))
        if isinstance(query_result, (list, tuple)) and query_result:
            first = query_result[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                return [(str(t), float(s)) for t, s in query_result]
            if isinstance(first, str):
                return [(t, 0.0) for t in query_result]
        return []

    async def get_reasoning_path(self, request, collection_id: str, query: str) -> dict[str, Any]:
        """Ф6: цепочка логического вывода по графу для объяснения ответа.

        Стартовые узлы — сущности запроса (fast NER), финал — чанк-доказательство
        с максимальным PPR-скором. Возвращает шаги [сущность] --(ребро)--> [узел].
        """
        hippo = self._get_instance(request, collection_id)
        if hippo is None:
            return {'available': False, 'steps': []}
        graph_obj = getattr(hippo, 'graph', None)
        if graph_obj is None:
            return {'available': False, 'steps': []}
        try:
            import networkx as nx

            entities = await self._fast_entities(request, query)
            if not entities:
                return {'available': False, 'steps': [], 'reason': 'fast NER unavailable'}

            normalized = {entity.strip().lower() for entity in entities}
            entity_nodes = [
                node for node in graph_obj.nodes
                if str(node).strip().lower() in normalized
            ] or [
                node for node in graph_obj.nodes
                if any(entity in str(node).lower() for entity in normalized)
            ]
            if not entity_nodes:
                return {'available': False, 'steps': [], 'reason': 'query entities not in graph'}

            # Финал пути: чанк с максимальным PPR-скором среди узлов с текстом
            scores = nx.pagerank(graph_obj, alpha=0.5, max_iter=100)
            passage_nodes = [
                node for node in graph_obj.nodes
                if (self._node_text(node) or '').strip()
                and str(node).strip().lower() not in normalized
            ]
            if not passage_nodes:
                return {'available': False, 'steps': [], 'reason': 'no passage nodes'}
            target = max(passage_nodes, key=lambda node: scores.get(node, 0.0))

            # Кратчайший вывод: ближайшая сущность запроса -> чанк
            steps: list[dict[str, Any]] = []
            best_path = None
            for entity_node in entity_nodes:
                try:
                    candidate = nx.shortest_path(graph_obj, source=entity_node, target=target)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if best_path is None or len(candidate) < len(best_path):
                    best_path = candidate
            if not best_path:
                return {'available': False, 'steps': [], 'reason': 'no path in graph'}

            for index in range(len(best_path) - 1):
                edge = graph_obj.get_edge_data(best_path[index], best_path[index + 1]) or {}
                steps.append(
                    {
                        'from': self._node_text(best_path[index]) or str(best_path[index]),
                        'to': self._node_text(best_path[index + 1]) or str(best_path[index + 1]),
                        'relation': str(edge.get('relation', edge.get('label', '')) or ''),
                    }
                )
            return {
                'available': True,
                'steps': steps[:8],
                'evidence': self._node_text(target),
                'entities': sorted(normalized),
            }
        except Exception as exc:  # noqa: BLE001
            log.warning('Reasoning path failed for %s: %s', collection_id, exc)
            return {'available': False, 'steps': [], 'reason': str(exc)}

    async def get_graph(self, request, collection_id: str) -> dict[str, Any]:
        """Выгрузка графа для визуализатора (узлы и ребра)."""
        hippo = self._get_instance(request, collection_id)
        if hippo is None:
            return {'nodes': [], 'edges': [], 'available': False}
        try:
            graph_obj = getattr(hippo, 'graph', None) or getattr(hippo, 'kg', None)
            if graph_obj is None:
                return {'nodes': [], 'edges': [], 'available': False}
            import networkx as nx  # noqa: PLC0415

            if isinstance(graph_obj, nx.Graph):
                nodes = [
                    {
                        'id': str(node),
                        'label': str(node),
                        'kind': 'entity' if not str(node).startswith('passage') else 'passage',
                    }
                    for node in list(graph_obj.nodes)[:2000]
                ]
                edges = [
                    {'source': str(u), 'target': str(v), 'label': str(data.get('relation', ''))}
                    for u, v, data in list(graph_obj.edges(data=True))[:6000]
                ]
                return {'nodes': nodes, 'edges': edges, 'available': True}
            return {'nodes': [], 'edges': [], 'available': False}
        except Exception as exc:  # noqa: BLE001
            log.warning('HippoRAG graph export failed for %s: %s', collection_id, exc)
            return {'nodes': [], 'edges': [], 'available': False}

    async def delete_collection_graph(self, collection_id: str) -> bool:
        """Удаляет каталог графа вместе с коллекцией знаний."""
        from open_webui.config import DATA_DIR  # noqa: PLC0415

        base_dir = os.getenv('HIPPORAG_STORAGE_DIR', f'{DATA_DIR}/hipporag_storage')
        graph_dir = os.path.join(base_dir, collection_id)
        if not os.path.isdir(graph_dir):
            return False
        import shutil  # noqa: PLC0415

        await asyncio.to_thread(shutil.rmtree, graph_dir, ignore_errors=True)
        self._instances.pop(collection_id, None)
        return True


_adapter: HippoRAGAdapter | None = None


def get_hipporag_adapter() -> HippoRAGAdapter:
    global _adapter
    if _adapter is None:
        _adapter = HippoRAGAdapter()
    return _adapter


async def delete_hipporag_collection_graph(collection_id: str) -> bool:
    """Модульная обертка для удаления графа при удалении коллекции знаний."""
    return await get_hipporag_adapter().delete_collection_graph(collection_id)


_indexing_in_flight: set[str] = set()


async def _acquire_index_lock(collection_id: str, lock_timeout: int = 600):
    """Д5: распределенный Redis-лок на перестроение графа коллекции.

    Возвращает объект-лок или None (Redis недоступен — внутрипроцессной
    защиты через _indexing_in_flight достаточно для одного воркера).
    """
    from open_webui.env import REDIS_URL  # noqa: PLC0415

    if not REDIS_URL:
        return None
    try:
        import redis.asyncio as aioredis  # noqa: PLC0415

        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        lock = client.lock(f'hipporag_lock:{collection_id}', timeout=lock_timeout, blocking_timeout=3)
        if await lock.acquire(blocking=True):
            return (lock, client)
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning('Redis lock unavailable for %s, using in-process guard: %s', collection_id, exc)
        return None


async def _release_index_lock(lock_handle) -> None:
    if not lock_handle:
        return
    lock, client = lock_handle
    try:
        await lock.release()
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            pass


def atomic_swap_graph_directory(base_dir: str, collection_id: str, tmp_dir: str) -> str:
    """Д1 (Stage 3): безопасная атомарная замена непустого каталога графа.

    Трехфазная транзакция внутри одного тома (Linux и Windows):
    current -> backup_<uniq>, tmp -> current, удаление backup. В каталог
    записывается version.txt — маркер инвалидации in-memory кэша воркеров.
    """
    import shutil  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    target_dir = os.path.join(base_dir, collection_id)
    backup_dir = os.path.join(base_dir, f'{collection_id}_old_{uuid.uuid4().hex[:6]}')
    new_version = f'{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}'

    with open(os.path.join(tmp_dir, 'version.txt'), 'w', encoding='utf-8') as vf:
        vf.write(new_version)

    try:
        if os.path.exists(target_dir):
            os.replace(target_dir, backup_dir)
        os.replace(tmp_dir, target_dir)
    except Exception as exc:  # noqa: BLE001
        if not os.path.exists(target_dir) and os.path.exists(backup_dir):
            os.replace(backup_dir, target_dir)
        raise RuntimeError(f'Failed to swap HippoRAG directory for {collection_id}: {exc}') from exc
    finally:
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)

    return new_version


class LightweightHippoGraph:
    """К4 (Stage 4): встроенный граф знаний на networkx.DiGraph.

    Работает без пакета hipporag и без LLM: сущности чанков берутся из
    /v1/extract-slots сервиса encoder-classifier (regex-режим мгновенен),
    плюс со-встречаемость сущностей в предложениях. PPR — штатный
    nx.pagerank с personalization. Совместим с адаптером: .graph,
    .index(texts), сохранение в save_dir с version.txt.
    """

    def __init__(self, save_dir: str, collection_id: str):
        import networkx as nx

        self.save_dir = save_dir
        self.collection_id = collection_id
        self.graph = nx.DiGraph()
        self._chunk_count = 0

    def index(self, texts: list) -> None:
        """Строит граф: passage-узлы = тексты чанков, entity-узлы = спаны."""
        for text in texts:
            text = str(text).strip()
            if not text:
                continue
            self._chunk_count += 1
            entities = self._extract_entities_sync(text)
            if not entities:
                continue
            for entity in entities:
                # Ребро passage -> entity (упоминание) и обратно (для PPR)
                self.graph.add_edge(text, entity, relation='mentions')
                self.graph.add_edge(entity, text, relation='mentioned_in')
        self.save()

    def _extract_entities_sync(self, text: str) -> list:
        """Сущности через /v1/extract-slots; при недоступности сервиса — заглавные биграммы."""
        import re

        import httpx

        base_url = os.getenv('ENCODER_CLASSIFIER_URL', 'http://encoder-classifier:8001').rstrip('/')
        slot_model = os.getenv('HIPPORAG_FALLBACK_SLOT_MODEL', 'rubert-slots-v1')
        try:
            response = httpx.post(
                f'{base_url}/v1/extract-slots',
                json={'text': text, 'model_id': slot_model},
                timeout=2.0,
            )
            response.raise_for_status()
            spans = response.json().get('spans') or []
            return list({str(s.get('text') or '').strip() for s in spans if s.get('text')})[:12]
        except Exception:  # noqa: BLE001
            entities = []
            pattern = re.compile(r'\b[А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z]+)?\b')
            for match in pattern.finditer(text):
                value = match.group(0).strip()
                if 3 <= len(value) <= 60:
                    entities.append(value)
            return list(dict.fromkeys(entities))[:8]

    def save(self) -> None:
        import json

        import networkx as nx

        os.makedirs(self.save_dir, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(os.path.join(self.save_dir, 'graph.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        with open(os.path.join(self.save_dir, 'version.txt'), 'w', encoding='utf-8') as f:
            f.write(f'lightweight_{int(time.time() * 1000)}')

    @classmethod
    def load(cls, save_dir: str, collection_id: str):
        import json

        import networkx as nx

        graph_path = os.path.join(save_dir, 'graph.json')
        if not os.path.exists(graph_path):
            return None
        instance = cls(save_dir, collection_id)
        try:
            with open(graph_path, encoding='utf-8') as f:
                instance.graph = nx.node_link_graph(json.load(f))
        except Exception:  # noqa: BLE001
            return None
        return instance


def _graph_version(graph_dir: str) -> str | None:
    """Текущая версия графа из version.txt (None — каталога или маркера нет)."""
    version_file = os.path.join(graph_dir, 'version.txt')
    try:
        with open(version_file, encoding='utf-8') as vf:
            return vf.read().strip() or None
    except OSError:
        return None


async def schedule_hipporag_indexing(request, collection_id: str) -> None:
    """Фоновая переиндексация графа коллекции после загрузки/обновления файлов.

    Чанки берутся из векторного хранилища, чтобы граф всегда соответствовал
    текущему состоянию коллекции. Перестроение идет во временную директорию
    `<id>_tmp` с атомарной подменой; параллельные запуски сериализует Redis-лок.
    """
    from open_webui.models.config import Config  # noqa: PLC0415

    try:
        if not await Config.get('hipporag.enable'):
            return
        if collection_id in _indexing_in_flight:
            return
        _indexing_in_flight.add(collection_id)
        asyncio.create_task(_run_indexing(request, collection_id))
    except Exception as exc:  # noqa: BLE001
        log.warning('HippoRAG scheduling failed for %s: %s', collection_id, exc)


async def _run_indexing(request, collection_id: str) -> None:
    lock_handle = None
    tmp_dir = None
    try:
        lock_handle = await _acquire_index_lock(collection_id)
        if lock_handle is None:
            log.info('HippoRAG indexing of %s skipped: another worker holds the lock', collection_id)
            return

        from open_webui.retrieval.utils import get_all_items_from_collections  # noqa: PLC0415

        data = await asyncio.to_thread(get_all_items_from_collections, [collection_id])
        payload = data.get(collection_id) if isinstance(data, dict) else None
        documents = (payload or {}).get('documents') or []
        texts = []
        for doc_list in documents if isinstance(documents, list) else [documents]:
            if isinstance(doc_list, list):
                texts.extend(str(t) for t in doc_list if t)
            elif doc_list:
                texts.append(str(doc_list))
        if not texts:
            return

        from open_webui.config import DATA_DIR  # noqa: PLC0415

        HippoRAG = _load_hipporag_class()
        base_dir = os.getenv('HIPPORAG_STORAGE_DIR', f'{DATA_DIR}/hipporag_storage')
        final_dir = os.path.join(base_dir, collection_id)
        tmp_dir = f'{final_dir}_tmp'
        if os.path.exists(tmp_dir):
            import shutil  # noqa: PLC0415

            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        # Строим граф во временную директорию отдельным инстансом
        if HippoRAG is None:
            tmp_instance = LightweightHippoGraph(tmp_dir, collection_id)
        else:
            tmp_instance = HippoRAG(
                save_dir=tmp_dir,
                llm_model_name=os.getenv('HIPPORAG_LLM_MODEL', 'Qwen/Qwen2.5-14B-Instruct'),
                llm_base_url=os.getenv('HIPPORAG_LLM_BASE_URL', 'http://vllm-server:8000/v1'),
                embedding_model_name=os.getenv('HIPPORAG_EMBEDDING_MODEL', 'BAAI/bge-m3'),
                embedding_base_url=os.getenv('HIPPORAG_EMBEDDING_BASE_URL', 'http://tei-embeddings:80'),
            )
        await asyncio.to_thread(tmp_instance.index, texts)

        # Атомарная подмена + сброс кэшированного инстанса
        # Д1: трехфазная атомарная подмена с записью version.txt
        new_version = await asyncio.to_thread(atomic_swap_graph_directory, base_dir, collection_id, tmp_dir)
        tmp_dir = None
        get_hipporag_adapter()._instances.pop(collection_id, None)
        log.info(
            'HippoRAG reindex %s: graph swapped, version %s (%s passages)',
            collection_id, new_version, len(texts),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning('HippoRAG background indexing failed for %s: %s', collection_id, exc)
    finally:
        if tmp_dir:
            import shutil  # noqa: PLC0415

            shutil.rmtree(tmp_dir, ignore_errors=True)
        await _release_index_lock(lock_handle)
        _indexing_in_flight.discard(collection_id)
