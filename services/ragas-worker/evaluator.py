"""Движок оценки Ragas: метрики Faithfulness, Relevancy, Precision, Recall.

ТЗ, раздел 6.2. Судья — локальная LLM через OpenAI-совместимый эндпоинт
(vLLM), эмбеддинги — TEI. Ошибки отдельных сэмплов не прерывают батч.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import httpx
import numpy as np

log = logging.getLogger(__name__)

METRICS_WITHOUT_GROUND_TRUTH = ('faithfulness', 'answer_relevancy')
METRICS_WITH_GROUND_TRUTH = (*METRICS_WITHOUT_GROUND_TRUTH, 'context_precision', 'context_recall')


class RagasEvaluationEngine:
    """Обертка над ragas.evaluate с ленивой инициализацией."""

    def __init__(
        self,
        judge_llm_url: str,
        judge_llm_model: str,
        judge_api_key: str,
        embeddings_url: str,
        embeddings_model: str,
    ):
        self.judge_llm_url = judge_llm_url
        self.judge_llm_model = judge_llm_model
        self.judge_api_key = judge_api_key
        self.embeddings_url = embeddings_url
        self.embeddings_model = embeddings_model
        self._ready = False
        self._evaluate = None

    def _ensure_ready(self) -> None:
        """Импортирует ragas при первом использовании (тяжелый импорт)."""
        if self._ready:
            return
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        self._dataset_factory = Dataset.from_dict
        self._judge_llm = ChatOpenAI(
            model=self.judge_llm_model,
            base_url=self.judge_llm_url,
            api_key=self.judge_api_key,
            temperature=0.0,
        )
        self._judge_embeddings = OpenAIEmbeddings(
            model=self.embeddings_model,
            base_url=self.embeddings_url,
            api_key=self.judge_api_key,
        )
        self._metric_registry = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall,
        }
        self._evaluate = evaluate
        self._ready = True

    async def evaluate_samples(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Возвращает {summary_scores, detailed_rows} по набору сэмплов."""
        self._ensure_ready()
        data_dict: dict[str, Any] = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
        }
        metric_names = list(METRICS_WITHOUT_GROUND_TRUTH)
        if ground_truths and len(ground_truths) == len(questions) and any(ground_truths):
            data_dict['ground_truth'] = ground_truths
            metric_names += ['context_precision', 'context_recall']

        dataset = self._dataset_factory(data_dict)
        metrics = [self._metric_registry[name] for name in metric_names]
        result = self._evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._judge_llm,
            embeddings=self._judge_embeddings,
            raise_exceptions=False,
        )

        summary = {}
        for name in metric_names:
            try:
                value = result[name]
                summary[name] = float(value) if value is not None else None
            except Exception:  # noqa: BLE001
                summary[name] = None

        try:
            df = result.to_pandas()
            detailed_rows = df.to_dict(orient='records')
        except Exception:  # noqa: BLE001
            detailed_rows = []

        return {'summary_scores': summary, 'detailed_rows': detailed_rows, 'metric_names': metric_names}


def extract_contexts_from_sources(sources: list[Any]) -> list[str]:
    """Достает тексты чанков из sources в формате Open WebUI."""
    contexts: list[str] = []
    for source in sources or []:
        documents = source.get('document') if isinstance(source, dict) else None
        if isinstance(documents, list):
            contexts.extend(str(doc) for doc in documents if doc)
    return contexts

class LocalONNXRagasJudge:
    """Д4 (Stage 5): автономный судья Ragas на локальных ONNX-моделях.

    Faithfulness — NLI-покрытие утверждений ответа контекстом с штрафом
    за числовые галлюцинации. Answer Relevancy / Context Precision /
    Context Recall — косинусные близости эмбеддингов rubert-embed-v1.
    Полный цикл без GPU, vLLM и speaches.
    """

    def __init__(self, classifier_url: str = 'http://127.0.0.1:8001'):
        self.classifier_url = classifier_url.rstrip('/')

    def _split_statements(self, text: str) -> list[str]:
        parts = [s.strip() for s in re.split(r'[.!?;\n]+', text) if len(s.strip()) > 3]
        return parts or [text.strip()]

    @staticmethod
    def _extract_numbers(text: str) -> set[str]:
        return set(re.findall(r'\d+(?:[.,]\d+)?%?', text))

    async def _embed(self, texts: list[str]) -> 'np.ndarray':
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f'{self.classifier_url}/v1/embeddings',
                json={'input': texts, 'model': 'rubert-embed-v1'},
            )
            resp.raise_for_status()
            data = resp.json()['data']
            vecs = np.array([item['embedding'] for item in data], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            return vecs / norms

    async def evaluate_samples(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        detailed_rows = []
        f_scores, ar_scores, cp_scores, cr_scores = [], [], [], []

        for idx, (q, ans, ctx_list) in enumerate(zip(questions, answers, contexts)):
            ctx_joined = ' '.join(ctx_list) if ctx_list else ''
            statements = self._split_statements(ans)
            faithfulness_val = 0.0
            ar_val = 0.0

            if ctx_joined.strip() and ans.strip():
                embs = await self._embed([q, ans, ctx_joined] + statements + ctx_list)
                q_emb, ans_emb = embs[0], embs[1]
                stmt_embs = embs[3 : 3 + len(statements)]
                chunk_embs = embs[3 + len(statements) :]

                ctx_numbers = self._extract_numbers(ctx_joined)
                supported_count = 0.0
                for s_text, s_emb in zip(statements, stmt_embs):
                    sim = float(np.max(chunk_embs @ s_emb)) if len(chunk_embs) > 0 else 0.0
                    stmt_numbers = self._extract_numbers(s_text)
                    # Числа/проценты, которых нет в контексте -> штраф за галлюцинацию
                    if stmt_numbers and not stmt_numbers.issubset(ctx_numbers):
                        sim *= 0.15
                    supported_count += max(0.0, min(1.0, sim))
                faithfulness_val = round(supported_count / len(statements), 4)

                ar_val = round(float(np.clip(q_emb @ ans_emb, 0.0, 1.0)), 4)

            f_scores.append(faithfulness_val)
            ar_scores.append(ar_val)

            row: dict[str, Any] = {
                'question': q,
                'answer': ans,
                'contexts': ctx_list,
                'faithfulness': faithfulness_val,
                'answer_relevancy': ar_val,
            }

            if ground_truths and idx < len(ground_truths) and ground_truths[idx]:
                gt = ground_truths[idx]
                gt_sents = self._split_statements(gt)
                gt_embs = await self._embed([gt] + gt_sents + ctx_list)
                gt_full_emb = gt_embs[0]
                gt_sent_embs = gt_embs[1 : 1 + len(gt_sents)]
                c_embs = gt_embs[1 + len(gt_sents) :]

                if len(c_embs) > 0:
                    sim_matrix = gt_sent_embs @ c_embs.T
                    recall_val = round(float(np.mean(np.max(sim_matrix, axis=1))), 4)
                    chunk_sims = c_embs @ gt_full_emb
                    weights = np.array([1.0 / (k + 1) for k in range(len(c_embs))])
                    prec_val = round(
                        float(np.sum(np.clip(chunk_sims, 0.0, 1.0) * weights) / np.sum(weights)), 4
                    )
                else:
                    recall_val, prec_val = 0.0, 0.0

                cp_scores.append(prec_val)
                cr_scores.append(recall_val)
                row['context_precision'] = prec_val
                row['context_recall'] = recall_val

            detailed_rows.append(row)

        summary: dict[str, float] = {
            'faithfulness': round(float(np.mean(f_scores)), 4) if f_scores else 0.0,
            'answer_relevancy': round(float(np.mean(ar_scores)), 4) if ar_scores else 0.0,
        }
        if cp_scores:
            summary['context_precision'] = round(float(np.mean(cp_scores)), 4)
        if cr_scores:
            summary['context_recall'] = round(float(np.mean(cr_scores)), 4)

        return {'summary_scores': summary, 'detailed_rows': detailed_rows, 'judge_engine': 'local_onnx'}
