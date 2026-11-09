"""Модели оценки качества RAG по фреймворку Ragas (ТЗ, раздел 6)."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from open_webui.internal.db import Base, get_async_db_context
from pydantic import BaseModel, ConfigDict
from sqlalchemy import JSON, BigInteger, Column, Float, Index, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession


class RagasDataset(Base):
    """Эталонный датасет (Golden Dataset): вопросы и эталонные ответы."""

    __tablename__ = 'ragas_dataset'

    id = Column(String, primary_key=True, unique=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    items = Column(JSON, nullable=False, default=list)  # [{"question", "ground_truth", "meta"}]
    meta = Column(JSON, nullable=True)
    user_id = Column(String, nullable=True)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)


class RagasRun(Base):
    """Запуск оценки: batch-прогон эталона или live-сэмпл продуктивного трафика."""

    __tablename__ = 'ragas_run'
    __table_args__ = (Index('ix_ragas_run_status_created', 'status', 'created_at'),)

    id = Column(String, primary_key=True, unique=True)
    name = Column(Text, nullable=True)
    mode = Column(String, nullable=False, default='batch')  # batch | live
    dataset_id = Column(String, nullable=True)
    bot_model_id = Column(String, nullable=True)
    status = Column(String, nullable=False, default='pending')  # pending|running|completed|failed
    metrics = Column(JSON, nullable=True)  # сводные метрики
    config = Column(JSON, nullable=True)  # веса поиска, версия промпта и т.п.
    sample_count = Column(BigInteger, nullable=False, default=0)
    error = Column(Text, nullable=True)
    created_at = Column(BigInteger, nullable=False)
    completed_at = Column(BigInteger, nullable=True)


class RagasSample(Base):
    """Результат оценки одного диалогового обмена (вопрос/контекст/ответ)."""

    __tablename__ = 'ragas_sample'
    __table_args__ = (Index('ix_ragas_sample_run_created', 'run_id', 'created_at'),)

    id = Column(String, primary_key=True, unique=True)
    run_id = Column(String, nullable=False, index=True)
    question = Column(Text, nullable=False)
    rewritten_query = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    contexts = Column(JSON, nullable=True)  # список чанков
    ground_truth = Column(Text, nullable=True)
    scores = Column(JSON, nullable=True)  # {"faithfulness": .., "answer_relevancy": ..}
    claims = Column(JSON, nullable=True)  # утверждения с подсветкой
    retrieval_trace = Column(JSON, nullable=True)  # откуда пришёл каждый чанк
    flag = Column(Text, nullable=True)  # "hallucination" при Faithfulness < порога
    bot_model_id = Column(String, nullable=True)
    chat_id = Column(String, nullable=True)
    message_id = Column(String, nullable=True)
    created_at = Column(BigInteger, nullable=False)


class RagasDatasetModel(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    items: list[dict[str, Any]] = []
    meta: Optional[dict[str, Any]] = None
    user_id: Optional[str] = None
    created_at: int
    updated_at: int
    model_config = ConfigDict(from_attributes=True)


class RagasRunModel(BaseModel):
    id: str
    name: Optional[str] = None
    mode: str = 'batch'
    dataset_id: Optional[str] = None
    bot_model_id: Optional[str] = None
    status: str = 'pending'
    metrics: Optional[dict[str, Any]] = None
    config: Optional[dict[str, Any]] = None
    sample_count: int = 0
    error: Optional[str] = None
    created_at: int
    completed_at: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)


class RagasSampleModel(BaseModel):
    id: str
    run_id: str
    question: str
    rewritten_query: Optional[str] = None
    answer: Optional[str] = None
    contexts: Optional[list[Any]] = None
    ground_truth: Optional[str] = None
    scores: Optional[dict[str, Any]] = None
    claims: Optional[list[Any]] = None
    retrieval_trace: Optional[list[Any]] = None
    flag: Optional[str] = None
    bot_model_id: Optional[str] = None
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    created_at: int
    model_config = ConfigDict(from_attributes=True)


class RagasDatasetsTable:
    async def get_datasets(self, db: AsyncSession | None = None) -> list[RagasDatasetModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(RagasDataset).order_by(RagasDataset.created_at.desc()))
            return [RagasDatasetModel.model_validate(d) for d in result.scalars().all()]

    async def get_dataset_by_id(self, id: str, db: AsyncSession | None = None) -> Optional[RagasDatasetModel]:
        async with get_async_db_context(db) as db:
            dataset = await db.get(RagasDataset, id)
            return RagasDatasetModel.model_validate(dataset) if dataset else None

    async def insert_new_dataset(self, form: dict[str, Any], db: AsyncSession | None = None) -> RagasDatasetModel:
        async with get_async_db_context(db) as db:
            now = int(time.time())
            record = RagasDataset(
                id=form.get('id') or str(uuid.uuid4()),
                created_at=now,
                updated_at=now,
                **{
                    k: v
                    for k, v in form.items()
                    if k in ('name', 'description', 'items', 'meta', 'user_id')
                },
            )
            db.add(record)
            await db.commit()
            return RagasDatasetModel.model_validate(record)

    async def update_dataset_by_id(
        self, id: str, fields: dict[str, Any], db: AsyncSession | None = None
    ) -> Optional[RagasDatasetModel]:
        async with get_async_db_context(db) as db:
            dataset = await db.get(RagasDataset, id)
            if not dataset:
                return None
            for key, value in fields.items():
                if hasattr(dataset, key) and key not in ('id', 'created_at'):
                    setattr(dataset, key, value)
            dataset.updated_at = int(time.time())
            await db.commit()
            return RagasDatasetModel.model_validate(dataset)

    async def delete_dataset_by_id(self, id: str, db: AsyncSession | None = None) -> bool:
        async with get_async_db_context(db) as db:
            dataset = await db.get(RagasDataset, id)
            if not dataset:
                return False
            await db.delete(dataset)
            await db.commit()
            return True


class RagasRunsTable:
    async def get_runs(self, limit: int = 50, db: AsyncSession | None = None) -> list[RagasRunModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(RagasRun).order_by(RagasRun.created_at.desc()).limit(limit)
            )
            return [RagasRunModel.model_validate(r) for r in result.scalars().all()]

    async def get_run_by_id(self, id: str, db: AsyncSession | None = None) -> Optional[RagasRunModel]:
        async with get_async_db_context(db) as db:
            run = await db.get(RagasRun, id)
            return RagasRunModel.model_validate(run) if run else None

    async def insert_new_run(self, form: dict[str, Any], db: AsyncSession | None = None) -> RagasRunModel:
        async with get_async_db_context(db) as db:
            now = int(time.time())
            record = RagasRun(
                id=form.get('id') or str(uuid.uuid4()),
                created_at=now,
                **{
                    k: v
                    for k, v in form.items()
                    if k in ('name', 'mode', 'dataset_id', 'bot_model_id', 'status', 'metrics', 'config', 'sample_count')
                },
            )
            db.add(record)
            await db.commit()
            return RagasRunModel.model_validate(record)

    async def update_run_by_id(
        self, id: str, fields: dict[str, Any], db: AsyncSession | None = None
    ) -> Optional[RagasRunModel]:
        async with get_async_db_context(db) as db:
            run = await db.get(RagasRun, id)
            if not run:
                return None
            for key, value in fields.items():
                if hasattr(run, key) and key not in ('id', 'created_at'):
                    setattr(run, key, value)
            await db.commit()
            return RagasRunModel.model_validate(run)

    async def delete_run_by_id(self, id: str, db: AsyncSession | None = None) -> bool:
        async with get_async_db_context(db) as db:
            run = await db.get(RagasRun, id)
            if not run:
                return False
            await db.delete(run)
            await db.commit()
            return True


class RagasSamplesTable:
    async def get_samples_by_run_id(
        self, run_id: str, db: AsyncSession | None = None
    ) -> list[RagasSampleModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(RagasSample).filter_by(run_id=run_id).order_by(RagasSample.created_at)
            )
            return [RagasSampleModel.model_validate(s) for s in result.scalars().all()]

    async def get_flagged_samples(
        self, flag: str = 'hallucination', limit: int = 100, db: AsyncSession | None = None
    ) -> list[RagasSampleModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(RagasSample)
                .filter_by(flag=flag)
                .order_by(RagasSample.created_at.desc())
                .limit(limit)
            )
            return [RagasSampleModel.model_validate(s) for s in result.scalars().all()]

    async def insert_new_sample(self, form: dict[str, Any], db: AsyncSession | None = None) -> RagasSampleModel:
        async with get_async_db_context(db) as db:
            record = RagasSample(
                id=form.get('id') or str(uuid.uuid4()),
                created_at=int(time.time()),
                **{
                    k: v
                    for k, v in form.items()
                    if k in (
                        'run_id', 'question', 'rewritten_query', 'answer', 'contexts',
                        'ground_truth', 'scores', 'claims', 'retrieval_trace', 'flag',
                        'bot_model_id', 'chat_id', 'message_id',
                    )
                },
            )
            db.add(record)
            await db.commit()
            return RagasSampleModel.model_validate(record)

    async def delete_samples_by_run_id(self, run_id: str, db: AsyncSession | None = None) -> int:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(RagasSample).filter_by(run_id=run_id))
            samples = result.scalars().all()
            for sample in samples:
                await db.delete(sample)
            await db.commit()
            return len(samples)


RagasDatasets = RagasDatasetsTable()
RagasRuns = RagasRunsTable()
RagasSamples = RagasSamplesTable()
