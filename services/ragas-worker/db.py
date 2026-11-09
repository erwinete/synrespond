"""Запись результатов оценки в таблицы ragas_run / ragas_sample.

Схема таблиц совпадает с миграцией e7a1f4c2b9d8 форка Open WebUI.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from sqlalchemy import BigInteger, Column, JSON, MetaData, String, Table, Text, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import DATABASE_URL

metadata = MetaData()

ragas_run = Table(
    'ragas_run',
    metadata,
    Column('id', String, primary_key=True),
    Column('name', Text, nullable=True),
    Column('mode', String, nullable=False),
    Column('dataset_id', String, nullable=True),
    Column('bot_model_id', String, nullable=True),
    Column('status', String, nullable=False),
    Column('metrics', JSON, nullable=True),
    Column('config', JSON, nullable=True),
    Column('sample_count', BigInteger, nullable=False),
    Column('error', Text, nullable=True),
    Column('created_at', BigInteger, nullable=False),
    Column('completed_at', BigInteger, nullable=True),
)

ragas_sample = Table(
    'ragas_sample',
    metadata,
    Column('id', String, primary_key=True),
    Column('run_id', String, nullable=False),
    Column('question', Text, nullable=False),
    Column('rewritten_query', Text, nullable=True),
    Column('answer', Text, nullable=True),
    Column('contexts', JSON, nullable=True),
    Column('ground_truth', Text, nullable=True),
    Column('scores', JSON, nullable=True),
    Column('claims', JSON, nullable=True),
    Column('retrieval_trace', JSON, nullable=True),
    Column('flag', Text, nullable=True),
    Column('bot_model_id', String, nullable=True),
    Column('chat_id', String, nullable=True),
    Column('message_id', String, nullable=True),
    Column('created_at', BigInteger, nullable=False),
)

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def init_db_if_sqlite() -> None:
    """Д4 (Stage 5): для локального sqlite создаем таблицы при старте.
    На Postgres схему управляет миграция ядра Open WebUI."""
    if not DATABASE_URL.startswith('sqlite'):
        return
    from sqlalchemy.ext.asyncio import create_async_engine as _ce  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


async def ensure_live_run(session: AsyncSession, run_id: str, bot_model_id: Optional[str]) -> None:
    """Создает суточный live-ран, если его еще нет."""
    existing = await session.execute(select(ragas_run.c.id).where(ragas_run.c.id == run_id))
    if existing.first() is None:
        await session.execute(
            insert(ragas_run).values(
                id=run_id,
                name=f'Live production sampling {run_id}',
                mode='live',
                bot_model_id=bot_model_id,
                status='running',
                sample_count=0,
                created_at=int(time.time()),
            )
        )
        await session.commit()


async def insert_sample(payload: dict[str, Any]) -> None:
    """Пишет один оцененный диалог и инкрементирует счетчик запуска."""
    async with SessionFactory() as session:
        await session.execute(
            insert(ragas_sample).values(
                id=str(uuid.uuid4()),
                created_at=int(time.time()),
                **{k: v for k, v in payload.items() if k in ragas_sample.c.keys()},
            )
        )
        if payload.get('run_id'):
            run = await session.execute(
                select(ragas_run.c.sample_count).where(ragas_run.c.id == payload['run_id'])
            )
            current = (run.scalar() or 0) + 1
            await session.execute(
                update(ragas_run).where(ragas_run.c.id == payload['run_id']).values(
                    sample_count=current, status='running'
                )
            )
        await session.commit()


async def finalize_run(run_id: str, metrics: dict[str, Any], sample_count: int, error: Optional[str] = None) -> None:
    """Обновляет сводные метрики и статус запуска."""
    async with SessionFactory() as session:
        await session.execute(
            update(ragas_run)
            .where(ragas_run.c.id == run_id)
            .values(
                status='failed' if error else 'completed',
                metrics=metrics or None,
                error=error,
                sample_count=sample_count,
                completed_at=int(time.time()),
            )
        )
        await session.commit()


async def set_run_status(run_id: str, status: str) -> None:
    async with SessionFactory() as session:
        await session.execute(update(ragas_run).where(ragas_run.c.id == run_id).values(status=status))
        await session.commit()
