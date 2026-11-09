"""Модели маршрутизации интентов: энкодеры-классификаторы и правила (ТЗ, раздел 4.2)."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from open_webui.internal.db import Base, get_async_db_context
from pydantic import BaseModel, ConfigDict
from sqlalchemy import JSON, BigInteger, Boolean, Column, Float, Index, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession


class EncoderModel(Base):
    """Подключенный энкодер-классификатор (ONNX-модель на микросервисе)."""

    __tablename__ = 'encoder_model'
    __table_args__ = (Index('ix_encoder_model_id_active', 'id', 'is_active'),)

    id = Column(String, primary_key=True, unique=True)
    name = Column(Text, nullable=False)
    task = Column(Text, nullable=False, default='sequence_classification')
    endpoint_url = Column(Text, nullable=True)  # URL микросервиса encoder-classifier
    model_path = Column(Text, nullable=True)  # подкаталог в MODELS_DIR
    labels_json = Column(JSON, nullable=True)
    thresholds = Column(JSON, nullable=True)  # {"high": 0.85, "low": 0.5}
    is_active = Column(Boolean, nullable=False, default=True)
    meta = Column(JSON, nullable=True)
    user_id = Column(String, nullable=True)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)


class IntentRoute(Base):
    """Правило маршрутизации: интент -> действие (RAG / прямой ответ / вебхук / промпт)."""

    __tablename__ = 'intent_route'
    __table_args__ = (Index('ix_intent_route_model_intent', 'bot_model_id', 'intent_name'),)

    id = Column(String, primary_key=True, unique=True)
    bot_model_id = Column(String, nullable=False, index=True)  # Workspace Model бота
    intent_name = Column(Text, nullable=False)  # метка класса энкодера
    examples = Column(JSON, nullable=True)  # примеры фраз для UI и дообучения
    min_confidence = Column(Float, nullable=False, default=0.80)  # tau_high
    tau_low = Column(Float, nullable=False, default=0.50)
    action_type = Column(Text, nullable=False, default='RAG_COLLECTION')
    # RAG_COLLECTION | DIRECT_RESPONSE | WEBHOOK_TOOL | SYSTEM_PROMPT_OVERRIDE
    target_collection_ids = Column(JSON, nullable=True)  # список knowledge_id
    use_hipporag = Column(Boolean, nullable=False, default=False)
    hybrid_weights = Column(JSON, nullable=True)  # {"dense": .4, "bm25": .2, "hipporag": .4}
    system_prompt_template = Column(Text, nullable=True)
    direct_response = Column(Text, nullable=True)  # шаблон мгновенного ответа
    webhook_config = Column(JSON, nullable=True)  # {"url", "headers", "body_template"}
    required_slots = Column(JSON, nullable=True)  # [{"name", "pattern", "question"}]
    allowed_channels = Column(JSON, nullable=True)  # ["chat", "webrtc", "sip"]
    mcp_harness_config = Column(JSON, nullable=True)  # {"harness_id", "tool_name", "argument_mapping", ...}
    priority = Column(BigInteger, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)


class EncoderModelModel(BaseModel):
    id: str
    name: str
    task: str = 'sequence_classification'
    endpoint_url: Optional[str] = None
    model_path: Optional[str] = None
    labels_json: Optional[dict[str, Any]] = None
    thresholds: Optional[dict[str, float]] = None
    is_active: bool = True
    meta: Optional[dict[str, Any]] = None
    user_id: Optional[str] = None
    created_at: int
    updated_at: int
    model_config = ConfigDict(from_attributes=True)


class IntentRouteModel(BaseModel):
    id: str
    bot_model_id: str
    intent_name: str
    examples: Optional[list[str]] = None
    min_confidence: float = 0.80
    tau_low: float = 0.50
    action_type: str = 'RAG_COLLECTION'
    target_collection_ids: Optional[list[str]] = None
    use_hipporag: bool = False
    hybrid_weights: Optional[dict[str, float]] = None
    system_prompt_template: Optional[str] = None
    direct_response: Optional[str] = None
    webhook_config: Optional[dict[str, Any]] = None
    required_slots: Optional[list[dict[str, Any]]] = None
    allowed_channels: Optional[list[str]] = None
    mcp_harness_config: Optional[dict[str, Any]] = None
    priority: int = 0
    is_active: bool = True
    meta: Optional[dict[str, Any]] = None
    created_at: int
    updated_at: int
    model_config = ConfigDict(from_attributes=True)


class EncoderModelsTable:
    async def get_available_encoders(self, db: AsyncSession | None = None) -> list[EncoderModelModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(EncoderModel).filter_by(is_active=True))
            return [EncoderModelModel.model_validate(e) for e in result.scalars().all()]

    async def get_encoder_by_id(self, id: str, db: AsyncSession | None = None) -> Optional[EncoderModelModel]:
        async with get_async_db_context(db) as db:
            encoder = await db.get(EncoderModel, id)
            return EncoderModelModel.model_validate(encoder) if encoder else None

    async def insert_new_encoder(self, form: dict[str, Any], db: AsyncSession | None = None) -> EncoderModelModel:
        async with get_async_db_context(db) as db:
            now = int(time.time())
            record = EncoderModel(id=form.get('id') or str(uuid.uuid4()), created_at=now, updated_at=now, **{
                k: v for k, v in form.items() if k in
                ('name', 'task', 'endpoint_url', 'model_path', 'labels_json', 'thresholds', 'is_active', 'meta', 'user_id')
            })
            db.add(record)
            await db.commit()
            return EncoderModelModel.model_validate(record)

    async def update_encoder_by_id(
        self, id: str, fields: dict[str, Any], db: AsyncSession | None = None
    ) -> Optional[EncoderModelModel]:
        async with get_async_db_context(db) as db:
            encoder = await db.get(EncoderModel, id)
            if not encoder:
                return None
            for key, value in fields.items():
                if hasattr(encoder, key) and key not in ('id', 'created_at'):
                    setattr(encoder, key, value)
            encoder.updated_at = int(time.time())
            await db.commit()
            return EncoderModelModel.model_validate(encoder)

    async def delete_encoder_by_id(self, id: str, db: AsyncSession | None = None) -> bool:
        async with get_async_db_context(db) as db:
            encoder = await db.get(EncoderModel, id)
            if not encoder:
                return False
            await db.delete(encoder)
            await db.commit()
            return True


class IntentRoutesTable:
    async def get_routes(self, db: AsyncSession | None = None) -> list[IntentRouteModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(IntentRoute).order_by(IntentRoute.priority.desc(), IntentRoute.intent_name)
            )
            return [IntentRouteModel.model_validate(r) for r in result.scalars().all()]

    async def get_active_routes_by_model(
        self, bot_model_id: str, db: AsyncSession | None = None
    ) -> list[IntentRouteModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(IntentRoute)
                .filter_by(bot_model_id=bot_model_id, is_active=True)
                .order_by(IntentRoute.priority.desc())
            )
            return [IntentRouteModel.model_validate(r) for r in result.scalars().all()]

    async def get_route_by_id(self, id: str, db: AsyncSession | None = None) -> Optional[IntentRouteModel]:
        async with get_async_db_context(db) as db:
            route = await db.get(IntentRoute, id)
            return IntentRouteModel.model_validate(route) if route else None

    async def insert_new_route(self, form: dict[str, Any], db: AsyncSession | None = None) -> IntentRouteModel:
        async with get_async_db_context(db) as db:
            now = int(time.time())
            record = IntentRoute(id=str(uuid.uuid4()), created_at=now, updated_at=now, **{
                k: v for k, v in form.items() if k in
                (
                    'bot_model_id', 'intent_name', 'examples', 'min_confidence', 'tau_low',
                    'action_type', 'target_collection_ids', 'use_hipporag', 'hybrid_weights',
                    'system_prompt_template', 'direct_response', 'webhook_config', 'required_slots',
                    'allowed_channels', 'mcp_harness_config', 'priority', 'is_active', 'meta',
                )
            })
            db.add(record)
            await db.commit()
            return IntentRouteModel.model_validate(record)

    async def update_route_by_id(
        self, id: str, fields: dict[str, Any], db: AsyncSession | None = None
    ) -> Optional[IntentRouteModel]:
        async with get_async_db_context(db) as db:
            route = await db.get(IntentRoute, id)
            if not route:
                return None
            for key, value in fields.items():
                if hasattr(route, key) and key not in ('id', 'created_at'):
                    setattr(route, key, value)
            route.updated_at = int(time.time())
            await db.commit()
            return IntentRouteModel.model_validate(route)

    async def delete_route_by_id(self, id: str, db: AsyncSession | None = None) -> bool:
        async with get_async_db_context(db) as db:
            route = await db.get(IntentRoute, id)
            if not route:
                return False
            await db.delete(route)
            await db.commit()
            return True

    async def delete_routes_by_model(self, bot_model_id: str, db: AsyncSession | None = None) -> int:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(IntentRoute).filter_by(bot_model_id=bot_model_id))
            routes = result.scalars().all()
            for route in routes:
                await db.delete(route)
            await db.commit()
            return len(routes)


class IntentFeedback(Base):
    """Запрос из серой зоны для Active Learning (киллер-фича №1)."""

    __tablename__ = 'intent_feedback'
    __table_args__ = (Index('ix_intent_feedback_status_created', 'status', 'created_at'),)

    id = Column(String, primary_key=True, unique=True)
    bot_model_id = Column(String, nullable=True, index=True)
    text = Column(Text, nullable=False)
    predicted_intent = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    top_candidates = Column(JSON, nullable=True)  # top-k с вероятностями
    hint_intent = Column(Text, nullable=True)  # подсказка LLM-разметчика
    hint_confidence = Column(Float, nullable=True)
    labeled_intent = Column(Text, nullable=True)  # решение аналитика
    status = Column(String, nullable=False, default='pending')  # pending|labeled|trained|dismissed
    source = Column(String, nullable=False, default='gray_zone')  # gray_zone|manual
    meta = Column(JSON, nullable=True)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)


class IntentFeedbackModel(BaseModel):
    id: str
    bot_model_id: Optional[str] = None
    text: str
    predicted_intent: Optional[str] = None
    confidence: Optional[float] = None
    top_candidates: Optional[list[dict[str, Any]]] = None
    hint_intent: Optional[str] = None
    hint_confidence: Optional[float] = None
    labeled_intent: Optional[str] = None
    status: str = 'pending'
    source: str = 'gray_zone'
    meta: Optional[dict[str, Any]] = None
    created_at: int
    updated_at: int
    model_config = ConfigDict(from_attributes=True)


class IntentFeedbackTable:
    async def get_feedback(
        self,
        status: Optional[str] = None,
        bot_model_id: Optional[str] = None,
        limit: int = 200,
        db: AsyncSession | None = None,
    ) -> list[IntentFeedbackModel]:
        async with get_async_db_context(db) as db:
            query = select(IntentFeedback).order_by(IntentFeedback.created_at.desc()).limit(limit)
            if status:
                query = query.filter_by(status=status)
            if bot_model_id:
                query = query.filter_by(bot_model_id=bot_model_id)
            result = await db.execute(query)
            return [IntentFeedbackModel.model_validate(row) for row in result.scalars().all()]

    async def insert_new_feedback(self, form: dict[str, Any], db: AsyncSession | None = None) -> IntentFeedbackModel:
        async with get_async_db_context(db) as db:
            now = int(time.time())
            record = IntentFeedback(
                id=str(uuid.uuid4()),
                created_at=now,
                updated_at=now,
                **{
                    k: v
                    for k, v in form.items()
                    if k in (
                        'bot_model_id', 'text', 'predicted_intent', 'confidence', 'top_candidates',
                        'hint_intent', 'hint_confidence', 'labeled_intent', 'status', 'source', 'meta',
                    )
                },
            )
            db.add(record)
            await db.commit()
            return IntentFeedbackModel.model_validate(record)

    async def update_feedback_by_id(
        self, id: str, fields: dict[str, Any], db: AsyncSession | None = None
    ) -> Optional[IntentFeedbackModel]:
        async with get_async_db_context(db) as db:
            record = await db.get(IntentFeedback, id)
            if not record:
                return None
            for key, value in fields.items():
                if hasattr(record, key) and key not in ('id', 'created_at'):
                    setattr(record, key, value)
            record.updated_at = int(time.time())
            await db.commit()
            return IntentFeedbackModel.model_validate(record)

    async def get_labeled_feedback(
        self, bot_model_id: Optional[str] = None, limit: int = 5000, db: AsyncSession | None = None
    ) -> list[IntentFeedbackModel]:
        """Размеченные примеры для дообучения энкодера."""
        async with get_async_db_context(db) as db:
            query = select(IntentFeedback).filter_by(status='labeled').limit(limit)
            if bot_model_id:
                query = query.filter_by(bot_model_id=bot_model_id)
            result = await db.execute(query)
            return [IntentFeedbackModel.model_validate(row) for row in result.scalars().all()]

    async def delete_feedback_by_id(self, id: str, db: AsyncSession | None = None) -> bool:
        async with get_async_db_context(db) as db:
            record = await db.get(IntentFeedback, id)
            if not record:
                return False
            await db.delete(record)
            await db.commit()
            return True


class MCPHarness(Base):
    """Внешний MCP-харнес: инструменты вызываются из диалога (Stage 4, направление Б)."""

    __tablename__ = 'mcp_harness'

    id = Column(String, primary_key=True, unique=True)
    name = Column(Text, nullable=False)
    description_for_agents = Column(Text, nullable=True)
    transport = Column(String, nullable=False, default='streamable_http')
    endpoint_or_cmd = Column(Text, nullable=False)
    auth_token = Column(Text, nullable=True)
    discovered_tools_schema = Column(JSON, nullable=True)
    timeout_ms = Column(BigInteger, nullable=False, default=5000)
    is_active = Column(Boolean, nullable=False, default=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)


class MCPHarnessModel(BaseModel):
    id: str
    name: str
    description_for_agents: Optional[str] = None
    transport: str = 'streamable_http'
    endpoint_or_cmd: str
    auth_token: Optional[str] = None
    discovered_tools_schema: Optional[dict[str, Any]] = None
    timeout_ms: int = 5000
    is_active: bool = True
    meta: Optional[dict[str, Any]] = None
    created_at: int
    updated_at: int
    model_config = ConfigDict(from_attributes=True)


class MCPHarnessesTable:
    async def get_harnesses(
        self, active_only: bool = False, db: AsyncSession | None = None
    ) -> list[MCPHarnessModel]:
        async with get_async_db_context(db) as db:
            query = select(MCPHarness)
            if active_only:
                query = query.filter_by(is_active=True)
            result = await db.execute(query)
            return [MCPHarnessModel.model_validate(row) for row in result.scalars().all()]

    async def get_harness_by_id(self, id: str, db: AsyncSession | None = None) -> Optional[MCPHarnessModel]:
        async with get_async_db_context(db) as db:
            row = await db.get(MCPHarness, id)
            return MCPHarnessModel.model_validate(row) if row else None

    async def insert_new_harness(self, form: dict[str, Any], db: AsyncSession | None = None) -> MCPHarnessModel:
        async with get_async_db_context(db) as db:
            now = int(time.time())
            record = MCPHarness(
                id=form.get('id') or str(uuid.uuid4()),
                created_at=now,
                updated_at=now,
                **{
                    k: v
                    for k, v in form.items()
                    if k in (
                        'name', 'description_for_agents', 'transport', 'endpoint_or_cmd',
                        'auth_token', 'discovered_tools_schema', 'timeout_ms', 'is_active', 'meta',
                    )
                },
            )
            db.add(record)
            await db.commit()
            return MCPHarnessModel.model_validate(record)

    async def update_harness_by_id(
        self, id: str, fields: dict[str, Any], db: AsyncSession | None = None
    ) -> Optional[MCPHarnessModel]:
        async with get_async_db_context(db) as db:
            row = await db.get(MCPHarness, id)
            if not row:
                return None
            for key, value in fields.items():
                if hasattr(row, key) and key not in ('id', 'created_at'):
                    setattr(row, key, value)
            row.updated_at = int(time.time())
            await db.commit()
            return MCPHarnessModel.model_validate(row)

    async def delete_harness_by_id(self, id: str, db: AsyncSession | None = None) -> bool:
        async with get_async_db_context(db) as db:
            row = await db.get(MCPHarness, id)
            if not row:
                return False
            await db.delete(row)
            await db.commit()
            return True


Encoders = EncoderModelsTable()
IntentRoutes = IntentRoutesTable()
IntentFeedbacks = IntentFeedbackTable()
MCPHarnesses = MCPHarnessesTable()
