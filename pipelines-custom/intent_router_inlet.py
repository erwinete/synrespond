"""
title: Intent Router Inlet (encoder-classifier)
author: enterprise-ai-platform
version: 1.0
description: Перехватчик inlet — классифицирует запрос через encoder-classifier и подставляет системный промпт/коллекции по правилам интента.
required_open_webui_version: 0.6.0
"""

import os
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field


class Pipeline:
    """Inlet-пайплайн: маршрутизация интентов без вызова LLM.

    Используется в рантайме open-webui/pipelines как альтернатива нативным
    хукам middleware в форке. Правила хранятся в переменных окружения или
    в body["features"]["intent_routes"].
    """

    class Valves(BaseModel):
        classifier_url: str = Field(
            default=os.getenv("ENCODER_CLASSIFIER_URL", "http://encoder-classifier:8001")
        )
        model_id: str = Field(default=os.getenv("INTENT_MODEL_ID", "rubert-intents-v1"))
        tau_high: float = Field(default=0.85)
        tau_low: float = Field(default=0.50)
        timeout_s: float = Field(default=0.5)

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        routes: list[dict[str, Any]] = (body.get("features") or {}).get("intent_routes") or []
        if not routes:
            return body

        text = ""
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                text = msg.get("content", "")
                break
        if not text:
            return body

        try:
            async with httpx.AsyncClient(timeout=self.valves.timeout_s) as client:
                resp = await client.post(
                    f"{self.valves.classifier_url}/v1/classify",
                    json={
                        "text": text,
                        "model_id": self.valves.model_id,
                        "tau_high": self.valves.tau_high,
                        "tau_low": self.valves.tau_low,
                    },
                )
                resp.raise_for_status()
                verdict = resp.json()
        except Exception:
            # Классификатор недоступен — пропускаем запрос без маршрутизации
            return body

        action = verdict.get("action")
        intent_name = verdict.get("intent_name")
        rule = next((r for r in routes if r.get("intent_name") == intent_name), None)

        if action == "direct" and rule:
            feature_update = {
                "routed_intent": intent_name,
                "intent_confidence": verdict.get("confidence"),
            }
            if rule.get("system_prompt_template"):
                body["system_prompt_override"] = rule["system_prompt_template"]
            if rule.get("target_collection_ids"):
                feature_update["target_collection_ids"] = rule["target_collection_ids"]
            if rule.get("use_hipporag") is not None:
                feature_update["use_hipporag"] = rule["use_hipporag"]
            features = body.get("features") or {}
            features.update(feature_update)
            body["features"] = features

        return body
