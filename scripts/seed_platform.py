#!/usr/bin/env python3
"""Идемпотентный сидер платформы synrespond (PERF-OPS-02).

Дожидается старта ядра :8080, создает или авторизует администратора,
регистрирует 4 стартовые ONNX-модели с реальными метками из label_map.json
и создает стартовый Golden Dataset, если база пуста. Повторный запуск
ничего не ломает и не дублирует.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

DEFAULT_ADMIN_EMAIL = os.getenv("SYNRESPOND_ADMIN_EMAIL", "admin@local.dev")
DEFAULT_ADMIN_PASSWORD = os.getenv("SYNRESPOND_ADMIN_PASSWORD", "Admin-Platform-2026!")
DEFAULT_ADMIN_NAME = "Synrespond Admin"

ENCODER_SPECS = [
    ("rubert-intents-v1", "ruBERT Intents v1", "sequence_classification"),
    ("rubert-nli-v1", "ruBERT Zero-Shot NLI v1", "zero_shot_nli"),
    ("rubert-slots-v1", "ruBERT NER Slots v1", "token_classification"),
    ("rubert-embed-v1", "ruBERT Embeddings & Rerank v1", "feature_extraction"),
]


def wait_for_health(base_url: str, timeout_s: int = 90) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.5)
    return False


def acquire_admin_token(base_url: str) -> str | None:
    with httpx.Client(timeout=10.0) as client:
        # 1. Пробуем войти существующим админом
        r = client.post(
            f"{base_url}/api/v1/auths/signin",
            json={"email": DEFAULT_ADMIN_EMAIL, "password": DEFAULT_ADMIN_PASSWORD},
        )
        if r.status_code == 200 and r.json().get("token"):
            return r.json()["token"]
        # 2. Если база свежая — создаем первого пользователя (он становится admin)
        r = client.post(
            f"{base_url}/api/v1/auths/signup",
            json={"name": DEFAULT_ADMIN_NAME, "email": DEFAULT_ADMIN_EMAIL, "password": DEFAULT_ADMIN_PASSWORD},
        )
        if r.status_code == 200 and r.json().get("token"):
            print(f"[seed] Создан администратор {DEFAULT_ADMIN_EMAIL}")
            return r.json()["token"]
    return None


def seed_encoders_and_demo(base_url: str, token: str, models_dir: Path) -> None:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    with httpx.Client(timeout=15.0, headers=headers) as client:
        existing_resp = client.get(f"{base_url}/api/v1/classifiers/encoders")
        existing_paths = {
            e.get("model_path") for e in (existing_resp.json() if existing_resp.status_code == 200 else [])
        }
        for model_id, display_name, task in ENCODER_SPECS:
            if model_id in existing_paths:
                continue
            label_map_path = models_dir / model_id / "label_map.json"
            labels_json: dict = {}
            thresholds = {"high": 0.85, "low": 0.50}
            if label_map_path.exists():
                try:
                    meta = json.loads(label_map_path.read_text(encoding="utf-8"))
                    labels_json = meta.get("labels") or {}
                    thresholds = meta.get("thresholds") or thresholds
                except Exception:  # noqa: BLE001
                    pass
            client.post(
                f"{base_url}/api/v1/classifiers/encoders/create",
                json={
                    "name": display_name,
                    "task": task,
                    "model_path": model_id,
                    "labels_json": labels_json,
                    "thresholds": thresholds,
                    "is_active": True,
                },
            )
            print(f"[seed] Зарегистрирован энкодер: {model_id} ({task})")

        # Демо-датасет Ragas для первого запуска (PERF-OPS-02, п.4)
        ds_resp = client.get(f"{base_url}/api/v1/ragas/datasets")
        datasets = ds_resp.json() if ds_resp.status_code == 200 else []
        if not datasets:
            client.post(
                f"{base_url}/api/v1/ragas/datasets/create",
                json={
                    "name": "Golden Dataset (стартовый)",
                    "description": "Эталонные вопросы для регрессионных прогонов и Auto-Tune",
                },
            )
            print("[seed] Создан стартовый Golden Dataset")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webui-url", default="http://127.0.0.1:8080")
    parser.add_argument("--models-dir", default="models/encoders")
    args = parser.parse_args()

    base_url = args.webui_url.rstrip("/")
    if not wait_for_health(base_url):
        print("[seed] Ядро не поднялось за отведенное время — сидирование пропущено")
        return 0
    token = acquire_admin_token(base_url)
    if not token:
        print("[seed] Не удалось получить токен администратора — сидирование пропущено")
        return 0
    seed_encoders_and_demo(base_url, token, Path(args.models_dir))
    print("[seed] Сидирование завершено")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
