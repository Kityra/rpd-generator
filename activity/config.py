"""Пороги алгоритма: config/activity_model.yaml и словарь областей знаний config/knowledge_areas.yaml."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml

from orchestration.semantic_check import load_analysis


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "activity_model.yaml"
AREAS_PATH = ROOT / "config" / "knowledge_areas.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Множители свежести общие с проверкой источников курса (config/analysis.yaml → coverage.freshness).
    config.setdefault("freshness", load_analysis()["coverage"]["freshness"])
    return config


@lru_cache(maxsize=4)
def load_areas(path: str | None = None) -> list[dict]:
    with open(path or AREAS_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["areas"]


def config_hash(config: dict | None = None, areas: list[dict] | None = None) -> str:
    """Хэш правил: версия модели воспроизводима только с тем же конфигом и словарём областей."""
    payload = {"config": config or load_config(), "areas": areas or load_areas()}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
