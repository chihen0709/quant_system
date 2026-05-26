from __future__ import annotations

from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INDUSTRY_MAP_PATH = BASE_DIR / 'data' / 'industry_map.yaml'


def load_industry_map(path: str | Path | None = None) -> dict:
    target = Path(path) if path else DEFAULT_INDUSTRY_MAP_PATH
    if not target.exists():
        return {'topics': []}

    try:
        payload = yaml.safe_load(target.read_text(encoding='utf-8')) or {}
    except Exception:
        return {'topics': []}

    topics = payload.get('topics')
    if isinstance(topics, list):
        return payload
    return {'topics': []}


def list_topics(path: str | Path | None = None) -> list[dict]:
    payload = load_industry_map(path)
    return [topic for topic in payload.get('topics', []) if isinstance(topic, dict)]


def get_topic_map(topic: str, path: str | Path | None = None) -> dict | None:
    wanted = str(topic or '').strip().lower()
    for item in list_topics(path):
        key = str(item.get('key') or '').strip().lower()
        aliases = [str(alias).strip().lower() for alias in item.get('aliases', []) if alias]
        if wanted and (wanted == key or wanted in aliases):
            return item
    return None


def topic_symbols(topic_data: dict | None) -> list[str]:
    symbols = []
    seen = set()
    if not isinstance(topic_data, dict):
        return symbols

    for lane in topic_data.get('lanes', []) or []:
        for company in lane.get('companies', []) or []:
            symbol = str(company.get('ticker') or company.get('symbol') or '').strip()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
    return symbols
