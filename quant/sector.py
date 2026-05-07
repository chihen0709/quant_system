"""Sector/industry aggregation helpers for ranked scan results."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any


SECTOR_STRONG_THRESHOLD = float(os.environ.get("SECTOR_STRONG_THRESHOLD", "65"))


def _item_industry(item: dict[str, Any]) -> str:
    profile = item.get("profile_info") or {}
    industry = str(profile.get("industry") or "N/A").strip()
    return industry if industry else "N/A"


def _is_breakout(item: dict[str, Any]) -> bool:
    conditions = (item.get("tech_pack") or {}).get("conditions", {})
    keys = (
        "vcp_setup",
        "bb_breakout",
        "c_box_breakout",
        "c_engulfing_5d",
        "c_bb_squeeze_breakout",
        "c_bb_momentum_breakout",
    )
    return any(bool(conditions.get(key)) for key in keys)


def _is_up(item: dict[str, Any]) -> bool:
    df = (item.get("tech_pack") or {}).get("df")
    try:
        if df is None or len(df) < 2:
            return False
        return float(df["Close"].iloc[-1]) > float(df["Close"].iloc[-2])
    except Exception:
        return False


def summarize_sectors(ranked: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ranked:
        buckets[_item_industry(item)].append(item)

    summaries: dict[str, dict[str, Any]] = {}
    for industry, items in buckets.items():
        count = len(items)
        avg_score = sum(float(item.get("total_score", 0.0)) for item in items) / max(1, count)
        up_ratio = sum(1 for item in items if _is_up(item)) / max(1, count)
        breakout_ratio = sum(1 for item in items if _is_breakout(item)) / max(1, count)
        strength = min(100.0, max(0.0, avg_score * 0.65 + up_ratio * 20.0 + breakout_ratio * 15.0))
        summaries[industry] = {
            "industry": industry,
            "count": count,
            "avg_score": avg_score,
            "up_ratio": up_ratio,
            "breakout_ratio": breakout_ratio,
            "sector_strength_score": strength,
        }
    return summaries


def annotate_sector_strength(
    ranked: list[dict[str, Any]],
    strong_threshold: float = SECTOR_STRONG_THRESHOLD,
) -> list[dict[str, Any]]:
    summaries = summarize_sectors(ranked)
    annotated = []
    for item in ranked:
        copied = dict(item)
        industry = _item_industry(item)
        sector = summaries.get(industry, {})
        tags: list[str] = []
        if sector.get("count", 0) >= 2 and sector.get("sector_strength_score", 0) >= strong_threshold:
            tags.append("▲同族連動")
        if sector.get("breakout_ratio", 0) >= 0.34:
            tags.append("◆新資金")
        if float(item.get("total_score", 0.0)) >= float(sector.get("avg_score", 0.0)) + 5:
            tags.append("◎族群領先")
        copied["sector_info"] = sector
        copied["sector_tags"] = tags
        annotated.append(copied)
    return annotated
