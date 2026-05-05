"""Chip/ownership flow scoring helpers."""

from __future__ import annotations

import os


US_MIN_AVG_VOLUME = int(os.environ.get("US_MIN_AVG_VOLUME", "3000000"))


def calc_chip_score(snapshot: dict, is_us: bool = False) -> float:
    if is_us:
        score = 0.0
        institutional = snapshot.get("institutional_ownership_pct")
        short_float = snapshot.get("short_percent_float")
        short_ratio = snapshot.get("short_ratio")
        avg_volume = snapshot.get("avg_volume_3m")
        avg_volume_10d = snapshot.get("avg_volume_10d")
        latest_volume = snapshot.get("latest_volume")

        if institutional is not None:
            if 20 <= institutional <= 85:
                score += 26
            elif 10 <= institutional < 20 or 85 < institutional <= 95:
                score += 16
            elif institutional > 95:
                score += 8
        if short_float is not None:
            if short_float < 5:
                score += 24
            elif short_float < 10:
                score += 16
            elif short_float < 20:
                score += 6
            else:
                score -= 12
        if short_ratio is not None:
            if short_ratio < 3:
                score += 16
            elif short_ratio < 6:
                score += 8
            elif short_ratio > 10:
                score -= 8
        if avg_volume is not None:
            if avg_volume >= 10_000_000:
                score += 18
            elif avg_volume >= US_MIN_AVG_VOLUME:
                score += 12
        if avg_volume and latest_volume:
            rel_vol = latest_volume / avg_volume
            if rel_vol >= 1.5:
                score += 16
            elif rel_vol >= 1.0:
                score += 10
            elif rel_vol < 0.6:
                score -= 6
        if avg_volume and avg_volume_10d and avg_volume_10d > avg_volume * 1.15:
            score += 8
        return max(0.0, min(float(score), 100.0))

    score = 0.0
    t2 = snapshot.get("total_2d")
    t5 = snapshot.get("total_5d")
    t10 = snapshot.get("total_10d")
    f5 = snapshot.get("foreign_5d")
    if t2 is not None:
        score += 12 if t2 > 0 else -6
    if t5 is not None:
        score += 24 if t5 > 5000 else (18 if t5 > 1000 else (10 if t5 > 0 else (-16 if t5 < -5000 else -8)))
    if t10 is not None:
        score += 14 if t10 > 0 else -8
    if f5 is not None:
        score += 10 if f5 > 0 else -5
    return max(0.0, min(float(score), 100.0))

