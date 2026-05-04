"""Quantitative approximation of Minervini VCP morphology."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _segment_depth(high: np.ndarray, low: np.ndarray) -> float:
    seg_high = np.nanmax(high)
    seg_low = np.nanmin(low)
    if not np.isfinite(seg_high) or seg_high <= 0 or not np.isfinite(seg_low):
        return np.nan
    return float((seg_high - seg_low) / seg_high * 100.0)


def add_vcp_features(
    df: pd.DataFrame,
    lookback: int = 80,
    contractions: int = 4,
    pivot_lookback: int = 20,
) -> pd.DataFrame:
    """Add VCP score and pivot columns without using future bars."""
    out = df.copy()
    scores = np.zeros(len(out), dtype=float)
    pivots = np.full(len(out), np.nan, dtype=float)
    is_vcp = np.zeros(len(out), dtype=int)

    highs = out["High"].to_numpy(dtype=float)
    lows = out["Low"].to_numpy(dtype=float)
    closes = out["Close"].to_numpy(dtype=float)
    volumes = out["Volume"].to_numpy(dtype=float)

    for i in range(lookback - 1, len(out)):
        start = i - lookback + 1
        window_high = highs[start : i + 1]
        window_low = lows[start : i + 1]
        window_close = closes[start : i + 1]
        window_volume = volumes[start : i + 1]

        split_high = np.array_split(window_high, contractions)
        split_low = np.array_split(window_low, contractions)
        depths = np.array([_segment_depth(h, l) for h, l in zip(split_high, split_low)], dtype=float)
        if np.isnan(depths).any():
            continue

        decreasing = np.mean(depths[:-1] > depths[1:])
        depth_shape = 1.0 if depths[0] >= 10 and depths[-1] <= max(12.0, depths[0] * 0.70) else 0.0

        prior_volume = np.nanmean(window_volume[-60:-10]) if len(window_volume) >= 60 else np.nanmean(window_volume[:-10])
        recent_volume = np.nanmean(window_volume[-10:])
        dry_up = 1.0 if np.isfinite(prior_volume) and prior_volume > 0 and recent_volume < prior_volume * 0.75 else 0.0

        recent_close = window_close[-10:]
        close_mean = np.nanmean(recent_close)
        tightness = 0.0
        if np.isfinite(close_mean) and close_mean > 0:
            tightness = 1.0 - min(float(np.nanstd(recent_close) / close_mean) / 0.05, 1.0)

        pivot_slice = window_high[-pivot_lookback:]
        pivot = float(np.nanmax(pivot_slice)) if len(pivot_slice) else np.nan
        pivots[i] = pivot
        near_pivot = 1.0 if np.isfinite(pivot) and pivot > 0 and pivot * 0.90 <= closes[i] <= pivot * 1.03 else 0.0

        score = 0.40 * decreasing + 0.20 * depth_shape + 0.20 * dry_up + 0.10 * tightness + 0.10 * near_pivot
        scores[i] = float(np.clip(score, 0.0, 1.0))
        is_vcp[i] = int(score >= 0.65 and near_pivot > 0)

    out["vcp_score"] = scores
    out["vcp_pivot"] = pivots
    out["is_vcp"] = is_vcp
    return out
