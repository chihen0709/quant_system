"""Lightweight swing-point pattern recognition for chart morphology."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:  # pragma: no cover - fallback is covered when scipy is unavailable.
    from scipy.signal import argrelextrema
except Exception:  # pragma: no cover
    argrelextrema = None


PATTERN_SWING_ORDER = int(os.environ.get("PATTERN_SWING_ORDER", "5"))


def _local_extrema(values: np.ndarray, order: int, mode: str) -> np.ndarray:
    if len(values) < order * 2 + 1:
        return np.array([], dtype=int)
    if argrelextrema is not None:
        comparator = np.greater_equal if mode == "max" else np.less_equal
        return argrelextrema(values, comparator, order=order)[0]

    idxs: list[int] = []
    for idx in range(order, len(values) - order):
        window = values[idx - order : idx + order + 1]
        center = values[idx]
        if mode == "max" and np.isfinite(center) and center >= np.nanmax(window):
            idxs.append(idx)
        if mode == "min" and np.isfinite(center) and center <= np.nanmin(window):
            idxs.append(idx)
    return np.asarray(idxs, dtype=int)


def _fit_slope(points: np.ndarray, values: np.ndarray) -> float | None:
    if len(points) < 2:
        return None
    y = values[points]
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return None
    x = points[valid].astype(float)
    y = y[valid].astype(float)
    slope, _ = np.polyfit(x, y, 1)
    denom = float(np.nanmean(np.abs(y))) or 1.0
    return float(slope / denom)


def triangle_contraction_score(high: np.ndarray, low: np.ndarray, order: int = PATTERN_SWING_ORDER) -> float:
    """Score converging triangle behavior from local highs/lows in one window."""
    high_points = _local_extrema(high, order, "max")
    low_points = _local_extrema(low, order, "min")
    if len(high_points) < 2 or len(low_points) < 2:
        return 0.0

    high_slope = _fit_slope(high_points[-4:], high)
    low_slope = _fit_slope(low_points[-4:], low)
    if high_slope is None or low_slope is None:
        return 0.0

    compressing = high_slope < 0 and low_slope > 0
    if not compressing:
        return 0.0

    latest_range = float(np.nanmax(high[-20:]) - np.nanmin(low[-20:]))
    prior_range = float(np.nanmax(high[:20]) - np.nanmin(low[:20])) if len(high) >= 40 else latest_range
    compression = 1.0 - min(max(latest_range / prior_range, 0.0), 1.0) if prior_range > 0 else 0.0
    slope_strength = min(abs(high_slope) + abs(low_slope), 0.04) / 0.04
    return float(np.clip(0.55 * compression + 0.45 * slope_strength, 0.0, 1.0))


def inverse_head_shoulders_score(low: np.ndarray, order: int = PATTERN_SWING_ORDER) -> float:
    """Score a rough inverse head-and-shoulders structure from the last three swing lows."""
    low_points = _local_extrema(low, order, "min")
    if len(low_points) < 3:
        return 0.0
    points = low_points[-3:]
    left, head, right = low[points[0]], low[points[1]], low[points[2]]
    if not all(np.isfinite([left, head, right])):
        return 0.0
    if not (head < left and head < right):
        return 0.0

    shoulder_avg = (left + right) / 2.0
    if shoulder_avg <= 0:
        return 0.0
    head_depth = (shoulder_avg - head) / shoulder_avg
    shoulder_symmetry = 1.0 - min(abs(left - right) / shoulder_avg / 0.08, 1.0)
    depth_score = min(max(head_depth / 0.12, 0.0), 1.0)
    return float(np.clip(0.55 * depth_score + 0.45 * shoulder_symmetry, 0.0, 1.0))


def add_pattern_features(
    df: pd.DataFrame,
    lookback: int = 120,
    order: int = PATTERN_SWING_ORDER,
) -> pd.DataFrame:
    """Add rolling pattern morphology scores without using future bars."""
    out = df.copy()
    highs = pd.to_numeric(out["High"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(out["Low"], errors="coerce").to_numpy(dtype=float)
    triangle = np.zeros(len(out), dtype=float)
    inverse_hs = np.zeros(len(out), dtype=float)

    for end in range(lookback - 1, len(out)):
        start = end - lookback + 1
        high_window = highs[start : end + 1]
        low_window = lows[start : end + 1]
        triangle[end] = triangle_contraction_score(high_window, low_window, order)
        inverse_hs[end] = inverse_head_shoulders_score(low_window, order)

    out["triangle_contraction_score"] = triangle
    out["inverse_head_shoulders_score"] = inverse_hs
    out["pattern_score"] = np.maximum(triangle, inverse_hs).clip(0.0, 1.0)
    return out

