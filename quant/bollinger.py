"""Bollinger squeeze and breakout features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _last_percentile(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    last = valid[-1]
    return float((valid <= last).sum() / len(valid) * 100.0)


def add_bollinger_features(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    percentile_lookback: int = 252,
    squeeze_percentile: float = 10.0,
    breakout_volume_mult: float = 1.5,
) -> pd.DataFrame:
    """Add Bollinger Band width, squeeze percentile and breakout columns."""
    out = df.copy()
    mid = out["Close"].rolling(window, min_periods=window).mean()
    std = out["Close"].rolling(window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std

    out["BBMid"] = mid
    out["BBUpper"] = upper
    out["BBLower"] = lower
    out["bb_width"] = ((upper - lower) / mid.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["bb_width_pctile"] = out["bb_width"].rolling(
        percentile_lookback,
        min_periods=max(30, percentile_lookback // 4),
    ).apply(_last_percentile, raw=True)

    volume_avg = out.get("volume_avg20", out["Volume"].rolling(20, min_periods=10).mean())
    squeeze = out["bb_width_pctile"] <= squeeze_percentile
    squeeze_recent = squeeze.rolling(5, min_periods=1).max().fillna(0).astype(bool)
    volume_breakout = out["Volume"] > (volume_avg * breakout_volume_mult)
    price_breakout = out["Close"] > upper

    out["bb_squeeze"] = squeeze.fillna(False).astype(int)
    out["bb_breakout"] = (squeeze_recent & volume_breakout & price_breakout).fillna(False).astype(int)
    pressure = (1.0 - out["bb_width_pctile"] / 100.0).clip(0, 1).fillna(0)
    out["bb_score"] = np.maximum(out["bb_breakout"].astype(float), pressure * 0.6)
    return out
