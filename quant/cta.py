"""Rule-based CTA pattern features for breakout-style equity systems."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


BOX_LOOKBACK = int(os.environ.get("BOX_LOOKBACK", "20"))
BOX_WIDTH_THRESHOLD = float(os.environ.get("BOX_WIDTH_THRESHOLD", "0.10"))
BOX_BREAKOUT_VOLUME_MULT = float(os.environ.get("BOX_BREAKOUT_VOLUME_MULT", "1.5"))
ENGULFING_LOOKBACK = int(os.environ.get("ENGULFING_LOOKBACK", "5"))
ENGULFING_BODY_PCT = float(os.environ.get("ENGULFING_BODY_PCT", "0.02"))
BB_WIDTH_THRESHOLD = float(os.environ.get("BB_WIDTH_THRESHOLD", "0.10"))
BB_BREAKOUT_VOLUME_MULT = float(os.environ.get("BB_BREAKOUT_VOLUME_MULT", "1.5"))


def add_cta_features(
    df: pd.DataFrame,
    box_lookback: int = BOX_LOOKBACK,
    box_width_threshold: float = BOX_WIDTH_THRESHOLD,
    box_volume_mult: float = BOX_BREAKOUT_VOLUME_MULT,
    engulfing_lookback: int = ENGULFING_LOOKBACK,
    engulfing_body_pct: float = ENGULFING_BODY_PCT,
    bb_width_threshold: float = BB_WIDTH_THRESHOLD,
    bb_volume_mult: float = BB_BREAKOUT_VOLUME_MULT,
) -> pd.DataFrame:
    """Add no-lookahead box, engulfing and Bollinger breakout signals."""
    out = df.copy()

    close = pd.to_numeric(out["Close"], errors="coerce")
    open_ = pd.to_numeric(out["Open"], errors="coerce")
    high = pd.to_numeric(out["High"], errors="coerce")
    volume = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0)

    if "BBMid" not in out.columns:
        out["BBMid"] = close.rolling(20, min_periods=20).mean()
    if "BBUpper" not in out.columns or "BBLower" not in out.columns:
        bb_mid = out["BBMid"]
        bb_std = close.rolling(20, min_periods=20).std(ddof=0)
        out["BBUpper"] = bb_mid + 2.0 * bb_std
        out["BBLower"] = bb_mid - 2.0 * bb_std
    out["BBWidth"] = ((out["BBUpper"] - out["BBLower"]) / out["BBMid"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    out["Close_Max_20"] = close.rolling(box_lookback, min_periods=box_lookback).max()
    out["Close_Min_20"] = close.rolling(box_lookback, min_periods=box_lookback).min()
    out["Close_Max_20_Prior"] = close.shift(1).rolling(box_lookback, min_periods=box_lookback).max()
    out["Close_Min_20_Prior"] = close.shift(1).rolling(box_lookback, min_periods=box_lookback).min()
    out["Box_Width"] = (
        (out["Close_Max_20_Prior"] - out["Close_Min_20_Prior"]) / out["Close_Min_20_Prior"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    out["High_Max_5"] = high.rolling(engulfing_lookback, min_periods=engulfing_lookback).max()
    out["High_Max_5_Prior"] = high.shift(1).rolling(engulfing_lookback, min_periods=engulfing_lookback).max()
    out["volume_avg20"] = out.get("volume_avg20", volume.rolling(20, min_periods=10).mean())
    prior_volume_avg = out["volume_avg20"].shift(1)

    body_pct = ((close - open_) / open_.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["engulfing_5d"] = (
        (close > open_)
        & (close > out["High_Max_5_Prior"])
        & (body_pct > engulfing_body_pct)
    ).fillna(False).astype(int)

    out["close_box_breakout"] = (
        (out["Box_Width"] < box_width_threshold)
        & (close > out["Close_Max_20_Prior"])
        & (volume > prior_volume_avg * box_volume_mult)
    ).fillna(False).astype(int)

    out["bb_squeeze_breakout"] = (
        (close > out["BBUpper"])
        & (out["BBWidth"].shift(1) < bb_width_threshold)
        & (volume > prior_volume_avg * bb_volume_mult)
    ).fillna(False).astype(int)

    out["bb_momentum_breakout"] = (
        (close > out["BBUpper"])
        & (pd.to_numeric(out.get("RSI", 0), errors="coerce") > 60)
        & (pd.to_numeric(out.get("MACD_Osc", 0), errors="coerce") > 0)
    ).fillna(False).astype(int)

    out["cta_score"] = (
        out["close_box_breakout"].astype(float) * 0.35
        + out["engulfing_5d"].astype(float) * 0.20
        + out["bb_squeeze_breakout"].astype(float) * 0.30
        + out["bb_momentum_breakout"].astype(float) * 0.15
    ).clip(0.0, 1.0)
    return out


def row_strategy_tags(row: pd.Series | dict) -> list[str]:
    """Return human-readable strategy tags for a latest feature row."""
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    tags: list[str] = []
    if bool(getter("close_box_breakout", 0)):
        tags.append("無雜訊箱型突破")
    if bool(getter("engulfing_5d", 0)):
        tags.append("五日陣吞噬")
    if bool(getter("bb_squeeze_breakout", 0)):
        tags.append("布林壓縮突破")
    if bool(getter("bb_momentum_breakout", 0)):
        tags.append("布林動能突破")
    return tags

