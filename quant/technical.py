"""Technical feature engineering for Minervini-style trend systems."""

from __future__ import annotations

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MA_PERIODS = (5, 20, 50, 60, 150, 200, 240)


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV frame with a datetime index and title-case columns."""
    if df is None or df.empty:
        raise ValueError("OHLCV dataframe is empty")

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    rename_map = {}
    for col in out.columns:
        normalized = str(col).strip().lower()
        if normalized in {"open", "high", "low", "close", "volume"}:
            rename_map[col] = normalized.title()
        elif normalized in {"adj close", "adj_close"} and "Close" not in out.columns:
            rename_map[col] = "Close"
    out = out.rename(columns=rename_map)

    missing = [col for col in OHLCV_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")

    out = out.loc[:, list(OHLCV_COLUMNS)].copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    for col in OHLCV_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0.0)
    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages, trend, momentum and distance features."""
    out = ensure_ohlcv(df)

    for period in MA_PERIODS:
        out[f"MA{period}"] = out["Close"].rolling(period, min_periods=period).mean()

    out["High52W"] = out["High"].rolling(250, min_periods=120).max()
    out["Low52W"] = out["Low"].rolling(250, min_periods=120).min()
    out["RSI"] = _rsi(out["Close"], 14)

    ema12 = out["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["MACD_Osc"] = out["MACD"] - out["MACD_Signal"]

    for period in (5, 20, 60, 240):
        ma = out[f"MA{period}"].replace(0, np.nan)
        out[f"BIAS{period}"] = (out["Close"] - ma) / ma * 100

    out["volume_avg20"] = out["Volume"].rolling(20, min_periods=10).mean()
    return add_minervini_features(out)


def add_minervini_features(df: pd.DataFrame, rs_rating: pd.Series | None = None) -> pd.DataFrame:
    """Add the hard Minervini trend template and a 0..1 trend score."""
    out = df.copy()
    for period in (50, 150, 200):
        if f"MA{period}" not in out.columns:
            out[f"MA{period}"] = out["Close"].rolling(period, min_periods=period).mean()

    high52 = out.get("High52W", out["High"].rolling(250, min_periods=120).max())
    low52 = out.get("Low52W", out["Low"].rolling(250, min_periods=120).min())

    conditions = pd.DataFrame(index=out.index)
    conditions["price_above_ma150_200"] = (out["Close"] > out["MA150"]) & (out["Close"] > out["MA200"])
    conditions["ma150_above_ma200"] = out["MA150"] > out["MA200"]
    conditions["ma200_rising"] = out["MA200"] > out["MA200"].shift(20)
    conditions["ma50_above_long_mas"] = (out["MA50"] > out["MA150"]) & (out["MA50"] > out["MA200"])
    conditions["price_above_ma50"] = out["Close"] > out["MA50"]
    conditions["price_off_low"] = out["Close"] > (low52 * 1.30)
    conditions["price_near_high"] = out["Close"] > (high52 * 0.75)

    if rs_rating is not None:
        conditions["rs_rating_ok"] = rs_rating.reindex(out.index).fillna(0) >= 70

    clean_conditions = conditions.fillna(False)
    out["pass_minervini"] = clean_conditions.all(axis=1).astype(int)
    out["trend_score"] = clean_conditions.mean(axis=1).clip(0, 1)
    return out
