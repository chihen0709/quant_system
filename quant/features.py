"""Public feature-frame builder for hybrid backtests and optimizers."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import yfinance as yf

from .bollinger import add_bollinger_features
from .scoring import add_hybrid_score
from .technical import add_technical_indicators, ensure_ohlcv
from .vcp import add_vcp_features


FEATURE_COLUMNS = [
    "pass_minervini",
    "trend_score",
    "vcp_score",
    "vcp_pivot",
    "bb_width",
    "bb_width_pctile",
    "bb_breakout",
    "fund_score",
    "chip_score",
    "dl_signal",
    "trend_probability",
    "breakout_probability",
    "risk_probability",
    "hybrid_score",
]


def infer_market(ticker: str, market: str | None = None) -> str:
    if market:
        return market.upper()
    t = ticker.upper()
    return "TW" if t.endswith(".TW") or t.endswith(".TWO") or t.isdigit() else "US"


def normalize_ticker(ticker: str, market: str | None = None) -> str:
    t = ticker.strip().upper()
    m = infer_market(t, market)
    if m == "TW" and t.isdigit():
        return f"{t}.TW"
    return t


def download_ohlcv(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if start or end:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    else:
        df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
    return ensure_ohlcv(df)


def _assign_score(out: pd.DataFrame, name: str, value: Any, default: float = 0.0) -> None:
    if isinstance(value, pd.Series):
        out[name] = value.reindex(out.index).fillna(default)
    else:
        out[name] = float(default if value is None else value)


def build_feature_frame(
    ticker: str,
    market: str | None = None,
    start: str | None = None,
    end: str | None = None,
    df: pd.DataFrame | None = None,
    weights: dict | None = None,
    model_path: str | None = None,
    fund_score: float | pd.Series = 0.0,
    chip_score: float | pd.Series = 0.0,
    strict_model: bool = False,
) -> pd.DataFrame:
    """Build the unified OHLCV + signal schema used by DL, Backtrader and optimizers."""
    ticker = normalize_ticker(ticker, market)
    market = infer_market(ticker, market)

    raw = ensure_ohlcv(df) if df is not None else download_ohlcv(ticker, start=start, end=end)
    out = add_technical_indicators(raw)
    out = add_vcp_features(out)
    out = add_bollinger_features(out)

    _assign_score(out, "fund_score", fund_score, 0.0)
    _assign_score(out, "chip_score", chip_score if market == "TW" else 0.0, 0.0)

    out["dl_signal"] = 0.5
    out["trend_probability"] = out["trend_score"].clip(0, 1)
    out["breakout_probability"] = out[["vcp_score", "bb_breakout"]].max(axis=1).clip(0, 1)
    out["risk_probability"] = (1.0 - out["trend_score"]).clip(0, 1)

    if model_path and os.path.exists(model_path):
        try:
            from .dl import predict_dl_signals

            prediction_frame = predict_dl_signals(out, model_path)
            for col in ("dl_signal", "trend_probability", "breakout_probability", "risk_probability"):
                out[col] = prediction_frame[col].reindex(out.index).fillna(out[col])
        except Exception as exc:
            if strict_model:
                raise
            out.attrs["dl_warning"] = str(exc)

    out = add_hybrid_score(out, weights)
    out.attrs["ticker"] = ticker
    out.attrs["market"] = market

    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0

    return out
