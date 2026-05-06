"""Hybrid score composition for technical, VCP, Bollinger and ML signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "tech_weight": 0.28,
    "vcp_weight": 0.13,
    "bb_weight": 0.10,
    "cta_weight": 0.10,
    "pattern_weight": 0.06,
    "fund_weight": 0.09,
    "chip_weight": 0.05,
    "ml_weight": 0.19,
}


def normalize_weights(weights: dict | None = None) -> dict:
    merged = DEFAULT_WEIGHTS.copy()
    if weights:
        merged.update({k: float(v) for k, v in weights.items() if k in merged and v is not None})
    total = sum(max(v, 0.0) for v in merged.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {k: max(v, 0.0) / total for k, v in merged.items()}


def _unit_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if s.abs().max() > 1.5:
        s = s / 100.0
    return s.clip(0.0, 1.0)


def add_hybrid_score(df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Add component scores and the final 0..1 hybrid score."""
    out = df.copy()
    w = normalize_weights(weights)

    if "trend_score" not in out.columns:
        out["trend_score"] = 0.0
    if "vcp_score" not in out.columns:
        out["vcp_score"] = 0.0
    if "bb_score" not in out.columns:
        if "bb_breakout" in out.columns:
            out["bb_score"] = pd.to_numeric(out["bb_breakout"], errors="coerce").fillna(0.0)
        else:
            out["bb_score"] = 0.0
    if "fund_score" not in out.columns:
        out["fund_score"] = 0.0
    if "chip_score" not in out.columns:
        out["chip_score"] = 0.0
    if "cta_score" not in out.columns:
        cta_cols = [
            col for col in ("close_box_breakout", "engulfing_5d", "bb_squeeze_breakout", "bb_momentum_breakout")
            if col in out.columns
        ]
        out["cta_score"] = out[cta_cols].max(axis=1) if cta_cols else 0.0
    if "pattern_score" not in out.columns:
        pattern_cols = [
            col for col in ("triangle_contraction_score", "inverse_head_shoulders_score")
            if col in out.columns
        ]
        out["pattern_score"] = out[pattern_cols].max(axis=1) if pattern_cols else 0.0
    if "dl_signal" not in out.columns:
        out["dl_signal"] = 0.5

    components = {
        "tech_weight": _unit_series(out["trend_score"]),
        "vcp_weight": _unit_series(out["vcp_score"]),
        "bb_weight": _unit_series(out["bb_score"]),
        "cta_weight": _unit_series(out["cta_score"]),
        "pattern_weight": _unit_series(out["pattern_score"]),
        "fund_weight": _unit_series(out["fund_score"]),
        "chip_weight": _unit_series(out["chip_score"]),
        "ml_weight": _unit_series(out["dl_signal"]),
    }
    score = np.zeros(len(out), dtype=float)
    for key, values in components.items():
        score += values.to_numpy(dtype=float) * w[key]

    out["hybrid_score"] = np.clip(score, 0.0, 1.0)
    return out
