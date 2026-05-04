"""Backtrader PandasData feed carrying precomputed hybrid signals."""

from __future__ import annotations

import backtrader as bt


class MultiSignalDataFeed(bt.feeds.PandasData):
    lines = (
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
    )

    params = tuple((line, -1) for line in lines)
