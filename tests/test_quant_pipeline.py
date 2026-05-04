import numpy as np
import pandas as pd

from backtests.run_backtest import run_single_backtest
from quant.bollinger import add_bollinger_features
from quant.features import build_feature_frame
from quant.risk import calculate_position_size
from quant.scoring import add_hybrid_score
from quant.technical import add_technical_indicators
from quant.vcp import add_vcp_features


def synthetic_ohlcv(rows=320):
    idx = pd.date_range("2020-01-01", periods=rows, freq="B")
    trend = np.linspace(50, 150, rows)
    wave = np.sin(np.linspace(0, 12, rows)) * 3
    close = trend + wave
    open_ = close * 0.995
    high = close * 1.015
    low = close * 0.985
    volume = np.linspace(1_000_000, 600_000, rows)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def test_feature_frame_schema_and_score_range():
    frame = build_feature_frame("TEST", market="US", df=synthetic_ohlcv())
    required = [
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
        "hybrid_score",
    ]
    for column in required:
        assert column in frame.columns
    assert frame["hybrid_score"].dropna().between(0, 1).all()


def test_minervini_vcp_bollinger_and_scoring_are_numeric():
    frame = add_technical_indicators(synthetic_ohlcv())
    frame = add_vcp_features(frame)
    frame = add_bollinger_features(frame)
    frame = add_hybrid_score(frame)
    assert frame["trend_score"].dropna().between(0, 1).all()
    assert frame["vcp_score"].dropna().between(0, 1).all()
    assert frame["bb_score"].dropna().between(0, 1).all()
    assert frame["hybrid_score"].dropna().between(0, 1).all()


def test_position_size_caps_risk_and_cash():
    size = calculate_position_size(
        cash=1_000_000,
        entry_price=100,
        stop_loss_pct=0.08,
        risk_per_trade=0.01,
        max_cash_fraction=0.95,
    )
    assert size == 1250


def test_backtrader_smoke_runs_on_precomputed_features():
    frame = build_feature_frame("TEST", market="US", df=synthetic_ohlcv())
    metrics = run_single_backtest(
        frame,
        {
            "score_threshold": 0.40,
            "risk_per_trade": 0.01,
            "stop_loss_pct": 0.08,
            "exit_ma_period": 20,
        },
    )
    assert "total_return_pct" in metrics
    assert metrics["bars"] == len(frame)
