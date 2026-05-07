import numpy as np
import pandas as pd

from backtests.run_backtest import run_single_backtest
from quant.bollinger import add_bollinger_features
from quant.chip import calc_chip_score
from quant.cta import add_cta_features
from quant.data_sources import SQLiteTTLCache
from quant.features import build_feature_frame
from quant.features import normalize_ticker as normalize_feature_ticker
from quant.fundamental import calc_fundamental_score
from quant.patterns import add_pattern_features
from quant.risk import calculate_position_size
from quant.scoring import add_hybrid_score
from quant.sector import annotate_sector_strength
from quant.technical import add_technical_indicators
from quant.telegram_bot import sanitize_telegram_error, validate_telegram_token
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


def test_ticker_normalization_tw_and_us():
    assert normalize_feature_ticker("2330", market="TW") == "2330.TW"
    assert normalize_feature_ticker("2330.TW") == "2330.TW"
    assert normalize_feature_ticker("2330.TWO") == "2330.TWO"
    assert normalize_feature_ticker("aapl") == "AAPL"
    assert normalize_feature_ticker("BRK.B") == "BRK-B"
    assert normalize_feature_ticker("BRK-B") == "BRK-B"


def test_telegram_token_validation_and_sanitization():
    secret = "123456789:" + "A" * 32
    assert validate_telegram_token(secret)
    assert not validate_telegram_token("not-a-real-token")
    assert not validate_telegram_token("123456789")
    sanitized = sanitize_telegram_error(Exception(f"failed for {secret}"))
    assert secret not in sanitized
    assert "<redacted-token>" in sanitized


def test_fundamental_and_chip_scoring_ranges():
    tw_snapshot = {
        "single_month_yoy": 35,
        "single_month_mom": 10,
        "eps_latest_quarter": 12,
        "eps_ttm": 45,
        "total_2d": 100,
        "total_5d": 6000,
        "total_10d": 3000,
        "foreign_5d": 2000,
    }
    us_snapshot = {
        "single_month_yoy": 12,
        "eps_ttm": 8,
        "profit_margin_pct": 18,
        "earnings_growth_pct": 30,
        "institutional_ownership_pct": 62,
        "short_percent_float": 4,
        "short_ratio": 2,
        "avg_volume_3m": 12_000_000,
        "avg_volume_10d": 15_000_000,
        "latest_volume": 20_000_000,
    }
    for score in (
        calc_fundamental_score(tw_snapshot, is_us=False),
        calc_chip_score(tw_snapshot, is_us=False),
        calc_fundamental_score(us_snapshot, is_us=True),
        calc_chip_score(us_snapshot, is_us=True),
    ):
        assert 0 <= score <= 100


def test_sqlite_ttl_cache_hit_and_expire(tmp_path):
    cache = SQLiteTTLCache(str(tmp_path / "cache.sqlite3"))
    cache.set("unit", "hit", {"value": 1}, ttl_hours=1)
    assert cache.get("unit", "hit") == {"value": 1}
    cache.set("unit", "expired", {"value": 2}, ttl_hours=-0.001)
    assert cache.get("unit", "expired") is None


def test_close_box_breakout_uses_prior_box_not_today_high():
    frame = synthetic_ohlcv(45)
    frame["Close"] = 100.0
    frame["Open"] = 99.0
    frame["High"] = 101.0
    frame["Low"] = 98.0
    frame["Volume"] = 1_000_000
    frame.iloc[-1, frame.columns.get_loc("Close")] = 111.0
    frame.iloc[-1, frame.columns.get_loc("Open")] = 108.0
    frame.iloc[-1, frame.columns.get_loc("High")] = 112.0
    frame.iloc[-1, frame.columns.get_loc("Volume")] = 2_000_000

    out = add_cta_features(frame)
    latest = out.iloc[-1]
    assert latest["Close_Max_20"] == 111.0
    assert latest["Close_Max_20_Prior"] == 100.0
    assert latest["close_box_breakout"] == 1


def test_five_day_engulfing_requires_red_body_and_prior_high_break():
    frame = synthetic_ohlcv(30)
    frame["High"] = 100.0
    frame["Close"] = 98.0
    frame["Open"] = 97.0
    frame.iloc[-1, frame.columns.get_loc("Open")] = 100.0
    frame.iloc[-1, frame.columns.get_loc("Close")] = 104.0
    frame.iloc[-1, frame.columns.get_loc("High")] = 105.0
    out = add_cta_features(frame)
    assert out["engulfing_5d"].iloc[-1] == 1

    frame.iloc[-1, frame.columns.get_loc("Close")] = 101.0
    out = add_cta_features(frame)
    assert out["engulfing_5d"].iloc[-1] == 0


def test_pattern_scores_and_sector_tags_are_bounded():
    frame = synthetic_ohlcv(180)
    patterned = add_pattern_features(frame)
    assert patterned["triangle_contraction_score"].between(0, 1).all()
    assert patterned["inverse_head_shoulders_score"].between(0, 1).all()

    ranked = annotate_sector_strength(
        [
            {
                "ticker": "AAA",
                "total_score": 80,
                "profile_info": {"industry": "AI"},
                "tech_pack": {"conditions": {"c_box_breakout": True}, "df": frame},
            },
            {
                "ticker": "BBB",
                "total_score": 75,
                "profile_info": {"industry": "AI"},
                "tech_pack": {"conditions": {"bb_breakout": True}, "df": frame},
            },
        ]
    )
    assert ranked[0]["sector_info"]["count"] == 2
    assert ranked[0]["sector_info"]["sector_strength_score"] > 0
    assert ranked[0]["sector_tags"]
