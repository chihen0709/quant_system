"""Backtest entrypoints for precomputed hybrid feature frames."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import backtrader as bt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtests.datafeed import MultiSignalDataFeed
from backtests.strategies import MinerviniHybridStrategy
from quant.features import build_feature_frame


DEFAULT_BACKTEST_PARAMS = {
    "score_threshold": 0.75,
    "risk_per_trade": 0.015,
    "stop_loss_pct": 0.075,
    "exit_ma_period": 20,
    "risk_exit_threshold": 0.85,
    "walk_forward_folds": 1,
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _extract_trade_metrics(trade_analysis: dict) -> tuple[int, int, float]:
    total = int(trade_analysis.get("total", {}).get("closed", 0) or 0)
    won = int(trade_analysis.get("won", {}).get("total", 0) or 0)
    win_rate = won / total * 100.0 if total else 0.0
    return total, won, win_rate


def _return_stats(timereturns: dict) -> tuple[float, float]:
    returns = np.asarray(list(timereturns.values()), dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0, 0.0
    mean_daily = returns.mean()
    std_daily = returns.std(ddof=1)
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0.0
    sortino = mean_daily / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    return float(sharpe), float(sortino)


def run_single_backtest(
    feature_frame: pd.DataFrame,
    params: dict | None = None,
    initial_cash: float = 1_000_000.0,
    commission: float = 0.002,
    plot: bool = False,
) -> dict:
    """Run Backtrader for one precomputed feature frame."""
    cfg = DEFAULT_BACKTEST_PARAMS.copy()
    if params:
        cfg.update(params)

    data_frame = feature_frame.copy()
    data_frame.index = pd.to_datetime(data_frame.index)
    data_frame = data_frame.sort_index()
    data_frame = data_frame.dropna(subset=["Open", "High", "Low", "Close"])
    if data_frame.empty:
        raise ValueError("feature_frame has no OHLC rows")

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(MultiSignalDataFeed(dataname=data_frame))
    cerebro.addstrategy(
        MinerviniHybridStrategy,
        score_threshold=float(cfg["score_threshold"]),
        risk_per_trade=float(cfg["risk_per_trade"]),
        stop_loss_pct=float(cfg["stop_loss_pct"]),
        exit_ma_period=int(cfg["exit_ma_period"]),
        risk_exit_threshold=float(cfg.get("risk_exit_threshold", 0.85)),
        printlog=bool(cfg.get("printlog", False)),
    )
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    result = cerebro.run()[0]
    final_value = cerebro.broker.getvalue()
    total_return_pct = (final_value / initial_cash - 1.0) * 100.0
    drawdown = result.analyzers.drawdown.get_analysis()
    trades = result.analyzers.trades.get_analysis()
    total_trades, won_trades, win_rate = _extract_trade_metrics(trades)
    sharpe, sortino = _return_stats(result.analyzers.timereturn.get_analysis())
    bars = len(data_frame)

    if plot:
        cerebro.plot(style="candlestick", volume=True)

    return {
        "ticker": data_frame.attrs.get("ticker", ""),
        "bars": bars,
        "initial_cash": initial_cash,
        "final_value": final_value,
        "total_return_pct": float(total_return_pct),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": _safe_float(drawdown.get("max", {}).get("drawdown", 0.0)),
        "total_trades": total_trades,
        "won_trades": won_trades,
        "win_rate_pct": float(win_rate),
        "turnover": float(total_trades / max(1, bars)),
    }


def _iter_frames(feature_frames: dict[str, pd.DataFrame] | Iterable[pd.DataFrame]) -> list[pd.DataFrame]:
    if isinstance(feature_frames, dict):
        return list(feature_frames.values())
    return list(feature_frames)


def _walk_forward_slices(frame: pd.DataFrame, folds: int, warmup_bars: int = 260) -> list[pd.DataFrame]:
    if folds <= 1 or len(frame) < warmup_bars + 80:
        return [frame]
    edges = np.linspace(warmup_bars, len(frame), folds + 1).astype(int)
    slices = []
    for i in range(folds):
        start = max(0, edges[i] - 40)
        end = edges[i + 1]
        if end - start >= 80:
            slices.append(frame.iloc[start:end].copy())
    return slices or [frame]


def run_walk_forward_backtest(
    feature_frames: dict[str, pd.DataFrame] | Iterable[pd.DataFrame],
    params: dict | None = None,
    initial_cash: float = 1_000_000.0,
    commission: float = 0.002,
) -> dict:
    """Evaluate a parameter set over many tickers/folds and aggregate metrics."""
    cfg = DEFAULT_BACKTEST_PARAMS.copy()
    if params:
        cfg.update(params)

    results = []
    for frame in _iter_frames(feature_frames):
        for test_frame in _walk_forward_slices(frame, int(cfg.get("walk_forward_folds", 1))):
            try:
                results.append(run_single_backtest(test_frame, cfg, initial_cash, commission, plot=False))
            except Exception as exc:
                results.append(
                    {
                        "ticker": frame.attrs.get("ticker", ""),
                        "error": str(exc),
                        "bars": len(test_frame),
                        "total_return_pct": -100.0,
                        "sharpe_ratio": -5.0,
                        "sortino_ratio": -5.0,
                        "max_drawdown_pct": 100.0,
                        "total_trades": 0,
                        "won_trades": 0,
                        "win_rate_pct": 0.0,
                        "turnover": 1.0,
                    }
                )

    if not results:
        raise ValueError("no backtest results were produced")

    total_trades = sum(r["total_trades"] for r in results)
    won_trades = sum(r.get("won_trades", 0) for r in results)
    bars = sum(r.get("bars", 0) for r in results)
    risk_adjusted = [
        np.nanmean([r.get("sharpe_ratio", 0.0), r.get("sortino_ratio", 0.0)])
        for r in results
    ]

    aggregate = {
        "runs": results,
        "run_count": len(results),
        "total_return_pct": float(np.nanmean([r["total_return_pct"] for r in results])),
        "sharpe_ratio": float(np.nanmean([r["sharpe_ratio"] for r in results])),
        "sortino_ratio": float(np.nanmean([r["sortino_ratio"] for r in results])),
        "risk_adjusted_return": float(np.nanmean(risk_adjusted)),
        "max_drawdown_pct": float(np.nanmax([r["max_drawdown_pct"] for r in results])),
        "total_trades": int(total_trades),
        "win_rate_pct": float(won_trades / total_trades * 100.0) if total_trades else 0.0,
        "turnover": float(total_trades / max(1, bars)),
    }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid feature-frame backtests.")
    parser.add_argument("--tickers", nargs="+", default=["2454.TW", "2330.TW", "AAPL", "NVDA"])
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.75)
    parser.add_argument("--stop-loss-pct", type=float, default=0.075)
    parser.add_argument("--risk-per-trade", type=float, default=0.015)
    parser.add_argument("--exit-ma-period", type=int, default=20)
    parser.add_argument("--folds", type=int, default=1)
    args = parser.parse_args()

    frames = {}
    for idx, ticker in enumerate(args.tickers, start=1):
        print(f"[BACKTEST] ({idx}/{len(args.tickers)}) building features for {ticker}...", flush=True)
        frames[ticker] = build_feature_frame(ticker, start=args.start, end=args.end, model_path=args.model_path)
        print(f"[BACKTEST] ({idx}/{len(args.tickers)}) {ticker}: {len(frames[ticker])} bars ready", flush=True)
    print("[BACKTEST] running walk-forward backtest...", flush=True)
    metrics = run_walk_forward_backtest(
        frames,
        {
            "score_threshold": args.score_threshold,
            "stop_loss_pct": args.stop_loss_pct,
            "risk_per_trade": args.risk_per_trade,
            "exit_ma_period": args.exit_ma_period,
            "walk_forward_folds": args.folds,
        },
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
