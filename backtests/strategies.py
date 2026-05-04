"""Hybrid Minervini strategy that only consumes precomputed signals."""

from __future__ import annotations

import math

import backtrader as bt

from quant.risk import calculate_position_size


class MinerviniHybridStrategy(bt.Strategy):
    params = (
        ("score_threshold", 0.75),
        ("risk_per_trade", 0.015),
        ("stop_loss_pct", 0.075),
        ("exit_ma_period", 20),
        ("risk_exit_threshold", 0.85),
        ("lot_size", 1),
        ("max_cash_fraction", 0.95),
        ("printlog", False),
    )

    def log(self, message: str) -> None:
        if self.p.printlog:
            dt = self.datas[0].datetime.date(0)
            print(f"[{dt.isoformat()}] {message}")

    def __init__(self):
        self.close = self.datas[0].close
        self.exit_ma = bt.indicators.SMA(self.close, period=self.p.exit_ma_period)
        self.order = None
        self.entry_price = None
        self.stop_price = None

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = float(order.executed.price)
                self.stop_price = self.entry_price * (1.0 - self.p.stop_loss_pct)
                self.log(f"BUY {order.executed.size} @ {order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL {order.executed.size} @ {order.executed.price:.2f}")
                self.entry_price = None
                self.stop_price = None
        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.log("ORDER CANCELED/MARGIN/REJECTED")
        self.order = None

    def next(self):
        if self.order:
            return

        close = float(self.close[0])
        if not math.isfinite(close) or close <= 0:
            return

        if self.position:
            risk_probability = float(self.data.risk_probability[0])
            hard_stop_hit = self.stop_price is not None and close <= self.stop_price
            exit_ma_hit = math.isfinite(float(self.exit_ma[0])) and close < float(self.exit_ma[0])
            risk_exit = math.isfinite(risk_probability) and risk_probability >= self.p.risk_exit_threshold
            if hard_stop_hit or exit_ma_hit or risk_exit:
                self.order = self.sell(size=self.position.size)
            return

        pass_filter = float(self.data.pass_minervini[0]) >= 0.5
        score = float(self.data.hybrid_score[0])
        if not pass_filter or not math.isfinite(score) or score < self.p.score_threshold:
            return

        size = calculate_position_size(
            cash=float(self.broker.get_cash()),
            entry_price=close,
            stop_loss_pct=float(self.p.stop_loss_pct),
            risk_per_trade=float(self.p.risk_per_trade),
            lot_size=int(self.p.lot_size),
            max_cash_fraction=float(self.p.max_cash_fraction),
        )
        if size > 0:
            self.order = self.buy(size=size)
