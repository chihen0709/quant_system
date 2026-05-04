"""Risk and sizing helpers shared by scanners and backtests."""

from __future__ import annotations

import math


def calculate_position_size(
    cash: float,
    entry_price: float,
    stop_loss_pct: float,
    risk_per_trade: float = 0.015,
    lot_size: int = 1,
    max_cash_fraction: float = 0.95,
) -> int:
    """Return a risk-capped share count."""
    if cash <= 0 or entry_price <= 0 or stop_loss_pct <= 0 or risk_per_trade <= 0:
        return 0

    stop_distance = entry_price * stop_loss_pct
    risk_amount = cash * risk_per_trade
    risk_size = math.floor(risk_amount / stop_distance)
    cash_size = math.floor((cash * max_cash_fraction) / entry_price)
    size = max(0, min(risk_size, cash_size))

    if lot_size > 1:
        size = (size // lot_size) * lot_size
    return int(size)
