"""Small reporting helpers for the gradual quant_pro.py refactor."""

from __future__ import annotations


def score_bucket(score: float) -> str:
    if score >= 80:
        return "strong"
    if score >= 65:
        return "watch"
    return "neutral"


def format_score_summary(ticker: str, total_score: float, tech_score: float, fund_score: float, chip_score: float) -> str:
    return (
        f"{ticker} total={total_score:.1f} "
        f"tech={tech_score:.1f} fund={fund_score:.1f} chip={chip_score:.1f}"
    )

