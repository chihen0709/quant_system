"""Fundamental scoring helpers shared by scanners and reports."""

from __future__ import annotations


def calc_fundamental_score(snapshot: dict, is_us: bool = False) -> float:
    score = 0.0
    syoy = snapshot.get("single_month_yoy")
    smom = snapshot.get("single_month_mom")
    eq = snapshot.get("eps_latest_quarter")
    ettm = snapshot.get("eps_ttm")
    profit_margin = snapshot.get("profit_margin_pct")
    earnings_growth = snapshot.get("earnings_growth_pct")

    if syoy is not None:
        score += 25 if syoy >= 30 else (18 if syoy >= 15 else (10 if syoy >= 5 else (-10 if syoy < 0 else 0)))
    if smom is not None:
        score += 16 if smom >= 20 else (10 if smom >= 5 else (5 if smom >= 0 else -6))
    if eq is not None:
        score += 18 if eq >= 20 else (14 if eq >= 10 else (8 if eq > 0 else -8))
    if ettm is not None:
        score += 20 if ettm >= 40 else (14 if ettm >= 20 else (8 if ettm > 0 else -8))
    if is_us and profit_margin is not None:
        score += 16 if profit_margin >= 20 else (10 if profit_margin >= 10 else (5 if profit_margin > 0 else -8))
    if is_us and earnings_growth is not None:
        score += 14 if earnings_growth >= 25 else (9 if earnings_growth >= 5 else (3 if earnings_growth > 0 else -6))
    if is_us and score < 30 and (syoy is not None or ettm is not None):
        score += 20
    return max(0.0, min(float(score), 100.0))

