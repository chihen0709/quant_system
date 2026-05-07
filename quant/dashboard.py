"""HTML dashboard rendering for Telegram scan summaries."""

from __future__ import annotations

import html
import os
from typing import Any

from playwright.sync_api import sync_playwright


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        if value is None:
            return "N/A"
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return "N/A"


def _tone_class(value: float | None, good_threshold: float = 0.0) -> str:
    if value is None:
        return "neutral"
    return "pos" if value >= good_threshold else "neg"


def render_top_ranked_dashboard(
    ranked: list[dict[str, Any]],
    output_path: str,
    *,
    region: str,
    market_mode: str,
) -> str | None:
    if not ranked:
        return None
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = []
    for idx, item in enumerate(ranked[:10], start=1):
        tech_pack = item.get("tech_pack") or {}
        metrics = tech_pack.get("metrics") or {}
        profile = item.get("profile_info") or {}
        fin = item.get("fin_data") or {}
        sector = item.get("sector_info") or {}
        tags = list(tech_pack.get("strategy_tags") or []) + list(item.get("sector_tags") or [])
        tag_text = " / ".join(tags[:4]) if tags else "-"
        total_score = float(item.get("total_score", 0.0))
        tech_score = float(tech_pack.get("technical_score", 0.0))
        sector_score = sector.get("sector_strength_score")
        yoy = fin.get("single_month_yoy")
        chip = fin.get("total_5d") if region.upper() == "TW" else fin.get("institutional_ownership_pct")

        rows.append(
            f"""
            <tr>
              <td class="rank">{idx}</td>
              <td class="ticker">{html.escape(str(item.get("ticker", "")))}</td>
              <td>{html.escape(str(profile.get("industry", "N/A")))}</td>
              <td class="score">{_fmt(total_score)}</td>
              <td>{_fmt(tech_score)}</td>
              <td class="{_tone_class(yoy)}">{_fmt(yoy, 1, "%")}</td>
              <td class="{_tone_class(chip)}">{_fmt(chip, 1)}</td>
              <td>{_fmt(sector_score)}</td>
              <td class="tags">{html.escape(tag_text)}</td>
            </tr>
            """
        )

    html_content = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{
          margin: 0;
          padding: 24px;
          width: 1180px;
          font-family: "Noto Sans CJK TC", "Microsoft JhengHei", "Arial", sans-serif;
          background: #f4f6f8;
          color: #172033;
        }}
        .wrap {{
          background: #ffffff;
          border-radius: 10px;
          padding: 22px;
          box-shadow: 0 8px 28px rgba(20, 32, 50, 0.10);
        }}
        .header {{
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          margin-bottom: 16px;
          border-bottom: 1px solid #e5e9f0;
          padding-bottom: 12px;
        }}
        h1 {{
          margin: 0;
          font-size: 26px;
          letter-spacing: 0;
        }}
        .mode {{
          color: #5e6b7f;
          font-size: 14px;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          font-size: 14px;
        }}
        th {{
          text-align: left;
          color: #5e6b7f;
          background: #f8fafc;
          padding: 10px 9px;
          border-bottom: 2px solid #dce3ec;
        }}
        td {{
          padding: 11px 9px;
          border-bottom: 1px solid #edf1f5;
          vertical-align: middle;
        }}
        .rank {{
          width: 38px;
          font-weight: 700;
          color: #49627a;
        }}
        .ticker {{
          font-weight: 800;
          color: #111827;
        }}
        .score {{
          font-weight: 800;
          color: #b42318;
        }}
        .pos {{
          color: #b42318;
          background: #fff1f0;
        }}
        .neg {{
          color: #176b45;
          background: #ecfdf3;
        }}
        .neutral {{
          color: #5e6b7f;
        }}
        .tags {{
          color: #374151;
          max-width: 280px;
        }}
      </style>
    </head>
    <body>
      <div class="wrap" id="capture-area">
        <div class="header">
          <h1>{html.escape(region.upper())} Top 10 Quant Dashboard</h1>
          <div class="mode">Mode: {html.escape(market_mode)} | CTA / VCP / BB / Sector</div>
        </div>
        <table>
          <thead>
            <tr>
              <th>#</th><th>Ticker</th><th>Industry</th><th>Total</th><th>Tech</th>
              <th>YoY</th><th>Chip/Inst</th><th>Sector</th><th>Tags</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </body>
    </html>
    """

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1228, "height": 780})
            page.set_content(html_content)
            page.locator("#capture-area").screenshot(path=output_path, omit_background=True)
            browser.close()
        return output_path
    except Exception:
        return None

