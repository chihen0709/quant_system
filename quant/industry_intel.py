from __future__ import annotations

import importlib
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from quant import data_sources
from quant.industry_map import get_topic_map


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

INDUSTRY_INTEL_TTL_MINUTES = int(os.environ.get('INDUSTRY_INTEL_TTL_MINUTES', '360'))
INDUSTRY_INTEL_MAX_COMPANIES = int(os.environ.get('INDUSTRY_INTEL_MAX_COMPANIES', '8'))

OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', '').strip()
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', '').strip()
OLLAMA_TIMEOUT_SEC = float(os.environ.get('OLLAMA_TIMEOUT_SEC', '120'))
OLLAMA_KEEP_ALIVE = os.environ.get('OLLAMA_KEEP_ALIVE', '15m').strip() or '15m'


def _now_text() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, '', 'N/A', 'nan'):
            return None
        return float(value)
    except Exception:
        return None


def _ticker_code(ticker: str) -> str:
    text = str(ticker or '').upper().strip()
    match = re.search(r'(\d{4,6})', text)
    return match.group(1) if match else text


def _ticker_candidates(ticker: str) -> list[str]:
    text = str(ticker or '').upper().strip()
    code = _ticker_code(text)
    out = []
    for item in [text, code, f'{code}.TW' if code else '', f'{code}.TWO' if code else '']:
        if item and item not in out:
            out.append(item)
    return out


def _resolve_tw_ticker_without_ranked(ticker: str) -> str:
    text = str(ticker or '').upper().strip()
    code = _ticker_code(text)
    if not re.fullmatch(r'\d{4}', code):
        return text
    if text.endswith(('.TW', '.TWO')):
        return text

    try:
        qp = _load_quant_pro()
        mapping = qp.qpro_fetch_tw_stock_name_map(force=False)
        tw_key = f'{code}.TW'
        two_key = f'{code}.TWO'
        if tw_key in mapping and two_key not in mapping:
            return tw_key
        if two_key in mapping and tw_key not in mapping:
            return two_key
        return qp.normalize_ticker(code)
    except Exception:
        return f'{code}.TW'


def _intel_cache_path(topic_key: str) -> Path:
    safe_key = re.sub(r'[^a-z0-9_-]+', '_', str(topic_key or '').strip().lower())
    return DATA_DIR / f'industry_intel_{safe_key}.json'


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def _save_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def _parse_ts(text: str | None) -> datetime | None:
    try:
        return datetime.strptime(str(text or '').strip(), '%Y-%m-%d %H:%M:%S')
    except Exception:
        return None


def _is_fresh(payload: dict[str, Any], ttl_minutes: int) -> bool:
    if ttl_minutes <= 0:
        return False
    generated_at = _parse_ts(payload.get('generated_at'))
    if generated_at is None:
        return False
    return datetime.now() - generated_at <= timedelta(minutes=ttl_minutes)


def load_ranked(region: str = 'tw') -> dict[str, Any]:
    return _load_json(
        DATA_DIR / f'latest_ranked_{region.lower()}.json',
        {'region': region.upper(), 'updated_at': 'N/A', 'rows': []},
    )


def load_cached_industry_intel(topic: str, max_age_minutes: int | None = None) -> dict[str, Any] | None:
    topic_data = get_topic_map(topic)
    if not topic_data:
        return None
    path = _intel_cache_path(topic_data.get('key') or topic)
    payload = _load_json(path, None)
    if not isinstance(payload, dict):
        return None
    ttl_minutes = INDUSTRY_INTEL_TTL_MINUTES if max_age_minutes is None else int(max_age_minutes)
    if ttl_minutes > 0 and not _is_fresh(payload, ttl_minutes):
        return None
    return payload


def _build_rank_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        ticker = str(row.get('ticker') or '').upper().strip()
        code = _ticker_code(ticker)
        if ticker:
            lookup[ticker] = row
        if code:
            lookup[code] = row
    return lookup


def _iter_topic_companies(topic_data: dict[str, Any], ranked_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _build_rank_lookup(ranked_rows)
    companies: list[dict[str, Any]] = []
    seen: set[str] = set()

    for lane in topic_data.get('lanes', []) or []:
        for company in lane.get('companies', []) or []:
            symbol = str(company.get('ticker') or company.get('symbol') or '').strip()
            if not symbol:
                continue

            row = None
            for candidate in _ticker_candidates(symbol):
                row = lookup.get(candidate)
                if row:
                    break

            ticker = str((row or {}).get('ticker') or symbol).strip()
            if not row:
                ticker = _resolve_tw_ticker_without_ranked(ticker)
            code = _ticker_code(ticker)
            if code in seen:
                continue
            seen.add(code)

            companies.append({
                'ticker': ticker,
                'code': code,
                'name': company.get('name') or (row or {}).get('company') or ticker,
                'role': company.get('role') or '',
                'focus': company.get('focus') or '',
                'thesis': company.get('thesis') or '',
                'lane_key': lane.get('key') or '',
                'lane_title': lane.get('title') or '',
                'lane_description': lane.get('description') or '',
                'ranked_row': row or {},
            })

    companies.sort(
        key=lambda item: (
            1 if item.get('ranked_row') else 0,
            1 if (item.get('ranked_row') or {}).get('is_priority') else 0,
            _safe_float((item.get('ranked_row') or {}).get('total_score')) or 0.0,
        ),
        reverse=True,
    )
    return companies


def _load_quant_pro():
    return importlib.import_module('quant_pro')


def _maybe_yf_info(ticker: str) -> dict[str, Any] | None:
    text = str(ticker or '').strip().upper()
    if text.endswith('.TW') or text.endswith('.TWO') or _ticker_code(text).isdigit():
        return None
    try:
        return data_sources.get_yahoo_info(text, ttl_hours=12)
    except Exception:
        return None


def _build_company_signals(company: dict[str, Any]) -> tuple[list[str], list[str], str]:
    ranked = company.get('ranked_row') or {}
    total_score = _safe_float(ranked.get('total_score'))
    tech_score = _safe_float(ranked.get('tech_score'))
    yoy = _safe_float(company.get('single_month_yoy'))
    eps_ttm = _safe_float(company.get('eps_ttm'))
    chip_5d = _safe_float(company.get('total_5d'))

    positives: list[str] = []
    risks: list[str] = []

    if total_score is not None and total_score >= 85:
        positives.append('量化總分維持高檔，屬於目前主題內優先追蹤對象。')
    elif total_score is not None and total_score >= 75:
        positives.append('量化總分中高，已有趨勢訊號但還要觀察延續性。')
    elif total_score is not None:
        risks.append('量化總分尚未站上強勢門檻，容易受題材輪動影響。')

    if tech_score is not None and tech_score >= 80:
        positives.append('技術面分數偏強，代表價格與趨勢結構相對健康。')

    if yoy is not None and yoy >= 20:
        positives.append(f'單月營收年增約 {yoy:.1f}%，基本面動能偏正向。')
    elif yoy is not None and yoy < 0:
        risks.append(f'單月營收年增為 {yoy:.1f}%，需求擴散仍需確認。')

    if eps_ttm is not None and eps_ttm > 0:
        positives.append(f'TTM EPS 約 {eps_ttm:.2f}，獲利基底仍在。')
    elif eps_ttm is not None and eps_ttm <= 0:
        risks.append('TTM EPS 未轉正，題材想像需要更多營運驗證。')

    if chip_5d is not None and chip_5d > 0:
        positives.append(f'近 5 日法人合計偏多，約買超 {chip_5d:.0f} 張。')
    elif chip_5d is not None and chip_5d < 0:
        risks.append(f'近 5 日法人偏空，約賣超 {abs(chip_5d):.0f} 張。')

    if not positives:
        positives.append('目前偏向等待更多量價與基本面同步轉強的訊號。')
    if not risks:
        risks.append('短線仍要留意題材過熱、估值先跑與訂單遞延風險。')

    summary = positives[0]
    return positives[:4], risks[:4], summary


def _collect_company_intel(company: dict[str, Any]) -> dict[str, Any]:
    qp = _load_quant_pro()
    ticker = str(company.get('ticker') or '').strip()
    code = str(company.get('code') or _ticker_code(ticker)).strip()
    ranked = company.get('ranked_row') or {}

    yf_info = _maybe_yf_info(ticker)
    profile = qp.get_company_profile(code or ticker, ticker_full=ticker, yf_info=yf_info)
    snapshot = qp.merge_financial_snapshot(ticker or code, profile.get('raw_text'), yf_info=yf_info)

    payload = {
        'ticker': ticker,
        'code': code,
        'company': company.get('name') or profile.get('company_name') or ticker,
        'lane_title': company.get('lane_title') or '',
        'role': company.get('role') or '',
        'focus': company.get('focus') or '',
        'thesis': company.get('thesis') or '',
        'industry': ranked.get('industry') or profile.get('industry') or '',
        'profile': profile.get('profile') or '',
        'close': ranked.get('close'),
        'total_score': ranked.get('total_score'),
        'tech_score': ranked.get('tech_score'),
        'fund_score': ranked.get('fund_score'),
        'chip_score': ranked.get('chip_score'),
        'priority_score': ranked.get('priority_score'),
        'estimated_win_rate_pct': ranked.get('estimated_win_rate_pct'),
        'strategy': ranked.get('strategy'),
        'is_priority': bool(ranked.get('is_priority')),
        'tags': ranked.get('tags') or [],
        'single_month_revenue': snapshot.get('single_month_revenue'),
        'single_month_mom': snapshot.get('single_month_mom'),
        'single_month_yoy': snapshot.get('single_month_yoy'),
        'eps_latest_quarter': snapshot.get('eps_latest_quarter'),
        'eps_ttm': snapshot.get('eps_ttm'),
        'chips_summary': snapshot.get('chips_summary') or '',
        'foreign_5d': snapshot.get('foreign_5d'),
        'trust_5d': snapshot.get('trust_5d'),
        'dealer_5d': snapshot.get('dealer_5d'),
        'total_5d': snapshot.get('total_5d'),
        'sources': snapshot.get('sources') or snapshot.get('source') or [],
    }
    positives, risks, summary = _build_company_signals(payload | {'ranked_row': ranked})
    payload['bull_signals'] = positives
    payload['risk_flags'] = risks
    payload['summary'] = summary
    return payload


def _build_rule_based_topic_summary(topic_data: dict[str, Any], intel_rows: list[dict[str, Any]], all_companies: list[dict[str, Any]]) -> dict[str, Any]:
    ranked_rows = [row for row in intel_rows if row.get('total_score') is not None]
    ranked_rows.sort(key=lambda item: _safe_float(item.get('total_score')) or 0.0, reverse=True)
    leaders = ranked_rows[:3]

    positive_yoy = [row for row in intel_rows if (_safe_float(row.get('single_month_yoy')) or -999) > 0]
    positive_chip = [row for row in intel_rows if (_safe_float(row.get('total_5d')) or 0) > 0]

    highlights = [
        f"主題共追蹤 {len(all_companies)} 檔，這次深入解析 {len(intel_rows)} 檔。",
        f"其中 {len(ranked_rows)} 檔有最新量化分數，{sum(1 for row in intel_rows if row.get('is_priority'))} 檔屬於 priority 名單。",
        f"已解析公司裡有 {len(positive_yoy)} 檔營收年增為正、{len(positive_chip)} 檔近 5 日法人偏多。",
    ]

    top_names = [f"{row.get('ticker')} {row.get('company')}" for row in leaders]
    if top_names:
        highlights.append('主題內目前量化分數最突出的公司：' + '、'.join(top_names) + '。')

    risks = [
        '主題股若只剩少數龍頭有分數，其餘公司沒有同步進榜，通常代表資金擴散還不完整。',
        '若營收動能、法人流向和價格趨勢不同步，題材很容易先漲故事再回頭修正。',
    ]

    return {
        'summary': topic_data.get('description') or '',
        'highlights': highlights,
        'risks': risks,
        'leaders': [
            {
                'ticker': row.get('ticker'),
                'company': row.get('company'),
                'total_score': row.get('total_score'),
                'reason': row.get('summary'),
            }
            for row in leaders
        ],
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or '').strip()
    if not raw:
        raise ValueError('empty model response')
    try:
        return json.loads(raw)
    except Exception:
        pass

    match = re.search(r'(\{.*\})', raw, re.DOTALL)
    if not match:
        raise ValueError('json object not found')
    return json.loads(match.group(1))


def _ollama_topic_analysis(topic_data: dict[str, Any], intel_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not OLLAMA_BASE_URL or not OLLAMA_MODEL:
        return {
            'status': 'disabled',
            'reason': '未設定 OLLAMA_BASE_URL 或 OLLAMA_MODEL，先回傳規則式分析。',
        }

    compact_rows = []
    for row in intel_rows:
        compact_rows.append({
            'ticker': row.get('ticker'),
            'company': row.get('company'),
            'lane': row.get('lane_title'),
            'role': row.get('role'),
            'total_score': row.get('total_score'),
            'tech_score': row.get('tech_score'),
            'fund_score': row.get('fund_score'),
            'chip_score': row.get('chip_score'),
            'strategy': row.get('strategy'),
            'single_month_yoy': row.get('single_month_yoy'),
            'eps_ttm': row.get('eps_ttm'),
            'total_5d': row.get('total_5d'),
            'summary': row.get('summary'),
        })

    prompt = '\n'.join([
        '請你扮演台股產業研究助理，根據提供的資料輸出 JSON。',
        '限制：只能根據提供資料推論，不要捏造沒有出現的訂單、法說或新聞。',
        '語氣：繁體中文，簡潔但具投資研究感。',
        '請輸出欄位：summary, demand_drivers, bottlenecks, leaders, risks, next_checks。',
        'leaders 請輸出陣列，每筆包含 ticker, company, reason。',
        '',
        f"主題：{topic_data.get('title')}",
        f"副標：{topic_data.get('subtitle')}",
        f"主題描述：{topic_data.get('description')}",
        f"市場命題：{(topic_data.get('deep_dive') or {}).get('market_question', '')}",
        '公司資料：',
        json.dumps(compact_rows, ensure_ascii=False),
    ])

    body = {
        'model': OLLAMA_MODEL,
        'system': '你是謹慎的台股產業分析模型，重視不確定性揭露與資料來源限制。',
        'prompt': prompt,
        'format': 'json',
        'stream': False,
        'keep_alive': OLLAMA_KEEP_ALIVE,
        'options': {
            'temperature': 0.2,
        },
    }

    try:
        response = requests.post(
            OLLAMA_BASE_URL.rstrip('/') + '/api/generate',
            json=body,
            timeout=OLLAMA_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
        parsed = _extract_json_object(payload.get('response', ''))
        return {
            'status': 'ready',
            'provider': 'ollama',
            'model': OLLAMA_MODEL,
            'generated_at': _now_text(),
            'summary': parsed.get('summary') or '',
            'demand_drivers': parsed.get('demand_drivers') or [],
            'bottlenecks': parsed.get('bottlenecks') or [],
            'leaders': parsed.get('leaders') or [],
            'risks': parsed.get('risks') or [],
            'next_checks': parsed.get('next_checks') or [],
            'raw_total_duration_ns': payload.get('total_duration'),
        }
    except Exception as exc:
        return {
            'status': 'error',
            'provider': 'ollama',
            'model': OLLAMA_MODEL,
            'generated_at': _now_text(),
            'reason': str(exc)[:240],
        }


def build_industry_intel(topic: str, region: str = 'tw', force: bool = False) -> dict[str, Any] | None:
    topic_data = get_topic_map(topic)
    if not topic_data:
        return None

    cache_path = _intel_cache_path(topic_data.get('key') or topic)
    if not force:
        cached = load_cached_industry_intel(topic_data.get('key') or topic)
        if cached:
            return cached

    ranked = load_ranked(region)
    all_companies = _iter_topic_companies(topic_data, ranked.get('rows', []))
    detailed_companies = all_companies[: max(1, INDUSTRY_INTEL_MAX_COMPANIES)]
    intel_rows = [_collect_company_intel(company) for company in detailed_companies]

    payload = {
        'topic_key': topic_data.get('key') or topic,
        'topic_title': topic_data.get('title') or topic,
        'generated_at': _now_text(),
        'source_ranked_file': f'data/latest_ranked_{region.lower()}.json',
        'analysis_window': {
            'tracked_companies': len(all_companies),
            'deep_analyzed_companies': len(intel_rows),
            'max_companies': INDUSTRY_INTEL_MAX_COMPANIES,
        },
        'topic_brief': _build_rule_based_topic_summary(topic_data, intel_rows, all_companies),
        'companies': intel_rows,
        'ai_analysis': _ollama_topic_analysis(topic_data, intel_rows),
    }
    _save_json(cache_path, payload)
    return payload
