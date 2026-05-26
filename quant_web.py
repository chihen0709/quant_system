from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from quant.industry_intel import build_industry_intel, load_cached_industry_intel
from quant.industry_map import get_topic_map, list_topics, topic_symbols


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
REPORT_DIR = BASE_DIR / 'reports'
STATIC_DIR = BASE_DIR / 'static'
TEMPLATES_DIR = BASE_DIR / 'templates'

DATA_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

app = FastAPI(title='Stock Minervini Dashboard')
app.mount('/reports', StaticFiles(directory=str(REPORT_DIR)), name='reports')
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

NAV_ITEMS = [
    {'href': '/', 'label': '首頁', 'key': 'home'},
    {'href': '/tw', 'label': '台股推薦', 'key': 'tw'},
    {'href': '/macro', 'label': '四維雷達', 'key': 'macro'},
    {'href': '/industry/ic-substrate?activeTab=ai', 'label': '產業地圖', 'key': 'industry'},
]

TOPIC_FALLBACK_IMAGES = {
    'ic-substrate': 'industry_ic_substrate.png',
    'cowos': 'industry_cowos.png',
    'silicon-photonics': 'industry_silicon_photonics.png',
    'cooling': 'industry_cooling.png',
    'pcb': 'industry_pcb.png',
}


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def load_ranked(region='tw') -> dict:
    return load_json(
        DATA_DIR / f'latest_ranked_{region.lower()}.json',
        {'region': region.upper(), 'updated_at': 'N/A', 'rows': []},
    )


def load_macro(region='tw') -> dict:
    return load_json(
        DATA_DIR / f'latest_macro_{region.lower()}.json',
        {'region': region.upper(), 'updated_at': 'N/A', 'latest_mode': 'neutral', 'latest_score': None, 'rows': []},
    )


def to_float(value):
    try:
        if value in (None, ''):
            return None
        return float(value)
    except Exception:
        return None


def ticker_code(ticker: str) -> str:
    text = str(ticker or '').upper().strip()
    return text.replace('.TW', '').replace('.TWO', '')


def ticker_candidates(ticker: str) -> list[str]:
    text = str(ticker or '').upper().strip()
    code = ticker_code(text)
    out = []
    for item in [text, code, f'{code}.TW' if code else '', f'{code}.TWO' if code else '']:
        if item and item not in out:
            out.append(item)
    return out


def fmt_num(value, digits=1, default='N/A') -> str:
    number = to_float(value)
    if number is None:
        return default
    return f'{number:.{digits}f}'


def fmt_price(value) -> str:
    number = to_float(value)
    if number is None:
        return 'N/A'
    if abs(number) >= 100:
        return f'{number:.1f}'
    return f'{number:.2f}'


def fmt_pct(value, digits=1) -> str:
    number = to_float(value)
    if number is None:
        return 'N/A'
    return f'{number:+.{digits}f}%'


def mode_label(mode: str) -> str:
    return {
        'offensive': '多方進攻',
        'defensive': '防守保守',
        'neutral': '中性觀望',
    }.get(str(mode or 'neutral'), '中性觀望')


def mode_class(mode: str) -> str:
    text = str(mode or 'neutral')
    if text not in {'offensive', 'neutral', 'defensive'}:
        return 'neutral'
    return text


def build_ranked_lookup(rows: list[dict]) -> dict[str, dict]:
    lookup = {}
    for row in rows or []:
        ticker = str(row.get('ticker') or '').upper().strip()
        code = ticker_code(ticker)
        if ticker:
            lookup[ticker] = row
        if code:
            lookup[code] = row
    return lookup


def find_stock_row(rows: list[dict], ticker: str) -> dict | None:
    for key in ticker_candidates(ticker):
        for row in rows or []:
            current = str(row.get('ticker') or '').upper().strip()
            if key == current or key == ticker_code(current):
                return row
    return None


def file_url_if_exists(directory: Path, file_name: str) -> str | None:
    if not file_name:
        return None
    path = directory / file_name
    if path.exists() and path.is_file():
        return str(file_name)
    return None


def collect_report_assets(ticker: str) -> list[dict]:
    assets = []
    seen = set()
    suffix_map = [
        ('analysis_card', '分析卡'),
        ('report', '個股報告'),
        ('strategy', '策略圖'),
        ('bot_backtest', '回測圖'),
    ]

    for candidate in ticker_candidates(ticker):
        for suffix, label in suffix_map:
            file_name = f'{candidate}_{suffix}.png'
            if file_name in seen:
                continue
            if file_url_if_exists(REPORT_DIR, file_name):
                seen.add(file_name)
                assets.append({'file_name': file_name, 'label': label, 'url': f'/reports/{file_name}'})
    return assets


def build_topic_report_asset(topic_data: dict) -> dict | None:
    key = str(topic_data.get('key') or '').strip()
    report_name = str(topic_data.get('report_image') or TOPIC_FALLBACK_IMAGES.get(key) or '').strip()
    if not report_name:
        return None
    if not file_url_if_exists(REPORT_DIR, report_name):
        return None
    return {'file_name': report_name, 'label': topic_data.get('title') or key, 'url': f'/reports/{report_name}'}


def stock_row_brief(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        'ticker': row.get('ticker'),
        'company': row.get('company'),
        'industry': row.get('industry'),
        'close': row.get('close'),
        'total_score': row.get('total_score'),
        'tech_score': row.get('tech_score'),
        'fund_score': row.get('fund_score'),
        'chip_score': row.get('chip_score'),
        'strategy': row.get('strategy'),
        'priority_score': row.get('priority_score'),
        'estimated_win_rate_pct': row.get('estimated_win_rate_pct'),
        'is_priority': bool(row.get('is_priority')),
        'tags': row.get('tags') or [],
    }


def related_topics_for_ticker(ticker: str) -> list[dict]:
    wanted = ticker_code(ticker)
    related = []
    for topic in list_topics():
        symbols = {ticker_code(symbol) for symbol in topic_symbols(topic)}
        if wanted and wanted in symbols:
            related.append({
                'key': topic.get('key'),
                'title': topic.get('title'),
                'subtitle': topic.get('subtitle'),
                'url': f"/industry/{topic.get('key')}?activeTab=ai",
            })
    return related


def build_topic_context(topic_key: str, active_tab: str, region='tw') -> dict | None:
    topic_data = get_topic_map(topic_key)
    if not topic_data:
        return None

    ranked = load_ranked(region)
    ranked_rows = ranked.get('rows', [])
    lookup = build_ranked_lookup(ranked_rows)
    tabs = topic_data.get('tabs') or [{'key': 'ai', 'label': 'AI 需求鏈', 'summary': topic_data.get('description', '')}]
    tab_keys = {str(tab.get('key')) for tab in tabs}
    current_tab = active_tab if active_tab in tab_keys else str(tabs[0].get('key'))
    active_tab_meta = next((tab for tab in tabs if str(tab.get('key')) == current_tab), tabs[0])
    tab_panels = topic_data.get('tab_panels') or {}
    active_panel = tab_panels.get(current_tab, {})

    lanes = []
    table_rows = []
    leaders = []
    seen_leaders = set()
    total_companies = 0
    in_rank_count = 0
    priority_count = 0

    for lane in topic_data.get('lanes', []) or []:
        lane_tabs = {str(item) for item in lane.get('tabs', []) if item}
        if lane_tabs and current_tab != 'leaders' and current_tab not in lane_tabs:
            continue

        companies = []
        for company in lane.get('companies', []) or []:
            total_companies += 1
            symbol = str(company.get('ticker') or company.get('symbol') or '').strip()
            row = None
            for candidate in ticker_candidates(symbol):
                row = lookup.get(candidate)
                if row:
                    break

            stock_ticker = row.get('ticker') if row else symbol
            company_entry = {
                'ticker': stock_ticker,
                'code': ticker_code(stock_ticker),
                'name': company.get('name') or (row or {}).get('company') or symbol,
                'role': company.get('role') or '',
                'focus': company.get('focus') or '',
                'thesis': company.get('thesis') or '',
                'industry': (row or {}).get('industry') or '',
                'in_rank': bool(row),
                'is_priority': bool((row or {}).get('is_priority')),
                'total_score': (row or {}).get('total_score'),
                'tech_score': (row or {}).get('tech_score'),
                'close': (row or {}).get('close'),
                'strategy': (row or {}).get('strategy'),
                'estimated_win_rate_pct': (row or {}).get('estimated_win_rate_pct'),
                'tags': (row or {}).get('tags') or [],
                'stock_url': f'/stock/{stock_ticker}',
            }
            companies.append(company_entry)
            table_rows.append({**company_entry, 'lane_title': lane.get('title') or ''})

            if row:
                in_rank_count += 1
                if company_entry['is_priority']:
                    priority_count += 1
                if stock_ticker not in seen_leaders:
                    leaders.append(company_entry)
                    seen_leaders.add(stock_ticker)

        lanes.append({
            'key': lane.get('key'),
            'title': lane.get('title'),
            'description': lane.get('description') or '',
            'notes': lane.get('notes') or [],
            'companies': companies,
        })

    leaders.sort(key=lambda item: (1 if item.get('is_priority') else 0, to_float(item.get('total_score')) or 0.0), reverse=True)
    table_rows.sort(key=lambda item: (1 if item.get('in_rank') else 0, 1 if item.get('is_priority') else 0, to_float(item.get('total_score')) or 0.0), reverse=True)

    coverage = {
        'tracked': total_companies,
        'in_rank': in_rank_count,
        'priority': priority_count,
        'updated_at': ranked.get('updated_at', 'N/A'),
    }

    return {
        'topic': topic_data,
        'tabs': tabs,
        'active_tab': current_tab,
        'active_tab_meta': active_tab_meta,
        'active_panel': active_panel,
        'lanes': lanes,
        'leaders': leaders[:8],
        'table_rows': table_rows,
        'coverage': coverage,
        'report_asset': build_topic_report_asset(topic_data),
        'deep_dive': topic_data.get('deep_dive') or {},
        'chain_steps': topic_data.get('chain_steps') or [],
        'watch_metrics': topic_data.get('watch_metrics') or [],
        'highlights': topic_data.get('highlights') or [],
        'symbol_count': len(topic_symbols(topic_data)),
        'ranked_rows': ranked_rows,
    }


def base_context(active_page: str) -> dict:
    topics = list_topics()
    return {
        'nav_items': NAV_ITEMS,
        'active_page': active_page,
        'topic_index': [
            {
                'key': topic.get('key'),
                'title': topic.get('title'),
                'subtitle': topic.get('subtitle'),
                'url': f"/industry/{topic.get('key')}?activeTab=ai",
            }
            for topic in topics
        ],
    }


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    ranked = load_ranked('tw')
    macro = load_macro('tw')
    rows = ranked.get('rows', [])
    topics = list_topics()
    macro_row = (macro.get('rows') or [{}])[0]

    context = {
        **base_context('home'),
        'request': request,
        'page_title': 'Stock Minervini Dashboard',
        'ranked': ranked,
        'macro': macro,
        'macro_row': macro_row,
        'spotlight_rows': rows[:3],
        'topic_cards': [
            {
                'key': topic.get('key'),
                'title': topic.get('title'),
                'subtitle': topic.get('subtitle'),
                'description': topic.get('description'),
                'url': f"/industry/{topic.get('key')}?activeTab=ai",
            }
            for topic in topics
        ],
        'fmt_num': fmt_num,
        'mode_label': mode_label,
    }
    return templates.TemplateResponse(request=request, name='home.html', context=context)


@app.get('/tw', response_class=HTMLResponse)
def tw_dashboard(request: Request):
    ranked = load_ranked('tw')
    rows = ranked.get('rows', [])
    top_pick = rows[0] if rows else None

    context = {
        **base_context('tw'),
        'request': request,
        'page_title': '台股推薦 Top 10',
        'ranked': ranked,
        'rows': rows[:30],
        'top_pick': top_pick,
        'priority_report': file_url_if_exists(REPORT_DIR, 'TW_priority_top10.png'),
        'top_report': file_url_if_exists(REPORT_DIR, 'TW_top10_dashboard.png'),
        'fmt_num': fmt_num,
        'fmt_price': fmt_price,
    }
    return templates.TemplateResponse(request=request, name='tw.html', context=context)


@app.get('/macro', response_class=HTMLResponse)
def macro_dashboard(request: Request):
    macro = load_macro('tw')
    rows = macro.get('rows', [])
    latest = rows[0] if rows else {}

    context = {
        **base_context('macro'),
        'request': request,
        'page_title': '四維大盤雷達',
        'macro': macro,
        'latest': latest,
        'history_rows': rows,
        'macro_report': file_url_if_exists(REPORT_DIR, 'TW_macro_dashboard.png'),
        'fmt_num': fmt_num,
        'mode_label': mode_label,
        'mode_class': mode_class,
    }
    return templates.TemplateResponse(request=request, name='macro.html', context=context)


@app.get('/stock/{ticker}', response_class=HTMLResponse)
def stock_page(request: Request, ticker: str):
    ranked = load_ranked('tw')
    rows = ranked.get('rows', [])
    target = find_stock_row(rows, ticker)
    if not target:
        context = {
            **base_context('tw'),
            'request': request,
            'page_title': ticker,
            'message': f'{ticker} 目前沒有這檔股票的最新掃描資料。',
            'back_href': '/tw',
            'back_label': '回台股推薦',
        }
        return templates.TemplateResponse(request=request, name='not_found.html', context=context, status_code=404)

    stock_ticker = str(target.get('ticker') or ticker)
    context = {
        **base_context('tw'),
        'request': request,
        'page_title': stock_ticker,
        'stock': target,
        'report_assets': collect_report_assets(stock_ticker),
        'related_topics': related_topics_for_ticker(stock_ticker),
        'fmt_num': fmt_num,
        'fmt_price': fmt_price,
    }
    return templates.TemplateResponse(request=request, name='stock.html', context=context)


@app.get('/industry/{topic}', response_class=HTMLResponse)
def industry_page(request: Request, topic: str, activeTab: str = 'ai', refreshIntel: int = 0):
    context = build_topic_context(topic, activeTab, region='tw')
    if not context:
        payload = {
            **base_context('industry'),
            'request': request,
            'page_title': topic,
            'message': f'找不到 {topic} 這個產業主題。',
            'back_href': '/',
            'back_label': '回首頁',
        }
        return templates.TemplateResponse(request=request, name='not_found.html', context=payload, status_code=404)

    intel = None if refreshIntel else load_cached_industry_intel(topic)
    if refreshIntel or intel is None:
        try:
            intel = build_industry_intel(topic, region='tw', force=bool(refreshIntel))
        except Exception as exc:
            intel = {
                'generated_at': None,
                'analysis_window': {'tracked_companies': context['coverage']['tracked'], 'deep_analyzed_companies': 0},
                'topic_brief': {'summary': '', 'highlights': [], 'risks': []},
                'companies': [],
                'ai_analysis': {'status': 'error', 'reason': str(exc)[:240]},
            }

    payload = {
        **base_context('industry'),
        'request': request,
        'page_title': context['topic'].get('title') or topic,
        'industry': context,
        'industry_intel': intel,
        'fmt_num': fmt_num,
        'fmt_price': fmt_price,
    }
    return templates.TemplateResponse(request=request, name='industry.html', context=payload)


@app.get('/api/tw/top10')
def api_tw_top10():
    ranked = load_ranked('tw')
    return JSONResponse({
        'region': 'TW',
        'updated_at': ranked.get('updated_at'),
        'source': 'data/latest_ranked_tw.json',
        'rows': ranked.get('rows', [])[:10],
    })


@app.get('/api/macro')
def api_macro():
    macro = load_macro('tw')
    macro['source'] = 'data/latest_macro_tw.json'
    return JSONResponse(macro)


@app.get('/api/stock/{ticker}')
def api_stock(ticker: str):
    ranked = load_ranked('tw')
    stock = find_stock_row(ranked.get('rows', []), ticker)
    if not stock:
        raise HTTPException(status_code=404, detail='找不到這檔股票的最新掃描資料')
    return JSONResponse({
        'updated_at': ranked.get('updated_at'),
        'stock': stock_row_brief(stock),
        'report_assets': collect_report_assets(stock.get('ticker') or ticker),
        'related_topics': related_topics_for_ticker(stock.get('ticker') or ticker),
    })


@app.get('/api/industry/{topic}')
def api_industry(topic: str, activeTab: str = 'ai'):
    context = build_topic_context(topic, activeTab, region='tw')
    if not context:
        raise HTTPException(status_code=404, detail='找不到這個產業主題')

    return JSONResponse({
        'topic': context['topic'],
        'active_tab': context['active_tab'],
        'active_tab_meta': context['active_tab_meta'],
        'coverage': context['coverage'],
        'leaders': context['leaders'],
        'table_rows': context['table_rows'],
        'watch_metrics': context['watch_metrics'],
        'chain_steps': context['chain_steps'],
    })


@app.get('/api/industry/{topic}/intel')
def api_industry_intel(topic: str, refresh: int = 0):
    topic_data = get_topic_map(topic)
    if not topic_data:
        raise HTTPException(status_code=404, detail='找不到這個產業主題')

    if refresh:
        payload = build_industry_intel(topic, region='tw', force=True)
    else:
        payload = load_cached_industry_intel(topic) or build_industry_intel(topic, region='tw', force=False)

    if not payload:
        raise HTTPException(status_code=500, detail='產業分析暫時無法建立')
    return JSONResponse(payload)
