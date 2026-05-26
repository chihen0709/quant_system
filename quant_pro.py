import os
import re
import time
import math
import json
import warnings
import threading
import subprocess
import traceback
import sys
import html
from io import StringIO
from datetime import datetime
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd
import requests
from playwright.sync_api import sync_playwright  
import schedule
import ta
import telebot
import yfinance as yf
import joblib  

from quant.chip import calc_chip_score as shared_calc_chip_score
from quant.cta import add_cta_features, row_strategy_tags
from quant import data_sources
from quant.dashboard import render_top_ranked_dashboard
from quant.fundamental import calc_fundamental_score as shared_calc_fundamental_score
from quant.industry_intel import build_industry_intel
from quant.industry_map import list_topics
from quant.patterns import add_pattern_features
from quant.sector import annotate_sector_strength

FINMIND_TOKEN = os.environ.get('FINMIND_TOKEN', '').strip()

try:
    from FinMind.data import DataLoader
    dl = DataLoader()
    if FINMIND_TOKEN:
        try:
            dl.login_by_token(api_token=FINMIND_TOKEN)
        except TypeError:
            dl.login_by_token(FINMIND_TOKEN)
except Exception as e:
    DataLoader = None
    dl = None
    print(f"[FinMind-WARN] FinMind unavailable: {e}")

# ==========================================
# Paths and settings
# ==========================================
# QPRO_FIX_20260513_2225_SCAN_POOL_RECOVERY
# QPRO_LAYOUT_FIX_20260513_1435: macro report vertical/mobile-safe layout; conclusion card full-width below charts
warnings.filterwarnings('ignore')

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK TC', 'Noto Sans CJK SC', 'Microsoft JhengHei',
    'PingFang TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans'
]

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID') or os.environ.get('CHAT_ID', '')

if not TELEGRAM_TOKEN or not CHAT_ID:
    print("⚠️ 找不到 TELEGRAM_TOKEN 或 TELEGRAM_CHAT_ID，Telegram bot will be disabled.")

if TELEGRAM_TOKEN and ':' not in TELEGRAM_TOKEN:
    print("⚠️ TELEGRAM_TOKEN format looks invalid. Telegram bot will be disabled.")
    TELEGRAM_TOKEN = ''

TEST_MODE = False
HOLD_DAYS = 20

REPORT_DIR = 'reports'
DATA_DIR = 'data'
MODEL_DIR = 'models'
CONFIG_DIR = 'config'

MY_TW_COVERAGE_PATH = os.environ.get('MY_TW_COVERAGE_PATH', './My-TW-Coverage')

MACRO_MODEL_PATH = os.path.join(MODEL_DIR, 'macro_rf_model.pkl')
PARAMS_FILE_PATH = os.path.join(CONFIG_DIR, 'best_params.json')

GIT_PULL_TIMEOUT = 30
FINANCIAL_UPDATE_TIMEOUT = 300
TELEGRAM_RETRY = 2
SCAN_PROGRESS_STEP = 100 

TECHNICAL_PRESCREEN_LIMIT = 20
FINAL_TOP_N = 10
WEIGHT_TECH = 0.50
WEIGHT_FUND = 0.35
WEIGHT_CHIP = 0.15
SCAN_DEEP_LIMIT_US = int(os.environ.get('SCAN_DEEP_LIMIT_US', '120'))
SCAN_DEEP_LIMIT_TW = int(os.environ.get('SCAN_DEEP_LIMIT_TW', '120'))
# QPRO_FIX_20260513_2325_SCAN_FAST_FULL: /scan 快掃、/scan_full 全市場。
# QPRO_FIX_20260513_2340_SCAN_LIMIT_OFFICIAL_FIRST: /scan 真正限制 500 檔，台股 K 線官方優先。
TW_SCAN_POOL_LIMIT = int(os.environ.get('TW_SCAN_POOL_LIMIT', '500'))
TW_FULL_SCAN_POOL_LIMIT = int(os.environ.get('TW_FULL_SCAN_POOL_LIMIT', '0'))  # 0 = 不限制
LAST_SCAN_RANKED = {}
LAST_MACRO_DATA = {}

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN and ':' in TELEGRAM_TOKEN else None

# ==========================================
# Telegram sender
# ==========================================
def safe_send_message(chat_id, text, parse_mode=None):
    if not bot or not chat_id: return False
    for _ in range(TELEGRAM_RETRY):
        try:
            bot.send_message(chat_id, text, parse_mode=parse_mode)
            return True
        except Exception as e:
            log(f"[Telegram 傳送警告] {e}")
            time.sleep(2)
            
    if parse_mode:
        try:
            bot.send_message(chat_id, text, parse_mode=None)
            log("[Telegram] 成功以純文字降級發送。")
            return True
        except Exception as e: log(f"[Telegram 降級傳送失敗] {e}")
    return False

def safe_send_photo(chat_id, photo_path, caption=None, parse_mode=None):
    if not bot or not chat_id or not os.path.exists(photo_path): return False
    for _ in range(TELEGRAM_RETRY):
        try:
            with open(photo_path, 'rb') as f: bot.send_photo(chat_id, f, caption=caption, parse_mode=parse_mode)
            return True
        except Exception as e: 
            log(f"[Telegram 圖片傳送警告] {e}")
            time.sleep(2)
    return False

def safe_reply_to(message, text, parse_mode=None):
    if not bot: return False
    try: bot.reply_to(message, text, parse_mode=parse_mode); return True
    except Exception: return False

# ==========================================
# Configuration loader
# ==========================================
def load_best_params():
    default_params = {
        'hard_stop': 0.075,
        'ma_period': 20,
        'tech_weight': WEIGHT_TECH,
        'fund_weight': WEIGHT_FUND,
        'chip_weight': WEIGHT_CHIP,
        'vcp_weight': 0.15,
        'bb_weight': 0.10,
        'ml_weight': 0.25,
        'score_threshold': 0.75,
        'stop_loss_pct': 0.075,
        'risk_per_trade': 0.015,
        'exit_ma_period': 20,
        'model_path': None,
    }
    if os.path.exists(PARAMS_FILE_PATH):
        try:
            with open(PARAMS_FILE_PATH, 'r') as f:
                loaded = json.load(f)
            default_params.update(loaded)
            return default_params
        except Exception as e: log(f"[CONFIG-WARN] 讀取參數失敗: {e}")
    return default_params

SYS_PARAMS = load_best_params()

# ==========================================
# Market monitor
# ==========================================
def _download_yf_df(ticker, period='6mo'):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    except Exception as e:
        log_exception(f'[YF-DATA-ERROR] {ticker}', e)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    return df.dropna(how='all').copy()


def _download_macro_index_df(region='TW', period='6mo'):
    index_ticker = '^TWII' if region == 'TW' else '^GSPC'
    df = _download_yf_df(index_ticker, period=period)

    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame()

    df['MA20'] = ta.trend.sma_indicator(df['Close'], 20)
    df['MA50'] = ta.trend.sma_indicator(df['Close'], 50)
    df['RET'] = df['Close'].pct_change() * 100

    if 'Volume' in df.columns:
        df['VOL_MA20'] = df['Volume'].rolling(20).mean()
        df['VOL_RATIO'] = np.where(df['VOL_MA20'] > 0, df['Volume'] / df['VOL_MA20'], np.nan)
    else:
        df['VOL_MA20'] = np.nan
        df['VOL_RATIO'] = np.nan

    return df


def _download_vix_series(period='2mo'):
    df = _download_yf_df('^VIX', period=period)
    if df.empty or 'Close' not in df.columns:
        return pd.Series(dtype=float)
    return df['Close'].dropna()


def _date_range(days=45):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    return start_date, end_date


def _as_date_col(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    x = df.copy()
    if 'date' not in x.columns:
        for c in x.columns:
            if str(c).lower() in ('日期', 'datetime', 'trade_date', 'trading_date'):
                x = x.rename(columns={c: 'date'})
                break
    if 'date' in x.columns:
        x['date'] = pd.to_datetime(x['date'], errors='coerce')
        x = x.dropna(subset=['date']).sort_values('date')
    return x


def _find_column(df, include_keywords, exclude_keywords=None):
    if df is None or df.empty:
        return None
    exclude_keywords = exclude_keywords or []
    for c in df.columns:
        name = str(c).lower()
        if all(k.lower() in name for k in include_keywords) and not any(k.lower() in name for k in exclude_keywords):
            return c
    return None




def _pick_first_column(df, candidates, exclude_keywords=None):
    """Pick the first matching column from mixed FinMind / Goodinfo / TWSE schemas.

    The project pulls data from several Taiwan data sources whose column names
    differ by API version and language.  This helper accepts a list of aliases
    and returns the first column that matches by normalized exact/partial match.
    """
    if df is None or df.empty:
        return None

    exclude_keywords = exclude_keywords or []

    def _norm_col_name(value):
        return (
            str(value)
            .strip()
            .lower()
            .replace(' ', '')
            .replace('_', '')
            .replace('-', '')
            .replace('\n', '')
            .replace('\r', '')
            .replace('(', '')
            .replace(')', '')
            .replace('（', '')
            .replace('）', '')
        )

    cols = list(df.columns)
    cand_norms = [_norm_col_name(c) for c in candidates]
    excl_norms = [_norm_col_name(e) for e in exclude_keywords]

    def _excluded(col_norm):
        return any(e and e in col_norm for e in excl_norms)

    # 1) Exact normalized match.
    for col in cols:
        col_norm = _norm_col_name(col)
        if col_norm in cand_norms and not _excluded(col_norm):
            return col

    # 2) Candidate is contained in column name.
    for col in cols:
        col_norm = _norm_col_name(col)
        if any(c and c in col_norm for c in cand_norms) and not _excluded(col_norm):
            return col

    # 3) Column name is contained in candidate alias.
    for col in cols:
        col_norm = _norm_col_name(col)
        if any(c and col_norm in c for c in cand_norms if col_norm) and not _excluded(col_norm):
            return col

    return None

def _numeric_series(df, col):
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors='coerce')


def _call_finmind(method_names, **kwargs):
    if dl is None:
        return pd.DataFrame()
    for method_name in method_names:
        try:
            fn = getattr(dl, method_name, None)
            if fn is None:
                continue
            df = fn(**kwargs)
            if df is not None and not df.empty:
                return _as_date_col(df)
        except TypeError:
            try:
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in ('start_date', 'end_date', 'stock_id')}
                df = fn(**filtered_kwargs)
                if df is not None and not df.empty:
                    return _as_date_col(df)
            except Exception:
                continue
        except Exception:
            continue
    return pd.DataFrame()


def _latest_by_trade_dates(series_df, trade_dates, value_col='value', display_func=None, default_display='N/A'):
    out = []
    if series_df is None or series_df.empty or 'date' not in series_df.columns or value_col not in series_df.columns:
        for _ in trade_dates:
            out.append({'value': None, 'display': default_display, 'score': 0.0})
        return out

    x = series_df.copy()
    x['date'] = pd.to_datetime(x['date'], errors='coerce')
    x = x.dropna(subset=['date']).sort_values('date')
    x[value_col] = pd.to_numeric(x[value_col], errors='coerce')

    for d in trade_dates:
        d = pd.to_datetime(d)
        part = x[x['date'] <= d]
        value = None if part.empty else safe_float(part.iloc[-1][value_col])
        display = default_display if value is None else (display_func(value) if display_func else f'{value:.2f}')
        score = 0.0
        if value is not None and 'score' in x.columns:
            score = safe_float(part.iloc[-1].get('score'), 0.0) or 0.0
        out.append({'value': value, 'display': display, 'score': score})
    return out


def _macro_values_by_trade_dates(series_df, trade_dates, value_col='value', display_func=None, default_display='N/A', source_label='宏觀資料', carry_forward=False, repeated_policy='latest_only'):
    """Align macro dimensions to trade dates without creating fake history.

    The old `_latest_by_trade_dates()` forward-filled the latest available value.
    That is fine for some slow-moving indicators, but it is misleading for daily
    TAIFEX/TWSE dimensions when an endpoint only returns a latest snapshot or the
    HTML parser accidentally reads the same row for multiple query dates.

    repeated_policy='latest_only': if all fetched values are identical, show the
    value only on the latest dashboard row and render older rows as N/A. This
    avoids the false impression that 十大交易人 / 韭菜指數 were unchanged every day.
    """
    out = []
    if series_df is None or series_df.empty or 'date' not in series_df.columns or value_col not in series_df.columns:
        for _ in trade_dates:
            out.append({'value': None, 'display': default_display, 'score': 0.0, 'stale': False})
        return out

    x = series_df.copy()
    x['date'] = pd.to_datetime(x['date'], errors='coerce').dt.normalize()
    x = x.dropna(subset=['date']).sort_values('date')
    x[value_col] = pd.to_numeric(x[value_col], errors='coerce')
    x = x.dropna(subset=[value_col])

    if x.empty:
        for _ in trade_dates:
            out.append({'value': None, 'display': default_display, 'score': 0.0, 'stale': False})
        return out

    vals = pd.to_numeric(x[value_col], errors='coerce').dropna()
    repeated = len(vals) >= 3 and vals.nunique(dropna=True) == 1
    latest_trade_date = pd.to_datetime(max(trade_dates)).normalize() if trade_dates else None
    if repeated and repeated_policy == 'latest_only':
        try:
            log(f'[DATA-WARN] {source_label} values are identical across fetched rows; showing latest snapshot only to avoid fake daily history.')
        except Exception:
            pass

    for d in trade_dates:
        d_norm = pd.to_datetime(d).normalize()
        value = None
        score = 0.0
        stale = False

        if repeated and repeated_policy == 'latest_only' and latest_trade_date is not None and d_norm != latest_trade_date:
            out.append({'value': None, 'display': default_display, 'score': 0.0, 'stale': True})
            continue

        if carry_forward:
            part = x[x['date'] <= d_norm]
            if not part.empty:
                row = part.iloc[-1]
                value = safe_float(row.get(value_col))
                if pd.to_datetime(row.get('date')).normalize() != d_norm:
                    stale = True
        else:
            part = x[x['date'] == d_norm]
            if not part.empty:
                row = part.iloc[-1]
                value = safe_float(row.get(value_col))

        if value is not None and 'score' in x.columns:
            score = safe_float(row.get('score'), 0.0) or 0.0
        display = default_display if value is None else (display_func(value) if display_func else f'{value:.2f}')
        out.append({'value': value, 'display': display, 'score': score, 'stale': stale})
    return out


def _normalize_score(score):
    return max(-1.0, min(1.0, safe_float(score, 0.0) or 0.0))


def _score_label(score):
    score = safe_float(score, 0.0) or 0.0
    if score >= 0.25:
        return '偏多'
    if score <= -0.25:
        return '偏空'
    return '中性'


def _score_class(score):
    return 'pos' if (safe_float(score, 0.0) or 0.0) >= 0 else 'neg'


def _td_class_by_value(value):
    try:
        if isinstance(value, str) and value.startswith('+'):
            return 'pos'
        if isinstance(value, str) and value.startswith('-'):
            return 'neg'
        fv = float(str(value).replace('%', '').replace('+', '').replace(',', ''))
        return 'pos' if fv >= 0 else 'neg'
    except Exception:
        return ''


def _format_signed_number(value, digits=0):
    if value is None:
        return 'N/A'
    try:
        return f'{float(value):+,.{digits}f}'
    except Exception:
        return 'N/A'


def _format_ratio(value, digits=1):
    if value is None:
        return 'N/A'
    try:
        return f'{float(value):.{digits}f}'
    except Exception:
        return 'N/A'


def _score_by_threshold(value, pos_threshold, neg_threshold, pos_score=0.25, neg_score=-0.25):
    value = safe_float(value)
    if value is None:
        return 0.0
    if value >= pos_threshold:
        return pos_score
    if value <= neg_threshold:
        return neg_score
    return 0.0


# ==========================================
# Official TW macro data helpers
# Priority for Macro Wave Engine:
#   TWSE T86 spot flow -> TAIFEX futures/PCR/large-trader -> FinMind fallback
# ==========================================
def _official_headers():
    return {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125 Safari/537.36',
        'Accept': 'application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8',
        'Referer': 'https://www.twse.com.tw/',
    }


def _fetch_text(url, params=None, timeout=20):
    try:
        r = requests.get(url, params=params, headers=_official_headers(), timeout=timeout)
        if r.status_code != 200 or not r.text:
            return ''
        r.encoding = r.apparent_encoding or 'utf-8'
        return r.text
    except Exception:
        return ''


def _flatten_table_columns(df):
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [' '.join(str(i).strip() for i in c if str(i).strip() not in ('', 'nan', 'None')) for c in x.columns]
    else:
        x.columns = [str(c).strip() for c in x.columns]
    x.columns = [re.sub(r'\s+', '', str(c).replace('\n', '').replace('\r', '')) for c in x.columns]
    return x


def _read_html_tables_from_text(text):
    if not text:
        return []
    try:
        return [_flatten_table_columns(t) for t in pd.read_html(StringIO(text))]
    except Exception:
        return []


def _clean_num(v):
    if v is None:
        return None
    s = str(v).replace(',', '').replace('%', '').replace('−', '-').replace('－', '-').strip()
    m = re.search(r'[-+]?\d+(?:\.\d+)?', s)
    return float(m.group(0)) if m else None


def _official_date_iter(start_date=None, end_date=None, calendar_days=35):
    if end_date:
        end = pd.to_datetime(end_date, errors='coerce')
        if pd.isna(end):
            end = pd.Timestamp.today().normalize()
    else:
        end = pd.Timestamp.today().normalize()

    if start_date:
        start = pd.to_datetime(start_date, errors='coerce')
        if pd.isna(start):
            start = end - pd.Timedelta(days=calendar_days)
    else:
        start = end - pd.Timedelta(days=calendar_days)

    cur = end.normalize()
    while cur >= start.normalize():
        # skip weekends to reduce failed official requests
        if cur.weekday() < 5:
            yield cur
        cur -= pd.Timedelta(days=1)


def _find_col_contains(df, *keywords, exclude=()):
    if df is None or df.empty:
        return None
    for c in df.columns:
        cs = str(c)
        if all(k in cs for k in keywords) and not any(e in cs for e in exclude):
            return c
    return None


def _twse_t86_json(date_yyyymmdd):
    # TWSE current endpoint. Old /fund/T86 and new /rwd/zh/fund/T86 are both tried.
    endpoints = [
        'https://www.twse.com.tw/rwd/zh/fund/T86',
        'https://www.twse.com.tw/fund/T86',
    ]
    params_list = [
        {'response': 'json', 'date': date_yyyymmdd, 'type': 'ALLBUT0999'},
        {'response': 'json', 'date': date_yyyymmdd, 'selectType': 'ALLBUT0999'},
    ]
    for url in endpoints:
        for params in params_list:
            try:
                r = requests.get(url, params=params, headers=_official_headers(), timeout=20)
                if r.status_code == 200:
                    js = r.json()
                    if js.get('data') and js.get('fields'):
                        return js
            except Exception:
                continue
    return None


def _twse_t86_market_one_day(date_yyyymmdd):
    js = _twse_t86_json(date_yyyymmdd)
    if not js:
        return None
    fields = js.get('fields') or []
    data = js.get('data') or []
    if not fields or not data:
        return None

    def idx_by(*keys):
        for i, f in enumerate(fields):
            fs = str(f).replace(' ', '')
            if all(k in fs for k in keys):
                return i
        return None

    foreign_i = idx_by('外陸資買賣超股數') or idx_by('外陸資', '買賣超')
    trust_i = idx_by('投信買賣超股數') or idx_by('投信', '買賣超')
    dealer_i = idx_by('自營商買賣超股數') or idx_by('自營商', '買賣超')
    total_i = idx_by('三大法人買賣超股數') or idx_by('三大法人', '買賣超')

    sums = {'foreign': 0.0, 'trust': 0.0, 'dealer': 0.0, 'total': 0.0}
    counts = {'foreign': 0, 'trust': 0, 'dealer': 0, 'total': 0}
    for row in data:
        for key, idx in [('foreign', foreign_i), ('trust', trust_i), ('dealer', dealer_i), ('total', total_i)]:
            if idx is None or idx >= len(row):
                continue
            v = _clean_num(row[idx])
            if v is not None:
                sums[key] += v / 1000.0  # shares -> lots/張
                counts[key] += 1

    if not any(counts.values()):
        return None
    return {
        'date': pd.to_datetime(date_yyyymmdd, format='%Y%m%d', errors='coerce'),
        'foreign': sums['foreign'] if counts['foreign'] else None,
        'trust': sums['trust'] if counts['trust'] else None,
        'dealer': sums['dealer'] if counts['dealer'] else None,
        'total': sums['total'] if counts['total'] else None,
    }


def fetch_twse_market_spot_feature(start_date=None, end_date=None):
    """TWSE T86 official market spot flow, unit = 張. Aggregates all listed stocks."""
    rows = []
    for d in _official_date_iter(start_date, end_date, calendar_days=35):
        one = _twse_t86_market_one_day(d.strftime('%Y%m%d'))
        if one and one.get('foreign') is not None:
            rows.append(one)
        if len(rows) >= 12:
            break
        time.sleep(0.12)
    if not rows:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    out = pd.DataFrame(rows).sort_values('date')
    out['value'] = pd.to_numeric(out['foreign'], errors='coerce')
    # market-level foreign net buy/sell in lots; +/-100k lots is meaningful for broad market bias
    out['score'] = out['value'].apply(lambda x: _score_by_threshold(x, 100000, -100000, 0.25, -0.25))
    return out[['date', 'value', 'score']].dropna(subset=['date'])


def fetch_twse_market_total_spot_feature(start_date=None, end_date=None):
    """TWSE T86 official three-institution total spot flow, unit = 張."""
    rows = []
    for d in _official_date_iter(start_date, end_date, calendar_days=35):
        one = _twse_t86_market_one_day(d.strftime('%Y%m%d'))
        if one and one.get('total') is not None:
            rows.append(one)
        if len(rows) >= 12:
            break
        time.sleep(0.12)
    if not rows:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    out = pd.DataFrame(rows).sort_values('date')
    out['value'] = pd.to_numeric(out['total'], errors='coerce')
    out['score'] = out['value'].apply(lambda x: _score_by_threshold(x, 100000, -100000, 0.25, -0.25))
    return out[['date', 'value', 'score']].dropna(subset=['date'])


def fetch_taifex_pcr_official_feature(start_date=None, end_date=None):
    """TAIFEX official Put/Call Ratio. Prefer open-interest PCR if present."""
    url = 'https://www.taifex.com.tw/cht/3/pcRatioExcel'
    text = _fetch_text(url, timeout=20)
    tables = _read_html_tables_from_text(text)
    rows = []
    for df in tables:
        if df.empty:
            continue
        cols = list(df.columns)
        date_col = None
        for c in cols:
            if '日期' in str(c) or 'Date' in str(c):
                date_col = c
                break
        if date_col is None:
            continue
        ratio_col = None
        # Prefer OI PCR; fallback volume PCR.
        for c in cols:
            cs = str(c)
            if ('未平倉' in cs and '比率' in cs) or ('未平倉量比率' in cs):
                ratio_col = c
                break
        if ratio_col is None:
            for c in cols:
                cs = str(c)
                if '比率' in cs or 'Put/Call' in cs or 'Ratio' in cs:
                    ratio_col = c
                    break
        if ratio_col is None:
            continue
        for _, r in df.iterrows():
            dt = pd.to_datetime(str(r.get(date_col)), errors='coerce')
            val = _clean_num(r.get(ratio_col))
            if pd.notna(dt) and val is not None:
                rows.append({'date': dt, 'value': val})
    if not rows:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    out = pd.DataFrame(rows).drop_duplicates('date').sort_values('date')
    if start_date:
        out = out[out['date'] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out['date'] <= pd.to_datetime(end_date)]
    def pcr_score(x):
        x = safe_float(x)
        if x is None:
            return 0.0
        if x >= 130:
            return 0.20
        if x >= 100:
            return 0.10
        if x <= 80:
            return -0.20
        return 0.0
    out['score'] = out['value'].apply(pcr_score)
    return out[['date', 'value', 'score']]


def _taifex_fut_contract_one_day(date_slash):
    """TAIFEX major institutional traders, TX futures, one day. value = FINI net OI contracts."""
    urls = [
        'https://www.taifex.com.tw/cht/3/futContractsDate',
        'https://www.taifex.com.tw/cht/3/futContractsDateView',
    ]
    params_candidates = [
        {'queryDate': date_slash, 'commodityId': 'TX'},
        {'queryDate': date_slash, 'commodity_id': 'TX'},
        {'date': date_slash, 'commodityId': 'TX'},
    ]
    for url in urls:
        for params in params_candidates:
            text = _fetch_text(url, params=params, timeout=20)
            tables = _read_html_tables_from_text(text)
            for df in tables:
                if df.empty:
                    continue
                # Find row for foreign institutional investors.
                row_idx = None
                for idx, row in df.iterrows():
                    row_text = ' '.join(str(x) for x in row.values)
                    if '外資' in row_text or 'FINI' in row_text or 'Foreign' in row_text:
                        row_idx = idx
                        break
                if row_idx is None:
                    continue
                row = df.loc[row_idx]
                # Prefer column with both 未平倉 and 多空淨額.
                net_col = None
                for c in df.columns:
                    cs = str(c)
                    if ('未平倉' in cs or 'OpenInterest' in cs or 'OI' in cs) and ('淨' in cs or 'Net' in cs or '多空' in cs):
                        net_col = c
                        break
                if net_col is None:
                    # Fallback: last numeric cell in the foreign row often is net OI / amount.
                    nums = [_clean_num(x) for x in row.values]
                    nums = [x for x in nums if x is not None]
                    val = nums[-1] if nums else None
                else:
                    val = _clean_num(row.get(net_col))
                if val is not None:
                    return val
    return None


def fetch_taifex_foreign_futures_official_feature(start_date=None, end_date=None):
    rows = []
    for d in _official_date_iter(start_date, end_date, calendar_days=35):
        val = _taifex_fut_contract_one_day(d.strftime('%Y/%m/%d'))
        if val is not None:
            rows.append({'date': d, 'value': val})
        if len(rows) >= 12:
            break
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    out = pd.DataFrame(rows).sort_values('date')
    out['score'] = out['value'].apply(lambda x: _score_by_threshold(x, 5000, -5000, 0.25, -0.25))
    return out[['date', 'value', 'score']]


def _taifex_large_trader_one_day(date_slash):
    """TAIFEX large trader futures structure. value = top10 buyer - top10 seller contracts.

    Uses official largeTraderFutQry. If the site ignores historical query parameters and only
    returns a latest snapshot, snapshot filtering later prevents backfilling fake history.
    """
    date_ts = pd.to_datetime(date_slash, errors='coerce')
    paths = ['/cht/3/largeTraderFutQry']
    for text in _qpro_fetch_taifex_texts(paths, date_ts=date_ts, commodity_id='TX', timeout=20):
        val = _qpro_extract_top10_net_from_text(text)
        if val is not None:
            return val

    # Original table fallback for environments where pd.read_html preserves columns well.
    url = 'https://www.taifex.com.tw/cht/3/largeTraderFutQry'
    for params in [{'queryDate': date_slash, 'commodityId': 'TX'}, {'date': date_slash, 'commodityId': 'TX'}, {}]:
        text = _fetch_text(url, params=params, timeout=20)
        tables = _read_html_tables_from_text(text)
        for df in tables:
            if df.empty:
                continue
            target = None
            for idx, row in df.iterrows():
                txt = ' '.join(str(x) for x in row.values)
                if ('所有' in txt and '契約' in txt and ('臺股期貨' in txt or '台股期貨' in txt or 'TX' in txt)):
                    target = row
                    break
            if target is None:
                for idx, row in df.iterrows():
                    txt = ' '.join(str(x) for x in row.values)
                    if '所有' in txt and '契約' in txt:
                        target = row
                        break
            if target is None:
                continue
            nums = []
            for x in target.values:
                sx = str(x)
                if '%' in sx:
                    continue
                v = _clean_num(sx)
                if v is not None:
                    nums.append(v)
            nums = _qpro_dedupe_adjacent_numbers(nums)
            if len(nums) >= 4:
                return nums[1] - nums[3]
    return None


def fetch_taifex_top10_official_feature(start_date=None, end_date=None):
    rows = []
    for d in _official_date_iter(start_date, end_date, calendar_days=35):
        val = _taifex_large_trader_one_day(d.strftime('%Y/%m/%d'))
        if val is not None:
            rows.append({'date': d, 'value': val})
        if len(rows) >= 12:
            break
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    out = pd.DataFrame(rows).sort_values('date')
    out['score'] = out['value'].apply(lambda x: _score_by_threshold(x, 1000, -1000, 0.25, -0.25))
    return out[['date', 'value', 'score']]


def fetch_tw_foreign_futures_feature(start_date=None, end_date=None):
    start_date, end_date = (start_date, end_date) if start_date and end_date else _date_range(90)
    df = _call_finmind(
        [
            'taiwan_futures_institutional_investors',
            'taiwan_futures_institutional_investors_report',
            'taiwan_futures_institutional_investors_open_interest',
        ],
        start_date=start_date,
        end_date=end_date,
    )
    if df.empty:
        return pd.DataFrame(columns=['date', 'value', 'score'])

    work = df.copy()
    for c in work.columns:
        cs = str(c).lower()
        if cs in ('name', '身份別', 'institutional_investors', 'investor', 'investor_type'):
            work = work[work[c].astype(str).str.contains('外資|Foreign', case=False, na=False)]
            break

    for c in work.columns:
        cs = str(c).lower()
        if cs in ('commodity_id', '契約', 'contract', 'contract_name', 'futures_id'):
            tx = work[work[c].astype(str).str.contains('TX|臺股期貨|台股期貨|加權', case=False, na=False)]
            if not tx.empty:
                work = tx
            break

    long_col = short_col = net_col = None
    for ks in [['open', 'interest', 'buy'], ['open_interest', 'buy'], ['多方', '未平倉'], ['買方', '未平倉'], ['long', 'open'], ['long']]:
        long_col = _find_column(work, ks)
        if long_col is not None:
            break
    for ks in [['open', 'interest', 'sell'], ['open_interest', 'sell'], ['空方', '未平倉'], ['賣方', '未平倉'], ['short', 'open'], ['short']]:
        short_col = _find_column(work, ks)
        if short_col is not None:
            break
    for ks in [['net'], ['未平倉', '淨'], ['多空', '淨'], ['買賣', '淨']]:
        net_col = _find_column(work, ks)
        if net_col is not None:
            break

    if long_col is not None and short_col is not None:
        work['value'] = _numeric_series(work, long_col) - _numeric_series(work, short_col)
    elif net_col is not None:
        work['value'] = _numeric_series(work, net_col)
    else:
        return pd.DataFrame(columns=['date', 'value', 'score'])

    if 'date' not in work.columns:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    out = work.groupby('date', as_index=False)['value'].sum().dropna()
    out['score'] = out['value'].apply(lambda x: _score_by_threshold(x, 5000, -5000))
    return out[['date', 'value', 'score']]


def fetch_tw_pcr_feature(start_date=None, end_date=None):
    start_date, end_date = (start_date, end_date) if start_date and end_date else _date_range(90)
    df = _call_finmind(
        [
            'taiwan_option_put_call_ratio',
            'taiwan_option_put_call_ratio_report',
            'taiwan_option_put_call_ratio_daily',
        ],
        start_date=start_date,
        end_date=end_date,
    )
    if df.empty:
        return pd.DataFrame(columns=['date', 'value', 'score'])

    ratio_col = None
    for ks in [['put', 'call', 'ratio'], ['pcr'], ['賣權', '買權', '比'], ['買賣權', '比']]:
        ratio_col = _find_column(df, ks)
        if ratio_col is not None:
            break

    if ratio_col is None:
        put_col = _find_column(df, ['put']) or _find_column(df, ['賣權'])
        call_col = _find_column(df, ['call']) or _find_column(df, ['買權'])
        if put_col is not None and call_col is not None:
            denom = _numeric_series(df, call_col).replace(0, np.nan)
            df['value'] = _numeric_series(df, put_col) / denom * 100
        else:
            return pd.DataFrame(columns=['date', 'value', 'score'])
    else:
        df['value'] = _numeric_series(df, ratio_col)

    out = df[['date', 'value']].dropna().copy()
    def pcr_score(x):
        x = safe_float(x)
        if x is None:
            return 0.0
        if x >= 130:
            return 0.20
        if x >= 100:
            return 0.10
        if x <= 80:
            return -0.20
        return 0.0
    out['score'] = out['value'].apply(pcr_score)
    return out[['date', 'value', 'score']]


def _taifex_fetch_tables(url, params_candidates, timeout=20):
    for params in params_candidates:
        text = _fetch_text(url, params=params, timeout=timeout)
        tables = _read_html_tables_from_text(text)
        if tables:
            return tables
    return []


def _numeric_values_from_row(row):
    vals = []
    for x in getattr(row, 'values', []):
        v = _clean_num(x)
        if v is not None:
            vals.append(v)
    return vals


# QPRO_FIX_20260513_2140: official TAIFEX macro parser hardening.
# Uses official TAIFEX pages first; does not use PCR as a fake retail substitute.
def _qpro_text_lines(text):
    if not text:
        return []
    raw = re.sub(r'<br\s*/?>', '\n', str(text), flags=re.I)
    raw = re.sub(r'<[^>]+>', '\n', raw)
    raw = html.unescape(raw)
    return [x.strip() for x in raw.replace('\r', '\n').split('\n') if x.strip()]


def _qpro_is_number_token(token):
    return _clean_num(token) is not None and not ('%' in str(token))


def _qpro_collect_number_tokens(tokens, start, stop_pred=None, max_scan=80):
    vals = []
    j = start
    scanned = 0
    while j < len(tokens) and scanned < max_scan:
        t = str(tokens[j]).strip()
        if stop_pred and stop_pred(t, j):
            break
        if _qpro_is_number_token(t):
            vals.append(_clean_num(t))
        j += 1
        scanned += 1
    return vals


def _qpro_dedupe_adjacent_numbers(vals, tol=1e-9):
    out = []
    for v in vals:
        if not out or abs(float(out[-1]) - float(v)) > tol:
            out.append(v)
    return out


def _qpro_fetch_taifex_texts(paths, date_ts=None, commodity_id=None, timeout=20):
    """Try current and bq888 TAIFEX hosts with common query parameter names."""
    hosts = ['https://www.taifex.com.tw', 'https://www.bq888.taifex.com.tw']
    date_slash = pd.to_datetime(date_ts).strftime('%Y/%m/%d') if date_ts is not None else None
    params_list = [{}]
    if date_slash:
        params_list += [
            {'queryDate': date_slash},
            {'date': date_slash},
            {'queryDate': date_slash, 'commodityId': commodity_id or ''},
            {'queryDate': date_slash, 'commodity_id': commodity_id or ''},
            {'date': date_slash, 'commodityId': commodity_id or ''},
        ]
    if commodity_id:
        params_list += [{'commodityId': commodity_id}, {'commodity_id': commodity_id}]

    texts = []
    seen = set()
    for host in hosts:
        for path in paths:
            url = host + path
            for params in params_list:
                params = {k: v for k, v in params.items() if v not in (None, '')}
                key = (url, tuple(sorted(params.items())))
                if key in seen:
                    continue
                seen.add(key)
                txt = _fetch_text(url, params=params, timeout=timeout)
                if txt and len(txt) > 500:
                    texts.append(txt)
    return texts


def _qpro_extract_contract_inst_net_oi_from_text(text, commodity_id='MTX'):
    """Extract three-institution open-interest net contracts from TAIFEX futContractsDate text.

    On TAIFEX futContractsDate each product has 自營商 / 投信 / 外資 rows.
    The second-last non-percent numeric value in each investor row is usually 未平倉多空淨額口數.
    """
    tokens = _qpro_text_lines(text)
    product_keywords = {
        'MTX': ['小型臺指期貨', '小型台指期貨', '小型臺指', '小型台指', 'MTX'],
        'TX': ['臺股期貨', '台股期貨', 'TX'],
    }.get(str(commodity_id).upper(), [str(commodity_id).upper()])
    investor_tokens = ['自營商', '投信', '外資', '外資及陸資']

    total = 0.0
    found = 0
    active = False
    for i, tok in enumerate(tokens):
        if any(k in tok for k in product_keywords):
            active = True
            continue
        # stop when another product section starts after we were active
        if active and ('期貨' in tok or tok in ('電子期貨', '金融期貨', '臺股期貨', '台股期貨')) and not any(k in tok for k in product_keywords) and tok not in investor_tokens:
            # not always safe to stop, so only deactivate if we already found some rows
            if found:
                break
        if not active:
            continue
        if any(inv == tok or inv in tok for inv in investor_tokens):
            nums = _qpro_collect_number_tokens(
                tokens,
                i + 1,
                stop_pred=lambda t, j: any(inv == t or inv in t for inv in investor_tokens) or ('期貨' in t and not _qpro_is_number_token(t)),
                max_scan=40,
            )
            if len(nums) >= 12:
                total += nums[-2]
                found += 1
            elif len(nums) >= 6:
                total += nums[-2]
                found += 1
    return total if found else None


def _qpro_extract_contract_market_oi_from_tables(text, commodity_id='MTX'):
    product_keywords = {
        'MTX': ['MTX', '小型臺指', '小型台指'],
        'TX': ['TX', '臺股期貨', '台股期貨'],
    }.get(str(commodity_id).upper(), [str(commodity_id).upper()])
    tables = _read_html_tables_from_text(text)
    for df in tables:
        if df is None or df.empty:
            continue
        oi_col = None
        for c in df.columns:
            cs = str(c).replace('\n', '').replace(' ', '')
            if '未沖銷契約' in cs or '未平倉' in cs or 'OpenInterest' in cs:
                oi_col = c
                break
        if oi_col is None:
            continue
        total = 0.0
        found = 0
        for _, row in df.iterrows():
            row_text = ' '.join(str(x) for x in row.values)
            if any(k in row_text for k in product_keywords):
                v = _clean_num(row.get(oi_col))
                if v is not None and v > 0:
                    total += v
                    found += 1
        if found and total > 0:
            return total
    return None


def _qpro_extract_contract_market_oi_from_text(text, commodity_id='MTX'):
    """Fallback token parser for TAIFEX futDailyMarketReport.
    It picks plausible open-interest numbers after MTX product tokens.
    """
    tokens = _qpro_text_lines(text)
    product_keywords = {
        'MTX': ['小型臺指', '小型台指', 'MTX'],
        'TX': ['臺股期貨', '台股期貨', 'TX'],
    }.get(str(commodity_id).upper(), [str(commodity_id).upper()])
    candidates = []
    for i, tok in enumerate(tokens):
        if any(k in tok for k in product_keywords):
            nums = _qpro_collect_number_tokens(tokens, i + 1, max_scan=45)
            # Daily market row has volume and OI as larger integer-ish values; choose largest plausible OI.
            plausible = [v for v in nums if v is not None and v > 1000]
            if plausible:
                candidates.append(max(plausible))
    return max(candidates) if candidates else None


def _qpro_extract_top10_net_from_text(text):
    """Extract top10 buyer - seller position from TAIFEX largeTraderFutQry text."""
    tokens = _qpro_text_lines(text)
    product_keys = ['臺股期貨(TX+MTX/4+TMF/20)', '台股期貨(TX+MTX/4+TMF/20)', '臺股期貨', '台股期貨']
    product_seen = False
    best = None
    for i, tok in enumerate(tokens):
        if any(k in tok for k in product_keys):
            product_seen = True
            continue
        if not product_seen:
            continue
        # Prefer 所有契約 row. In extracted text it may be split into two tokens: 所有 / 契約.
        if tok == '所有' or '所有契約' in tok:
            nums = _qpro_collect_number_tokens(tokens, i + 1, max_scan=80)
            nums = [v for v in nums if v is not None]
            nums = _qpro_dedupe_adjacent_numbers(nums)
            # Expected after de-dup: buy top5, buy top10, sell top5, sell top10, market OI.
            if len(nums) >= 4:
                best = nums[1] - nums[3]
                break
        # fallback: current month row before 所有契約
        if re.match(r'^\d{4}$', tok) or re.match(r'^\d{2}$', tok):
            nums = _qpro_collect_number_tokens(tokens, i + 1, max_scan=70)
            nums = _qpro_dedupe_adjacent_numbers([v for v in nums if v is not None])
            if len(nums) >= 4:
                best = nums[1] - nums[3]
    return best


def _qpro_market_status_label(score, real_count=None):
    s = safe_float(score, 0.0) or 0.0
    if s >= 0.35:
        return '強多'
    if s >= 0.15:
        return '偏多'
    if s <= -0.35:
        return '強空'
    if s <= -0.25:
        return '偏空'
    return '震盪'


def _taifex_contract_institutional_net_oi_one_day(date_ts, commodity_id='MTX'):
    """TAIFEX three-institution net OI for a contract, official pages first.

    Returns contract count, not TWD amount.  Robust against messy merged HTML tables.
    """
    paths = ['/cht/3/futContractsDateExcel', '/cht/3/futContractsDate']
    for text in _qpro_fetch_taifex_texts(paths, date_ts=date_ts, commodity_id=commodity_id, timeout=20):
        val = _qpro_extract_contract_inst_net_oi_from_text(text, commodity_id)
        if val is not None:
            return val
    return None


def _taifex_contract_market_oi_one_day(date_ts, commodity_id='MTX'):
    """TAIFEX total market open interest for one futures contract."""
    paths = ['/cht/3/futDailyMarketReport', '/cht/3/futDailyMarketReportExcel', '/cht/3/futDailyMarketExcel']
    for text in _qpro_fetch_taifex_texts(paths, date_ts=date_ts, commodity_id=commodity_id, timeout=20):
        val = _qpro_extract_contract_market_oi_from_tables(text, commodity_id)
        if val is None:
            val = _qpro_extract_contract_market_oi_from_text(text, commodity_id)
        if val is not None and val > 0:
            return val
    return None


def fetch_taifex_mtx_retail_proxy_feature(start_date=None, end_date=None):
    """MTX retail long/short proxy from official TAIFEX data.

    retail_proxy_pct = -1 * three_institution_net_oi(MTX) / total_market_oi(MTX) * 100

    Guard rails are intentionally strict:
      - MTX total market OI must be a real open-interest denominator, normally tens of thousands+.
      - abs(retail_proxy_pct) above 200% is parser failure, not market truth.
      - invalid rows are hidden as N/A rather than shown as fake precision.
    """
    rows = []
    for d in _official_date_iter(start_date, end_date, calendar_days=35):
        inst_net = _taifex_contract_institutional_net_oi_one_day(d, 'MTX')
        market_oi = _taifex_contract_market_oi_one_day(d, 'MTX')

        inst_net_f = safe_float(inst_net)
        market_oi_f = safe_float(market_oi)

        if inst_net_f is None or market_oi_f is None:
            log(f'[MTX-RETAIL-WARN] {pd.to_datetime(d).strftime("%Y-%m-%d")} missing numerator/denominator: inst_net={inst_net}, market_oi={market_oi}')
            continue

        # MTX open interest should not be tiny.  A tiny denominator usually means
        # the parser grabbed volume/change/percent instead of open interest.
        if market_oi_f < 10000:
            log(f'[MTX-RETAIL-WARN] {pd.to_datetime(d).strftime("%Y-%m-%d")} invalid MTX OI denominator: inst_net={inst_net_f:,.0f}, market_oi={market_oi_f:,.0f}; row hidden as N/A')
            continue

        value = -1.0 * inst_net_f / market_oi_f * 100.0

        # A retail ratio beyond +/-200% is mechanically impossible / parser error
        # for this proxy.  Hide it instead of polluting macro_history.csv.
        if not np.isfinite(value) or abs(value) > 200:
            log(f'[MTX-RETAIL-WARN] {pd.to_datetime(d).strftime("%Y-%m-%d")} impossible retail proxy: {value:.1f}% from inst_net={inst_net_f:,.0f}, market_oi={market_oi_f:,.0f}; row hidden as N/A')
            continue

        log(f'[MTX-RETAIL] {pd.to_datetime(d).strftime("%Y-%m-%d")} inst_net={inst_net_f:,.0f}, market_oi={market_oi_f:,.0f}, retail_proxy={value:+.1f}%')

        if value >= 20:
            score = -0.25
        elif value <= -20:
            score = 0.25
        elif value >= 10:
            score = -0.10
        elif value <= -10:
            score = 0.10
        else:
            score = 0.0
        rows.append({'date': pd.to_datetime(d), 'value': value, 'score': score})

        if len(rows) >= 12:
            break
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    return pd.DataFrame(rows).sort_values('date')[['date', 'value', 'score']]


def fetch_tw_retail_sentiment_feature(start_date=None, end_date=None):
    """Retail sentiment: official MTX proxy first, FinMind fallback second."""
    official = fetch_taifex_mtx_retail_proxy_feature(start_date, end_date)
    if official is not None and not official.empty:
        return official

    start_date, end_date = (start_date, end_date) if start_date and end_date else _date_range(90)
    df = _call_finmind(
        [
            'taiwan_futures_retail_investors',
            'taiwan_futures_retail_long_short_ratio',
            'taiwan_futures_retail_sentiment',
            'taiwan_futures_small_trader_ratio',
        ],
        start_date=start_date,
        end_date=end_date,
    )
    if df.empty:
        return pd.DataFrame(columns=['date', 'value', 'score'])

    ratio_col = None
    for ks in [['retail'], ['散戶'], ['小台'], ['small', 'trader'], ['long', 'short', 'ratio'], ['多空', '比']]:
        ratio_col = _find_column(df, ks)
        if ratio_col is not None:
            break
    if ratio_col is None:
        return pd.DataFrame(columns=['date', 'value', 'score'])

    out = df[['date']].copy()
    out['value'] = _numeric_series(df, ratio_col)
    out = out.dropna()
    def retail_score(x):
        x = safe_float(x)
        if x is None:
            return 0.0
        if x >= 20:
            return -0.25
        if x <= -20:
            return 0.25
        if x >= 10:
            return -0.10
        if x <= -10:
            return 0.10
        return 0.0
    out['score'] = out['value'].apply(retail_score)
    return out[['date', 'value', 'score']]


def fetch_tw_top10_traders_feature(start_date=None, end_date=None):
    start_date, end_date = (start_date, end_date) if start_date and end_date else _date_range(90)
    df = _call_finmind(
        [
            'taiwan_futures_top10_traders',
            'taiwan_futures_top10_dealers',
            'taiwan_futures_large_trader_open_interest',
            'taiwan_futures_trader_and_volume_report',
        ],
        start_date=start_date,
        end_date=end_date,
    )
    if df.empty:
        return pd.DataFrame(columns=['date', 'value', 'score'])

    long_col = short_col = net_col = None
    for ks in [['top', '10', 'net'], ['十大', '淨'], ['net']]:
        net_col = _find_column(df, ks)
        if net_col is not None:
            break
    for ks in [['top', '10', 'long'], ['十大', '多'], ['long']]:
        long_col = _find_column(df, ks)
        if long_col is not None:
            break
    for ks in [['top', '10', 'short'], ['十大', '空'], ['short']]:
        short_col = _find_column(df, ks)
        if short_col is not None:
            break

    if net_col is not None:
        df['value'] = _numeric_series(df, net_col)
    elif long_col is not None and short_col is not None:
        df['value'] = _numeric_series(df, long_col) - _numeric_series(df, short_col)
    else:
        return pd.DataFrame(columns=['date', 'value', 'score'])

    out = df.groupby('date', as_index=False)['value'].sum().dropna()
    out['score'] = out['value'].apply(lambda x: _score_by_threshold(x, 1000, -1000))
    return out[['date', 'value', 'score']]




# Override macro fetchers: official source first, FinMind second.
def fetch_tw_foreign_futures_feature(start_date=None, end_date=None):
    official = fetch_taifex_foreign_futures_official_feature(start_date, end_date)
    if official is not None and not official.empty:
        return official
    df = _call_finmind(
        [
            'taiwan_futures_institutional_investors',
            'taiwan_futures_institutional_investors_report',
            'taiwan_futures_institutional_investors_open_interest',
        ],
        start_date=start_date, end_date=end_date,
    )
    if df.empty:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    work = df.copy()
    for c in work.columns:
        cs = str(c).lower()
        if cs in ('name', '身份別', 'institutional_investors', 'investor', 'investor_type'):
            work = work[work[c].astype(str).str.contains('外資|Foreign|FINI', case=False, na=False)]
            break
    for c in work.columns:
        cs = str(c).lower()
        if cs in ('commodity_id', '契約', 'contract', 'contract_name', 'futures_id'):
            tx = work[work[c].astype(str).str.contains('TX|臺股期貨|台股期貨|加權', case=False, na=False)]
            if not tx.empty:
                work = tx
            break
    long_col = short_col = net_col = None
    for ks in [['open', 'interest', 'buy'], ['open_interest', 'buy'], ['多方', '未平倉'], ['買方', '未平倉'], ['long', 'open'], ['long']]:
        long_col = _find_column(work, ks)
        if long_col is not None: break
    for ks in [['open', 'interest', 'sell'], ['open_interest', 'sell'], ['空方', '未平倉'], ['賣方', '未平倉'], ['short', 'open'], ['short']]:
        short_col = _find_column(work, ks)
        if short_col is not None: break
    for ks in [['net'], ['未平倉', '淨'], ['多空', '淨'], ['買賣', '淨']]:
        net_col = _find_column(work, ks)
        if net_col is not None: break
    if long_col is not None and short_col is not None:
        work['value'] = _numeric_series(work, long_col) - _numeric_series(work, short_col)
    elif net_col is not None:
        work['value'] = _numeric_series(work, net_col)
    else:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    if 'date' not in work.columns:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    out = work.groupby('date', as_index=False)['value'].sum().dropna()
    out['score'] = out['value'].apply(lambda x: _score_by_threshold(x, 5000, -5000))
    return out[['date', 'value', 'score']]


def fetch_tw_pcr_feature(start_date=None, end_date=None):
    official = fetch_taifex_pcr_official_feature(start_date, end_date)
    if official is not None and not official.empty:
        return official
    df = _call_finmind(
        ['taiwan_option_put_call_ratio', 'taiwan_option_put_call_ratio_report', 'taiwan_option_put_call_ratio_daily'],
        start_date=start_date, end_date=end_date,
    )
    if df.empty:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    ratio_col = None
    for ks in [['put', 'call', 'ratio'], ['pcr'], ['賣權', '買權', '比'], ['買賣權', '比']]:
        ratio_col = _find_column(df, ks)
        if ratio_col is not None: break
    if ratio_col is None:
        put_col = _find_column(df, ['put']) or _find_column(df, ['賣權'])
        call_col = _find_column(df, ['call']) or _find_column(df, ['買權'])
        if put_col is not None and call_col is not None:
            denom = _numeric_series(df, call_col).replace(0, np.nan)
            df['value'] = _numeric_series(df, put_col) / denom * 100
        else:
            return pd.DataFrame(columns=['date', 'value', 'score'])
    else:
        df['value'] = _numeric_series(df, ratio_col)
    out = df[['date', 'value']].dropna().copy()
    def pcr_score(x):
        x = safe_float(x)
        if x is None: return 0.0
        if x >= 130: return 0.20
        if x >= 100: return 0.10
        if x <= 80: return -0.20
        return 0.0
    out['score'] = out['value'].apply(pcr_score)
    return out[['date', 'value', 'score']]


def fetch_tw_top10_traders_feature(start_date=None, end_date=None):
    official = fetch_taifex_top10_official_feature(start_date, end_date)
    if official is not None and not official.empty:
        return official
    return pd.DataFrame(columns=['date', 'value', 'score'])


def fetch_tw_foreign_spot_feature(start_date=None, end_date=None):
    official = fetch_twse_market_spot_feature(start_date, end_date)
    if official is not None and not official.empty:
        return official
    return pd.DataFrame(columns=['date', 'value', 'score'])

def get_tw_four_dimensional_data(trade_dates):
    start_date, end_date = _date_range(120)
    foreign_fut = fetch_tw_foreign_futures_feature(start_date, end_date)
    pcr = fetch_tw_pcr_feature(start_date, end_date)
    retail = fetch_tw_retail_sentiment_feature(start_date, end_date)
    top10 = fetch_tw_top10_traders_feature(start_date, end_date)

    ff_values = _latest_by_trade_dates(foreign_fut, trade_dates, display_func=lambda x: _format_signed_number(x, 0))
    pcr_values = _latest_by_trade_dates(pcr, trade_dates, display_func=lambda x: _format_ratio(x, 1))
    retail_values = _latest_by_trade_dates(retail, trade_dates, display_func=lambda x: _format_ratio(x, 1))
    top10_values = _latest_by_trade_dates(top10, trade_dates, display_func=lambda x: _format_signed_number(x, 0))

    dims = []
    for i in range(len(trade_dates)):
        vals = [ff_values[i], pcr_values[i], retail_values[i], top10_values[i]]
        real_count = sum(1 for v in vals if v['value'] is not None)
        score = sum(v['score'] for v in vals) / real_count if real_count else 0.0
        dims.append({
            'Foreign_Fut': ff_values[i],
            'PCR_Ratio': pcr_values[i],
            'Retail_Sentiment': retail_values[i],
            'Top_10_Traders': top10_values[i],
            'score': _normalize_score(score),
            'real_count': real_count,
        })
    return dims
def fetch_tw_foreign_spot_feature(start_date=None, end_date=None):
    """TWSE official T86 foreign spot net buy/sell first; FinMind fallback."""
    official = fetch_twse_market_spot_feature(start_date, end_date)
    if official is not None and not official.empty:
        return official

    start_date, end_date = (start_date, end_date) if start_date and end_date else _date_range(90)
    candidates = []
    if dl is not None:
        for kwargs in [
            {'stock_id': 'MI_INDEX', 'start_date': start_date, 'end_date': end_date},
            {'stock_id': 'TAIEX', 'start_date': start_date, 'end_date': end_date},
            {'start_date': start_date, 'end_date': end_date},
        ]:
            try:
                fn = getattr(dl, 'taiwan_stock_institutional_investors', None)
                if fn is None:
                    continue
                df = fn(**kwargs)
                if df is not None and not df.empty:
                    candidates.append(_as_date_col(df))
            except Exception:
                continue

    if not candidates:
        return pd.DataFrame(columns=['date', 'value', 'score'])
    work = pd.concat(candidates, ignore_index=True)
    work = _as_date_col(work)
    if work.empty or 'date' not in work.columns:
        return pd.DataFrame(columns=['date', 'value', 'score'])

    name_col = None
    for c in work.columns:
        if str(c).lower() in ('name', 'institutional_investors', 'investor', 'investor_type') or str(c) in ('身份別', '法人', '投資人'):
            name_col = c
            break
    if name_col is not None:
        filtered = work[work[name_col].astype(str).str.contains('外資|Foreign', case=False, na=False)]
        if not filtered.empty:
            work = filtered

    buy_col = _find_column(work, ['buy']) or _find_column(work, ['買'])
    sell_col = _find_column(work, ['sell']) or _find_column(work, ['賣'])
    net_col = _find_column(work, ['net']) or _find_column(work, ['買賣', '超']) or _find_column(work, ['淨'])
    if buy_col is not None and sell_col is not None:
        work['value'] = _numeric_series(work, buy_col) - _numeric_series(work, sell_col)
    elif net_col is not None:
        work['value'] = _numeric_series(work, net_col)
    else:
        return pd.DataFrame(columns=['date', 'value', 'score'])

    out = work.groupby('date', as_index=False)['value'].sum().dropna()
    if not out.empty and out['value'].abs().median() > 1_000_000:
        out['value'] = out['value'] / 100_000_000
    out['score'] = out['value'].apply(lambda x: _score_by_threshold(x, 50, -50, 0.25, -0.25))
    return out[['date', 'value', 'score']]


def _feature_looks_repeated(df, min_rows=3):
    """Return True when a fetched official feature looks like one page was reused for many query dates."""
    try:
        if df is None or df.empty or 'value' not in df.columns or len(df.dropna(subset=['value'])) < min_rows:
            return False
        vals = pd.to_numeric(df['value'], errors='coerce').dropna()
        if len(vals) < min_rows:
            return False
        return vals.nunique(dropna=True) == 1
    except Exception:
        return False


def _drop_repeated_official_feature(df, source_label='官方資料'):
    """Keep official values even when identical across rows; log only."""
    if _feature_looks_repeated(df):
        log(f'[DATA-WARN] {source_label} values are identical across days; keeping official/latest snapshot instead of hiding as N/A.')
    return df




# QPRO_FIX_20260515_0815_TWSE_FINMIND_MACRO_DATE
# QPRO_FIX_20260515_2255_TWO_CODE_COMPANY_CHIP: fix .TWO -> code normalization, company name, chip lookup
# Macro report trade-date/index source priority:
#   TWSE official MI_INDEX -> FinMind TAIEX/index daily -> macro_history dates -> yfinance ^TWII fallback.
# This prevents the macro dashboard from being stuck at stale yfinance ^TWII dates.
def _qpro_parse_twse_mi_index_one_day(date_yyyymmdd):
    """Return one TWSE official TAIEX index row for date_yyyymmdd.

    The TWSE MI_INDEX schema changes occasionally.  This parser avoids hard
    relying on a single column index and searches for the 發行量加權股價指數 row.
    """
    endpoints = [
        'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX',
        'https://www.twse.com.tw/exchangeReport/MI_INDEX',
    ]
    params_candidates = [
        {'response': 'json', 'date': date_yyyymmdd, 'type': 'MS'},
        {'response': 'json', 'date': date_yyyymmdd, 'type': 'ALL'},
    ]
    target_keys = ['發行量加權股價指數', 'TAIEX', '加權股價指數']

    for url in endpoints:
        for params in params_candidates:
            try:
                r = requests.get(url, params=params, headers=_official_headers(), timeout=15)
                if r.status_code != 200:
                    continue
                js = r.json()
            except Exception:
                continue

            blocks = []
            for k in ('data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data'):
                v = js.get(k) if isinstance(js, dict) else None
                if isinstance(v, list) and v:
                    blocks.append(v)

            for block in blocks:
                for row in block:
                    if not isinstance(row, (list, tuple)):
                        continue
                    row_text = ' '.join(str(x) for x in row)
                    if not any(key in row_text for key in target_keys):
                        continue

                    nums = [_clean_num(x) for x in row]
                    nums = [x for x in nums if x is not None]
                    if not nums:
                        continue

                    # Close should be an index-level number (roughly thousands to tens of thousands).
                    close_candidates = [x for x in nums if 1000 <= abs(x) <= 100000]
                    close = close_candidates[0] if close_candidates else nums[0]

                    # Return percentage is usually the last small number with percent semantics.
                    ret = None
                    for cell in reversed(row):
                        s = str(cell)
                        v = _clean_num(s)
                        if v is None:
                            continue
                        if '%' in s or abs(v) < 20:
                            ret = v
                            break

                    dt = pd.to_datetime(date_yyyymmdd, format='%Y%m%d', errors='coerce')
                    if pd.isna(dt):
                        continue
                    return {'Date': dt, 'Close': close, 'RET': ret}
    return None


def _qpro_fetch_twse_index_df(days=45, calendar_days=120):
    rows = []
    for d in _official_date_iter(calendar_days=calendar_days):
        one = _qpro_parse_twse_mi_index_one_day(d.strftime('%Y%m%d'))
        if one is not None:
            rows.append(one)
        if len(rows) >= days:
            break
        time.sleep(0.08)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).dropna(subset=['Date']).drop_duplicates('Date').sort_values('Date')
    df = df.set_index('Date')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    if 'RET' not in df.columns or df['RET'].isna().all():
        df['RET'] = df['Close'].pct_change() * 100
    else:
        df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
        df['RET'] = df['RET'].fillna(df['Close'].pct_change() * 100)
    df['MA20'] = df['Close'].rolling(20, min_periods=5).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=10).mean()
    df['Volume'] = np.nan
    df['VOL_MA20'] = np.nan
    df['VOL_RATIO'] = np.nan
    log(f'[TW-INDEX] source=TWSE official latest={df.index.max().strftime("%Y-%m-%d")} rows={len(df)}')
    return df


def _qpro_fetch_finmind_taiex_index_df(start_date=None, end_date=None):
    if dl is None:
        return pd.DataFrame()
    if not start_date or not end_date:
        start_date, end_date = _date_range(180)

    candidates = []
    calls = [
        ('taiwan_stock_daily', {'stock_id': 'TAIEX', 'start_date': start_date, 'end_date': end_date}),
        ('taiwan_stock_price', {'stock_id': 'TAIEX', 'start_date': start_date, 'end_date': end_date}),
        ('taiwan_stock_price', {'stock_id': 'MI_INDEX', 'start_date': start_date, 'end_date': end_date}),
        ('taiwan_stock_daily', {'stock_id': 'MI_INDEX', 'start_date': start_date, 'end_date': end_date}),
    ]
    for method, kwargs in calls:
        try:
            fn = getattr(dl, method, None)
            if fn is None:
                continue
            df = fn(**kwargs)
            if df is not None and not df.empty:
                candidates.append(_as_date_col(df))
        except Exception:
            continue

    if not candidates:
        return pd.DataFrame()
    x = pd.concat(candidates, ignore_index=True)
    x = _as_date_col(x)
    if x.empty or 'date' not in x.columns:
        return pd.DataFrame()

    close_col = _pick_first_column(x, ['close', '收盤價', '收盤指數', '收盤', 'TAIEX'])
    if close_col is None:
        # Fallback to any numeric-looking column with index magnitude.
        for c in x.columns:
            vals = pd.to_numeric(x[c], errors='coerce')
            if vals.dropna().between(1000, 100000).mean() > 0.5:
                close_col = c
                break
    if close_col is None:
        return pd.DataFrame()

    out = x[['date', close_col]].copy().rename(columns={'date': 'Date', close_col: 'Close'})
    out['Date'] = pd.to_datetime(out['Date'], errors='coerce')
    out['Close'] = pd.to_numeric(out['Close'], errors='coerce')
    out = out.dropna(subset=['Date', 'Close']).drop_duplicates('Date').sort_values('Date').set_index('Date')
    if out.empty:
        return pd.DataFrame()
    out['RET'] = out['Close'].pct_change() * 100
    out['MA20'] = out['Close'].rolling(20, min_periods=5).mean()
    out['MA50'] = out['Close'].rolling(50, min_periods=10).mean()
    out['Volume'] = np.nan
    out['VOL_MA20'] = np.nan
    out['VOL_RATIO'] = np.nan
    log(f'[TW-INDEX] source=FinMind latest={out.index.max().strftime("%Y-%m-%d")} rows={len(out)}')
    return out


def _qpro_fetch_macro_history_index_stub(region='TW', days=45):
    """Use macro_history dates only as a last-resort date calendar.

    Close is unavailable, but RET can still be taken from IndexRet when present,
    so the dashboard can advance to the latest official macro date even if
    yfinance is stale.
    """
    path = _qpro_macro_history_path(region) if '_qpro_macro_history_path' in globals() else os.path.join(REPORT_DIR, f'{str(region).upper()}_macro_history.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        h = pd.read_csv(path)
        if h.empty or 'Date' not in h.columns:
            return pd.DataFrame()
        h['Date'] = pd.to_datetime(h['Date'], errors='coerce')
        h = h.dropna(subset=['Date']).drop_duplicates('Date').sort_values('Date').tail(days)
        if h.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=h['Date'])
        out['Close'] = np.nan
        out['RET'] = pd.to_numeric(h.get('IndexRet'), errors='coerce').values if 'IndexRet' in h.columns else np.nan
        out['MA20'] = np.nan
        out['MA50'] = np.nan
        out['Volume'] = np.nan
        out['VOL_MA20'] = np.nan
        out['VOL_RATIO'] = np.nan
        log(f'[TW-INDEX] source=macro_history latest={out.index.max().strftime("%Y-%m-%d")} rows={len(out)}')
        return out
    except Exception as e:
        try:
            log(f'[TW-INDEX-WARN] macro_history fallback failed: {e}')
        except Exception:
            pass
        return pd.DataFrame()


# Override old yfinance-first macro index downloader.  US still uses yfinance;
# TW uses official/FinMind first so the macro dashboard date does not lag behind.
def _download_macro_index_df(region='TW', period='6mo'):
    region = 'US' if str(region).upper() == 'US' else 'TW'
    if region == 'TW':
        df = _qpro_fetch_twse_index_df(days=60, calendar_days=140)
        if df is not None and not df.empty:
            return df

        start_date, end_date = _date_range(220)
        df = _qpro_fetch_finmind_taiex_index_df(start_date, end_date)
        if df is not None and not df.empty:
            return df

        df = _qpro_fetch_macro_history_index_stub(region='TW', days=60)
        if df is not None and not df.empty:
            return df

    index_ticker = '^TWII' if region == 'TW' else '^GSPC'
    df = _download_yf_df(index_ticker, period=period)
    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame()
    df['MA20'] = ta.trend.sma_indicator(df['Close'], 20)
    df['MA50'] = ta.trend.sma_indicator(df['Close'], 50)
    df['RET'] = df['Close'].pct_change() * 100
    if 'Volume' in df.columns:
        df['VOL_MA20'] = df['Volume'].rolling(20).mean()
        df['VOL_RATIO'] = np.where(df['VOL_MA20'] > 0, df['Volume'] / df['VOL_MA20'], np.nan)
    else:
        df['VOL_MA20'] = np.nan
        df['VOL_RATIO'] = np.nan
    try:
        log(f'[TW-INDEX] source=yfinance fallback latest={df.index.max().strftime("%Y-%m-%d")} rows={len(df)}')
    except Exception:
        pass
    return df.dropna(how='all').copy()


def fetch_real_macro_data(days=5):
    """Build Taiwan Macro Wave dataframe from official sources.

    Dimensions:
      Index momentum, foreign spot, foreign futures, PCR, MTX retail proxy, top10 traders.
    """
    log('[DATA] 啟動官方資料優先的大盤籌碼波段引擎...')

    index_df = _download_macro_index_df(region='TW', period='6mo')
    if index_df.empty:
        return pd.DataFrame(columns=['Date', 'Spot', 'Future', 'PCR', 'Retail', 'Top10', 'Score', 'RealCount'])

    last_rows = index_df.tail(days).copy()
    trade_dates = list(last_rows.index)
    start_date = (pd.to_datetime(trade_dates[0]) - pd.Timedelta(days=20)).strftime('%Y-%m-%d') if trade_dates else None
    end_date = pd.to_datetime(trade_dates[-1]).strftime('%Y-%m-%d') if trade_dates else None

    spot_df = fetch_tw_foreign_spot_feature(start_date, end_date)
    future_df = _drop_repeated_official_feature(fetch_tw_foreign_futures_feature(start_date, end_date), '外資期貨')
    pcr_df = fetch_tw_pcr_feature(start_date, end_date)
    retail_df = fetch_tw_retail_sentiment_feature(start_date, end_date)
    top10_df = _drop_repeated_official_feature(fetch_tw_top10_traders_feature(start_date, end_date), '十大交易人')

    spot_values = _latest_by_trade_dates(spot_df, trade_dates, display_func=lambda x: _format_signed_number(x, 0))
    future_values = _latest_by_trade_dates(future_df, trade_dates, display_func=lambda x: _format_signed_number(x, 0))
    pcr_values = _latest_by_trade_dates(pcr_df, trade_dates, display_func=lambda x: _format_ratio(x, 1))
    retail_values = _macro_values_by_trade_dates(retail_df, trade_dates, display_func=lambda x: f'{x:+.1f}%', source_label='MTX散戶多空', carry_forward=False, repeated_policy='latest_only')
    top10_values = _macro_values_by_trade_dates(top10_df, trade_dates, display_func=lambda x: _format_signed_number(x, 0), source_label='十大交易人', carry_forward=False, repeated_policy='latest_only')

    records = []
    for i, d in enumerate(trade_dates):
        vals = {
            'Spot': spot_values[i],
            'Future': future_values[i],
            'PCR': pcr_values[i],
            'Retail': retail_values[i],
            'Top10': top10_values[i],
        }
        real_items = [v for v in vals.values() if v.get('value') is not None]
        real_count = len(real_items)
        dim_score = sum(safe_float(v.get('score'), 0.0) or 0.0 for v in real_items) / real_count if real_count else 0.0
        index_score = _macro_score_from_index_row(last_rows.iloc[i])
        final_score = _normalize_score(index_score * 0.35 + dim_score * 0.65) if real_count else index_score

        idx_ret = safe_float(last_rows.iloc[i].get('RET'))
        records.append({
            'Date': pd.to_datetime(d).strftime('%Y-%m-%d'),
            'Spot': vals['Spot']['value'],
            'Future': vals['Future']['value'],
            'PCR': vals['PCR']['value'],
            'Retail': vals['Retail']['value'],
            'Top10': vals['Top10']['value'],
            'SpotDisplay': vals['Spot']['display'],
            'FutureDisplay': vals['Future']['display'],
            'PCRDisplay': vals['PCR']['display'],
            'RetailDisplay': vals['Retail']['display'],
            'Top10Display': vals['Top10']['display'],
            'IndexRet': idx_ret,
            'IndexRetDisplay': 'N/A' if idx_ret is None else f'{idx_ret:+.2f}%',
            'Score': round(final_score, 3),
            'RealCount': real_count + (1 if idx_ret is not None else 0),
        })

    return pd.DataFrame(records)


def get_us_breadth_series(trade_dates):
    tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'RSP']
    records = {}
    for t in tickers:
        df = _download_yf_df(t, period='6mo')
        if df.empty or 'Close' not in df.columns:
            continue
        df = df.copy()
        df['MA20'] = ta.trend.sma_indicator(df['Close'], 20)
        df['above'] = np.where(df['Close'] >= df['MA20'], 1, 0)
        records[t] = df[['above']]

    out = []
    for d in trade_dates:
        vals = []
        for df in records.values():
            part = df[df.index <= d]
            if not part.empty:
                vals.append(safe_float(part.iloc[-1]['above']))
        if vals:
            pct = sum(vals) / len(vals) * 100
            score = 0.25 if pct >= 70 else (-0.25 if pct <= 40 else 0.0)
            out.append({'value': pct, 'display': f'{pct:.0f}%', 'score': score})
        else:
            out.append({'value': None, 'display': 'N/A', 'score': 0.0})
    return out


def get_us_risk_appetite_series(trade_dates):
    hyg = _download_yf_df('HYG', period='6mo')
    tlt = _download_yf_df('TLT', period='6mo')
    if hyg.empty or tlt.empty or 'Close' not in hyg.columns or 'Close' not in tlt.columns:
        return [{'value': None, 'display': 'N/A', 'score': 0.0} for _ in trade_dates]

    spread = pd.DataFrame(index=hyg.index.union(tlt.index)).sort_index()
    spread['HYG'] = hyg['Close'].reindex(spread.index).ffill()
    spread['TLT'] = tlt['Close'].reindex(spread.index).ffill()
    spread['value'] = spread['HYG'].pct_change(20) * 100 - spread['TLT'].pct_change(20) * 100
    spread = spread.dropna(subset=['value']).reset_index()
    if spread.empty:
        return [{'value': None, 'display': 'N/A', 'score': 0.0} for _ in trade_dates]
    date_col = 'date' if 'date' in spread.columns else spread.columns[0]
    spread = spread.rename(columns={date_col: 'date'})
    spread['score'] = spread['value'].apply(lambda x: 0.25 if x >= 2 else (-0.25 if x <= -2 else 0.0))
    return _latest_by_trade_dates(spread[['date', 'value', 'score']], trade_dates, display_func=lambda x: f'{x:+.1f}%')


def get_us_four_dimensional_data(trade_dates, index_df):
    index_values = []
    for d in trade_dates:
        part = index_df[index_df.index <= d]
        if part.empty:
            index_values.append({'value': None, 'display': 'N/A', 'score': 0.0})
            continue
        row = part.iloc[-1]
        ret = safe_float(row.get('RET'))
        close = safe_float(row.get('Close'))
        ma20 = safe_float(row.get('MA20'))
        score = 0.0
        if close is not None and ma20 is not None:
            score += 0.20 if close >= ma20 else -0.20
        if ret is not None:
            score += 0.15 if ret >= 0 else -0.15
        index_values.append({'value': ret, 'display': 'N/A' if ret is None else f'{ret:+.2f}%', 'score': _normalize_score(score)})

    vix_series = _download_vix_series(period='2mo')
    if not vix_series.empty:
        vix_df = vix_series.reset_index()
        vix_df.columns = ['date', 'value']
        def vix_score(x):
            x = safe_float(x)
            if x is None: return 0.0
            if x >= 25: return -0.30
            if x >= 20: return -0.15
            if x <= 15: return 0.15
            return 0.0
        vix_df['score'] = vix_df['value'].apply(lambda x: _normalize_score(vix_score(x)))
        vix_values = _latest_by_trade_dates(vix_df[['date', 'value', 'score']], trade_dates, display_func=lambda x: f'{x:.1f}')
    else:
        vix_values = [{'value': None, 'display': 'N/A', 'score': 0.0} for _ in trade_dates]

    breadth_values = get_us_breadth_series(trade_dates)
    risk_appetite_values = get_us_risk_appetite_series(trade_dates)

    dims = []
    for i in range(len(trade_dates)):
        vals = [index_values[i], vix_values[i], breadth_values[i], risk_appetite_values[i]]
        real_count = sum(1 for x in vals if x['value'] is not None)
        score = sum(x['score'] for x in vals) / real_count if real_count else 0.0
        dims.append({
            'Index_Momentum': index_values[i],
            'VIX_Risk': vix_values[i],
            'Breadth_Proxy': breadth_values[i],
            'Risk_Appetite': risk_appetite_values[i],
            'score': _normalize_score(score),
            'real_count': real_count,
        })
    return dims


def _macro_score_from_index_row(row):
    score = 0.0
    close = safe_float(row.get('Close'))
    ma20 = safe_float(row.get('MA20'))
    ret = safe_float(row.get('RET'))
    vol_ratio = safe_float(row.get('VOL_RATIO'))

    if close is not None and ma20 is not None:
        score += 0.35 if close >= ma20 else -0.35
    if ret is not None:
        if ret >= 1.5:
            score += 0.25
        elif ret >= 0:
            score += 0.10
        elif ret <= -1.5:
            score -= 0.25
        else:
            score -= 0.10
    if vol_ratio is not None and ret is not None:
        if vol_ratio >= 1.2 and ret > 0:
            score += 0.15
        elif vol_ratio >= 1.2 and ret < 0:
            score -= 0.15
    return _normalize_score(score)


def get_macro_dashboard_data(region='TW'):
    region = 'US' if str(region).upper() == 'US' else 'TW'
    index_df = _download_macro_index_df(region=region, period='6mo')

    if index_df.empty:
        rows = []
        for i in range(3):
            rows.append({'label': '今日' if i == 0 else f'T-{i}', 'score': 0.0, 'real_count': 0, 'fields': [], 'index_ret_display': 'N/A', 'sentiment': '中性', 'data_status': '無資料'})
        return {'rows': rows, 'latest_score': 0.0, 'latest_mode': 'neutral', 'region': region}

    last_rows = index_df.tail(3).copy().iloc[::-1]
    trade_dates = list(last_rows.index)

    if region == 'TW':
        start_date = (pd.to_datetime(trade_dates[-1]) - pd.Timedelta(days=45)).strftime('%Y-%m-%d') if trade_dates else None
        end_date = pd.to_datetime(trade_dates[0]).strftime('%Y-%m-%d') if trade_dates else None

        spot_df = fetch_tw_foreign_spot_feature(start_date, end_date)
        future_df = _drop_repeated_official_feature(fetch_tw_foreign_futures_feature(start_date, end_date), '外資期貨')
        pcr_df = fetch_tw_pcr_feature(start_date, end_date)
        retail_df = fetch_tw_retail_sentiment_feature(start_date, end_date)
        top10_df = _drop_repeated_official_feature(fetch_tw_top10_traders_feature(start_date, end_date), '十大交易人')

        spot_values = _latest_by_trade_dates(spot_df, trade_dates, display_func=lambda x: _format_signed_number(x, 0))
        future_values = _latest_by_trade_dates(future_df, trade_dates, display_func=lambda x: _format_signed_number(x, 0))
        pcr_values = _latest_by_trade_dates(pcr_df, trade_dates, display_func=lambda x: _format_ratio(x, 1))
        retail_values = _macro_values_by_trade_dates(retail_df, trade_dates, display_func=lambda x: f'{x:+.1f}%', source_label='MTX散戶多空', carry_forward=False, repeated_policy='latest_only')
        top10_values = _macro_values_by_trade_dates(top10_df, trade_dates, display_func=lambda x: _format_signed_number(x, 0), source_label='十大交易人', carry_forward=False, repeated_policy='latest_only')

        dims = []
        for i in range(len(trade_dates)):
            items = {
                'Foreign_Spot': spot_values[i],
                'Foreign_Fut': future_values[i],
                'PCR_Ratio': pcr_values[i],
                'Retail_Sentiment': retail_values[i],
                'Top_10_Traders': top10_values[i],
            }
            real_items = [v for v in items.values() if v.get('value') is not None]
            real_count = len(real_items)
            dim_score = sum(safe_float(v.get('score'), 0.0) or 0.0 for v in real_items) / real_count if real_count else 0.0
            dims.append({**items, 'score': _normalize_score(dim_score), 'real_count': real_count})

        field_names = [
            ('Index_Momentum', '指數動能'),
            ('Foreign_Spot', '外資現貨'),
            ('Foreign_Fut', '外資期貨'),
            ('PCR_Ratio', 'PCR'),
            ('Retail_Sentiment', '散戶多空'),
            ('Top_10_Traders', '十大交易人'),
        ]
    else:
        dims = get_us_four_dimensional_data(trade_dates, index_df)
        field_names = [('Index_Momentum', '指數動能'), ('VIX_Risk', 'VIX'), ('Breadth_Proxy', '市場廣度'), ('Risk_Appetite', '風險偏好')]

    rows = []
    for idx, ((date, index_row), dim) in enumerate(zip(last_rows.iterrows(), dims)):
        label = '今日' if idx == 0 else f'T-{idx}'
        index_score = _macro_score_from_index_row(index_row)
        dim_score = safe_float(dim.get('score'), 0.0) or 0.0
        real_count = int(dim.get('real_count', 0) or 0)
        total_score = _normalize_score(index_score * 0.35 + dim_score * 0.65) if real_count > 0 else index_score

        ret = safe_float(index_row.get('RET'))
        fields = []
        if region == 'TW':
            fields.append({'key': 'Index_Momentum', 'label': '指數動能', 'display': 'N/A' if ret is None else f'{ret:+.2f}%', 'value': ret, 'score': index_score})
            for key, label_name in field_names:
                if key == 'Index_Momentum':
                    continue
                item = dim.get(key, {'display': 'N/A', 'value': None, 'score': 0.0})
                fields.append({'key': key, 'label': label_name, 'display': item.get('display', 'N/A'), 'value': item.get('value'), 'score': safe_float(item.get('score'), 0.0) or 0.0})
            total_real = real_count + (1 if ret is not None else 0)
        else:
            for key, label_name in field_names:
                item = dim.get(key, {'display': 'N/A', 'value': None, 'score': 0.0})
                fields.append({'key': key, 'label': label_name, 'display': item.get('display', 'N/A'), 'value': item.get('value'), 'score': safe_float(item.get('score'), 0.0) or 0.0})
            total_real = real_count

        rows.append({
            'label': label,
            'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
            'index_ret': ret,
            'index_ret_display': 'N/A' if ret is None else f'{ret:+.2f}%',
            'score': total_score,
            'sentiment': _score_label(total_score),
            'fields': fields,
            'real_count': total_real,
            'data_status': f'{total_real}/{len(fields)} real',
        })

    latest_score = safe_float(rows[0]['score'], 0.0) or 0.0
    if latest_score <= -0.25:
        latest_mode = 'defensive'
    elif latest_score >= 0.15:
        latest_mode = 'offensive'
    else:
        latest_mode = 'neutral'
    return {'rows': rows, 'latest_score': latest_score, 'latest_mode': latest_mode, 'region': region}


def _tw_rf_features_from_macro_data(macro_data):
    if not macro_data or macro_data.get('region') != 'TW' or not macro_data.get('rows'):
        return None
    row = macro_data['rows'][0]
    values = {f.get('key'): f.get('value') for f in row.get('fields', [])}
    required = ['Foreign_Fut', 'PCR_Ratio', 'Retail_Sentiment', 'Top_10_Traders']
    if not all(values.get(k) is not None for k in required):
        return None
    return pd.DataFrame([{
        'Foreign_Fut': float(values['Foreign_Fut']),
        'PCR_Ratio': float(values['PCR_Ratio']),
        'Retail_Sentiment': float(values['Retail_Sentiment']),
        'Top_10_Traders': float(values['Top_10_Traders']),
    }])


def check_market_status(region='TW'):
    log(f'[MARKET] 正在評估 {region} 大盤系統風險...')
    try:
        macro_data = get_macro_dashboard_data(region)
        LAST_MACRO_DATA[region] = macro_data
        score = safe_float(macro_data.get('latest_score'), 0.0) or 0.0
        mode = macro_data.get('latest_mode', 'neutral')

        if region == 'TW' and os.path.exists(MACRO_MODEL_PATH):
            try:
                rf_features = _tw_rf_features_from_macro_data(macro_data)
                if rf_features is not None:
                    rf_model = joblib.load(MACRO_MODEL_PATH)
                    is_bull = rf_model.predict(rf_features)[0]
                    return ('offensive', max(score, 0.35)) if is_bull == 1 else ('defensive', min(score, -0.35))
                real_count = int(macro_data.get('rows', [{}])[0].get('real_count', 0) or 0)
                log(f'[MARKET-WARN] TW macro dimensions incomplete ({real_count} real). RF skipped; using rule score.')
            except Exception as e:
                log(f'[MARKET-WARN] RF model skipped: {e}')

        if score <= -0.25:
            return 'defensive', score
        if score >= 0.15:
            return 'offensive', score
        return 'neutral', score
    except Exception as e:
        log_exception('[MARKET-ERROR]', e)
        return 'neutral', 0.0



def _qpro_macro_mode_from_score(score):
    score = safe_float(score, 0.0) or 0.0
    if score >= 0.15:
        return 'offensive'
    if score <= -0.25:
        return 'defensive'
    return 'neutral'


def _qpro_macro_mode_style(mode):
    mode = str(mode or 'neutral')
    if mode == 'offensive':
        return {
            'emoji': '🟢',
            'text': '偏多進攻 (Wave Up)',
            'short': '偏多',
            'color': '#10b981',
            'bg': '#064e3b',
            'advice': '盤勢偏多，優先觀察強勢突破、VCP、Minervini 趨勢股；可正常分批，但仍避免追高。',
        }
    if mode == 'defensive':
        return {
            'emoji': '🔴',
            'text': '偏空防守 (Wave Down)',
            'short': '偏空',
            'color': '#ef4444',
            'bg': '#7f1d1d',
            'advice': '波段風險升高，降低持股與追價比例；偏向現金、防禦股、高股息或 ETF 觀察。',
        }
    return {
        'emoji': '🟡',
        'text': '震盪中性 (Neutral)',
        'short': '震盪',
        'color': '#f59e0b',
        'bg': '#78350f',
        'advice': '多空分歧，建議降低倉位，只挑技術、基本面、籌碼同時偏強的標的，採分批進出。',
    }


def _qpro_macro_history_path(region='TW'):
    return os.path.join(REPORT_DIR, f'{str(region).upper()}_macro_history.csv')


def _qpro_update_macro_history(region, macro_df):
    """Persist macro rows so the report can build a true 10/20-day trend over time."""
    try:
        if macro_df is None or macro_df.empty or 'Date' not in macro_df.columns:
            return macro_df
        path = _qpro_macro_history_path(region)
        os.makedirs(REPORT_DIR, exist_ok=True)
        keep_cols = [
            'Date', 'IndexRet', 'IndexRetDisplay',
            'Spot', 'SpotDisplay', 'Future', 'FutureDisplay', 'PCR', 'PCRDisplay',
            'Retail', 'RetailDisplay', 'Top10', 'Top10Display', 'Score', 'RealCount'
        ]
        x = macro_df.copy()
        for c in keep_cols:
            if c not in x.columns:
                x[c] = np.nan
        x = x[keep_cols]

        # Purge bogus MTX retail proxy values generated by older parsers.
        # Normal values should be within +/-200%; values like -20000% mean the
        # denominator was not MTX open interest.
        def _qpro_sanitize_retail_history(df):
            try:
                if df is None or df.empty or 'Retail' not in df.columns:
                    return df
                r = pd.to_numeric(df['Retail'], errors='coerce')
                bad = r.notna() & ((r.abs() > 200) | (~np.isfinite(r)))
                if bad.any():
                    try:
                        log(f'[MACRO-HISTORY-WARN] Purged {int(bad.sum())} bogus Retail rows from macro history.')
                    except Exception:
                        pass
                    df.loc[bad, 'Retail'] = np.nan
                    if 'RetailDisplay' in df.columns:
                        df.loc[bad, 'RetailDisplay'] = 'N/A'
                return df
            except Exception:
                return df

        x = _qpro_sanitize_retail_history(x)
        if os.path.exists(path):
            old = pd.read_csv(path)
            old = _qpro_sanitize_retail_history(old)
            for c in keep_cols:
                if c not in old.columns:
                    old[c] = np.nan
            x = pd.concat([old[keep_cols], x], ignore_index=True)
        x['Date'] = pd.to_datetime(x['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        x = x.dropna(subset=['Date']).drop_duplicates('Date', keep='last').sort_values('Date')
        x.tail(260).to_csv(path, index=False)
        return x
    except Exception as e:
        try:
            log(f'[MACRO-HISTORY-WARN] {e}')
        except Exception:
            pass
        return macro_df


def _qpro_macro_score_from_value(kind, value):
    v = safe_float(value)
    if v is None:
        return 0.0
    kind = str(kind)
    if kind == 'IndexRet':
        if v >= 1.0: return 0.25
        if v >= 0.0: return 0.10
        if v <= -1.0: return -0.25
        return -0.10
    if kind == 'Spot':
        return _score_by_threshold(v, 50, -50, 0.25, -0.25)
    if kind == 'Future':
        return _score_by_threshold(v, 5000, -5000, 0.25, -0.25)
    if kind == 'PCR':
        if v >= 170: return 0.10   # 過熱不再過度加分
        if v >= 130: return 0.20
        if v <= 80: return -0.20
        return 0.0
    if kind == 'Retail':
        # 散戶偏多是反指標偏空；散戶偏空是反指標偏多
        if v <= -20: return 0.20
        if v >= 20: return -0.20
        return 0.0
    if kind == 'Top10':
        return _score_by_threshold(v, 1000, -1000, 0.25, -0.25)
    return 0.0


def _qpro_macro_dim_label(kind, value):
    v = safe_float(value)
    if v is None:
        return 'N/A'
    score = _qpro_macro_score_from_value(kind, v)
    if kind == 'Retail':
        return '散戶偏空' if score > 0 else ('散戶偏多' if score < 0 else '中性')
    if score > 0:
        return '偏多'
    if score < 0:
        return '偏空'
    return '中性'


def _qpro_macro_cell_class(kind, value):
    v = safe_float(value)
    if v is None:
        return 'neutral'
    score = _qpro_macro_score_from_value(kind, v)
    if score > 0:
        return 'pos'
    if score < 0:
        return 'neg'
    return 'neutral'


def _qpro_macro_svg_bar(dates, scores, width=650, height=230):
    if not dates:
        dates = []
    if not scores:
        scores = []
    n = max(len(scores), 1)
    pad_l, pad_r, pad_t, pad_b = 48, 18, 22, 42
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    y0 = pad_t + plot_h / 2
    parts = [f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"]
    parts.append("<rect width='100%' height='100%' rx='14' fill='#0f172a'/>")
    for i in range(5):
        y = pad_t + i * plot_h / 4
        parts.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width-pad_r}' y2='{y:.1f}' stroke='#334155' stroke-width='1'/>")
    parts.append(f"<line x1='{pad_l}' y1='{y0:.1f}' x2='{width-pad_r}' y2='{y0:.1f}' stroke='#94a3b8' stroke-width='1' opacity='.55'/>")
    bar_gap = 10
    bar_w = max(10, (plot_w - bar_gap * (n + 1)) / n)
    for i, sc in enumerate(scores):
        sc = max(-1.0, min(1.0, safe_float(sc, 0.0) or 0.0))
        x = pad_l + bar_gap + i * (bar_w + bar_gap)
        h = abs(sc) * plot_h / 2
        y = y0 - h if sc >= 0 else y0
        color = '#10b981' if sc >= 0 else '#ef4444'
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{max(h, 2):.1f}' rx='5' fill='{color}' opacity='.78'/>")
        label = str(dates[i])[-5:] if i < len(dates) else ''
        parts.append(f"<text x='{x+bar_w/2:.1f}' y='{height-16}' text-anchor='middle' fill='#94a3b8' font-size='11'>{html.escape(label)}</text>")
    parts.append("<text x='20' y='20' fill='#e2e8f0' font-size='14' font-weight='700'>20日交易口四維趨勢</text>")
    parts.append('</svg>')
    return ''.join(parts)


def _qpro_macro_svg_radar(labels, values, width=310, height=260, color='#f59e0b'):
    cx, cy = width / 2, height / 2 + 8
    rmax = min(width, height) * 0.34
    n = len(labels) or 1
    pts = []
    label_pts = []
    for i, val in enumerate(values):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        rr = rmax * max(0, min(100, safe_float(val, 50) or 50)) / 100
        pts.append((cx + math.cos(ang) * rr, cy + math.sin(ang) * rr))
        label_pts.append((cx + math.cos(ang) * (rmax + 28), cy + math.sin(ang) * (rmax + 28)))
    poly = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
    parts = [f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"]
    parts.append("<rect width='100%' height='100%' rx='14' fill='#0f172a'/>")
    for step in [0.25, 0.5, 0.75, 1.0]:
        ring=[]
        for i in range(n):
            ang=-math.pi/2+2*math.pi*i/n
            ring.append((cx+math.cos(ang)*rmax*step, cy+math.sin(ang)*rmax*step))
        parts.append("<polygon points='" + ' '.join(f'{x:.1f},{y:.1f}' for x,y in ring) + "' fill='none' stroke='#334155' stroke-width='1'/>")
    for i in range(n):
        ang=-math.pi/2+2*math.pi*i/n
        x=cx+math.cos(ang)*rmax; y=cy+math.sin(ang)*rmax
        parts.append(f"<line x1='{cx:.1f}' y1='{cy:.1f}' x2='{x:.1f}' y2='{y:.1f}' stroke='#334155'/>")
    parts.append(f"<polygon points='{poly}' fill='{color}55' stroke='{color}' stroke-width='2'/>")
    for (x,y) in pts:
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}'/>")
    for label,(x,y) in zip(labels,label_pts):
        parts.append(f"<text x='{x:.1f}' y='{y:.1f}' text-anchor='middle' dominant-baseline='middle' fill='#cbd5e1' font-size='11'>{html.escape(str(label))}</text>")
    parts.append('</svg>')
    return ''.join(parts)


def create_macro_dashboard_image(market_mode, macro_score, output_path, region='TW'):
    """Render macro dashboard as a four/six-dimensional trend report.

    TW output intentionally resembles a daily chip-flow report:
      - 10-day table
      - 20-day score trend
      - radar chart
      - conclusion card
    It uses real data only. Missing sources stay N/A.
    """
    region = 'US' if str(region).upper() == 'US' else 'TW'

    if region == 'TW':
        macro_df = macro_score if isinstance(macro_score, pd.DataFrame) else fetch_real_macro_data(days=20)
        if macro_df is None or macro_df.empty:
            macro_df = fetch_real_macro_data(days=5)
        macro_df = _qpro_update_macro_history(region, macro_df)
        if macro_df is None or macro_df.empty:
            macro_df = pd.DataFrame(columns=['Date','IndexRetDisplay','SpotDisplay','FutureDisplay','PCRDisplay','RetailDisplay','Top10Display','Score','RealCount'])

        # normalize date and keep last 20 rows
        macro_df = macro_df.copy()
        if 'Date' in macro_df.columns:
            macro_df['Date'] = pd.to_datetime(macro_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
            macro_df = macro_df.dropna(subset=['Date']).drop_duplicates('Date', keep='last').sort_values('Date')
        trend_df = macro_df.tail(20).copy()
        table_df = macro_df.tail(10).copy()
        latest = macro_df.iloc[-1].to_dict() if not macro_df.empty else {}
        latest_score = safe_float(latest.get('Score'), safe_float(macro_score, 0.0)) or 0.0
        market_mode = _qpro_macro_mode_from_score(latest_score)
        style = _qpro_macro_mode_style(market_mode)
        mode_color = style['color']

        def _disp(row, key):
            val = row.get(key + 'Display')
            if val is not None and str(val) not in ('nan','NaN','None'):
                return str(val)
            raw = row.get(key)
            if raw is None or pd.isna(raw):
                return 'N/A'
            if key in ('Spot','Future','Top10'):
                return _format_signed_number(raw, 0)
            if key == 'Retail':
                return f'{safe_float(raw):+.1f}%'
            if key == 'PCR':
                return _format_ratio(raw, 1)
            return str(raw)

        table_rows = ''
        for _, row in table_df.iterrows():
            score = safe_float(row.get('Score'), 0.0) or 0.0
            mode = _qpro_macro_mode_from_score(score)
            st = _qpro_macro_mode_style(mode)
            direction = st['short']
            real_count = int(safe_float(row.get('RealCount'), 0) or 0)
            idx_cls = _qpro_macro_cell_class('IndexRet', row.get('IndexRet'))
            spot_cls = _qpro_macro_cell_class('Spot', row.get('Spot'))
            fut_cls = _qpro_macro_cell_class('Future', row.get('Future'))
            pcr_cls = _qpro_macro_cell_class('PCR', row.get('PCR'))
            retail_cls = _qpro_macro_cell_class('Retail', row.get('Retail'))
            top_cls = _qpro_macro_cell_class('Top10', row.get('Top10'))
            table_rows += f"""
                <tr>
                    <td>{html.escape(str(row.get('Date','N/A'))[-5:])}</td>
                    <td class='{idx_cls}'>{html.escape(str(row.get('IndexRetDisplay','N/A')))}</td>
                    <td class='{spot_cls}'>{html.escape(_disp(row, 'Spot'))}</td>
                    <td class='{fut_cls}'>{html.escape(_disp(row, 'Future'))}</td>
                    <td class='{pcr_cls}'>{html.escape(_disp(row, 'PCR'))}</td>
                    <td class='{retail_cls}'>{html.escape(_disp(row, 'Retail'))}</td>
                    <td class='{top_cls}'>{html.escape(_disp(row, 'Top10'))}</td>
                    <td class='{ 'pos' if score >= 0.15 else ('neg' if score <= -0.25 else 'warn') }'><b>{score:+.2f}</b><br><span class='small'>{real_count}/6 real</span></td>
                    <td class='{ 'pos' if mode == 'offensive' else ('neg' if mode == 'defensive' else 'warn') }'>{html.escape(direction)}</td>
                    <td>{html.escape(_qpro_market_status_label(score, real_count))}</td>
                </tr>
            """

        trend_dates = trend_df['Date'].astype(str).tolist() if 'Date' in trend_df.columns else []
        trend_scores = [safe_float(x, 0.0) or 0.0 for x in trend_df.get('Score', pd.Series(dtype=float)).tolist()]
        trend_svg = _qpro_macro_svg_bar(trend_dates, trend_scores, width=790, height=235)

        radar_kinds = ['IndexRet', 'Spot', 'Future', 'PCR', 'Retail', 'Top10']
        radar_labels = ['指數', '現貨', '期貨', 'PCR', '散戶', '十大']
        radar_scores = []
        for k in radar_kinds:
            raw = latest.get(k if k != 'IndexRet' else 'IndexRet')
            radar_scores.append(max(0, min(100, 50 + _qpro_macro_score_from_value(k, raw) * 100)))
        radar_svg = _qpro_macro_svg_radar(radar_labels, radar_scores, width=300, height=235, color=mode_color)

        # conclusion details
        dim_scores = {k: _qpro_macro_score_from_value(k, latest.get(k)) for k in radar_kinds}
        valid_dims = [k for k in radar_kinds if safe_float(latest.get(k)) is not None]
        pos_dims = [k for k in valid_dims if dim_scores[k] > 0]
        neg_dims = [k for k in valid_dims if dim_scores[k] < 0]
        consistency = max(len(pos_dims), len(neg_dims)) / len(valid_dims) if valid_dims else 0.0
        lead_map = {'IndexRet':'指數動能', 'Spot':'外資現貨', 'Future':'外資期貨', 'PCR':'PCR', 'Retail':'散戶反指標', 'Top10':'十大交易人'}
        strongest = max(valid_dims, key=lambda k: abs(dim_scores.get(k, 0.0)), default=None)
        weakest = min(valid_dims, key=lambda k: dim_scores.get(k, 0.0), default=None)
        strongest_txt = lead_map.get(strongest, 'N/A') if strongest else 'N/A'
        weakest_txt = lead_map.get(weakest, 'N/A') if weakest else 'N/A'
        confidence = '高' if consistency >= 0.67 and len(valid_dims) >= 4 else ('中' if consistency >= 0.50 and len(valid_dims) >= 3 else '低')

        html_content = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset='UTF-8'>
            <style>
                body {{ margin:0; width:1280px; background:#f8fafc; font-family:'Noto Sans CJK TC','Microsoft JhengHei','Segoe UI',Arial,sans-serif; color:#0f172a; }}
                .wrap {{ padding:18px; background:#f8fafc; }}
                .panel {{ background:#ffffff; border:1px solid #dbe3ef; border-radius:18px; overflow:hidden; box-shadow:0 18px 55px rgba(15,23,42,.14); }}
                .head {{ padding:18px 22px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e2e8f0; background:linear-gradient(135deg,#ffffff,#f1f5f9); }}
                .title {{ font-size:28px; font-weight:950; letter-spacing:.5px; }}
                .sub {{ margin-top:6px; color:#64748b; font-size:13px; }}
                .badge {{ border:2px solid {mode_color}; color:{mode_color}; background:{mode_color}18; border-radius:14px; padding:10px 16px; font-size:20px; font-weight:950; white-space:nowrap; }}
                .content {{ padding:18px 20px 22px; overflow:hidden; }}
                table {{ width:100%; border-collapse:collapse; table-layout:fixed; font-size:13px; }}
                th {{ background:#e8eef6; color:#334155; padding:10px 6px; font-weight:900; border:1px solid #d7e0ec; }}
                td {{ padding:10px 6px; text-align:center; border:1px solid #e2e8f0; font-weight:700; overflow:hidden; text-overflow:ellipsis; }}
                tr:nth-child(even) td {{ background:#f8fafc; }}
                .pos {{ color:#047857; }} .neg {{ color:#b91c1c; }} .warn {{ color:#b45309; }} .neutral {{ color:#64748b; }}
                .small {{ font-size:11px; color:#64748b; font-weight:700; }}
                .grid {{ display:grid; grid-template-columns:330px 1fr; gap:14px; margin-top:18px; align-items:stretch; }}
                .box {{ background:#ffffff; border:1px solid #dbe3ef; border-radius:16px; padding:13px; min-width:0; overflow:hidden; }}
                .box h3 {{ margin:0 0 10px; font-size:18px; color:#111827; white-space:normal; }}
                .conclusion {{ background:#f8fafc; border-left:7px solid {mode_color}; grid-column:1 / -1; }}
                .kpi {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:10px 0; }}
                .kpi div {{ background:white; border:1px solid #e2e8f0; border-radius:11px; padding:8px; min-width:0; }}
                .k {{ color:#64748b; font-size:11px; font-weight:800; }} .v {{ margin-top:3px; font-size:16px; font-weight:950; white-space:normal; word-break:break-word; }}
                .note {{ margin-top:14px; color:#64748b; font-size:12px; line-height:1.55; }}
                ul {{ margin:6px 0 0 16px; padding:0; line-height:1.55; font-weight:700; font-size:13px; columns:2; column-gap:28px; }}
            </style>
        </head>
        <body>
            <div class='wrap'>
                <div id='capture-area' class='panel'>
                    <div class='head'>
                        <div>
                            <div class='title'>🌊 台股籌碼四維趨勢報告</div>
                            <div class='sub'>近 10 交易日表格｜20 日趨勢｜資料源：TWSE / TAIFEX / FinMind fallback｜產生時間 {now_str()}</div>
                        </div>
                        <div class='badge'>系統狀態：{style['emoji']} {style['text']}｜Score {latest_score:+.2f}</div>
                    </div>
                    <div class='content'>
                        <table>
                            <tr>
                                <th>日期</th><th>指數動能</th><th>外資現貨</th><th>外資期貨</th><th>PCR</th><th>散戶多空</th><th>十大交易人</th><th>四維分數</th><th>方向</th><th>市場狀態</th>
                            </tr>
                            {table_rows}
                        </table>
                        <div class='grid'>
                            <div class='box'>
                                <h3>📡 四維雷達</h3>
                                {radar_svg}
                                <div class='note'>雷達分數以 50 為中性；越外圈代表該維度越偏多。</div>
                            </div>
                            <div class='box'>
                                <h3>📈 20日交易口維趨勢</h3>
                                {trend_svg}
                                <div class='note'>快照型資料不會硬補歷史；若資料源缺漏，該日會顯示 N/A，避免假資料。</div>
                            </div>
                            <div class='box conclusion'>
                                <h3>🧭 四維結論：{style['short']}｜信心：{confidence}</h3>
                                <div class='kpi'>
                                    <div><div class='k'>總分</div><div class='v'>{latest_score:+.2f}</div></div>
                                    <div><div class='k'>一致性</div><div class='v'>{consistency:.2f}</div></div>
                                    <div><div class='k'>主導維度</div><div class='v'>{html.escape(strongest_txt)}</div></div>
                                    <div><div class='k'>風險維度</div><div class='v'>{html.escape(weakest_txt)}</div></div>
                                </div>
                                <b>今日重點：</b>
                                <ul>
                                    <li>外資現貨：{html.escape(_disp(latest, 'Spot'))}，{html.escape(_qpro_macro_dim_label('Spot', latest.get('Spot')))}</li>
                                    <li>外資期貨：{html.escape(_disp(latest, 'Future'))}，{html.escape(_qpro_macro_dim_label('Future', latest.get('Future')))}</li>
                                    <li>PCR：{html.escape(_disp(latest, 'PCR'))}，{html.escape(_qpro_macro_dim_label('PCR', latest.get('PCR')))}</li>
                                    <li>散戶多空：{html.escape(_disp(latest, 'Retail'))}，{html.escape(_qpro_macro_dim_label('Retail', latest.get('Retail')))}</li>
                                    <li>十大交易人：{html.escape(_disp(latest, 'Top10'))}，{html.escape(_qpro_macro_dim_label('Top10', latest.get('Top10')))}</li>
                                </ul>
                                <div class='note'><b>操作建議：</b>{html.escape(style['advice'])}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

    else:
        macro_data = get_macro_dashboard_data(region)
        rows = macro_data.get('rows', [])
        if rows:
            macro_score = safe_float(macro_data.get('latest_score'), macro_score) or macro_score
            market_mode = macro_data.get('latest_mode', market_mode)
        market_mode = _qpro_macro_mode_from_score(macro_score)
        style = _qpro_macro_mode_style(market_mode)
        table_rows = ''
        chart_labels = []
        chart_scores = []
        for r in rows[:5]:
            field_map = {f['key']: f for f in r.get('fields', [])}
            score = safe_float(r.get('score'), 0.0) or 0.0
            score_class = 'pos' if score >= 0.15 else ('neg' if score <= -0.25 else 'warn')
            idx_ret = r.get('index_ret_display', 'N/A')
            idx_class = _td_class_by_value(idx_ret)
            table_rows += f"""
                        <tr>
                            <td>{r.get('label', 'N/A')}<br><span class='small'>{r.get('date', '')}</span></td>
                            <td class='{idx_class}'>{idx_ret}</td>
                            <td>{field_map.get('VIX_Risk', {}).get('display', 'N/A')}</td>
                            <td>{field_map.get('Breadth_Proxy', {}).get('display', 'N/A')}</td>
                            <td>{field_map.get('Risk_Appetite', {}).get('display', 'N/A')}</td>
                            <td class='{score_class}' style='font-weight:bold;'>{score:+.2f}</td>
                        </tr>
            """
            chart_labels.append(r.get('label', 'N/A'))
            chart_scores.append(round(score, 3))
        trend_svg = _qpro_macro_svg_bar(list(reversed(chart_labels)), list(reversed(chart_scores)), width=640, height=240)
        html_content = f"""
        <!doctype html><html><head><meta charset='UTF-8'><style>
        body {{ margin:0; width:1100px; background:#0f172a; color:#f8fafc; font-family:'Segoe UI','Microsoft JhengHei',Arial,sans-serif; }}
        .dashboard {{ margin:20px; background:#1e293b; border:1px solid #334155; border-radius:18px; padding:28px; }}
        .header {{ display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #334155; padding-bottom:18px; }}
        .title {{ font-size:28px; font-weight:900; }} .sub {{ color:#94a3b8; margin-top:8px; }}
        .badge {{ color:{style['color']}; border:2px solid {style['color']}; background:{style['color']}22; border-radius:14px; padding:12px 18px; font-size:18px; font-weight:900; }}
        table {{ width:100%; border-collapse:collapse; margin-top:24px; }} th {{ background:#334155; padding:13px; }} td {{ border-bottom:1px solid #334155; padding:13px; text-align:center; }}
        .pos {{ color:#10b981; }} .neg {{ color:#ef4444; }} .warn {{ color:#f59e0b; }} .small {{ color:#94a3b8; font-size:11px; }}
        .chart {{ margin-top:22px; }}
        </style></head><body><div class='dashboard' id='capture-area'>
        <div class='header'><div><div class='title'>🇺🇸 美股四維風險偏好引擎</div><div class='sub'>S&P 500 / VIX / Breadth / HYG-TLT proxy｜{now_str()}</div></div><div class='badge'>{style['emoji']} {style['text']}｜Score {safe_float(macro_score,0):+.2f}</div></div>
        <table><tr><th>日期</th><th>S&P 動能</th><th>VIX</th><th>市場廣度</th><th>風險偏好</th><th>分數</th></tr>{table_rows}</table>
        <div class='chart'>{trend_svg}</div>
        </div></body></html>
        """

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1360 if region == 'TW' else 1100, 'height': 1120}, device_scale_factor=2)
            page.set_content(html_content, wait_until='networkidle')
            page.locator('#capture-area').screenshot(path=output_path)
            browser.close()
        return output_path
    except Exception as e:
        log_exception('[MACRO-DASHBOARD-IMG-ERROR]', e)
        return None

def get_defensive_etf_pool(region='TW'):
    if region == 'US':
        return ['SPY', 'QQQ', 'TLT', 'IEF', 'GLD', 'SH']
    return ['0050.TW', '0056.TW', '00713.TW', '00878.TW', '00679B.TWO', '00687B.TWO', '00632R.TW']

def get_us_defensive_etf_pool():
    return get_defensive_etf_pool('US')

# ==========================================
# Basic utilities
# ==========================================
def now_str(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def log(msg): print(f'[{now_str()}] {msg}', flush=True)
def log_exception(prefix, exc): log(f'{prefix}: {exc}'); print(traceback.format_exc(), flush=True)

def safe_float(v, default=None):
    try:
        if v is None: return default
        if isinstance(v, (int, float, np.integer, np.floating)):
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv): return default
            return fv
        s = str(v).strip()
        if s in ('', '-', '--', 'N/A', 'nan', 'None'): return default
        s = s.replace(',', '').replace('%', '').replace('元', '').replace('億', '').replace('張', '').replace('倍', '').strip()
        m = re.search(r'[-+]?\d+(?:\.\d+)?', s)
        return float(m.group()) if m else default
    except Exception: return default

def safe_pct_str(v): return 'N/A' if v is None else f'{v:.1f}%'
def safe_num_str(v, digits=2): return 'N/A' if v is None else f'{v:.{digits}f}'


# ==========================================
# UI helpers for dashboard rendering
# 台股慣例：紅 = 正向 / 買超 / 獲利，綠 = 負向 / 賣超 / 虧損
# 美股儀表板也沿用同一套顏色，避免 TW / US 版面邏輯不一致。
# ==========================================
def ui_tone_class(value, neutral_zero=True):
    v = safe_float(value)
    if v is None:
        return 'tone-neutral'
    if neutral_zero and abs(v) < 1e-12:
        return 'tone-neutral'
    return 'tone-good' if v > 0 else 'tone-bad'


def ui_tone_pct(value):
    return ui_tone_class(value)


def ui_tone_money(value):
    return ui_tone_class(value)


def ui_display_signed(value, digits=0, unit=''):
    v = safe_float(value)
    if v is None:
        return 'N/A'
    return f'{v:+,.{digits}f}{unit}'


def ui_company_name_from_raw_text(raw_text, ticker=''):
    if not raw_text:
        return None
    ticker_num = str(ticker).replace('.TWO', '').replace('.TW', '').strip()
    for line in str(raw_text).splitlines()[:50]:
        s = line.strip().replace('*', '').replace('#', '').strip()
        if not s or s.startswith('|'):
            continue
        for key in ['公司名稱', '股票名稱', '名稱']:
            if key in s and ('：' in s or ':' in s):
                name = s.split('：', 1)[-1].strip() if '：' in s else s.split(':', 1)[-1].strip()
                name = re.sub(r'[`\[\]()]', '', name).strip()
                if name and name not in ('N/A', '-', '--'):
                    return clip_text(name, 24)
        if ticker_num:
            m = re.match(rf'^{re.escape(ticker_num)}\s+(.+)$', s)
            if m:
                return clip_text(m.group(1).strip(), 24)
        m = re.match(r'^(\d{4})\s+(.+)$', s)
        if m:
            return clip_text(m.group(2).strip(), 24)
    return None


def ui_stock_name(item):
    ticker = str(item.get('ticker', '')).strip()
    profile = item.get('profile_info') or {}
    for key in ['company_name', 'shortName', 'longName', 'name']:
        val = profile.get(key)
        if val and str(val).strip() not in ('N/A', '-', '--'):
            return clip_text(str(val).strip(), 24)
    name = ui_company_name_from_raw_text(profile.get('raw_text'), ticker)
    return name or ticker


def ui_stock_cell(item):
    ticker = html.escape(str(item.get('ticker', '')).strip())
    name = html.escape(ui_stock_name(item))
    if name and name != ticker:
        return f'<div class="ticker">{ticker}</div><div class="cname">{name}</div>'
    return f'<div class="ticker">{ticker}</div>'


def ui_metric_cell(label, value, cls='tone-neutral'):
    return f'''
    <div class="metric {cls}">
        <div class="metric-label">{html.escape(str(label))}</div>
        <div class="metric-val">{html.escape(str(value))}</div>
    </div>
    '''


def ui_chip_summary(fin):
    total_5d = safe_float(fin.get('total_5d'))
    foreign_5d = safe_float(fin.get('foreign_5d'))
    trust_5d = safe_float(fin.get('trust_5d'))
    dealer_5d = safe_float(fin.get('dealer_5d'))

    main = total_5d if total_5d is not None else foreign_5d
    cls = ui_tone_class(main)
    lines = []
    if total_5d is not None:
        lines.append(f'三大5D {total_5d:+,.0f}張')
    if foreign_5d is not None:
        lines.append(f'外資5D {foreign_5d:+,.0f}張')
    if trust_5d is not None and abs(trust_5d) > 0:
        lines.append(f'投信5D {trust_5d:+,.0f}張')
    if dealer_5d is not None and abs(dealer_5d) > 0 and len(lines) < 2:
        lines.append(f'自營5D {dealer_5d:+,.0f}張')

    if not lines:
        return ui_metric_cell('籌碼面', 'N/A', 'tone-neutral')
    return ui_metric_cell('籌碼面', ' / '.join(lines[:2]), cls)


def ui_fund_summary(fin):
    eps = safe_float(fin.get('eps_ttm'))
    eps_q = safe_float(fin.get('eps_latest_quarter'))
    yoy = safe_float(fin.get('single_month_yoy'))

    # 基本面主色：EPS/營收有盈利成長偏紅；虧損/衰退偏綠；全 N/A 灰色。
    score_parts = []
    for v in [eps, eps_q, yoy]:
        if v is not None:
            score_parts.append(v)
    main = sum(score_parts) if score_parts else None
    cls = ui_tone_class(main)

    parts = []
    if eps is not None:
        parts.append(f'EPS {eps:.2f}')
    elif eps_q is not None:
        parts.append(f'EPS季 {eps_q:.2f}')
    if yoy is not None:
        parts.append(f'YoY {yoy:+.1f}%')

    if not parts:
        return ui_metric_cell('基本面', 'N/A', 'tone-neutral')
    return ui_metric_cell('基本面', ' / '.join(parts), cls)


def ui_card_cls(value):
    return ui_tone_class(value)

def clip_text(text, limit=180): return '' if not text else (str(text).strip() if len(str(text).strip()) <= limit else str(text).strip()[:limit] + '...')

def normalize_ticker(ticker):
    ticker = str(ticker or '').strip().upper()
    # Keep Taiwan suffixes exactly.  Important: .TWO is not .TW + O.
    if re.fullmatch(r'\d{4}\.TWO', ticker):
        return ticker
    if re.fullmatch(r'\d{4}\.TW', ticker):
        return ticker
    if re.fullmatch(r'\d{4}', ticker):
        return ticker + '.TW'
    # Yahoo Finance uses hyphenated class-share symbols, e.g. BRK-B/BF-B.
    if re.match(r'^[A-Z]{1,5}\.[A-Z]$', ticker):
        return ticker.replace('.', '-')
    return ticker

def run_cmd(cmd, cwd, timeout_sec, step_name):
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_sec)
        return {'ok': result.returncode == 0, 'timeout': False, 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired as e: return {'ok': False, 'timeout': True, 'stdout': getattr(e, 'stdout', ''), 'stderr': getattr(e, 'stderr', '')}
    except Exception as e: return {'ok': False, 'timeout': False, 'stdout': '', 'stderr': str(e)}

def update_my_tw_coverage(chat_id=None):
    safe_send_message(chat_id, '🔄 正在同步更新本地資料庫...')
    if not os.path.isdir(MY_TW_COVERAGE_PATH): return
    run_cmd(['git', 'pull'], MY_TW_COVERAGE_PATH, GIT_PULL_TIMEOUT, 'git pull')
    run_cmd([sys.executable, 'scripts/update_financials.py'], MY_TW_COVERAGE_PATH, FINANCIAL_UPDATE_TIMEOUT, 'update_financials.py')
    
def is_us_ticker(ticker):
    return not str(ticker).strip().upper().endswith(('.TW', '.TWO'))

# QPRO_FIX_20260515_2255_TWO_CODE_COMPANY_CHIP
def qpro_tw_code(ticker):
    """Return pure 4-digit Taiwan stock code.

    Critical fix: 6425.TWO must become 6425, not 6425O.
    Do NOT remove .TW before .TWO.
    """
    s = str(ticker or '').strip().upper()
    m = re.search(r'(\d{4})', s)
    return m.group(1) if m else ''

def qpro_tw_exchange_suffix(ticker, default='.TW'):
    s = str(ticker or '').strip().upper()
    if s.endswith('.TWO'):
        return '.TWO'
    if s.endswith('.TW'):
        return '.TW'
    return default

# ==========================================
# Data collection
# ==========================================
def get_company_profile(ticker_num, ticker_full=None, yf_info=None):
    is_us = is_us_ticker(ticker_full) if ticker_full else False
    if is_us:
        if yf_info:
            industry = yf_info.get('industry', 'N/A')
            desc = yf_info.get('longBusinessSummary', '查無美股業務描述')
            safe_desc = clip_text(desc.replace('*', '').replace('_', ''), 200)
            return {'profile': safe_desc, 'industry': industry, 'raw_text': None, 'company_name': yf_info.get('shortName') or yf_info.get('longName') or ticker_num}
        return {'profile': '無法取得美股資料', 'industry': 'N/A', 'raw_text': None, 'company_name': ticker_num}

    try:
        if not MY_TW_COVERAGE_PATH or not os.path.isdir(MY_TW_COVERAGE_PATH):
            return {'profile': '未設定 My-TW-Coverage，本地公司資料略過', 'industry': 'N/A', 'raw_text': None, 'company_name': str(ticker_num)}

        target_file = None
        for root, dirs, files in os.walk(MY_TW_COVERAGE_PATH):
            for file in files:
                if file.startswith(str(ticker_num)) and file.endswith('.md'):
                    target_file = os.path.join(root, file)
                    break
            if target_file: break

        if not target_file:
            return {'profile': '查無資料', 'industry': 'N/A', 'raw_text': None, 'company_name': str(ticker_num)}

        with open(target_file, 'r', encoding='utf-8') as f: content = f.read()

        industry = 'N/A'
        for line in content.splitlines():
            s = line.strip()
            if '產業' in s and ('：' in s or ':' in s):
                industry = s.split('：', 1)[-1].strip() if '：' in s else s.split(':', 1)[-1].strip()
                industry = clip_text(industry.replace('*', ''), 60)
                break

        desc = None
        for line in content.splitlines():
            s = line.strip()
            if len(s) > 20 and not s.startswith('#') and not s.startswith('|'):
                desc = clip_text(s, 180)
                break

        safe_desc = (desc or '查無業務描述').replace('*', '').replace('_', '')
        company_name = ui_company_name_from_raw_text(content, ticker_full or ticker_num) or str(ticker_num)
        return {'profile': safe_desc, 'industry': industry, 'raw_text': content, 'company_name': company_name}
    except Exception:
        return {'profile': '讀取失敗', 'industry': 'N/A', 'raw_text': None, 'company_name': str(ticker_num)}

def fetch_goodinfo_data(ticker_num):
    try:
        return data_sources.fetch_goodinfo_pages(str(ticker_num), ttl_hours=24)
    except Exception as e:
        log(f'[GOODINFO-WARN] {ticker_num} cached fetch failed: {e}')
        return "", ""

def parse_financials_from_mytwcoverage(md_text):
    result = {'eps_ttm': None, 'eps_latest_quarter': None, 'single_month_yoy': None, 'single_month_mom': None, 'source': []}
    if not md_text: return result
    text = md_text.replace(',', '')
    patterns = {
        'eps_ttm': [r'近四季\s*EPS[:：]?\s*([\-]?\d+(?:\.\d+)?)', r'TTM\s*EPS[:：]?\s*([\-]?\d+(?:\.\d+)?)'],
        'eps_latest_quarter': [r'最新一季\s*EPS[:：]?\s*([\-]?\d+(?:\.\d+)?)', r'單季\s*EPS[:：]?\s*([\-]?\d+(?:\.\d+)?)'],
        'single_month_yoy': [r'營收年增(?:率)?[:：]?\s*([\-]?\d+(?:\.\d+)?)\s*%', r'YoY[:：]?\s*([\-]?\d+(?:\.\d+)?)\s*%'],
        'single_month_mom': [r'營收月增(?:率)?[:：]?\s*([\-]?\d+(?:\.\d+)?)\s*%', r'MoM[:：]?\s*([\-]?\d+(?:\.\d+)?)\s*%']
    }
    for key, plist in patterns.items():
        for p in plist:
            m = re.search(p, text, re.IGNORECASE)
            if m: result[key] = safe_float(m.group(1)); break
    if any(v is not None for k, v in result.items() if k != 'source'): result['source'].append('My-TW-Coverage')
    return result

def normalize_goodinfo_table(df):
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [' '.join([str(c).strip() for c in col if str(c).strip() not in ('', 'nan')]) for col in x.columns]
    else: x.columns = [str(c).strip() for c in x.columns]
    x = x.dropna(how='all').dropna(axis=1, how='all')
    x.columns = [str(c).replace('\n', ' ').replace('\r', ' ').strip() for c in x.columns]
    return x

def get_goodinfo_tables(html):
    tables = []
    try:
        raw = pd.read_html(StringIO(html))
        for df in raw: tables.append(normalize_goodinfo_table(df))
    except Exception: pass
    return tables

def parse_goodinfo_revenue_from_text(html):
    text = re.sub(r'\s+', ' ', html.replace(',', ''))
    out = {'single_month_revenue': None, 'single_month_mom': None, 'single_month_yoy': None, 'ytd_revenue': None, 'ytd_yoy': None, 'source': []}
    patterns = {
        'single_month_revenue': [r'3月份單月.*?營收金額.*?(\d+(?:\.\d+)?)', r'單月營收.*?(\d+(?:\.\d+)?)', r'營收金額.*?(\d+(?:\.\d+)?)'],
        'single_month_mom': [r'月增率.*?([\-+]?\d+(?:\.\d+)?)'],
        'single_month_yoy': [r'年增率.*?([\-+]?\d+(?:\.\d+)?)']
    }
    for key, plist in patterns.items():
        for p in plist:
            m = re.search(p, text, re.IGNORECASE)
            if m: out[key] = safe_float(m.group(1)); break
    if any(v is not None for k, v in out.items() if k != 'source'): out['source'].append('Goodinfo-Text')
    return out

def parse_goodinfo_revenue_table(tables):
    result = {'single_month_revenue': None, 'single_month_mom': None, 'single_month_yoy': None, 'ytd_revenue': None, 'ytd_yoy': None, 'source': []}
    for df in tables:
        cols = [str(c).replace(' ', '').replace('\n', '') for c in df.columns]
        joined = ' | '.join(cols)
        if not (('營收' in joined) and ('月增' in joined or 'MoM' in joined) and ('年增' in joined or 'YoY' in joined)): continue
        try:
            work = df.copy().dropna(how='all')
            for ridx, row in work.iterrows():
                rev_col = mom_col = yoy_col = None
                for c in work.columns:
                    cs = str(c).replace(' ', '').replace('\n', '')
                    if rev_col is None and ('單月營收' in cs or '營收(億)' in cs or '月營收' in cs) and '增' not in cs and '累計' not in cs: rev_col = c
                    if mom_col is None and ('月增' in cs or 'MoM' in cs or 'M/M' in cs) and '累計' not in cs: mom_col = c
                    if yoy_col is None and ('年增' in cs or 'YoY' in cs or 'Y/Y' in cs) and '累計' not in cs: yoy_col = c
                single_rev = safe_float(row.get(rev_col))
                single_mom = safe_float(row.get(mom_col))
                single_yoy = safe_float(row.get(yoy_col))
                if any(v is not None for v in [single_rev, single_mom, single_yoy]):
                    result.update({'single_month_revenue': single_rev, 'single_month_mom': single_mom, 'single_month_yoy': single_yoy})
                    result['source'].append('Goodinfo-Table')
                    return result
        except Exception: pass
    return result

def parse_goodinfo_eps_table(tables):
    result = {'eps_latest_quarter': None, 'eps_ttm': None, 'eps_quarters': [], 'source': []}
    for df in tables:
        cols = [str(c).replace(' ', '').replace('\n', '') for c in df.columns]
        joined = ' | '.join(cols)
        if not ('EPS' in joined or '每股盈餘' in joined): continue
        if 'PER' in joined or 'PBR' in joined or '最高價' in joined: continue
        try:
            work = df.copy().dropna(how='all')
            eps_col = None
            for c in work.columns:
                cs = str(c).replace(' ', '').replace('\n', '')
                if eps_col is None and ('EPS' in cs or '每股盈餘' in cs) and '成長' not in cs and '平均' not in cs: eps_col = c
            if eps_col is None: continue
            vals = []
            for _, row in work.iterrows():
                v = safe_float(row[eps_col])
                if v is not None: vals.append(v)
                if len(vals) >= 4: break
            if vals:
                result['eps_latest_quarter'] = vals[0]
                result['eps_quarters'] = vals
                result['eps_ttm'] = round(sum(vals[:4]), 4)
                result['source'].append('Goodinfo-EPS')
                return result
        except Exception: pass
    return result

def parse_goodinfo_chip_table(tables):
    result = {
        'chips_summary': '近期法人動向平淡或無明顯建倉',
        'foreign_2d': 0, 'foreign_3d': 0, 'foreign_5d': 0, 'foreign_10d': 0,
        'trust_2d': 0, 'trust_3d': 0, 'trust_5d': 0, 'trust_10d': 0,
        'dealer_2d': 0, 'dealer_3d': 0, 'dealer_5d': 0, 'dealer_10d': 0,
        'total_2d': 0, 'total_3d': 0, 'total_5d': 0, 'total_10d': 0,
        'source': []
    }
    def parse_chip_value(val):
        if pd.isna(val): return 0
        s = str(val).replace(',', '').strip()
        if s in ('', '-', '--', 'N/A', 'nan', 'None'): return 0
        multiplier = 10000 if '萬' in s else 1
        s = s.replace('萬', '')
        m = re.search(r'[-+]?\d+(?:\.\d+)?', s)
        return float(m.group()) * multiplier if m else 0

    found_data = False
    for df in tables:
        cols_str = " ".join([str(c) for c in df.columns])
        if not ('2日' in cols_str and '5日' in cols_str and '10日' in cols_str): continue
        try:
            work = df.copy().dropna(how='all')
            current_actor = None
            for _, row in work.iterrows():
                row_str_no_space = (str(row.name) + " " + " ".join([str(x) for x in row.values])).replace(" ", "")
                if '外資' in row_str_no_space: current_actor = 'foreign'
                elif '投信' in row_str_no_space: current_actor = 'trust'
                elif '自營' in row_str_no_space: current_actor = 'dealer'
                elif '總計' in row_str_no_space: current_actor = None; continue
                
                if current_actor and ('買賣超張數' in row_str_no_space or ('買賣超' in row_str_no_space and '金額' not in row_str_no_space)):
                    for c in work.columns:
                        cs = str(c)
                        parsed_val = parse_chip_value(row[c])
                        if parsed_val != 0:
                            found_data = True
                            if '2日' in cs: result[f'{current_actor}_2d'] = parsed_val
                            elif '3日' in cs: result[f'{current_actor}_3d'] = parsed_val
                            elif '5日' in cs: result[f'{current_actor}_5d'] = parsed_val
                            elif '10日' in cs: result[f'{current_actor}_10d'] = parsed_val
        except Exception: pass

    if found_data:
        for d in ['2d', '3d', '5d', '10d']:
            result[f'total_{d}'] = result[f'foreign_{d}'] + result[f'trust_{d}'] + result[f'dealer_{d}']
        t5 = result['total_5d']
        result['chips_summary'] = f'近 5 日三大法人偏多，合計買超 {t5:.0f} 張' if t5 > 0 else (f'近 5 日三大法人偏空，賣超 {abs(t5):.0f} 張' if t5 < 0 else '三大法人中性')
        result['source'].append('Goodinfo-Chips')
    else:
        for k in result:
            if isinstance(result[k], (int, float)) and result[k] == 0: result[k] = None
    return result

def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None

def _merge_unique_sources(*source_lists):
    sources = []
    for source_list in source_lists:
        for source in source_list or []:
            if source and source not in sources:
                sources.append(source)
    return sources

def get_tw_finmind_financial_snapshot(ticker_full):
    """Fetch Taiwan revenue and EPS from FinMind before falling back elsewhere."""
    stock_id = str(ticker_full).replace('.TWO', '').replace('.TW', '').strip()
    result = {
        'single_month_revenue': None,
        'single_month_mom': None,
        'single_month_yoy': None,
        'eps_latest_quarter': None,
        'eps_ttm': None,
        'source': [],
    }
    if not stock_id.isdigit():
        return result

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.Timedelta(days=540)).strftime('%Y-%m-%d')

    revenue_df = pd.DataFrame()
    if dl is not None:
        for method_name in ['taiwan_stock_month_revenue', 'taiwan_stock_month_revenue_per_share']:
            fn = getattr(dl, method_name, None)
            if fn is None:
                continue
            try:
                revenue_df = fn(stock_id=stock_id, start_date=start_date, end_date=end_date)
                revenue_df = _as_date_col(revenue_df)
                if not revenue_df.empty:
                    break
            except TypeError:
                try:
                    revenue_df = fn(start_date=start_date, end_date=end_date, stock_id=stock_id)
                    revenue_df = _as_date_col(revenue_df)
                    if not revenue_df.empty:
                        break
                except Exception:
                    continue
            except Exception:
                continue

    if revenue_df.empty:
        revenue_df = _finmind_v4_dataset(
            dataset='TaiwanStockMonthRevenue',
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )

    try:
        if not revenue_df.empty:
            if 'stock_id' in revenue_df.columns:
                revenue_df = revenue_df[revenue_df['stock_id'].astype(str) == stock_id]
            revenue_col = (
                _pick_first_column(revenue_df, ['revenue', '營收', 'monthly_revenue'])
                or _find_column(revenue_df, ['revenue'])
                or _find_column(revenue_df, ['營收'])
            )
            if revenue_col is not None:
                work = revenue_df.copy().sort_values('date')
                work['revenue_value'] = _numeric_series(work, revenue_col)
                work = work.dropna(subset=['date', 'revenue_value'])
                if len(work) >= 1:
                    latest = work.iloc[-1]
                    result['single_month_revenue'] = safe_float(latest['revenue_value'])
                    if len(work) >= 2:
                        prev = safe_float(work.iloc[-2]['revenue_value'])
                        if prev not in (None, 0):
                            result['single_month_mom'] = (result['single_month_revenue'] - prev) / prev * 100
                    latest_date = pd.to_datetime(latest['date'])
                    yoy_candidates = work[
                        (pd.to_datetime(work['date']).dt.year == latest_date.year - 1)
                        & (pd.to_datetime(work['date']).dt.month == latest_date.month)
                    ]
                    if not yoy_candidates.empty:
                        prior = safe_float(yoy_candidates.iloc[-1]['revenue_value'])
                        if prior not in (None, 0):
                            result['single_month_yoy'] = (result['single_month_revenue'] - prior) / prior * 100
                    result['source'].append('FinMind-Revenue')
    except Exception as e:
        log(f'[FINMIND-FIN-WARN] revenue parse failed for {ticker_full}: {e}')

    fs_df = pd.DataFrame()
    if dl is not None:
        for method_name in ['taiwan_stock_financial_statement', 'taiwan_stock_financial_statements']:
            fn = getattr(dl, method_name, None)
            if fn is None:
                continue
            try:
                fs_df = fn(stock_id=stock_id, start_date=start_date, end_date=end_date)
                fs_df = _as_date_col(fs_df)
                if not fs_df.empty:
                    break
            except Exception:
                continue

    if fs_df.empty:
        fs_df = _finmind_v4_dataset(
            dataset='TaiwanStockFinancialStatements',
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )

    try:
        if not fs_df.empty:
            type_col = (
                _pick_first_column(fs_df, ['type', 'origin_name', 'name', 'statement'])
                or _find_column(fs_df, ['type'])
                or _find_column(fs_df, ['name'])
                or _find_column(fs_df, ['會計'])
            )
            value_col = _pick_first_column(fs_df, ['value', 'amount', 'EPS', '每股盈餘']) or _find_column(fs_df, ['value']) or _find_column(fs_df, ['金額'])
            if type_col is not None and value_col is not None:
                eps_rows = fs_df[
                    fs_df[type_col].astype(str).str.contains('EPS|每股盈餘|基本每股盈餘|Basic EPS', case=False, na=False)
                ].copy()
                if not eps_rows.empty:
                    eps_rows['eps_value'] = _numeric_series(eps_rows, value_col)
                    eps_rows = eps_rows.dropna(subset=['date', 'eps_value']).sort_values('date')
                    vals = [safe_float(v) for v in eps_rows['eps_value'].tail(4).tolist()]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        result['eps_latest_quarter'] = vals[-1]
                        result['eps_ttm'] = sum(vals[-4:])
                        result['source'].append('FinMind-EPS')
    except Exception as e:
        log(f'[FINMIND-FIN-WARN] EPS parse failed for {ticker_full}: {e}')

    return result

def merge_financial_snapshot(ticker_full, md_text, yf_info=None):
    is_us = is_us_ticker(ticker_full)
    if is_us:
        yf_info = yf_info or {}
        teps = safe_float(yf_info.get('trailingEps')) if yf_info else None
        rg = safe_float(yf_info.get('revenueGrowth')) if yf_info else None
        if rg is not None: rg = rg * 100
        profit_margin = safe_float(yf_info.get('profitMargins'))
        if profit_margin is not None and abs(profit_margin) <= 1: profit_margin *= 100
        earnings_growth = safe_float(yf_info.get('earningsGrowth'))
        if earnings_growth is not None and abs(earnings_growth) <= 1: earnings_growth *= 100
        institutional = safe_float(yf_info.get('heldPercentInstitutions'))
        if institutional is not None and institutional <= 1: institutional *= 100
        short_float = safe_float(yf_info.get('shortPercentOfFloat'))
        if short_float is not None and short_float <= 1: short_float *= 100
        short_ratio = safe_float(yf_info.get('shortRatio'))
        avg_volume = safe_float(yf_info.get('averageVolume'))
        avg_volume_10d = safe_float(yf_info.get('averageVolume10days') or yf_info.get('averageDailyVolume10Day'))
        latest_volume = safe_float(yf_info.get('volume'))

        inst_text = 'N/A' if institutional is None else f'{institutional:.1f}%'
        short_text = 'N/A' if short_float is None else f'{short_float:.1f}%'
        ratio_text = 'N/A' if short_ratio is None else f'{short_ratio:.1f}'
        return {
            'single_month_revenue': None, 'single_month_mom': None, 'single_month_yoy': rg,
            'eps_latest_quarter': None, 'eps_ttm': teps,
            'profit_margin_pct': profit_margin,
            'earnings_growth_pct': earnings_growth,
            'market_cap': safe_float(yf_info.get('marketCap')),
            'institutional_ownership_pct': institutional,
            'short_percent_float': short_float,
            'short_ratio': short_ratio,
            'avg_volume_3m': avg_volume,
            'avg_volume_10d': avg_volume_10d,
            'latest_volume': latest_volume,
            'chips_summary': f'美股籌碼代理：機構持股 {inst_text}，放空比例 {short_text}，Short ratio {ratio_text}',
            'foreign_2d': None, 'foreign_3d': None, 'foreign_5d': None, 'foreign_10d': None,
            'trust_2d': None, 'trust_3d': None, 'trust_5d': None, 'trust_10d': None,
            'dealer_2d': None, 'dealer_3d': None, 'dealer_5d': None, 'dealer_10d': None,
            'total_2d': None, 'total_3d': None, 'total_5d': None, 'total_10d': None,
            'sources': ['Yahoo Finance']
        }
        
    ticker_num = ticker_full.split('.')[0]
    from_finmind = get_tw_finmind_financial_snapshot(ticker_full)
    from_md = parse_financials_from_mytwcoverage(md_text)
    main_html, chip_html = fetch_goodinfo_data(ticker_num)
    main_tables = get_goodinfo_tables(main_html) if main_html else []
    chip_tables = get_goodinfo_tables(chip_html) if chip_html else []
    
    rev_text = parse_goodinfo_revenue_from_text(main_html)
    rev_table = parse_goodinfo_revenue_table(main_tables)
    eps = parse_goodinfo_eps_table(main_tables)
    chip = get_tw_chip_data(ticker_full)

    merged = {
        'single_month_revenue': _first_not_none(rev_table['single_month_revenue'], rev_text['single_month_revenue'], from_finmind['single_month_revenue']),
        'single_month_mom': _first_not_none(rev_table['single_month_mom'], rev_text['single_month_mom'], from_finmind['single_month_mom'], from_md['single_month_mom']),
        'single_month_yoy': _first_not_none(rev_table['single_month_yoy'], rev_text['single_month_yoy'], from_finmind['single_month_yoy'], from_md['single_month_yoy']),
        'eps_latest_quarter': _first_not_none(eps['eps_latest_quarter'], from_finmind['eps_latest_quarter'], from_md['eps_latest_quarter']),
        'eps_ttm': _first_not_none(eps['eps_ttm'], from_finmind['eps_ttm'], from_md['eps_ttm']),
        'chips_summary': chip['chips_summary'],
        'foreign_2d': chip['foreign_2d'], 'foreign_3d': chip['foreign_3d'], 'foreign_5d': chip['foreign_5d'], 'foreign_10d': chip['foreign_10d'],
        'trust_2d': chip['trust_2d'], 'trust_3d': chip['trust_3d'], 'trust_5d': chip['trust_5d'], 'trust_10d': chip['trust_10d'],
        'dealer_2d': chip['dealer_2d'], 'dealer_3d': chip['dealer_3d'], 'dealer_5d': chip['dealer_5d'], 'dealer_10d': chip['dealer_10d'],
        'total_2d': chip['total_2d'], 'total_3d': chip['total_3d'], 'total_5d': chip['total_5d'], 'total_10d': chip['total_10d'],
        'sources': _merge_unique_sources(from_finmind['source'], rev_table['source'], rev_text['source'], eps['source'], chip['source'], from_md['source'])
    }
    if merged['eps_ttm'] is None and yf_info:
        teps = safe_float(yf_info.get('trailingEps'))
        if teps is not None: merged['eps_ttm'] = teps; merged['sources'].append('YF-EPS')
    if merged['single_month_yoy'] is None and yf_info:
        rg = yf_info.get('revenueGrowth')
        if rg is not None: merged['single_month_yoy'] = float(rg) * 100; merged['sources'].append('YF-Rev')
    return merged

# ==========================================
# FinMind chip data helpers
# ==========================================
def _tw_numeric_stock_id(ticker):
    return qpro_tw_code(ticker)

def _init_tw_chip_result(summary='籌碼資料不足或查無資料'):
    return {
        'chips_summary': summary,
        'foreign_2d': None, 'foreign_3d': None, 'foreign_5d': None, 'foreign_10d': None,
        'trust_2d': None, 'trust_3d': None, 'trust_5d': None, 'trust_10d': None,
        'dealer_2d': None, 'dealer_3d': None, 'dealer_5d': None, 'dealer_10d': None,
        'total_2d': None, 'total_3d': None, 'total_5d': None, 'total_10d': None,
        'source': []
    }


def _tw_chip_has_data(chip_data):
    if not chip_data:
        return False
    keys = [
        'foreign_2d', 'foreign_3d', 'foreign_5d', 'foreign_10d',
        'trust_2d', 'trust_3d', 'trust_5d', 'trust_10d',
        'dealer_2d', 'dealer_3d', 'dealer_5d', 'dealer_10d',
        'total_2d', 'total_3d', 'total_5d', 'total_10d',
    ]
    return any(chip_data.get(k) is not None for k in keys)


def _append_source(result, source_name):
    if source_name and source_name not in result.get('source', []):
        result.setdefault('source', []).append(source_name)
    return result


def _sum_last_values(values, n):
    vals = [safe_float(v) for v in values if safe_float(v) is not None]
    if not vals:
        return None
    return float(sum(vals[-n:]))


def _fill_tw_chip_rollups(result, source_name):
    # Fill total_xd from actors only when total_xd is absent.  TWSE provides an
    # official total; FinMind / Goodinfo may only provide actor-level values.
    for n in [2, 3, 5, 10]:
        total_key = f'total_{n}d'
        if result.get(total_key) is None:
            vals = [result.get(f'{actor}_{n}d') for actor in ['foreign', 'trust', 'dealer']]
            valid = [safe_float(v) for v in vals if safe_float(v) is not None]
            result[total_key] = sum(valid) if valid else None

    for actor in ['foreign', 'trust', 'dealer', 'total']:
        for n in [2, 3, 5, 10]:
            key = f'{actor}_{n}d'
            value = safe_float(result.get(key))
            if value is not None:
                result[key] = float(round(value))

    t5 = result.get('total_5d')
    if t5 is not None:
        if t5 > 0:
            result['chips_summary'] = f'{source_name}：近 5 日三大法人合計買超 {t5:.0f} 張'
        elif t5 < 0:
            result['chips_summary'] = f'{source_name}：近 5 日三大法人合計賣超 {abs(t5):.0f} 張'
        else:
            result['chips_summary'] = f'{source_name}：近 5 日三大法人中性'

    return _append_source(result, source_name)


def _parse_twse_number_to_lots(v):
    """TWSE T86 returns shares; report displays lots/張."""
    x = safe_float(v)
    if x is None:
        return None
    return x / 1000.0


def _fetch_twse_t86_one_day(date_yyyymmdd, stock_id):
    """Fetch one day of official TWSE T86 institutional data for one stock."""
    url = 'https://www.twse.com.tw/fund/T86'
    params = {
        'response': 'json',
        'date': date_yyyymmdd,
        'selectType': 'ALLBUT0999',
    }
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json,text/plain,*/*',
        'Referer': 'https://www.twse.com.tw/',
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except Exception:
        return None

    data = payload.get('data') or []
    fields = payload.get('fields') or []
    if not data or not fields:
        return None

    code_idx = None
    for i, field in enumerate(fields):
        if '證券代號' in str(field):
            code_idx = i
            break
    if code_idx is None:
        return None

    row = None
    for item in data:
        if len(item) > code_idx and str(item[code_idx]).strip() == str(stock_id):
            row = item
            break
    if row is None:
        return None

    def get_by_keywords(*keywords):
        for i, field in enumerate(fields):
            field_text = str(field).replace(' ', '')
            if all(k in field_text for k in keywords) and i < len(row):
                return row[i]
        return None

    foreign = get_by_keywords('外陸資買賣超股數') or get_by_keywords('外陸資', '買賣超')
    trust = get_by_keywords('投信買賣超股數') or get_by_keywords('投信', '買賣超')
    dealer = get_by_keywords('自營商買賣超股數') or get_by_keywords('自營商', '買賣超')
    total = get_by_keywords('三大法人買賣超股數') or get_by_keywords('三大法人', '買賣超')

    return {
        'date': date_yyyymmdd,
        'foreign': _parse_twse_number_to_lots(foreign),
        'trust': _parse_twse_number_to_lots(trust),
        'dealer': _parse_twse_number_to_lots(dealer),
        'total': _parse_twse_number_to_lots(total),
    }


def fetch_twse_t86_chip_data(ticker, calendar_days=30):
    result = _init_tw_chip_result('TWSE T86 籌碼資料不足或查無資料')
    stock_id = _tw_numeric_stock_id(ticker)
    if not stock_id.isdigit():
        return result

    rows = []
    today = pd.Timestamp.today().normalize()
    for i in range(calendar_days):
        d = today - pd.Timedelta(days=i)
        one = _fetch_twse_t86_one_day(d.strftime('%Y%m%d'), stock_id)
        if one and any(one.get(k) is not None for k in ['foreign', 'trust', 'dealer', 'total']):
            rows.append(one)
        if len(rows) >= 10:
            break
        time.sleep(0.12)

    if not rows:
        return result

    rows = list(reversed(rows))
    foreign_vals = [r.get('foreign') for r in rows]
    trust_vals = [r.get('trust') for r in rows]
    dealer_vals = [r.get('dealer') for r in rows]
    total_vals = [r.get('total') for r in rows]

    for n in [2, 3, 5, 10]:
        result[f'foreign_{n}d'] = _sum_last_values(foreign_vals, n)
        result[f'trust_{n}d'] = _sum_last_values(trust_vals, n)
        result[f'dealer_{n}d'] = _sum_last_values(dealer_vals, n)
        result[f'total_{n}d'] = _sum_last_values(total_vals, n)

    result = _fill_tw_chip_rollups(result, 'TWSE-T86')
    if _tw_chip_has_data(result):
        result['chips_summary'] += f'；取近 {len(rows)} 個交易日'
    return result


def fetch_goodinfo_chip_data_fallback(ticker):
    result = _init_tw_chip_result('Goodinfo 籌碼資料不足或查無資料')
    stock_id = _tw_numeric_stock_id(ticker)
    if not stock_id.isdigit():
        return result

    try:
        main_html, chip_html = fetch_goodinfo_data(stock_id)
        htmls = [h for h in [chip_html, main_html] if h]
        tables = []
        for html_text in htmls:
            tables.extend(get_goodinfo_tables(html_text))
        if not tables:
            return result

        parsed = parse_goodinfo_chip_table(tables)
        if parsed and _tw_chip_has_data(parsed):
            for k, v in parsed.items():
                if k in result:
                    result[k] = v
            for src in parsed.get('source', []) or []:
                _append_source(result, src)
            result = _fill_tw_chip_rollups(result, 'Goodinfo-Chips')
            return result
    except Exception as e:
        result['chips_summary'] = f'Goodinfo 籌碼讀取失敗: {e}'

    return result


def fetch_finmind_chip_data_only(ticker, days=10):
    result = _init_tw_chip_result('FinMind 籌碼資料不足或查無資料')

    if dl is None:
        result['chips_summary'] = 'FinMind 未安裝或初始化失敗，請檢查 requirements.txt'
        return result

    stock_id = _tw_numeric_stock_id(ticker)
    if not stock_id.isdigit():
        return result

    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=45)).strftime('%Y-%m-%d')
        try:
            df = data_sources.fetch_finmind_institutional_investors(dl, stock_id, start_date, end_date, ttl_hours=6)
        except Exception:
            df = dl.taiwan_stock_institutional_investors(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date
            )

        if df is None or df.empty:
            result['chips_summary'] = f'FinMind 回傳 0 筆資料 (區間 {start_date} ~ {end_date})'
            return result

        df = _as_date_col(df.copy())
        if df.empty:
            result['chips_summary'] = 'FinMind 回傳資料缺少可解析日期欄位'
            return result

        buy_col = _pick_first_column(df, ['buy', '買進', '買進股數', 'buy_volume'])
        sell_col = _pick_first_column(df, ['sell', '賣出', '賣出股數', 'sell_volume'])
        net_col = _pick_first_column(df, ['net_buy', 'buy_sell', '買賣超', '買賣超股數', 'net'])

        if buy_col is not None and sell_col is not None:
            df['net_buy'] = _numeric_series(df, buy_col).fillna(0) - _numeric_series(df, sell_col).fillna(0)
        elif net_col is not None:
            df['net_buy'] = _numeric_series(df, net_col).fillna(0)
        else:
            result['chips_summary'] = f'FinMind 欄位無法解析: {list(df.columns)}'
            return result

        # Normalize FinMind unit to lots/張.
        # Some FinMind / cached-source schemas return raw shares, while the report displays 張.
        # Prefer column-name evidence first; then use magnitude as a safety guard.
        unit_text = ' '.join([str(c).lower() for c in [buy_col, sell_col, net_col] if c is not None])
        looks_like_shares = (
            '股' in unit_text
            or 'share' in unit_text
            or 'volume' in unit_text
            or df['net_buy'].abs().quantile(0.90) > 100000
        )
        if looks_like_shares:
            df['net_buy'] = df['net_buy'] / 1000.0

        actor_col = _pick_first_column(
            df,
            ['name', 'institutional_investors', 'investor', 'investor_type', '身份別', '法人', '投資人']
        )
        if actor_col is None:
            result['chips_summary'] = f'FinMind 法人欄位無法解析: {list(df.columns)}'
            return result

        def actor_mask(pattern):
            return df[actor_col].astype(str).str.contains(pattern, case=False, na=False)

        def sum_last(pattern, n):
            # Group by date first to avoid duplicate rows from category variants.
            tmp = df[actor_mask(pattern)].copy()
            if tmp.empty:
                return None
            daily = tmp.groupby('date', as_index=False)['net_buy'].sum().sort_values('date')
            return float(daily.tail(n)['net_buy'].sum()) if not daily.empty else None

        actor_map = {
            'foreign': r'外資|外陸資|Foreign',
            'trust': r'投信|Investment|Trust',
            'dealer': r'自營|Dealer',
        }
        for key, pattern in actor_map.items():
            for n in [2, 3, 5, 10]:
                result[f'{key}_{n}d'] = sum_last(pattern, n)

        result = _fill_tw_chip_rollups(result, 'FinMind-Chips')

    except Exception as e:
        result['chips_summary'] = f'FinMind 籌碼讀取失敗: {e}'

    return result


def get_tw_chip_data(ticker, days=10):
    """
    Taiwan chip-data priority, conservative unit order:
      1. Goodinfo chip table: usually already reports 買賣超張數.
      2. TWSE official T86: official listed-stock report, converted from shares to 張.
      3. FinMind: fallback only, because some environments return institutional data in raw shares.

    The returned schema stays compatible with the renderer:
      foreign/trust/dealer/total 2d,3d,5d,10d + chips_summary + source
    """
    errors = []

    gi = fetch_goodinfo_chip_data_fallback(ticker)
    if _tw_chip_has_data(gi):
        return gi
    errors.append(gi.get('chips_summary', 'Goodinfo 無資料'))

    twse = fetch_twse_t86_chip_data(ticker)
    if _tw_chip_has_data(twse):
        return twse
    errors.append(twse.get('chips_summary', 'TWSE T86 無資料'))

    fm = fetch_finmind_chip_data_only(ticker, days=days)
    if _tw_chip_has_data(fm):
        return fm
    errors.append(fm.get('chips_summary', 'FinMind 無資料'))

    out = _init_tw_chip_result('籌碼資料三源皆不足或解析失敗')
    out['chips_summary'] = ' | '.join([str(e) for e in errors if e])[:260]
    return out

def merge_finmind_chip_into_snapshot(fin_data, chip_data):
    if not chip_data: return fin_data
    merged = dict(fin_data)

    chip_keys = [
        'foreign_2d', 'foreign_3d', 'foreign_5d', 'foreign_10d',
        'trust_2d', 'trust_3d', 'trust_5d', 'trust_10d',
        'dealer_2d', 'dealer_3d', 'dealer_5d', 'dealer_10d',
        'total_2d', 'total_3d', 'total_5d', 'total_10d'
    ]

    has_fm_data = any(chip_data.get(k) is not None for k in chip_keys)
    if has_fm_data:
        if 'chips_summary' in chip_data:
            merged['chips_summary'] = chip_data['chips_summary']
        for k in chip_keys:
            if k in chip_data: merged[k] = chip_data[k]
        sources = list(merged.get('sources', []))
        for s in chip_data.get('source', []):
            if s not in sources: sources.append(s)
        merged['sources'] = sources
    elif not any(merged.get(k) is not None for k in chip_keys):
        merged['chips_summary'] = chip_data.get('chips_summary', merged.get('chips_summary'))
    return merged

# ==========================================
# Stock pools and technical filters
# ==========================================
def get_us_stock_pool():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers, timeout=15)
        
        table = pd.read_html(StringIO(response.text))
        symbols = [normalize_ticker(s) for s in table[0]['Symbol'].dropna().tolist()]
        return sorted(set(symbols))
        
    except Exception as e:
        log_exception("[US-POOL-ERROR]", e)
        return ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META', 'AMD', 'BRK-B', 'JPM']

def get_tw_stock_pool(mode='offensive'):
    tickers = []
    if mode == 'defensive': tickers.extend(get_defensive_etf_pool('TW'))
    for m in [2, 4]:
        try:
            res = requests.get(f'https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}', timeout=15)
            df = pd.read_html(StringIO(res.text))[0]
            df.columns = df.iloc[0]
            valid_codes = df.iloc[1:][df.iloc[1:]['CFICode'] == 'ESVUFR']['有價證券代號及名稱'].str.extract(r'^([0-9]{4})\b')[0].dropna()
            tickers.extend((valid_codes + ('.TW' if m == 2 else '.TWO')).tolist())
        except Exception: pass
    return list(set(tickers))

def download_stock_df(ticker):
    ticker = normalize_ticker(ticker)
    df = yf.download(ticker, period='5y', progress=False, auto_adjust=True)
    if df.empty and ticker.endswith('.TW'):
        alt = ticker.replace('.TW', '.TWO')
        df = yf.download(alt, period='5y', progress=False, auto_adjust=True)
        if not df.empty: ticker = alt
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    return ticker, df

def compute_indicators(df):
    df = df.copy()

    # Moving averages.
    for win in [5, 20, 50, 60, 150, 200, 240]:
        df[f'MA{win}'] = df['Close'].rolling(window=win, min_periods=win).mean()

    # 52-week high / low.
    df['High52W'] = df['High'].rolling(window=250, min_periods=1).max()
    df['Low52W'] = df['Low'].rolling(window=250, min_periods=1).min()

    # Bollinger Bands.
    bb_window = 20
    bb_std = 2.0
    df['BBMid'] = df['Close'].rolling(window=bb_window, min_periods=bb_window).mean()
    df['BBStd'] = df['Close'].rolling(window=bb_window, min_periods=bb_window).std()
    df['BBHigh'] = df['BBMid'] + bb_std * df['BBStd']
    df['BBLow'] = df['BBMid'] - bb_std * df['BBStd']
    df['BBWidth'] = np.where(
        df['BBMid'] != 0,
        (df['BBHigh'] - df['BBLow']) / df['BBMid'] * 100,
        np.nan
    )
    df['BBWidthPctile'] = df['BBWidth'].rolling(window=120, min_periods=20).rank(pct=True) * 100

    df['bb_width'] = df['BBWidth']
    df['bb_width_pctile'] = df['BBWidthPctile']
    vol_ma20 = df['Volume'].rolling(window=20, min_periods=20).mean() if 'Volume' in df.columns else np.nan
    df['bb_breakout'] = np.where(
        (df['Close'] > df['BBHigh']) & (df['Volume'] > vol_ma20 * 1.5),
        1,
        0
    ) if 'Volume' in df.columns else 0
    df['bb_squeeze'] = np.where(df['BBWidth'] < 10.0, 1, 0)
    df['vcp_score'] = np.where(df['BBWidth'] < 10.0, 0.70, 0.0)
    df['vcp_pivot'] = df['High'].rolling(window=20, min_periods=5).max()
    df['is_vcp'] = np.where(df['BBWidth'] < 10.0, 1, 0)
    df['is_engulfing'] = np.where(
        (df['Close'] > df['Open']) &
        (df['Close'] > df['High'].shift(1)) &
        (df['Open'] < df['Low'].shift(1)),
        1,
        0
    )

    # RSI.
    try:
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    except TypeError:
        df['RSI'] = ta.momentum.rsi(df['Close'], 14)
    except Exception:
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

    # MACD.
    try:
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Osc'] = macd.macd_diff()
    except Exception:
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Osc'] = df['MACD'] - df['MACD_Signal']

    # Bias ratio.
    for ma in [5, 20, 60, 240]:
        df[f'BIAS{ma}'] = np.where(
            df[f'MA{ma}'] != 0,
            (df['Close'] - df[f'MA{ma}']) / df[f'MA{ma}'] * 100,
            np.nan
        )

    # ATR (Average True Range)
    try:
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    except Exception:
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

    return df

def evaluate_technical(df, market_mode='offensive'):
    df = compute_indicators(df)
    try:
        from quant.vcp import add_vcp_features
        from quant.bollinger import add_bollinger_features

        df = add_vcp_features(df)
        df = add_bollinger_features(df)
        df = add_cta_features(df)
        df = add_pattern_features(df)
    except Exception as e:
        log(f"[FEATURE-WARN] VCP/BB/CTA/Pattern feature generation failed: {e}")
    latest = df.iloc[-1]
    
    c1 = bool(latest['Close'] > latest['MA50'] > latest['MA150'] > latest['MA200'])
    c2 = bool(latest['Close'] > latest['Low52W'] * 1.30) if not pd.isna(latest['Low52W']) else False
    c3 = bool(latest['Close'] > latest['High52W'] * 0.75) if not pd.isna(latest['High52W']) else False
    c4 = bool((latest['Close'] >= latest['BBMid']) and (latest['RSI'] > 60) and (latest['MACD_Osc'] > 0)) if not pd.isna(latest['BBMid']) else False
    c5 = bool(latest['Volume'] > 500_000) 
    c6 = bool(latest['Close'] > latest['MA5'] > latest['MA20'] > latest['MA60']) if not pd.isna(latest['MA60']) else False
    c7 = bool(latest['Close'] > latest['MA240']) if not pd.isna(latest['MA240']) else False

    weekly = compute_indicators(df[['Open', 'High', 'Low', 'Close', 'Volume']].resample('W-FRI').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna())
    monthly = compute_indicators(df[['Open', 'High', 'Low', 'Close', 'Volume']].resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna())
    
    wk_up = bool(weekly.iloc[-1]['Close'] > weekly.iloc[-1].get('MA20', 0)) if len(weekly) > 10 else False
    mo_up = bool(monthly.iloc[-1]['Close'] > monthly.iloc[-1].get('MA20', 0)) if len(monthly) > 10 else False
    wk_macd_pos = bool(weekly.iloc[-1].get('MACD_Osc', 0) > 0) if len(weekly) > 10 else False
    mo_macd_pos = bool(monthly.iloc[-1].get('MACD_Osc', 0) > 0) if len(monthly) > 10 else False

    bias20 = safe_float(latest.get('BIAS20'))
    bias60 = safe_float(latest.get('BIAS60'))
    vcp_score = safe_float(latest.get('vcp_score'), 0.0) or 0.0
    bb_score = safe_float(latest.get('bb_score'), 0.0) or 0.0
    pattern_score = safe_float(latest.get('pattern_score'), 0.0) or 0.0
    cta_score = safe_float(latest.get('cta_score'), 0.0) or 0.0
    c_engulfing_5d = bool(latest.get('engulfing_5d', 0))
    c_box_breakout = bool(latest.get('close_box_breakout', 0))
    c_bb_squeeze_breakout = bool(latest.get('bb_squeeze_breakout', 0))
    c_bb_momentum_breakout = bool(latest.get('bb_momentum_breakout', 0))
    c_triangle = bool((safe_float(latest.get('triangle_contraction_score'), 0.0) or 0.0) >= 0.55)
    c_inverse_hs = bool((safe_float(latest.get('inverse_head_shoulders_score'), 0.0) or 0.0) >= 0.55)
    
    score = 0
    if market_mode == 'offensive':
        score = sum([16*c1, 10*c2, 10*c3, 12*c4, 8*c5, 8*c6, 6*c7, 8*wk_up, 5*mo_up, 9*wk_macd_pos, 6*mo_macd_pos])
        score += 10 if vcp_score >= 0.65 else (5 if vcp_score >= 0.50 else 0)
        score += 15 if c_box_breakout else 0
        score += 10 if c_engulfing_5d else 0
        score += 12 if c_bb_squeeze_breakout else 0
        score += 8 if c_bb_momentum_breakout else 0
        score += 8 if c_triangle else 0
        score += 8 if c_inverse_hs else 0
        if bias20 is not None:
            if 0 <= bias20 <= 8: score += 6
            elif 8 < bias20 <= 15: score += 3
            elif bias20 > 25: score -= 6
            elif bias20 < -3: score -= 3
    else:
        score += 25 if c7 else 0 
        if latest['High52W']:
            drop = (latest['High52W'] - latest['Close']) / latest['High52W']
            if drop <= 0.10: score += 25
            elif drop <= 0.20: score += 15
            else: score -= 10
        if bias20 is not None:
            if 0 <= bias20 <= 5: score += 15
            elif 5 < bias20 <= 10: score += 5
            elif bias20 < 0: score -= 5
        score += 15 if bool(latest['Close'] > latest['MA50']) else -10
        score += 10 if wk_macd_pos else 0
        score += 10 if c5 else 0

    opt_ma_period = SYS_PARAMS.get('ma_period', 20)
    opt_ma_val = latest.get(f'MA{opt_ma_period}', latest.get('MA20'))

    technical_score = max(0, min(score, 100))
    bb_breakout = bool(latest.get('bb_breakout', 0))
    bb_width_pctile = safe_float(latest.get('bb_width_pctile'))
    pattern_tag = '多頭排列'
    if c_bb_squeeze_breakout or bb_breakout:
        pattern_tag = '布林突破'
    elif c_box_breakout:
        pattern_tag = '箱型突破'
    elif bool(latest.get('is_vcp', 0)) or vcp_score >= 0.65:
        pattern_tag = 'VCP 收斂'
    elif c_triangle:
        pattern_tag = '收斂三角'
    elif c_inverse_hs:
        pattern_tag = '頭肩底'
    elif c_engulfing_5d or bool(latest.get('is_engulfing', 0)):
        pattern_tag = '型態吞噬'
    strategy_tags = row_strategy_tags(latest)
    if vcp_score >= 0.65: strategy_tags.append('VCP量縮收斂')
    if bb_breakout: strategy_tags.append('BB帶量突破')
    if c_triangle: strategy_tags.append('收斂三角')
    if c_inverse_hs: strategy_tags.append('頭肩底雛形')
    strategy_tags = list(dict.fromkeys(strategy_tags))

    return {
        'df': df, 'weekly': weekly, 'monthly': monthly, 'technical_score': technical_score, 'latest': latest, 'mode': market_mode,
        'pattern_tag': pattern_tag,
        'strategy_tags': strategy_tags,
        'conditions': {
            'trend_stack': c1, 'off_bottom': c2, 'near_high': c3, 'momentum': c4, 'liquidity': c5,
            'short_mid_ma_stack': c6, 'above_ma240': c7, 'weekly_up': wk_up, 'monthly_up': mo_up,
            'weekly_macd_positive': wk_macd_pos, 'monthly_macd_positive': mo_macd_pos,
            'vcp_setup': vcp_score >= 0.65, 'bb_breakout': bb_breakout,
            'c_engulfing_5d': c_engulfing_5d, 'c_box_breakout': c_box_breakout,
            'c_bb_squeeze_breakout': c_bb_squeeze_breakout,
            'c_bb_momentum_breakout': c_bb_momentum_breakout,
            'c_triangle_contraction': c_triangle, 'c_inverse_head_shoulders': c_inverse_hs
        },
        'metrics': {
            'latest_date': df.index[-1].strftime('%Y-%m-%d'), 'rsi': safe_float(latest['RSI']), 'macd_osc_d': safe_float(latest['MACD_Osc']),
            'macd_osc_w': safe_float(weekly.iloc[-1].get('MACD_Osc')) if len(weekly) else None,
            'macd_osc_m': safe_float(monthly.iloc[-1].get('MACD_Osc')) if len(monthly) else None,
            'close': safe_float(latest['Close']), 'volume': safe_float(latest['Volume']),
            'dist_high_pct': ((latest['High52W'] - latest['Close']) / latest['High52W']) * 100 if latest['High52W'] else None,
            'ma5': safe_float(latest['MA5']), 'ma20': safe_float(latest['MA20']), 'opt_ma': safe_float(opt_ma_val),
            'ma50': safe_float(latest['MA50']), 'ma240': safe_float(latest['MA240']),
            'bias5': safe_float(latest.get('BIAS5')), 'bias20': bias20, 'bias60': bias60, 'bias240': safe_float(latest.get('BIAS240')),
            'vcp_score': vcp_score, 'vcp_pivot': safe_float(latest.get('vcp_pivot')), 'bb_width_pctile': bb_width_pctile, 'bb_breakout': bb_breakout,
            'bb_score': bb_score, 'cta_score': cta_score, 'pattern_score': pattern_score,
            'triangle_contraction_score': safe_float(latest.get('triangle_contraction_score')),
            'inverse_head_shoulders_score': safe_float(latest.get('inverse_head_shoulders_score')),
            'box_width': safe_float(latest.get('Box_Width')), 'close_box_breakout': c_box_breakout,
            'atr': safe_float(latest.get('ATR'))
        }
    }

def calc_fundamental_score(f, is_us=False):
    return shared_calc_fundamental_score(f, is_us=is_us)

def calc_chip_score(f, is_us=False):
    return shared_calc_chip_score(f, is_us=is_us)

def final_total_score(t, f, c, is_us=False):
    tech_w = max(0.0, float(SYS_PARAMS.get('tech_weight', WEIGHT_TECH)))
    fund_w = max(0.0, float(SYS_PARAMS.get('fund_weight', WEIGHT_FUND)))
    chip_w = max(0.0, float(SYS_PARAMS.get('chip_weight', WEIGHT_CHIP)))
    total_w = tech_w + fund_w + chip_w
    if total_w <= 0:
        return t * WEIGHT_TECH + f * WEIGHT_FUND + c * WEIGHT_CHIP
    return (t * tech_w + f * fund_w + c * chip_w) / total_w

# ==========================================
# Report and card generation
# ==========================================
def save_exquisite_plot(df, weekly, monthly, ticker, ranking_info):
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0], hspace=0.24, wspace=0.16)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]

    def plot_k(ax, data, title):
        if data.empty: return
        data = data.tail(90).copy()
        xs = mdates.date2num(data.index.to_pydatetime())
        width = 0.6 * np.median(np.diff(xs)) if len(xs) > 1 else 0.5
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.22)
        for x, (_, row) in zip(xs, data.iterrows()):
            o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
            color = '#d62828' if c >= o else '#1d3557'
            ax.vlines(x, l, h, color=color, linewidth=0.8)
            ax.add_patch(Rectangle((x - width / 2, min(o, c)), width, max(abs(c - o), 0.001), facecolor=color, edgecolor=color, alpha=0.75))
        for ma, color in [('MA5', '#f4a261'), ('MA20', '#2a9d8f'), ('MA60', '#457b9d'), ('MA240', '#6d597a')]:
            if ma in data.columns: ax.plot(data.index, data[ma], label=ma, linewidth=1.0, color=color)
        ax.legend(loc='upper left', fontsize=8, ncol=4)
        ax.xaxis_date()
        ax.tick_params(axis='x', labelrotation=20)

    def plot_m(ax, data, title):
        if data.empty: return
        data = data.tail(90).copy()
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.22)
        colors = ['#d62828' if x >= 0 else '#457b9d' for x in data['MACD_Osc'].fillna(0)]
        ax.bar(data.index, data['MACD_Osc'], color=colors, alpha=0.35, label='MACD Hist')
        ax.plot(data.index, data['MACD'], color='#1d3557', linewidth=1.0, label='MACD')
        ax.plot(data.index, data['MACD_Signal'], color='#f4a261', linewidth=1.0, label='Signal')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.legend(loc='upper left', fontsize=8)
        ax.tick_params(axis='x', labelrotation=20)

    plot_k(axes[0], df, f'{ticker} Daily K')
    plot_m(axes[1], df, 'MACD Daily')
    plot_k(axes[2], compute_indicators(weekly) if len(weekly) else weekly, f'{ticker} Weekly K')
    plot_m(axes[3], compute_indicators(weekly) if len(weekly) else weekly, 'MACD Weekly')
    plot_k(axes[4], compute_indicators(monthly) if len(monthly) else monthly, f'{ticker} Monthly K')
    plot_m(axes[5], compute_indicators(monthly) if len(monthly) else monthly, 'MACD Monthly')

    fig.suptitle(f'{ticker} Quant Report ({ranking_info["mode"].upper()} MODE) | Total {ranking_info["total_score"]:.1f} | Tech {ranking_info["technical_score"]:.1f} | Fund {ranking_info["fundamental_score"]:.1f}', fontsize=16, fontweight='bold')
    file_path = os.path.join(REPORT_DIR, f'{ticker}_report.png')
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return file_path

def generate_advanced_trading_plan(ticker, close, atr, total_score, rsi, bias20, volume, avg_vol, capital=500000):
    """生成圖卡式資金控管與分批進場計畫文字。"""
    grade = 'A' if total_score >= 80 else ('B' if total_score >= 65 else 'C')
    regime = '趨勢多頭' if total_score >= 65 else '震盪/偏空'
    win_rate = min(0.85, 0.40 + (total_score / 200))

    warnings_list = []
    if rsi is not None and rsi > 70:
        warnings_list.append('RSI_HOT')
    if bias20 is not None and bias20 > 15:
        warnings_list.append('BETA_HIGH')
    if volume is not None and avg_vol is not None and volume < avg_vol * 0.7:
        warnings_list.append('VOL_SHRINK')
    warnings_str = ', '.join(warnings_list) if warnings_list else 'SAFE'
    deduction = len(warnings_list) * 3.5

    pos_pct = 0.20 if grade == 'A' else (0.125 if grade == 'B' else 0.05)
    pos_amt = capital * pos_pct
    action = 'ENTER（分批建倉）' if grade in ['A', 'B'] else 'WATCH（觀望）'

    if atr is None or math.isnan(atr) or atr == 0:
        atr = close * 0.03

    p1 = close
    p2 = close - (atr * 0.5)
    p3 = close - (atr * 1.0)

    amt1 = pos_amt * 0.50
    amt2 = pos_amt * 0.30
    amt3 = pos_amt * 0.20

    avg_cost = (p1 * 0.5) + (p2 * 0.3) + (p3 * 0.2)
    sl_price = avg_cost - (atr * 1.5)
    sl_pct = ((sl_price - avg_cost) / avg_cost) * 100 if avg_cost else 0
    tp1_price = avg_cost + (atr * 2.5)
    tp1_pct = ((tp1_price - avg_cost) / avg_cost) * 100 if avg_cost else 0

    plan_text = f"""
━━━━━━━━━━━━━━━━━━━━
📊 **{ticker} 評估結果**
━━━━━━━━━━━━━━━━━━━━
🧮 **總分**：`{total_score:.0f}/100`　`{grade}`
🧭 **Regime**：`{regime}`
🎯 **勝率 p**：`{win_rate:.3f}`
🚦 **警示燈**：`{warnings_str}`
📉 **扣分**：`{deduction:.1f}%`

📌 **建議倉位**：`{pos_pct*100:.1f}%`
💵 **建議金額**：`{pos_amt:,.0f} 元`
👉 **動作**：`{action}`

━━━━━━━━━━━━━━━━━━━━
📍 **{ticker} 進出計畫｜ATR 動態波幅**
━━━━━━━━━━━━━━━━━━━━
💰 **現價**：`{close:.2f}`
📏 **ATR(14)**：`{atr:.2f}`

📥 **進場區｜分 3 批**
A｜第一批：`{p1:.2f}`　`50%`　`{amt1:,.0f} 元`
B｜第二批：`{p2:.2f}`　`30%`　`{amt2:,.0f} 元`
C｜第三批：`{p3:.2f}`　`20%`　`{amt3:,.0f} 元`

📊 **三批全成交平均成本**：`{avg_cost:.2f}`

🛑 **停損**：`{sl_price:.2f}`（`{sl_pct:+.1f}%`）
💰 **停利第 1 段**：`{tp1_price:.2f}`（`{tp1_pct:+.1f}%`）賣一半鎖利
💰 **停利第 2 段**：從持有期最高點回落 `8%` 即出場
━━━━━━━━━━━━━━━━━━━━
"""
    return plan_text



def create_stock_analysis_card_image(ticker, tech_pack, fin_data, profile_info, total_score, rank, output_path):
    """Render one-card visual summary for the top recommendation detail pages."""
    try:
        latest = tech_pack.get('latest')
        if latest is None:
            latest = {}
        c = tech_pack.get('conditions') or {}
        m = tech_pack.get('metrics') or {}
        close = safe_float(latest.get('Close'))
        rsi = safe_float(m.get('rsi'))
        atr = safe_float(m.get('atr'))
        eps = safe_float(fin_data.get('eps_ttm')) or safe_float(fin_data.get('eps_latest_quarter'))
        yoy = safe_float(fin_data.get('single_month_yoy'))
        total_5d = safe_float(fin_data.get('total_5d'))
        foreign_5d = safe_float(fin_data.get('foreign_5d'))
        chip_main = total_5d if total_5d is not None else foreign_5d
        name = profile_info.get('company_name') or ui_company_name_from_raw_text(profile_info.get('raw_text'), ticker) or ticker
        grade = 'A' if total_score >= 80 else ('B' if total_score >= 65 else 'C')
        regime = '趨勢多頭' if total_score >= 65 else '震盪觀察'

        tech_items = [
            ('長天期多頭', c.get('trend_stack')),
            ('短中期順多', c.get('short_mid_ma_stack')),
            ('站上 240MA', c.get('above_ma240')),
            ('VCP 收斂', c.get('vcp_setup')),
            ('布林突破', c.get('c_bb_squeeze_breakout')),
        ]
        tech_html = ''.join([f'<span class="pill {"ok" if ok else "no"}">{"✅" if ok else "❌"} {html.escape(label)}</span>' for label, ok in tech_items])

        fund_cls = ui_card_cls((eps or 0) + (yoy or 0) if eps is not None or yoy is not None else None)
        chip_cls = ui_card_cls(chip_main)
        rsi_cls = 'tone-good' if rsi is not None and 55 <= rsi <= 75 else ('tone-neutral' if rsi is not None else 'tone-neutral')

        html_content = f"""
        <!doctype html><html><head><meta charset="utf-8">
        <style>
          body {{ margin:0; width:920px; background:#ffffff; font-family:"Noto Sans CJK TC","Microsoft JhengHei","Segoe UI",Arial,sans-serif; color:#111827; }}
          .wrap {{ padding:28px; background:#ffffff; }}
          .card {{ border:1px solid #e5e7eb; border-radius:28px; overflow:hidden; box-shadow:0 20px 60px rgba(15,23,42,.14); background:#fff; }}
          .hero {{ padding:26px 30px; background:linear-gradient(135deg,#f8fafc,#eef2ff); border-bottom:1px solid #e5e7eb; display:flex; justify-content:space-between; align-items:center; }}
          .title {{ font-size:34px; font-weight:950; letter-spacing:.2px; }}
          .name {{ margin-top:5px; color:#64748b; font-size:17px; font-weight:800; }}
          .rank {{ background:#111827; color:#fff; border-radius:999px; padding:10px 16px; font-size:16px; font-weight:950; }}
          .grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; padding:24px 26px; }}
          .box {{ border-radius:22px; padding:18px; min-height:104px; border:1px solid rgba(15,23,42,.06); }}
          .label {{ color:#64748b; font-size:13px; font-weight:900; }}
          .value {{ margin-top:8px; font-size:28px; font-weight:950; }}
          .sub {{ margin-top:7px; font-size:13px; color:#64748b; font-weight:700; }}
          .tone-good {{ background:#fee2e2; color:#991b1b; }}
          .tone-bad {{ background:#dcfce7; color:#166534; }}
          .tone-neutral {{ background:#f1f5f9; color:#334155; }}
          .tech {{ padding:0 26px 24px; }}
          .section-title {{ font-size:18px; font-weight:950; margin-bottom:12px; }}
          .pills {{ display:flex; flex-wrap:wrap; gap:10px; }}
          .pill {{ border-radius:999px; padding:9px 13px; font-size:13px; font-weight:900; }}
          .ok {{ background:#fee2e2; color:#991b1b; }}
          .no {{ background:#dcfce7; color:#166534; }}
          .footer {{ padding:17px 26px 24px; color:#64748b; font-size:13px; font-weight:700; border-top:1px solid #e5e7eb; }}
        </style></head><body>
          <div class="wrap"><div class="card" id="capture-area">
            <div class="hero">
              <div><div class="title">📊 {html.escape(str(ticker))}</div><div class="name">{html.escape(str(name))}｜{html.escape(str(profile_info.get('industry','N/A'))[:36])}</div></div>
              <div class="rank">Top #{rank if rank is not None else '-'}</div>
            </div>
            <div class="grid">
              <div class="box tone-neutral"><div class="label">模型分數</div><div class="value">{total_score:.1f}</div><div class="sub">Grade {grade}｜{regime}</div></div>
              <div class="box tone-neutral"><div class="label">盤面</div><div class="value">{safe_num_str(close,2)}</div><div class="sub">RSI {safe_num_str(rsi,1)}｜ATR {safe_num_str(atr,2)}</div></div>
              <div class="box {fund_cls}"><div class="label">💰 基本面</div><div class="value">EPS {safe_num_str(eps,2)}</div><div class="sub">YoY {safe_pct_str(yoy)}</div></div>
              <div class="box {chip_cls}"><div class="label">🏦 籌碼面</div><div class="value">{ui_display_signed(chip_main,0,'張')}</div><div class="sub">三大/外資 5D net flow</div></div>
              <div class="box {rsi_cls}"><div class="label">🧭 動能</div><div class="value">RSI {safe_num_str(rsi,1)}</div><div class="sub">MACD Hist {safe_num_str(m.get('macd_osc_d'),3)}</div></div>
              <div class="box tone-neutral"><div class="label">🎯 狀態</div><div class="value">{html.escape(regime)}</div><div class="sub">警示依 RSI / 乖離 / 量能判斷</div></div>
            </div>
            <div class="tech"><div class="section-title">🧩 技術條件</div><div class="pills">{tech_html}</div></div>
            <div class="footer">紅底＝盈利、成長、買超、正向；綠底＝虧損、衰退、賣超、負向；灰底＝中性或無資料。</div>
          </div></div>
        </body></html>
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 920, "height": 900}, device_scale_factor=2)
            page.set_content(html_content, wait_until="networkidle")
            page.locator("#capture-area").screenshot(path=output_path)
            browser.close()
            return output_path
    except Exception as e:
        log_exception(f'[STOCK-CARD-IMAGE-ERROR] {ticker}', e)
        return None

def build_stock_report(ticker, tech_pack, fin_data, profile_info, rank=None):
    latest, c, m = tech_pack['latest'], tech_pack['conditions'], tech_pack['metrics']
    mode = tech_pack['mode']
    is_us = is_us_ticker(ticker)
    
    tech_score = tech_pack['technical_score']
    fund_score = calc_fundamental_score(fin_data, is_us)
    chip_score = calc_chip_score(fin_data, is_us)
    total_score = final_total_score(tech_score, fund_score, chip_score, is_us)

    img_path = save_exquisite_plot(tech_pack['df'], tech_pack['weekly'], tech_pack['monthly'], ticker, {
        'technical_score': tech_score, 'fundamental_score': fund_score, 'chip_score': chip_score, 'total_score': total_score, 'mode': mode
    })

    close_val = latest["Close"]
    opt_ma_val = safe_float(m.get('opt_ma')) or close_val

    report = ''
    if rank is not None: report += f'🏆 **排名 #{rank}**\n'
    
    is_etf = ('00' in ticker) or (ticker in get_defensive_etf_pool('US'))
    mode_text = '🛡️ ETF 防守避風港' if is_etf else ('🔥 攻擊型飆股' if mode == 'offensive' else '🛡️ RS相對強勢')

    report += f'📊 **【量化診斷：{ticker}】** ({mode_text})\n'
    
    if is_us:
        report += f'💰 最新收盤：`{latest["Close"]:.2f}` _({m["latest_date"]})_\n'
        report += f'🧮 總分：`{total_score:.1f}` | 技術：`{tech_score:.1f}` | 基本：`{fund_score:.1f}` | 美股籌碼：`{chip_score:.1f}`\n'
    else:
        report += f'💰 最新收盤：`{latest["Close"]:.2f}` _(資料日期: {m["latest_date"]})_\n'
        report += f'🧮 總分：`{total_score:.1f}` | 技術：`{tech_score:.1f}` | 基本：`{fund_score:.1f}` | 籌碼：`{chip_score:.1f}`\n'
        
    report += '------------------------\n'
    company_name = profile_info.get('company_name') or ui_company_name_from_raw_text(profile_info.get('raw_text'), ticker) or ticker
    report += f'🏢 **公司:** {company_name}\n'
    report += f'🏭 **產業:** {profile_info["industry"]}\n_{profile_info["profile"]}_\n'
    report_item = {
        'ticker': ticker,
        'tech_pack': tech_pack,
        'profile_info': profile_info,
        'fin_data': fin_data,
        'total_score': total_score,
        'sector_tags': tech_pack.get('sector_tags', []),
        'sector_info': tech_pack.get('sector_info', {}),
    }
    primary_category = tech_pack.get('primary_strategy') or max(_all_strategy_scores(report_item), key=_all_strategy_scores(report_item).get)
    primary_tags = _strategy_specific_tags(report_item, primary_category)
    report += f'🏷️ **主策略:** `{_strategy_display_name(primary_category)}`\n'
    report += f'🏷️ **主策略標籤:** `{" / ".join(primary_tags)}`\n'
    report += '------------------------\n'

    if is_etf:
        report += '🛡️ **ETF 防守狀態：**\n'
        report += f'└ 距離 52W 高點：`{safe_pct_str(m["dist_high_pct"])}`\n'
        report += f'└ 站穩年線 (240MA)：{"✅" if c["above_ma240"] else "❌"}\n'
        if not is_us:
            report += f'⚠️ **【警告】下單前請確認官網「即時折溢價」，溢價 > 1% 請勿追高！**\n'
    else:
        report += '🔍 **技術面分析：**\n'
        report += f'{"✅" if c["trend_stack"] else "❌"} 長天期多頭排列 (價>50>150>200)\n'
        report += f'{"✅" if c["short_mid_ma_stack"] else "❌"} 短中期均線順多 (價>5>20>60)\n'
        report += f'{"✅" if c["above_ma240"] else "❌"} 站上 240MA\n'
        report += f'{"✅" if c["off_bottom"] else "❌"} 已脱離 52W 低點至少 30%\n'
        report += f'{"✅" if c["near_high"] else "❌"} 靠近 52W 高點 25% 內\n'
        report += f'{"✅" if c["momentum"] else "❌"} 日線動能：RSI>60 且 MACD>0\n'
        report += f'{"✅" if c.get("vcp_setup") else "❌"} VCP 量縮收斂\n'
        report += f'{"✅" if c.get("c_box_breakout") else "❌"} 無雜訊收盤箱型突破\n'
        report += f'{"✅" if c.get("c_bb_squeeze_breakout") else "❌"} 布林壓縮帶量突破\n'
        report += f'{"✅" if c.get("c_triangle_contraction") else "❌"} 收斂三角 / {"✅" if c.get("c_inverse_head_shoulders") else "❌"} 頭肩底雛形\n'

    report += '\n📈 **技術數據面板：**\n'
    report += f'🔹 RSI(日)：`{safe_num_str(m["rsi"], 1)}`\n'
    report += f'🔹 MACD Hist 日/週/月：`{safe_num_str(m["macd_osc_d"], 3)}` / `{safe_num_str(m["macd_osc_w"], 3)}` / `{safe_num_str(m["macd_osc_m"], 3)}`\n'
    report += f'🔹 5/20/60/240MA乖離率：`{safe_pct_str(m["bias5"])}` / `{safe_pct_str(m["bias20"])}` / `{safe_pct_str(m["bias60"])}` / `{safe_pct_str(m["bias240"])}`\n'
    report += f'🔹 VCP/BB/CTA/Pattern：`{safe_num_str(m.get("vcp_score"), 2)}` / `{safe_num_str(m.get("bb_score"), 2)}` / `{safe_num_str(m.get("cta_score"), 2)}` / `{safe_num_str(m.get("pattern_score"), 2)}`\n'

    if not is_etf:
        report += '------------------------\n'
        if is_us:
            report += '💹 **基本面 (Yahoo Finance)：**\n'
            report += f'🔸 近四季 EPS (TTM)：`{safe_num_str(fin_data.get("eps_ttm"))}`\n'
            report += f'🔸 營收成長 (Y/Y)：`{safe_pct_str(fin_data.get("single_month_yoy"))}`\n'
        else:
            report += '💹 **基本面：**\n'
            report += f'🔸 最新一季 EPS：`{safe_num_str(fin_data.get("eps_latest_quarter"))}`\n🔸 近四季 EPS：`{safe_num_str(fin_data.get("eps_ttm"))}`\n'
            report += f'🔸 單月營收 Y/Y：`{safe_pct_str(fin_data.get("single_month_yoy"))}`\n🔸 月營收 M/M：`{safe_pct_str(fin_data.get("single_month_mom"))}`\n'

    report += '------------------------\n'
    if is_us:
        report += '🏦 **美股籌碼面：**\n'
        report += f'🔸 籌碼摘要：`{fin_data.get("chips_summary", "N/A")}`\n'
        report += f'🔸 機構持股：`{safe_pct_str(fin_data.get("institutional_ownership_pct"))}` | Float Short：`{safe_pct_str(fin_data.get("short_percent_float"))}` | Short Ratio：`{safe_num_str(fin_data.get("short_ratio"), 1)}`\n'
        report += f'🔸 3M/10D 均量：`{safe_num_str(fin_data.get("avg_volume_3m"), 0)}` / `{safe_num_str(fin_data.get("avg_volume_10d"), 0)}` | 最新量：`{safe_num_str(fin_data.get("latest_volume"), 0)}`\n'
    else:
        report += '🏦 **籌碼面：**\n'
        report += f'🔸 籌碼摘要：`{fin_data.get("chips_summary", "N/A")}`\n'
        report += f'🔸 外資 2/3/5/10日：`{safe_num_str(fin_data.get("foreign_2d"), 0)}` / `{safe_num_str(fin_data.get("foreign_3d"), 0)}` / `{safe_num_str(fin_data.get("foreign_5d"), 0)}` / `{safe_num_str(fin_data.get("foreign_10d"), 0)}`\n'
        report += f'🔸 投信 2/3/5/10日：`{safe_num_str(fin_data.get("trust_2d"), 0)}` / `{safe_num_str(fin_data.get("trust_3d"), 0)}` / `{safe_num_str(fin_data.get("trust_5d"), 0)}` / `{safe_num_str(fin_data.get("trust_10d"), 0)}`\n'
        report += f'🔸 自營 2/3/5/10日：`{safe_num_str(fin_data.get("dealer_2d"), 0)}` / `{safe_num_str(fin_data.get("dealer_3d"), 0)}` / `{safe_num_str(fin_data.get("dealer_5d"), 0)}` / `{safe_num_str(fin_data.get("dealer_10d"), 0)}`\n'
        report += f'🔸 三大法人合計：`{safe_num_str(fin_data.get("total_2d"), 0)}` / `{safe_num_str(fin_data.get("total_3d"), 0)}` / `{safe_num_str(fin_data.get("total_5d"), 0)}` / `{safe_num_str(fin_data.get("total_10d"), 0)}`\n'
        srcs = ', '.join(fin_data.get('sources', [])) if fin_data.get('sources') else 'N/A'
        report += f'🔸 資料來源：`{srcs}`\n'
        
    report += '------------------------\n'
    report += '🎯 **出場雷達 (Minervini 動態策略)：**\n'

    bias20 = safe_float(m.get("bias20"))
    if bias20 is not None and bias20 >= 18:
        report += f'🚨 **高潮噴出警報**：短線正乖離達 `{bias20:.1f}%`，強烈建議了結部分部位。\n'
    elif bias20 is not None and bias20 >= 10:
        report += f'⚠️ **動能過熱**：留意獲利回吐，持股者上移停利點。\n👉 **動作**：持股者上移停利點；空手者等待量縮測試。\n'
    if close_val < opt_ma_val:
        report += f'📉 **短線轉弱**：跌破動態防守均線 `{SYS_PARAMS.get("ma_period", 20)}MA` ({opt_ma_val:.2f})。\n👉 **動作**：部位過大應考慮減碼。\n'
    else:
        report += f'🛡️ **趨勢健康 (Hold)**：股價在防守均線之上且乖離正常。\n👉 **動作**：防守底線設於 `{SYS_PARAMS.get("ma_period", 20)}MA` ({opt_ma_val:.2f})。\n'

    report += '\n'
    if total_score >= 80: report += '🚀 **結論：結構極強，屬高優先級觀察名單。**'
    elif total_score >= 65: report += '🟡 **結論：結構偏強，可列入次高優先級。**'
    else: report += '⚪ **結論：有部分條件符合，尚未達到最強勢組。**'
    report += '\n\n'

    # === 新增終端機風格戰術面板 ===
    avg_vol = safe_float(tech_pack['df']['Volume'].rolling(20).mean().iloc[-1]) if len(tech_pack['df']) >= 20 else None
    terminal_plan = generate_advanced_trading_plan(
        ticker=ticker,
        close=close_val,
        atr=safe_float(m.get('atr')),
        total_score=total_score,
        rsi=safe_float(m.get('rsi')),
        bias20=safe_float(m.get('bias20')),
        volume=safe_float(m.get('volume')),
        avg_vol=avg_vol,
        capital=500000 # 💡 可以在這裡修改你的預設操盤本金
    )
    
    report += terminal_plan

    strategy_img_path = os.path.join(REPORT_DIR, f'{ticker}_analysis_card.png')
    strategy_img_path = create_stock_analysis_card_image(ticker, tech_pack, fin_data, profile_info, total_score, rank, strategy_img_path)

    return report, img_path, strategy_img_path

def create_scan_summary_image(ranked_list, output_path, region='TW'):
    if not ranked_list:
        return None

    title = f"{region} strong stock scan | {datetime.now().strftime('%Y-%m-%d')}"
    rows_html = ""
    for idx, item in enumerate(ranked_list[:10], start=1):
        tech_pack = item.get('tech_pack') or {}
        fin_data = item.get('fin_data') or {}
        latest = tech_pack.get('latest')
        if latest is None:
            latest = {}
        ticker = item.get('ticker', '')
        pattern_tag = tech_pack.get('pattern_tag', 'Trend')
        total_score = safe_float(item.get('total_score'), 0.0) or 0.0
        metrics = tech_pack.get('metrics') or {}

        tag_cls = 'tag-blue'
        if pattern_tag == '布林突破':
            tag_cls = 'tag-red'
        elif pattern_tag == 'VCP 收斂':
            tag_cls = 'tag-green'
        elif pattern_tag == '型態吞噬':
            tag_cls = 'tag-purple'

        fund_cell = ui_fund_summary(fin_data)
        chip_cell = ui_chip_summary(fin_data)

        rows_html += f"""
        <tr>
            <td class="rank">{idx}</td>
            <td class="stock">{ui_stock_cell(item)}</td>
            <td>
                <div class="price">收 {safe_num_str(safe_float(latest.get('Close')), 2)}</div>
                <div class="sub">RSI {safe_num_str(metrics.get('rsi'), 1)}</div>
            </td>
            <td><span class="tag {tag_cls}">{html.escape(str(pattern_tag))}</span></td>
            <td>{fund_cell}</td>
            <td>{chip_cell}</td>
            <td class="score">{total_score:.1f}</td>
        </tr>
        """

    html_content = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                width: 1120px;
                background: #ffffff;
                color: #111827;
                font-family: "Noto Sans CJK TC", "Microsoft JhengHei", "Segoe UI", Arial, sans-serif;
            }}
            .wrap {{ padding: 26px; background: #ffffff; }}
            .panel {{
                border: 1px solid #e5e7eb;
                border-radius: 18px;
                overflow: hidden;
                background: #ffffff;
                box-shadow: 0 18px 50px rgba(15, 23, 42, .10);
            }}
            .head {{
                padding: 24px 28px;
                background: linear-gradient(135deg, #f8fafc, #eef2ff);
                border-bottom: 1px solid #e5e7eb;
            }}
            h1 {{ margin: 0; font-size: 30px; letter-spacing: .2px; color: #111827; }}
            .meta {{ margin-top: 8px; color: #64748b; font-size: 14px; }}
            table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
            th {{ padding: 13px 12px; color: #475569; background: #f1f5f9; font-size: 14px; text-align: center; font-weight: 900; }}
            td {{ padding: 15px 12px; border-top: 1px solid #e5e7eb; text-align: center; font-size: 14px; vertical-align: middle; }}
            tr:nth-child(even) td {{ background: #fcfcfd; }}
            .rank {{ width: 44px; color: #94a3b8; font-weight: 900; }}
            .stock {{ text-align: left; width: 150px; }}
            .ticker {{ font-size: 19px; font-weight: 950; color: #111827; }}
            .cname {{ margin-top: 3px; font-size: 12px; color: #64748b; font-weight: 800; }}
            .price {{ font-weight: 900; color: #111827; }}
            .sub {{ margin-top: 4px; color: #64748b; font-size: 12px; }}
            .tag {{ display: inline-block; min-width: 92px; padding: 8px 11px; border-radius: 999px; font-weight: 900; }}
            .tag-red {{ background:#fee2e2; color:#991b1b; }}
            .tag-green {{ background:#dcfce7; color:#166534; }}
            .tag-purple {{ background:#ede9fe; color:#5b21b6; }}
            .tag-blue {{ background:#e0f2fe; color:#075985; }}
            .score {{ color: #b45309; font-weight: 950; font-size: 23px; }}
            .metric {{ border-radius: 14px; padding: 8px 9px; line-height: 1.25; min-height: 42px; display:flex; flex-direction:column; justify-content:center; }}
            .metric-label {{ font-size: 11px; opacity: .72; font-weight: 900; }}
            .metric-val {{ font-size: 12px; font-weight: 950; }}
            .tone-good {{ background:#fee2e2 !important; color:#991b1b !important; }}
            .tone-bad {{ background:#dcfce7 !important; color:#166534 !important; }}
            .tone-neutral {{ background:#f1f5f9 !important; color:#475569 !important; }}
        </style>
    </head>
    <body>
        <div class="wrap">
            <div class="panel" id="capture-area">
                <div class="head">
                    <h1>{html.escape(title)}</h1>
                    <div class="meta">Top 10 | VCP / Bollinger breakout / trend stack summary | red = profit/net buy, green = loss/net sell</div>
                </div>
                <table>
                    <tr>
                        <th>#</th><th>股票 / 公司</th><th>盤面</th><th>型態</th><th>基本面</th><th>籌碼面</th><th>模型</th>
                    </tr>
                    {rows_html}
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1120, "height": 920}, device_scale_factor=2)
            page.set_content(html_content, wait_until="networkidle")
            page.locator("#capture-area").screenshot(path=output_path)
            browser.close()
            return output_path
    except Exception as e:
        log_exception('[SCAN-SUMMARY-IMAGE-ERROR]', e)
        return None

def _strategy_category_score(item, category):
    tech_pack = item.get('tech_pack') or {}
    metrics = tech_pack.get('metrics') or {}
    conditions = tech_pack.get('conditions') or {}
    sector = item.get('sector_info') or {}
    total = safe_float(item.get('total_score'), 0.0) or 0.0

    if category == 'vcp':
        score = (safe_float(metrics.get('vcp_score'), 0.0) or 0.0) * 100
        if conditions.get('vcp_setup'): score += 25
        return score + total * 0.25
    if category == 'bollinger':
        score = (safe_float(metrics.get('bb_score'), 0.0) or 0.0) * 100
        if conditions.get('bb_breakout'): score += 18
        if conditions.get('c_bb_squeeze_breakout'): score += 25
        if conditions.get('c_bb_momentum_breakout'): score += 12
        return score + total * 0.25
    if category == 'pattern':
        score = (safe_float(metrics.get('pattern_score'), 0.0) or 0.0) * 100
        score += (safe_float(metrics.get('cta_score'), 0.0) or 0.0) * 45
        if conditions.get('c_triangle_contraction'): score += 22
        if conditions.get('c_inverse_head_shoulders'): score += 22
        if conditions.get('c_box_breakout'): score += 18
        if conditions.get('c_engulfing_5d'): score += 12
        return score + total * 0.20
    if category == 'minervini':
        score = safe_float(tech_pack.get('technical_score'), 0.0) or 0.0
        for key, bonus in [
            ('trend_stack', 18),
            ('off_bottom', 8),
            ('near_high', 10),
            ('momentum', 12),
            ('short_mid_ma_stack', 10),
            ('above_ma240', 8),
            ('weekly_up', 8),
            ('monthly_up', 5),
        ]:
            if conditions.get(key):
                score += bonus
        bias20 = safe_float(metrics.get('bias20'))
        if bias20 is not None and 0 <= bias20 <= 12:
            score += 8
        elif bias20 is not None and bias20 > 22:
            score -= 12
        return score + total * 0.25
    if category == 'sector':
        score = safe_float(sector.get('sector_strength_score'), 0.0) or 0.0
        score += len(item.get('sector_tags') or []) * 12
        return score + total * 0.20
    return total

def _all_strategy_scores(item):
    return {
        'vcp': _strategy_category_score(item, 'vcp'),
        'bollinger': _strategy_category_score(item, 'bollinger'),
        'pattern': _strategy_category_score(item, 'pattern'),
        'minervini': _strategy_category_score(item, 'minervini'),
        'sector': _strategy_category_score(item, 'sector'),
    }

def _strategy_display_name(category):
    return {
        'vcp': 'VCP 選股',
        'bollinger': '布林突破',
        'pattern': '形態學突破',
        'minervini': 'Minervini Trend',
        'sector': '族群連動',
    }.get(category, '綜合策略')

def _strategy_description(category):
    return {
        'vcp': '只看波動收斂、量縮、接近 pivot 的 VCP 結構',
        'bollinger': '只看布林帶寬壓縮、上軌突破、量能、RSI/MACD 動能',
        'pattern': '只看收斂三角、頭肩底、收盤箱型突破、五日陣吞噬',
        'minervini': '只看 Minervini 趨勢模板、均線排列、52W 強度與多週期趨勢',
        'sector': '只看同產業分數、上漲比例、突破比例與族群領先度',
    }.get(category, '綜合策略分數')

def _strategy_specific_tags(item, category):
    tech_pack = item.get('tech_pack') or {}
    metrics = tech_pack.get('metrics') or {}
    conditions = tech_pack.get('conditions') or {}
    sector = item.get('sector_info') or {}
    tags = []

    if category == 'vcp':
        vcp_score = safe_float(metrics.get('vcp_score'), 0.0) or 0.0
        if conditions.get('vcp_setup') or vcp_score >= 0.65:
            tags.append('VCP量縮收斂')
        if safe_float(metrics.get('vcp_pivot')) is not None:
            tags.append('Pivot附近')
        tags.append(f'VCP分 {vcp_score:.2f}')

    elif category == 'bollinger':
        if conditions.get('c_bb_squeeze_breakout') or conditions.get('bb_breakout'):
            tags.append('布林壓縮突破')
        if conditions.get('c_bb_momentum_breakout') or conditions.get('momentum'):
            tags.append('RSI/MACD動能確認')
        if conditions.get('liquidity'):
            tags.append('量能合格')
        bb_width = safe_float(metrics.get('bb_width_pctile'))
        if bb_width is not None:
            tags.append(f'BB寬度PR {bb_width:.1f}')

    elif category == 'pattern':
        if conditions.get('c_triangle_contraction'):
            tags.append('收斂三角')
        if conditions.get('c_inverse_head_shoulders'):
            tags.append('頭肩底雛形')
        if conditions.get('c_box_breakout'):
            tags.append('收盤箱型突破')
        if conditions.get('c_engulfing_5d'):
            tags.append('五日陣吞噬')
        pattern_score = safe_float(metrics.get('pattern_score'), 0.0) or 0.0
        tags.append(f'型態分 {pattern_score:.2f}')

    elif category == 'minervini':
        if conditions.get('trend_stack'):
            tags.append('價>50>150>200')
        if conditions.get('short_mid_ma_stack'):
            tags.append('價>5>20>60')
        if conditions.get('off_bottom'):
            tags.append('離52W低點+30%')
        if conditions.get('near_high'):
            tags.append('靠近52W高點')
        if conditions.get('weekly_up') or conditions.get('monthly_up'):
            tags.append('週/月趨勢向上')

    elif category == 'sector':
        tags.extend(item.get('sector_tags') or [])
        industry = (item.get('profile_info') or {}).get('industry')
        if industry and industry != 'N/A':
            tags.append(str(industry)[:18])
        strength = safe_float(sector.get('sector_strength_score'))
        if strength is not None:
            tags.append(f'族群強度 {strength:.1f}')
        count = sector.get('count')
        if count:
            tags.append(f'同族{count}檔')

    if not tags:
        tags.append(_strategy_display_name(category))
    return list(dict.fromkeys(tags))[:5]

def estimate_item_win_rate(item):
    """Rule-based estimated win rate used for priority ranking.

    This is a deterministic confidence proxy, not a historical realized win rate.
    It rewards total score, technical quality, confirmed breakout signals, and
    sector confirmation, then caps the estimate to avoid overclaiming certainty.
    """
    tech_pack = item.get('tech_pack') or {}
    metrics = tech_pack.get('metrics') or {}
    conditions = tech_pack.get('conditions') or {}
    total = safe_float(item.get('total_score'), 0.0) or 0.0
    tech = safe_float(tech_pack.get('technical_score'), 0.0) or 0.0

    win_rate = 0.38 + total * 0.0025 + tech * 0.0012
    signal_bonus = 0.0
    for key, bonus in [
        ('trend_stack', 0.018),
        ('momentum', 0.016),
        ('vcp_setup', 0.020),
        ('bb_breakout', 0.016),
        ('c_box_breakout', 0.020),
        ('c_bb_squeeze_breakout', 0.024),
        ('c_bb_momentum_breakout', 0.016),
        ('c_triangle_contraction', 0.018),
        ('c_inverse_head_shoulders', 0.018),
    ]:
        if conditions.get(key):
            signal_bonus += bonus

    signal_bonus += min((safe_float(metrics.get('vcp_score'), 0.0) or 0.0) * 0.030, 0.030)
    signal_bonus += min((safe_float(metrics.get('bb_score'), 0.0) or 0.0) * 0.025, 0.025)
    signal_bonus += min((safe_float(metrics.get('pattern_score'), 0.0) or 0.0) * 0.025, 0.025)
    signal_bonus += min(len(item.get('sector_tags') or []) * 0.012, 0.036)

    bias20 = safe_float(metrics.get('bias20'))
    if bias20 is not None and bias20 > 18:
        signal_bonus -= 0.035
    elif bias20 is not None and bias20 < -3:
        signal_bonus -= 0.025

    return max(0.35, min(0.88, win_rate + signal_bonus)) * 100

def select_priority_recommendations(ranked, limit=10):
    """Pick weighted Top 10 from each strategy bucket's Top 10 candidates."""
    candidate_map = {}
    for category in ['vcp', 'pattern', 'bollinger', 'minervini', 'sector']:
        for item in _rank_strategy_bucket(ranked, category, limit=limit):
            ticker = item.get('ticker')
            if ticker and ticker not in candidate_map:
                candidate_map[ticker] = item

    source_items = list(candidate_map.values()) or list(ranked)
    priority = []
    for item in source_items:
        copied = dict(item)
        strategy_scores = _all_strategy_scores(item)
        primary_strategy = max(strategy_scores, key=strategy_scores.get)
        best_strategy_score = strategy_scores.get(primary_strategy, 0.0)
        win_rate_pct = estimate_item_win_rate(item)
        total = safe_float(item.get('total_score'), 0.0) or 0.0
        strategy_average = sum(min(v, 140.0) for v in strategy_scores.values()) / max(1, len(strategy_scores))
        priority_score = total * 0.42 + win_rate_pct * 0.28 + min(best_strategy_score, 150.0) * 0.22 + strategy_average * 0.08
        copied['strategy_scores'] = strategy_scores
        copied['primary_strategy'] = primary_strategy
        copied['primary_strategy_name'] = _strategy_display_name(primary_strategy)
        copied['estimated_win_rate_pct'] = win_rate_pct
        copied['priority_score'] = priority_score
        priority.append(copied)

    priority.sort(
        key=lambda x: (
            safe_float(x.get('priority_score'), 0.0) or 0.0,
            safe_float(x.get('estimated_win_rate_pct'), 0.0) or 0.0,
            safe_float(x.get('total_score'), 0.0) or 0.0,
        ),
        reverse=True,
    )
    return priority[:limit]

def _rank_strategy_bucket(ranked, category, limit=10):
    scored = []
    for item in ranked:
        score = _strategy_category_score(item, category)
        copied = dict(item)
        copied['category_score'] = score
        scored.append(copied)

    scored.sort(key=lambda x: (safe_float(x.get('category_score'), 0.0) or 0.0, safe_float(x.get('total_score'), 0.0) or 0.0), reverse=True)
    selected = scored[:limit]
    if len(selected) < limit:
        seen = {x.get('ticker') for x in selected}
        fillers = [x for x in scored if x.get('ticker') not in seen]
        selected.extend(fillers[:limit - len(selected)])
    return selected[:limit]

def create_strategy_bucket_dashboard(ranked, output_path, region='US', category='vcp', title='Strategy Top 10'):
    if not ranked:
        return None

    rows_html = ''
    for idx, item in enumerate(ranked[:10], start=1):
        tech_pack = item.get('tech_pack') or {}
        metrics = tech_pack.get('metrics') or {}
        profile = item.get('profile_info') or {}
        fin = item.get('fin_data') or {}
        latest = tech_pack.get('latest', {})
        tag_text = ' / '.join(_strategy_specific_tags(item, category)) or '-'
        yoy_cls = ui_tone_pct(fin.get('single_month_yoy'))

        rows_html += f"""
        <tr>
            <td class="rank">{idx}</td>
            <td class="stock">{ui_stock_cell(item)}</td>
            <td class="industry">{html.escape(str(profile.get('industry', 'N/A'))[:34])}</td>
            <td class="num">{safe_num_str(safe_float(latest.get('Close')), 2)}</td>
            <td class="score-main">{safe_num_str(item.get('category_score'), 1)}</td>
            <td class="num">{safe_num_str(item.get('total_score'), 1)}</td>
            <td>{ui_fund_summary(fin)}</td>
            <td>{ui_chip_summary(fin)}</td>
            <td class="{yoy_cls} compact">{safe_pct_str(fin.get('single_month_yoy'))}</td>
            <td class="mini">{safe_num_str(metrics.get('vcp_score'), 2)}</td>
            <td class="mini">{safe_num_str(metrics.get('bb_score'), 2)}</td>
            <td class="mini">{safe_num_str(metrics.get('pattern_score'), 2)}</td>
            <td class="tags">{html.escape(tag_text)}</td>
        </tr>
        """

    html_content = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ margin:0; width:1280px; background:#ffffff; color:#111827; font-family:"Noto Sans CJK TC","Microsoft JhengHei","Segoe UI",Arial,sans-serif; }}
        .wrap {{ padding:26px; background:#ffffff; }}
        .panel {{ border:1px solid #e5e7eb; border-radius:18px; overflow:hidden; background:#ffffff; box-shadow:0 18px 50px rgba(15,23,42,.10); }}
        .head {{ display:flex; justify-content:space-between; align-items:flex-end; padding:24px 28px; background:linear-gradient(135deg,#f8fafc,#eef2ff); border-bottom:1px solid #e5e7eb; }}
        h1 {{ margin:0; font-size:30px; letter-spacing:.2px; color:#111827; }}
        .meta {{ color:#64748b; font-size:13px; margin-top:7px; }}
        table {{ width:100%; border-collapse:collapse; table-layout:fixed; font-size:13px; }}
        th {{ padding:12px 8px; background:#f1f5f9; color:#475569; text-align:center; font-weight:900; }}
        td {{ padding:13px 8px; border-top:1px solid #e5e7eb; text-align:center; vertical-align:middle; }}
        tr:nth-child(even) td {{ background:#fcfcfd; }}
        .rank {{ width:38px; color:#94a3b8; font-weight:900; }}
        .stock {{ text-align:left; width:130px; }}
        .ticker {{ font-size:18px; font-weight:950; color:#111827; }}
        .cname {{ margin-top:3px; font-size:12px; color:#64748b; font-weight:800; }}
        .industry {{ color:#334155; }}
        .num {{ font-variant-numeric:tabular-nums; font-weight:800; }}
        .score-main {{ color:#b45309; font-weight:950; font-size:20px; }}
        .mini {{ font-size:12px; color:#475569; }}
        .tags {{ color:#334155; text-align:left; line-height:1.45; font-size:12px; }}
        .compact {{ border-radius:12px; font-weight:950; font-variant-numeric:tabular-nums; }}
        .metric {{ border-radius:14px; padding:8px 9px; line-height:1.25; min-height:42px; display:flex; flex-direction:column; justify-content:center; }}
        .metric-label {{ font-size:11px; opacity:.72; font-weight:900; }}
        .metric-val {{ font-size:12px; font-weight:950; }}
        .tone-good {{ background:#fee2e2 !important; color:#991b1b !important; }}
        .tone-bad {{ background:#dcfce7 !important; color:#166534 !important; }}
        .tone-neutral {{ background:#f1f5f9 !important; color:#475569 !important; }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="panel" id="capture-area">
          <div class="head">
            <div>
              <h1>{html.escape(title)}</h1>
              <div class="meta">{html.escape(region)} | {html.escape(category.upper())} | {html.escape(_strategy_description(category))} | generated {now_str()}</div>
            </div>
            <div class="meta">Top 10 by category score</div>
          </div>
          <table>
            <tr>
              <th>#</th><th>股票 / 公司</th><th>Industry</th><th>Close</th><th>Cat</th><th>Total</th>
              <th>基本面</th><th>籌碼面</th><th>YoY</th><th>VCP</th><th>BB</th><th>Pattern</th><th>Tags</th>
            </tr>
            {rows_html}
          </table>
        </div>
      </div>
    </body>
    </html>
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=2)
            page.set_content(html_content, wait_until="networkidle")
            page.locator("#capture-area").screenshot(path=output_path)
            browser.close()
            return output_path
    except Exception as e:
        log_exception(f'[STRATEGY-DASHBOARD-ERROR] {category}', e)
        return None


def create_strategy_bucket_images(ranked, region='US', market_mode='offensive'):
    categories = [
        ('vcp', 'VCP 選股 Top 10'),
        ('pattern', '形態學突破 Top 10'),
        ('bollinger', '布林壓縮突破 Top 10'),
        ('minervini', 'Minervini Trend Top 10'),
        ('sector', '族群連動 Top 10'),
    ]
    outputs = []
    for key, title in categories:
        bucket = _rank_strategy_bucket(ranked, key, limit=10)
        output_path = os.path.join(REPORT_DIR, f'{region}_{key}_top10.png')
        rendered = create_strategy_bucket_dashboard(bucket, output_path, region=region, category=key, title=title)
        if rendered and os.path.exists(rendered):
            outputs.append((title, rendered))
    return outputs

def create_priority_recommendation_dashboard(ranked, output_path, region='TW', market_mode='offensive'):
    if not ranked:
        return None

    rows_html = ''
    for idx, item in enumerate(ranked[:10], start=1):
        tech_pack = item.get('tech_pack') or {}
        profile = item.get('profile_info') or {}
        fin = item.get('fin_data') or {}
        latest = tech_pack.get('latest', {})
        primary = item.get('primary_strategy', 'minervini')
        tag_text = ' / '.join(_strategy_specific_tags(item, primary)) or '-'

        rows_html += f"""
        <tr>
            <td class="rank">{idx}</td>
            <td class="stock">{ui_stock_cell(item)}</td>
            <td class="strategy">{html.escape(str(item.get('primary_strategy_name', '-')))}</td>
            <td class="win">{safe_num_str(item.get('estimated_win_rate_pct'), 1)}%</td>
            <td class="priority">{safe_num_str(item.get('priority_score'), 1)}</td>
            <td class="num">{safe_num_str(item.get('total_score'), 1)}</td>
            <td class="num">{safe_num_str(tech_pack.get('technical_score'), 1)}</td>
            <td class="num">{safe_num_str(safe_float(latest.get('Close')), 2)}</td>
            <td>{ui_fund_summary(fin)}</td>
            <td>{ui_chip_summary(fin)}</td>
            <td class="industry">{html.escape(str(profile.get('industry', 'N/A'))[:30])}</td>
            <td class="tags">{html.escape(tag_text)}</td>
        </tr>
        """

    html_content = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ margin:0; width:1280px; background:#ffffff; color:#111827; font-family:"Noto Sans CJK TC","Microsoft JhengHei","Segoe UI",Arial,sans-serif; }}
        .wrap {{ padding:26px; background:#ffffff; }}
        .panel {{ border:1px solid #e5e7eb; border-radius:18px; overflow:hidden; background:#ffffff; box-shadow:0 18px 50px rgba(15,23,42,.12); }}
        .head {{ display:flex; justify-content:space-between; align-items:flex-end; padding:24px 28px; background:linear-gradient(135deg,#fff7ed,#fee2e2); border-bottom:1px solid #fecaca; }}
        h1 {{ margin:0; font-size:30px; color:#7f1d1d; letter-spacing:.2px; }}
        .meta {{ color:#7f1d1d; opacity:.78; font-size:13px; margin-top:7px; }}
        table {{ width:100%; border-collapse:collapse; table-layout:fixed; font-size:13px; }}
        th {{ padding:12px 8px; background:#f8fafc; color:#475569; text-align:center; font-weight:900; }}
        td {{ padding:13px 8px; border-top:1px solid #e5e7eb; text-align:center; vertical-align:middle; }}
        tr:nth-child(even) td {{ background:#fcfcfd; }}
        .rank {{ width:38px; color:#94a3b8; font-weight:900; }}
        .stock {{ text-align:left; width:140px; }}
        .ticker {{ font-size:18px; font-weight:950; color:#111827; }}
        .cname {{ margin-top:3px; font-size:12px; color:#64748b; font-weight:800; }}
        .strategy {{ font-weight:900; color:#334155; }}
        .win {{ color:#2563eb; font-weight:950; }}
        .priority {{ color:#b45309; font-size:20px; font-weight:950; }}
        .num {{ font-variant-numeric:tabular-nums; font-weight:800; }}
        .industry {{ color:#334155; }}
        .tags {{ color:#334155; text-align:left; line-height:1.45; font-size:12px; }}
        .metric {{ border-radius:14px; padding:8px 9px; line-height:1.25; min-height:42px; display:flex; flex-direction:column; justify-content:center; }}
        .metric-label {{ font-size:11px; opacity:.72; font-weight:900; }}
        .metric-val {{ font-size:12px; font-weight:950; }}
        .tone-good {{ background:#fee2e2 !important; color:#991b1b !important; }}
        .tone-bad {{ background:#dcfce7 !important; color:#166534 !important; }}
        .tone-neutral {{ background:#f1f5f9 !important; color:#475569 !important; }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="panel" id="capture-area">
          <div class="head">
            <div>
              <h1>{html.escape(region)} 五策略綜合優先推薦 Top 10</h1>
              <div class="meta">VCP / 形態學突破 / 布林突破 / Minervini Trend / 族群連動 | Mode {html.escape(market_mode)} | {now_str()}</div>
            </div>
            <div class="meta">排序 = 分數 + 估勝率 + 策略分 + 現強衰分</div>
          </div>
          <table>
            <tr>
              <th>#</th><th>股票 / 公司</th><th>主策略</th><th>估勝率</th><th>優先分</th><th>總分</th>
              <th>技術</th><th>Close</th><th>基本面</th><th>籌碼面</th><th>Industry</th><th>Tags</th>
            </tr>
            {rows_html}
          </table>
        </div>
      </div>
    </body>
    </html>
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=2)
            page.set_content(html_content, wait_until="networkidle")
            page.locator("#capture-area").screenshot(path=output_path)
            browser.close()
            return output_path
    except Exception as e:
        log_exception('[PRIORITY-DASHBOARD-ERROR]', e)
        return None


def _qpro_industry_image_name(topic):
    report_name = str((topic or {}).get('report_image') or '').strip()
    if report_name:
        return report_name
    key = re.sub(r'[^a-z0-9_-]+', '_', str((topic or {}).get('key') or 'topic').lower()).strip('_')
    return f'industry_{key or "topic"}.png'


def _qpro_industry_rank_lookup(ranked):
    lookup = {}
    for item in ranked or []:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get('ticker') or '').strip().upper()
        if not ticker:
            continue
        lookup[ticker] = item
        code = qpro_tw_code(ticker)
        if code:
            lookup[code] = item
    return lookup


def create_industry_topic_dashboard(topic, ranked, output_path, region='TW'):
    if region != 'TW' or not isinstance(topic, dict):
        return None

    lookup = _qpro_industry_rank_lookup(ranked)
    lanes = []
    leaders = []
    seen_leaders = set()
    tracked_count = 0
    ranked_count = 0
    priority_count = 0

    for lane in topic.get('lanes', []) or []:
        companies = []
        for company in lane.get('companies', []) or []:
            tracked_count += 1
            symbol = str(company.get('ticker') or company.get('symbol') or '').strip()
            row = lookup.get(symbol.upper()) or lookup.get(qpro_tw_code(symbol))
            stock_ticker = str((row or {}).get('ticker') or symbol).strip()
            company_name = str(company.get('name') or (ui_stock_name(row) if row else symbol)).strip()
            in_rank = bool(row)
            is_priority = bool((row or {}).get('priority_score') is not None)
            if in_rank:
                ranked_count += 1
            if is_priority:
                priority_count += 1
            if row and stock_ticker not in seen_leaders:
                seen_leaders.add(stock_ticker)
                leaders.append(row)

            companies.append({
                'ticker': stock_ticker,
                'company_name': company_name,
                'role': str(company.get('role') or '').strip(),
                'thesis': str(company.get('thesis') or '').strip(),
                'industry': str((row or {}).get('profile_info', {}).get('industry') or (row or {}).get('industry') or '').strip(),
                'in_rank': in_rank,
                'is_priority': is_priority,
                'total_score': safe_float((row or {}).get('total_score')),
                'strategy': str((row or {}).get('primary_strategy_name') or (row or {}).get('strategy') or '').strip(),
                'close': safe_float(((row or {}).get('tech_pack') or {}).get('latest', {}).get('Close')) if row else safe_float((row or {}).get('close')),
                'tags': _full_signal_tags(row, (row or {}).get('primary_strategy')) if row else [],
            })
        lanes.append({
            'title': str(lane.get('title') or '').strip(),
            'description': str(lane.get('description') or '').strip(),
            'notes': list(lane.get('notes') or []),
            'companies': companies,
        })

    leaders.sort(key=lambda item: (
        1 if item.get('priority_score') is not None else 0,
        safe_float(item.get('total_score'), 0.0) or 0.0,
    ), reverse=True)

    leader_html = ''
    for item in leaders[:6]:
        profile = item.get('profile_info') or {}
        latest = qpro_get_latest(item.get('tech_pack') or {}) or {}
        leader_tags = ' / '.join(_full_signal_tags(item, item.get('primary_strategy'))[:3]) or '-'
        leader_html += f"""
        <div class="leader">
          <div class="leader-top">
            <div>
              <div class="leader-ticker">{html.escape(str(item.get('ticker', '')))}</div>
              <div class="leader-name">{html.escape(ui_stock_name(item))}</div>
            </div>
            <div class="score-chip">{safe_num_str(item.get('total_score'), 1)}</div>
          </div>
          <div class="leader-meta">{html.escape(str(profile.get('industry', 'N/A'))[:34])}</div>
          <div class="leader-sub">Strategy {html.escape(str(item.get('primary_strategy_name') or item.get('primary_strategy') or '-'))}</div>
          <div class="leader-sub">Close {safe_num_str(safe_float(latest.get('Close')), 2)}</div>
          <div class="leader-tags">{html.escape(leader_tags)}</div>
        </div>
        """
    if not leader_html:
        leader_html = '<div class="empty">目前這個主題還沒有公司進到最新量化排行。</div>'

    lane_html = ''
    for lane in lanes:
        companies_html = ''
        for company in lane['companies']:
            tag_text = ' / '.join(company.get('tags') or []) or '-'
            status_text = '已進最新排行' if company['in_rank'] else '尚未進最新排行'
            score_text = safe_num_str(company.get('total_score'), 1) if company['in_rank'] else '-'
            strategy_text = company.get('strategy') or '-'
            close_text = safe_num_str(company.get('close'), 2) if company['in_rank'] else '-'
            company_cls = 'company in-rank' if company['in_rank'] else 'company'
            companies_html += f"""
            <div class="{company_cls}">
              <div class="company-top">
                <div>
                  <div class="company-ticker">{html.escape(company.get('ticker', ''))}</div>
                  <div class="company-name">{html.escape(company.get('company_name', ''))}</div>
                </div>
                <div class="company-score">{score_text}</div>
              </div>
              <div class="company-role">{html.escape(company.get('role', ''))}</div>
              <div class="company-status">{html.escape(status_text)} | Strategy {html.escape(strategy_text)} | Close {close_text}</div>
              <div class="company-thesis">{html.escape(company.get('thesis', '') or company.get('industry', '') or '目前沒有補充說明')}</div>
              <div class="company-tags">{html.escape(tag_text)}</div>
            </div>
            """
        notes_html = ''.join(f'<li>{html.escape(str(note))}</li>' for note in lane.get('notes') or [])
        lane_html += f"""
        <div class="lane-card">
          <div class="lane-title">{html.escape(lane.get('title', ''))}</div>
          <div class="lane-desc">{html.escape(lane.get('description', ''))}</div>
          {'<ul class="lane-notes">' + notes_html + '</ul>' if notes_html else ''}
          <div class="company-list">{companies_html}</div>
        </div>
        """

    deep_dive = topic.get('deep_dive') or {}
    thesis_html = ''.join(f'<li>{html.escape(str(item))}</li>' for item in deep_dive.get('thesis') or [])
    risk_html = ''.join(f'<li>{html.escape(str(item))}</li>' for item in deep_dive.get('risks') or [])
    catalyst_html = ''.join(f'<li>{html.escape(str(item))}</li>' for item in deep_dive.get('catalysts') or [])
    chain_html = ''.join(
        f"""
        <div class="chain-card">
          <div class="chain-stage">{html.escape(str(item.get('stage') or ''))}</div>
          <div class="chain-summary">{html.escape(str(item.get('summary') or ''))}</div>
          <div class="chain-winners">{html.escape(' / '.join(str(x) for x in (item.get('winners') or [])) or '-')}</div>
        </div>
        """
        for item in topic.get('chain_steps') or []
    )
    watch_html = ''.join(
        f"""
        <div class="watch-card">
          <div class="watch-label">{html.escape(str(item.get('label') or ''))}</div>
          <div class="watch-why">{html.escape(str(item.get('why') or ''))}</div>
          <div class="watch-bull">偏多：{html.escape(str(item.get('bull') or ''))}</div>
          <div class="watch-bear">偏空：{html.escape(str(item.get('bear') or ''))}</div>
        </div>
        """
        for item in topic.get('watch_metrics') or []
    )
    highlight_html = ''.join(f'<span class="pill">{html.escape(str(item))}</span>' for item in topic.get('highlights') or [])

    html_content = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ margin:0; width:1440px; background:#eef7f0; color:#18212f; font-family:"Noto Sans CJK TC","Microsoft JhengHei","Segoe UI",Arial,sans-serif; }}
        .wrap {{ padding:24px; background:
          radial-gradient(circle at top left, rgba(11,118,109,.10), transparent 28%),
          radial-gradient(circle at top right, rgba(189,111,37,.12), transparent 24%),
          linear-gradient(180deg,#eef7f0 0%,#fffdf7 42%,#ffffff 100%); }}
        .canvas {{ border:1px solid #d8e1db; border-radius:28px; background:rgba(255,255,255,.92); overflow:hidden; box-shadow:0 22px 56px rgba(17,28,40,.10); }}
        .hero {{ display:grid; grid-template-columns:1.35fr .95fr; gap:18px; padding:24px; border-bottom:1px solid #e6ece8; }}
        .hero-copy h1 {{ margin:0; font-size:46px; line-height:1.02; color:#18212f; }}
        .eyebrow {{ color:#0b766d; font-size:12px; font-weight:950; text-transform:uppercase; letter-spacing:.08em; margin:0 0 10px; }}
        .hero-copy p {{ margin:16px 0 0; color:#5d6876; font-size:16px; line-height:1.72; }}
        .pill-row {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }}
        .pill {{ display:inline-flex; align-items:center; min-height:34px; padding:0 12px; border-radius:999px; background:#e5f7f3; color:#0b766d; font-size:12px; font-weight:900; }}
        .metric-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
        .metric {{ min-height:126px; border-radius:22px; border:1px solid #d8e1db; background:rgba(255,255,255,.88); padding:18px; box-shadow:0 14px 34px rgba(17,28,40,.06); }}
        .metric-label {{ color:#5d6876; font-size:12px; font-weight:900; text-transform:uppercase; letter-spacing:.05em; }}
        .metric-value {{ margin-top:10px; font-size:28px; font-weight:950; color:#18212f; }}
        .metric-note {{ margin-top:12px; color:#5d6876; font-size:13px; line-height:1.6; }}
        .section {{ padding:24px; border-top:1px solid #eef2ef; }}
        .section:first-of-type {{ border-top:0; }}
        .section-head {{ display:flex; justify-content:space-between; align-items:end; gap:16px; margin-bottom:16px; }}
        .section-head h2 {{ margin:0; font-size:28px; color:#18212f; }}
        .section-head p {{ margin:8px 0 0; color:#5d6876; font-size:14px; }}
        .split {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
        .card {{ border:1px solid #d8e1db; border-radius:22px; background:#ffffff; padding:20px; }}
        .card h3 {{ margin:0 0 10px; font-size:22px; }}
        .card p {{ margin:0; color:#5d6876; font-size:14px; line-height:1.7; }}
        .card ul {{ margin:12px 0 0; padding-left:18px; color:#5d6876; line-height:1.7; }}
        .leaders {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
        .leader {{ border:1px solid #d8e1db; border-radius:20px; background:#ffffff; padding:18px; }}
        .leader-top {{ display:flex; justify-content:space-between; gap:12px; align-items:start; }}
        .leader-ticker {{ font-size:20px; font-weight:950; color:#18212f; }}
        .leader-name {{ margin-top:4px; color:#5d6876; font-size:13px; font-weight:800; }}
        .leader-meta, .leader-sub, .leader-tags {{ margin-top:10px; color:#5d6876; font-size:13px; line-height:1.55; }}
        .score-chip {{ display:inline-flex; align-items:center; justify-content:center; min-width:66px; min-height:40px; padding:0 12px; border-radius:14px; background:#fff1e1; color:#bd6f25; font-weight:950; }}
        .empty {{ border:1px dashed #c6d0cb; border-radius:20px; padding:24px; color:#5d6876; background:#ffffff; }}
        .lane-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
        .lane-card {{ border:1px solid #d8e1db; border-radius:22px; background:#ffffff; padding:20px; }}
        .lane-title {{ font-size:22px; font-weight:950; color:#18212f; }}
        .lane-desc {{ margin-top:8px; color:#5d6876; line-height:1.7; }}
        .lane-notes {{ margin:12px 0 0; padding-left:18px; color:#5d6876; line-height:1.7; }}
        .company-list {{ display:grid; gap:10px; margin-top:16px; }}
        .company {{ border:1px solid #e6ece8; border-radius:16px; background:#fbfcfb; padding:14px; }}
        .company.in-rank {{ background:#eef9f6; border-color:#b9dfd7; }}
        .company-top {{ display:flex; justify-content:space-between; gap:10px; align-items:start; }}
        .company-ticker {{ font-size:18px; font-weight:950; color:#18212f; }}
        .company-name {{ margin-top:3px; color:#5d6876; font-size:13px; font-weight:800; }}
        .company-score {{ min-width:58px; text-align:center; color:#bd6f25; font-weight:950; }}
        .company-role, .company-status, .company-thesis, .company-tags {{ margin-top:9px; color:#5d6876; font-size:12px; line-height:1.55; }}
        .chain-grid, .watch-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
        .chain-card, .watch-card {{ border:1px solid #d8e1db; border-radius:18px; background:#ffffff; padding:16px; }}
        .chain-stage, .watch-label {{ font-size:18px; font-weight:950; color:#18212f; }}
        .chain-summary, .watch-why, .watch-bull, .watch-bear, .chain-winners {{ margin-top:10px; color:#5d6876; font-size:13px; line-height:1.6; }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="canvas" id="capture-area">
          <div class="hero">
            <div class="hero-copy">
              <div class="eyebrow">Industry Atlas</div>
              <h1>{html.escape(str(topic.get('title') or 'Industry Map'))}</h1>
              <p>{html.escape(str(topic.get('subtitle') or ''))}</p>
              <p>{html.escape(str(topic.get('description') or ''))}</p>
              <div class="pill-row">{highlight_html or '<span class="pill">以最新量化分數疊回供應鏈節點</span>'}</div>
            </div>
            <div class="metric-grid">
              <div class="metric">
                <div class="metric-label">Tracked Companies</div>
                <div class="metric-value">{tracked_count}</div>
                <div class="metric-note">目前主題地圖共整理 {tracked_count} 家供應鏈公司。</div>
              </div>
              <div class="metric">
                <div class="metric-label">In Ranked</div>
                <div class="metric-value">{ranked_count}</div>
                <div class="metric-note">有 {ranked_count} 家已經出現在最新量化排行裡。</div>
              </div>
              <div class="metric">
                <div class="metric-label">Priority Picks</div>
                <div class="metric-value">{priority_count}</div>
                <div class="metric-note">其中 {priority_count} 家屬於優先推薦名單。</div>
              </div>
              <div class="metric">
                <div class="metric-label">Updated</div>
                <div class="metric-value">{html.escape(now_str())}</div>
                <div class="metric-note">資料來源：Telegram 掃描結果與產業地圖共用同一批匯出。</div>
              </div>
            </div>
          </div>

          <div class="section">
            <div class="section-head">
              <div>
                <div class="eyebrow">Detailed Analysis</div>
                <h2>產業主軸與風險</h2>
                <p>先整理產業命題，再看量化排行是否開始支持這個故事。</p>
              </div>
            </div>
            <div class="split">
              <div class="card">
                <h3>投資主軸</h3>
                <p>{html.escape(str(deep_dive.get('market_question') or '目前尚未補充市場命題。'))}</p>
                {'<ul>' + thesis_html + '</ul>' if thesis_html else ''}
              </div>
              <div class="card">
                <h3>催化劑與風險</h3>
                {'<ul>' + catalyst_html + '</ul>' if catalyst_html else '<p>目前沒有額外催化劑說明。</p>'}
                {'<ul>' + risk_html + '</ul>' if risk_html else ''}
              </div>
            </div>
          </div>

          <div class="section">
            <div class="section-head">
              <div>
                <div class="eyebrow">Quant Overlay</div>
                <h2>量化焦點</h2>
                <p>哪些主題內公司已經被最新掃描抓到，這一塊最適合直接對照 Telegram 推播。</p>
              </div>
            </div>
            <div class="leaders">{leader_html}</div>
          </div>

          <div class="section">
            <div class="section-head">
              <div>
                <div class="eyebrow">Supply Nodes</div>
                <h2>供應鏈節點</h2>
                <p>每個節點都保留角色、產業敘事，以及是否已進量化排行。</p>
              </div>
            </div>
            <div class="lane-grid">{lane_html}</div>
          </div>

          <div class="section">
            <div class="section-head">
              <div>
                <div class="eyebrow">Value Chain</div>
                <h2>價值鏈路徑與追蹤指標</h2>
                <p>把故事拆成可以持續觀察的節奏，而不是只靠題材名稱判斷。</p>
              </div>
            </div>
            <div class="split">
              <div class="chain-grid">{chain_html or '<div class="empty">目前沒有價值鏈步驟資料。</div>'}</div>
              <div class="watch-grid">{watch_html or '<div class="empty">目前沒有追蹤指標資料。</div>'}</div>
            </div>
          </div>
        </div>
      </div>
    </body>
    </html>
    """

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1460, "height": 960}, device_scale_factor=2)
            page.set_content(html_content, wait_until="networkidle")
            page.locator("#capture-area").screenshot(path=output_path)
            browser.close()
            return output_path
    except Exception as e:
        log_exception(f'[INDUSTRY-DASHBOARD-ERROR] {topic.get("key")}', e)
        return None


def create_industry_map_images(ranked, region='TW'):
    if region != 'TW':
        return []

    outputs = []
    for topic in list_topics():
        output_name = _qpro_industry_image_name(topic)
        output_path = os.path.join(REPORT_DIR, output_name)
        rendered = create_industry_topic_dashboard(topic, ranked, output_path, region=region)
        if rendered and os.path.exists(rendered):
            outputs.append((str(topic.get('title') or topic.get('key') or 'Industry'), rendered))
    return outputs


def create_industry_intel_exports(region='TW'):
    if str(region).upper() != 'TW':
        return []

    outputs = []
    for topic in list_topics():
        topic_key = str(topic.get('key') or '').strip()
        if not topic_key:
            continue
        payload = build_industry_intel(topic_key, region=region.lower(), force=True)
        if isinstance(payload, dict):
            outputs.append({
                'topic': topic_key,
                'title': str(topic.get('title') or topic_key),
                'generated_at': payload.get('generated_at'),
                'companies': len(payload.get('companies') or []),
            })
    return outputs


def write_web_dashboard(region, market_mode, priority_ranked, full_ranked):
    os.makedirs(REPORT_DIR, exist_ok=True)
    payload = {
        'generated_at': now_str(),
        'region': region,
        'market_mode': market_mode,
        'priority': [],
        'strategy_buckets': {},
        'images': {
            'macro': f'{region}_macro_dashboard.png',
            'summary': f'{region}_scan_summary.png',
            'top10': f'{region}_top10_dashboard.png',
            'priority': f'{region}_priority_top10.png',
        },
    }

    for item in priority_ranked[:10]:
        tech_pack = item.get('tech_pack') or {}
        metrics = tech_pack.get('metrics') or {}
        profile = item.get('profile_info') or {}
        fin = item.get('fin_data') or {}
        payload['priority'].append({
            'ticker': item.get('ticker'),
            'company_name': ui_stock_name(item),
            'primary_strategy': item.get('primary_strategy_name'),
            'estimated_win_rate_pct': round(safe_float(item.get('estimated_win_rate_pct'), 0.0) or 0.0, 1),
            'priority_score': round(safe_float(item.get('priority_score'), 0.0) or 0.0, 1),
            'total_score': round(safe_float(item.get('total_score'), 0.0) or 0.0, 1),
            'technical_score': round(safe_float(tech_pack.get('technical_score'), 0.0) or 0.0, 1),
            'industry': profile.get('industry', 'N/A'),
            'close': safe_float((tech_pack.get('latest') if tech_pack.get('latest') is not None else {}).get('Close')),
            'yoy': safe_float(fin.get('single_month_yoy')),
            'vcp_score': safe_float(metrics.get('vcp_score')),
            'bb_score': safe_float(metrics.get('bb_score')),
            'pattern_score': safe_float(metrics.get('pattern_score')),
            'tags': _strategy_specific_tags(item, item.get('primary_strategy', 'minervini')),
        })

    for key, title in [('vcp', 'VCP'), ('pattern', 'Pattern'), ('bollinger', 'Bollinger'), ('minervini', 'Minervini'), ('sector', 'Sector')]:
        payload['strategy_buckets'][key] = [
            {
                'ticker': item.get('ticker'),
                'category_score': round(safe_float(item.get('category_score'), 0.0) or 0.0, 1),
                'total_score': round(safe_float(item.get('total_score'), 0.0) or 0.0, 1),
                'industry': (item.get('profile_info') or {}).get('industry', 'N/A'),
            }
            for item in _rank_strategy_bucket(full_ranked, key, limit=10)
        ]
        payload['images'][key] = f'{region}_{key}_top10.png'

    json_path = os.path.join(REPORT_DIR, f'{region}_web_dashboard.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    html_path = os.path.join(REPORT_DIR, f'{region}_web_dashboard.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(build_web_dashboard_html(payload))
    return html_path

def build_web_dashboard_html(payload):
    region = str(payload.get('region', 'TW'))
    cards = ''
    for key, label in [
        ('priority', '五策略優先推薦'),
        ('top10', '綜合 Top 10'),
        ('vcp', 'VCP Top 10'),
        ('pattern', '形態學 Top 10'),
        ('bollinger', '布林突破 Top 10'),
        ('minervini', 'Minervini Top 10'),
        ('sector', '族群連動 Top 10'),
    ]:
        image = (payload.get('images') or {}).get(key)
        if not image:
            continue
        cards += f"""
        <section>
          <h2>{html.escape(label)}</h2>
          <img src="{html.escape(image)}" alt="{html.escape(label)}">
        </section>
        """

    rows = ''
    for idx, item in enumerate(payload.get('priority', []), start=1):
        tags = ' / '.join(item.get('tags') or [])
        rows += f"""
        <tr>
          <td>{idx}</td><td><b>{html.escape(str(item.get('ticker', '')))}</b><br><small>{html.escape(str(item.get('company_name', '')))}</small></td>
          <td>{html.escape(str(item.get('primary_strategy', '')))}</td>
          <td>{item.get('estimated_win_rate_pct', 'N/A')}%</td>
          <td>{item.get('priority_score', 'N/A')}</td>
          <td>{item.get('total_score', 'N/A')}</td>
          <td>{html.escape(str(item.get('industry', 'N/A')))}</td>
          <td>{html.escape(tags)}</td>
        </tr>
        """

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{html.escape(region)} Quant Dashboard</title>
      <style>
        body {{ margin:0; font-family:"Noto Sans CJK TC","Microsoft JhengHei",Arial,sans-serif; background:#f4f6f8; color:#172033; }}
        header {{ padding:20px 28px; background:#111827; color:#f8fafc; display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; }}
        header h1 {{ margin:0; font-size:24px; }}
        nav a {{ color:#bfdbfe; margin-left:14px; text-decoration:none; font-weight:700; }}
        main {{ max-width:1220px; margin:0 auto; padding:22px; }}
        .meta {{ color:#64748b; margin-bottom:18px; }}
        section {{ background:white; border:1px solid #e5e7eb; border-radius:8px; padding:16px; margin-bottom:18px; box-shadow:0 8px 22px rgba(15,23,42,.06); }}
        h2 {{ margin:0 0 12px; font-size:18px; }}
        img {{ width:100%; height:auto; border-radius:6px; border:1px solid #e5e7eb; }}
        table {{ width:100%; border-collapse:collapse; background:white; }}
        th, td {{ padding:10px; border-bottom:1px solid #e5e7eb; text-align:left; font-size:14px; }}
        th {{ background:#f8fafc; color:#475569; }}
        .actions a {{ display:inline-block; background:#1d4ed8; color:white; padding:9px 12px; border-radius:6px; text-decoration:none; margin-right:8px; }}
      </style>
    </head>
    <body>
      <header>
        <div>
          <h1>{html.escape(region)} Quant System Web Dashboard</h1>
          <div>Mode {html.escape(str(payload.get('market_mode', 'N/A')))} | {html.escape(str(payload.get('generated_at', 'N/A')))}</div>
        </div>
        <nav><a href="/">首頁</a><a href="/refresh?region=TW">Refresh TW</a><a href="/refresh?region=US">Refresh US</a></nav>
      </header>
      <main>
        <div class="meta">優先推薦先由 VCP、形態學突破、布林突破、Minervini Trend、族群連動各取 Top 10，再用總分、規則估計勝率與策略加權分排序。估計勝率是策略信心分，不是歷史保證勝率。</div>
        <section>
          <h2>Priority Table</h2>
          <table>
            <tr><th>#</th><th>Ticker</th><th>主策略</th><th>估勝率</th><th>優先分</th><th>總分</th><th>Industry</th><th>Tags</th></tr>
            {rows}
          </table>
        </section>
        {cards}
      </main>
    </body>
    </html>
    """

def analyze_stock(ticker, market_mode='offensive', silent=False):
    try:
        ticker, df = download_stock_df(ticker)
        if df.empty or len(df) < 250: return (None, None, None) if silent else ('❌ 找不到資料', None, None)
        tech_pack = evaluate_technical(df, market_mode)
        
        is_us = is_us_ticker(ticker)
        yf_info = yf.Ticker(ticker).info if not TEST_MODE else {}
        ticker_num = ticker.split('.')[0]
        
        profile_info = get_company_profile(ticker_num, ticker_full=ticker, yf_info=yf_info)
        fin_data = merge_financial_snapshot(ticker, profile_info['raw_text'], yf_info=yf_info)
        
        if not is_us:
            chip_data = get_tw_chip_data(ticker)
            fin_data = merge_finmind_chip_into_snapshot(fin_data, chip_data)
        
        report, img_path, strategy_img_path = build_stock_report(ticker, tech_pack, fin_data, profile_info)
        return report, img_path, strategy_img_path
    
    except Exception as e: 
        log_exception(f'[ANALYZE-ERROR] {ticker}', e)
        return (None, None, None) if silent else (f'❌ 錯誤：{e}', None, None)

def process_single_scan(ticker, market_mode):
    try:
        tkr, df = download_stock_df(ticker)
        if df.empty or len(df) < 250:
            return None
        tech_pack = evaluate_technical(df, market_mode)
        return {'ticker': tkr, 'df': df, 'tech_pack': tech_pack}
    except Exception:
        return None
    return None

def scan_and_rank_market(chat_id=None, requested_by_user=False, market_mode='offensive', region='TW', full_scan=False):
    if region == 'TW':
        pool = get_tw_stock_pool(market_mode)
    else:
        pool = get_us_defensive_etf_pool() if market_mode == 'defensive' else get_us_stock_pool()

    pool = sorted(set(pool))
    if TEST_MODE:
        pool = pool[:15]

    if requested_by_user:
        safe_send_message(chat_id, f'📚 {region} 掃描池建立完成：`{len(pool)}` 檔，開始技術面初篩。', parse_mode='Markdown')

    precomputed_prescreen = []
    completed = 0
    max_workers = 10 if region == 'US' else 6
    scan_pool = list(pool)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_scan, ticker, market_mode) for ticker in scan_pool]
        for future in as_completed(futures):
            completed += 1
            if requested_by_user and (completed == 1 or completed % SCAN_PROGRESS_STEP == 0 or completed == len(scan_pool)):
                safe_send_message(chat_id, f'📡 {region} 技術面初篩中：`{completed}`/`{len(scan_pool)}` 檔；有效 `{len(precomputed_prescreen)}` 檔。', parse_mode='Markdown')
            try:
                res = future.result()
            except Exception:
                res = None
            if res:
                precomputed_prescreen.append(res)

    prescreen = precomputed_prescreen

    # Emergency retry: if concurrent YF requests all failed, retry a smaller liquid subset sequentially.
    if not prescreen and region == 'TW':
        retry_pool = qpro_static_tw_scan_pool()[:80]
        safe_send_message(chat_id, f'⚠️ TW 初篩有效名單為 0，啟動備援 K 線重試 `{len(retry_pool)}` 檔。', parse_mode=None)
        for idx, ticker in enumerate(retry_pool, start=1):
            try:
                if requested_by_user and (idx == 1 or idx % 20 == 0 or idx == len(retry_pool)):
                    safe_send_message(chat_id, f'🔁 TW 備援初篩：`{idx}`/`{len(retry_pool)}` 檔；有效 `{len(prescreen)}` 檔。', parse_mode='Markdown')
                tkr, df = download_stock_df(ticker)
                if df is None or df.empty or len(df) < 250:
                    continue
                tech_pack = evaluate_technical(df, market_mode)
                prescreen.append({'ticker': tkr, 'df': df, 'tech_pack': tech_pack})
            except Exception as e:
                log(f'[TW-RETRY-SCAN-WARN] {ticker}: {e}')
                continue

    prescreen.sort(key=lambda x: x['tech_pack']['technical_score'], reverse=True)
    deep_limit = SCAN_DEEP_LIMIT_US if region == 'US' else SCAN_DEEP_LIMIT_TW
    prescreen = prescreen[:max(FINAL_TOP_N, deep_limit)]

    if requested_by_user:
        safe_send_message(
            chat_id,
            f'✅ 第一階段資料有效名單完成，共 `{len(prescreen)}` 檔進入第二階段深度評分。',
            parse_mode='Markdown'
        )

    if not prescreen:
        log(f'[SCAN-WARN] {region} prescreen empty after primary and fallback scans. pool_size={len(pool)}')
        return []

    ranked = []
    for idx, item in enumerate(prescreen, start=1):
        ticker = item['ticker']
        try:
            is_us = is_us_ticker(ticker)
            try:
                yf_info = yf.Ticker(ticker).info
            except Exception:
                yf_info = {}
            ticker_num = ticker.split('.')[0]

            profile_info = get_company_profile(ticker_num, ticker_full=ticker, yf_info=yf_info)
            fin_data = merge_financial_snapshot(ticker, profile_info.get('raw_text'), yf_info=yf_info)

            if region == 'TW' and not is_us:
                if requested_by_user:
                    safe_send_message(chat_id, f'🐢 FinMind 籌碼精查 `{ticker}` ({idx}/{len(prescreen)})...')
                chip_data = get_tw_chip_data(ticker)
                fin_data = merge_finmind_chip_into_snapshot(fin_data, chip_data)
                time.sleep(1.2)

            t_score = item['tech_pack']['technical_score']
            f_score = calc_fundamental_score(fin_data, is_us)
            c_score = calc_chip_score(fin_data, is_us)
            ranked.append({
                'ticker': ticker,
                'tech_pack': item['tech_pack'],
                'fin_data': fin_data,
                'profile_info': profile_info,
                'total_score': final_total_score(t_score, f_score, c_score, is_us)
            })

        except Exception as e:
            log_exception(f'[RANK-ERROR] {ticker}', e)
            continue

    ranked.sort(key=lambda x: x['total_score'], reverse=True)
    ranked = annotate_sector_strength(ranked)
    LAST_SCAN_RANKED[region] = ranked
    return ranked[:FINAL_TOP_N]


# ==========================================
# QPRO_FIX_20260513_1735: company names + strategy labels
# ==========================================
_TW_STOCK_NAME_CACHE = {}
_TW_STOCK_NAME_CACHE_TS = 0


def qpro_clean_company_name(name, ticker=''):
    """Clean TW company name from My-TW-Coverage / ISIN / markdown links.

    Fixes UI issues like:
      - "-[啟達]" -> "啟達"
      - "[啟達]" -> "啟達"
      - numeric fallback "4106" shown as company name
    """
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None

    s = html.unescape(s)
    s = s.replace('`', '').replace('*', '').replace('#', '').strip()
    s = re.sub(r'^[\-–—•\s]+', '', s).strip()

    # Markdown link: [name](url) or [name]
    m = re.match(r'^\[([^\]]+)\]\([^\)]*\)$', s)
    if m:
        s = m.group(1).strip()
    m = re.match(r'^\[([^\]]+)\]$', s)
    if m:
        s = m.group(1).strip()

    # Remove residual brackets around names.
    s = s.strip('[]()（）【】「」『』')
    s = re.sub(r'\s+', ' ', s).strip()

    code = str(ticker).replace('.TWO', '').replace('.TW', '').strip()
    s = re.sub(rf'^{re.escape(code)}\s+', '', s).strip() if code else s

    bad_values = {'N/A', 'NA', '-', '--', 'None', 'null', ''}
    if s in bad_values:
        return None
    # Do not display pure code as company name.
    if re.fullmatch(r'\d{4}', s):
        return None
    if code and s == code:
        return None
    if len(s) > 32:
        s = s[:32]
    return s


def qpro_fetch_tw_stock_name_map(force=False):
    global _TW_STOCK_NAME_CACHE, _TW_STOCK_NAME_CACHE_TS
    now_ts = time.time()
    if _TW_STOCK_NAME_CACHE and not force and (now_ts - _TW_STOCK_NAME_CACHE_TS) < 6 * 3600:
        return _TW_STOCK_NAME_CACHE

    mapping = dict(_TW_STOCK_NAME_CACHE or {})
    for mode, suffix in [(2, '.TW'), (4, '.TWO')]:
        try:
            res = requests.get(f'https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}', timeout=20)
            res.encoding = res.apparent_encoding or 'big5'
            df = pd.read_html(StringIO(res.text))[0]
            df.columns = df.iloc[0]
            df = df.iloc[1:].copy()
            if '有價證券代號及名稱' not in df.columns:
                continue
            if 'CFICode' in df.columns:
                df = df[df['CFICode'].astype(str).eq('ESVUFR')]
            for raw in df['有價證券代號及名稱'].dropna().astype(str):
                m = re.match(r'^\s*(\d{4})\s+(.+?)\s*$', raw)
                if not m:
                    continue
                code, name = m.group(1), qpro_clean_company_name(m.group(2), code + suffix)
                if name:
                    mapping[code + suffix] = name
                    # fallback by code only, useful when ticker suffix changes TW <-> TWO.
                    mapping.setdefault(code, name)
        except Exception as e:
            log(f'[TW-NAME-MAP-WARN] ISIN mode {mode} failed: {e}')

    # Optional: learn names from My-TW-Coverage markdown files.
    try:
        if MY_TW_COVERAGE_PATH and os.path.isdir(MY_TW_COVERAGE_PATH):
            for root, _dirs, files in os.walk(MY_TW_COVERAGE_PATH):
                for file in files:
                    if not file.endswith('.md'):
                        continue
                    m = re.match(r'^(\d{4})[\s_\-]*(.*)\.md$', file)
                    if not m:
                        continue
                    code = m.group(1)
                    fname_name = qpro_clean_company_name(m.group(2), code)
                    if fname_name:
                        mapping.setdefault(code, fname_name)
    except Exception:
        pass

    _TW_STOCK_NAME_CACHE = mapping
    _TW_STOCK_NAME_CACHE_TS = now_ts
    return mapping


def ui_company_name_from_raw_text(raw_text, ticker=''):
    if not raw_text:
        return None
    ticker_num = str(ticker).replace('.TWO', '').replace('.TW', '').strip()
    for line in str(raw_text).splitlines()[:80]:
        s = line.strip()
        if not s or s.startswith('|'):
            continue
        cleaned_line = s.replace('*', '').replace('#', '').strip()
        for key in ['公司名稱', '股票名稱', '名稱', '公司']:
            if key in cleaned_line and ('：' in cleaned_line or ':' in cleaned_line):
                name = cleaned_line.split('：', 1)[-1].strip() if '：' in cleaned_line else cleaned_line.split(':', 1)[-1].strip()
                name = qpro_clean_company_name(name, ticker)
                if name:
                    return clip_text(name, 24)
        if ticker_num:
            m = re.match(rf'^[\-–—•\s]*\[?{re.escape(ticker_num)}\]?\s+(.+)$', cleaned_line)
            if m:
                name = qpro_clean_company_name(m.group(1), ticker)
                if name:
                    return clip_text(name, 24)
        # A very common markdown bullet/link: - [啟達]
        m = re.match(r'^[\-–—•\s]*\[([^\]]+)\]', cleaned_line)
        if m:
            name = qpro_clean_company_name(m.group(1), ticker)
            if name:
                return clip_text(name, 24)
    return None


def ui_stock_name(item):
    ticker = str(item.get('ticker', '')).strip()
    code = ticker.replace('.TWO', '').replace('.TW', '').strip()
    profile = item.get('profile_info') or {}

    # ISIN name map is the most stable for TW names.
    if ticker.endswith(('.TW', '.TWO')):
        mapping = qpro_fetch_tw_stock_name_map(force=False)
        name = qpro_clean_company_name(mapping.get(ticker) or mapping.get(code), ticker)
        if name:
            return clip_text(name, 24)

    for key in ['company_name', 'stock_name', 'stockName', 'shortName', 'longName', 'name', '公司名稱']:
        val = profile.get(key)
        name = qpro_clean_company_name(val, ticker)
        if name:
            return clip_text(name, 24)

    name = ui_company_name_from_raw_text(profile.get('raw_text'), ticker)
    if name:
        return clip_text(name, 24)

    return ticker


def ui_stock_cell(item):
    ticker = html.escape(str(item.get('ticker', '')).strip())
    name = html.escape(ui_stock_name(item))
    if name and name != ticker:
        return f'<div class="ticker">{ticker}</div><div class="cname">{name}</div>'
    return f'<div class="ticker">{ticker}</div>'


# QPRO_FIX_20260513_2225: TW scan pool recovery.
# Do not depend on the company-name map only; if ISIN / local DB name parsing fails,
# keep scanning with code-only tickers and a stable fallback universe.
def qpro_static_tw_scan_pool():
    # Liquid TW names used only as an emergency fallback when official / local pool fetch fails.
    base = [
        '1101','1102','1216','1301','1303','1326','1402','1476','1590','1605',
        '2002','2049','2105','2201','2207','2301','2303','2308','2317','2324',
        '2327','2330','2345','2352','2353','2354','2356','2357','2371','2376',
        '2379','2382','2395','2408','2409','2412','2449','2454','2474','2603',
        '2606','2610','2615','2633','2801','2880','2881','2882','2883','2884',
        '2885','2886','2887','2890','2891','2892','3008','3034','3037','3045',
        '3231','3443','3661','3711','4904','4938','5871','5876','5880','6505',
        '8046','8069','8299','8358','9910','9914','9921','9933','9945'
    ]
    return [c + '.TW' for c in base]


def qpro_fetch_tw_codes_from_local_coverage():
    out = []
    try:
        if MY_TW_COVERAGE_PATH and os.path.isdir(MY_TW_COVERAGE_PATH):
            for root, _dirs, files in os.walk(MY_TW_COVERAGE_PATH):
                for file in files:
                    m = re.match(r'^(\d{4})', str(file))
                    if m:
                        # Use .TW first; download_stock_df() automatically retries .TWO if .TW is empty.
                        out.append(m.group(1) + '.TW')
    except Exception as e:
        log(f'[TW-POOL-WARN] local coverage scan failed: {e}')
    return out


def qpro_fetch_tw_codes_from_isin_direct():
    out = []
    for mkt, suffix in [(2, '.TW'), (4, '.TWO')]:
        try:
            res = requests.get(f'https://isin.twse.com.tw/isin/C_public.jsp?strMode={mkt}', timeout=20)
            res.encoding = res.apparent_encoding or 'big5'
            tables = pd.read_html(StringIO(res.text))
            if not tables:
                continue
            df = tables[0]
            # The ISIN page occasionally returns unnamed columns depending on pandas/html parser.
            # Search every row's first text cell for a 4-digit stock code instead of relying only on CFICode.
            for _, row in df.iterrows():
                row_text = ' '.join(str(x) for x in row.values if str(x) != 'nan')
                code_match = re.search(r'\b(\d{4})\s+[^\s]+', row_text)
                if not code_match:
                    continue
                # Keep listed common stocks/OTC common stocks; skip obvious warrants/funds by CFICode when present.
                if 'ESVUFR' in row_text or ('股票' in row_text and not any(bad in row_text for bad in ['ETF', 'ETN', '受益證券', '認購', '認售'])):
                    out.append(code_match.group(1) + suffix)
        except Exception as e:
            log(f'[TW-POOL-WARN] ISIN direct mode {mkt} failed: {e}')
    return out


def get_tw_stock_pool(mode='offensive'):
    tickers = []
    if mode == 'defensive':
        tickers.extend(get_defensive_etf_pool('TW'))

    # 1) official ISIN/name map, when available
    try:
        mapping = qpro_fetch_tw_stock_name_map(force=True)
        for t in mapping.keys():
            if isinstance(t, str):
                tt = t.strip().upper()
                if re.match(r'^\d{4}(\.TW|\.TWO)?$', tt):
                    tickers.append(normalize_ticker(tt))
    except Exception as e:
        log(f'[TW-POOL-WARN] name map failed: {e}')

    # 2) direct ISIN parse independent of company-name map
    tickers.extend(qpro_fetch_tw_codes_from_isin_direct())

    # 3) local My-TW-Coverage markdown filenames
    tickers.extend(qpro_fetch_tw_codes_from_local_coverage())

    # 4) final fallback to keep the bot functional
    if len(set(tickers)) < 50:
        log(f'[TW-POOL-WARN] TW pool too small ({len(set(tickers))}); using static liquid fallback too.')
        tickers.extend(qpro_static_tw_scan_pool())

    result = sorted(set(normalize_ticker(t) for t in tickers if re.match(r'^\d{4}(\.TW|\.TWO)?$', str(t).upper())))
    log(f'[TW-POOL] resolved {len(result)} tickers for scan.')
    return result


def qpro_get_latest(tech_pack):
    latest = (tech_pack or {}).get('latest')
    return latest if latest is not None else {}


def _item_hit_strategy_keys(item):
    tech_pack = item.get('tech_pack') or {}
    metrics = tech_pack.get('metrics') or {}
    conditions = tech_pack.get('conditions') or {}
    sector = item.get('sector_info') or {}
    hits = []

    vcp_score = safe_float(metrics.get('vcp_score'), 0.0) or 0.0
    bb_score = safe_float(metrics.get('bb_score'), 0.0) or 0.0
    pattern_score = safe_float(metrics.get('pattern_score'), 0.0) or 0.0
    tech_score = safe_float(tech_pack.get('technical_score'), 0.0) or 0.0
    sector_strength = safe_float(sector.get('sector_strength_score'), 0.0) or 0.0

    if conditions.get('vcp_setup') or vcp_score >= 0.60:
        hits.append('vcp')
    if conditions.get('c_bb_squeeze_breakout') or conditions.get('bb_breakout') or bb_score >= 0.55:
        hits.append('bollinger')
    if (
        conditions.get('c_triangle_contraction') or conditions.get('c_inverse_head_shoulders') or
        conditions.get('c_box_breakout') or conditions.get('c_engulfing_5d') or pattern_score >= 0.55
    ):
        hits.append('pattern')
    if (
        (conditions.get('trend_stack') and (conditions.get('short_mid_ma_stack') or conditions.get('near_high'))) or
        (tech_score >= 88 and conditions.get('trend_stack'))
    ):
        hits.append('minervini')
    if item.get('sector_tags') or sector_strength >= 55:
        hits.append('sector')

    # include strategy bucket sources collected during priority selection.
    for k in item.get('source_strategies') or []:
        if k not in hits:
            hits.append(k)

    if not hits:
        scores = _all_strategy_scores(item)
        hits.append(max(scores, key=scores.get))
    return list(dict.fromkeys(hits))


def _strategy_short_name(category):
    return {
        'vcp': 'VCP',
        'bollinger': '布林突破',
        'pattern': '形態突破',
        'minervini': 'Minervini',
        'sector': '族群連動',
    }.get(category, str(category))


def _strategy_combo_label(item, max_items=3):
    keys = _item_hit_strategy_keys(item)
    # Prefer explicit non-Minervini structure first, then show Minervini if also true.
    order = ['vcp', 'pattern', 'bollinger', 'minervini', 'sector']
    keys = sorted(keys, key=lambda k: order.index(k) if k in order else 99)
    label = ' / '.join(_strategy_short_name(k) for k in keys[:max_items])
    return label or '綜合策略'


def _full_signal_tags(item, category=None, limit=7):
    tags = []
    tags.append(_strategy_combo_label(item, max_items=3))
    if category:
        tags.extend(_strategy_specific_tags(item, category))
    for k in _item_hit_strategy_keys(item):
        if k != category:
            tags.extend(_strategy_specific_tags(item, k)[:2])
    out = []
    for t in tags:
        t = str(t).strip()
        if t and t not in out:
            out.append(t)
    return out[:limit]


def select_priority_recommendations(ranked, limit=10):
    """Pick weighted Top 10 without labeling everything as Minervini.

    The old version used the raw maximum category score. Minervini scores are on a
    larger scale, so many rows were displayed as Minervini even when they entered
    from VCP / Bollinger / Pattern / Sector buckets. This version keeps bucket
    source information and displays a strategy combo label.
    """
    candidate_map = {}
    for category in ['vcp', 'pattern', 'bollinger', 'minervini', 'sector']:
        for rank_idx, item in enumerate(_rank_strategy_bucket(ranked, category, limit=limit), start=1):
            ticker = item.get('ticker')
            if not ticker:
                continue
            if ticker not in candidate_map:
                copied = dict(item)
                copied['source_strategies'] = []
                copied['source_bucket_ranks'] = {}
                candidate_map[ticker] = copied
            if category not in candidate_map[ticker]['source_strategies']:
                candidate_map[ticker]['source_strategies'].append(category)
            candidate_map[ticker]['source_bucket_ranks'][category] = rank_idx

    source_items = list(candidate_map.values()) or list(ranked)
    priority = []
    for item in source_items:
        copied = dict(item)
        strategy_scores = _all_strategy_scores(item)
        hits = _item_hit_strategy_keys(copied)
        primary_strategy = hits[0] if hits else max(strategy_scores, key=strategy_scores.get)
        best_strategy_score = max(strategy_scores.get(k, 0.0) for k in hits) if hits else max(strategy_scores.values())
        win_rate_pct = estimate_item_win_rate(item)
        total = safe_float(item.get('total_score'), 0.0) or 0.0
        strategy_average = sum(min(v, 140.0) for v in strategy_scores.values()) / max(1, len(strategy_scores))

        # Small bonus for multi-strategy confirmation.
        multi_bonus = min(len(hits), 4) * 2.2
        priority_score = total * 0.42 + win_rate_pct * 0.28 + min(best_strategy_score, 150.0) * 0.22 + strategy_average * 0.08 + multi_bonus

        copied['strategy_scores'] = strategy_scores
        copied['primary_strategy'] = primary_strategy
        copied['primary_strategy_name'] = _strategy_combo_label(copied, max_items=3)
        copied['strategy_hits'] = hits
        copied['estimated_win_rate_pct'] = win_rate_pct
        copied['priority_score'] = priority_score
        priority.append(copied)

    priority.sort(key=lambda x: x.get('priority_score', 0.0), reverse=True)
    return priority[:limit]


def create_scan_summary_image(ranked_list, output_path, region='TW'):
    if not ranked_list:
        return None

    title = f"{region} strong stock scan | {datetime.now().strftime('%Y-%m-%d')}"
    rows_html = ""
    for idx, item in enumerate(ranked_list[:10], start=1):
        tech_pack = item.get('tech_pack') or {}
        fin_data = item.get('fin_data') or {}
        latest = qpro_get_latest(tech_pack)
        pattern_tag = _strategy_combo_label(item, max_items=2)
        total_score = safe_float(item.get('total_score'), 0.0) or 0.0
        metrics = tech_pack.get('metrics') or {}

        tag_cls = 'tag-blue'
        if '布林' in pattern_tag:
            tag_cls = 'tag-red'
        elif 'VCP' in pattern_tag:
            tag_cls = 'tag-green'
        elif '形態' in pattern_tag:
            tag_cls = 'tag-purple'

        rows_html += f"""
        <tr>
            <td class="rank">{idx}</td>
            <td class="stock">{ui_stock_cell(item)}</td>
            <td><div class="price">收 {safe_num_str(safe_float(latest.get('Close')), 2)}</div><div class="sub">RSI {safe_num_str(metrics.get('rsi'), 1)}</div></td>
            <td><span class="tag {tag_cls}">{html.escape(str(pattern_tag))}</span></td>
            <td>{ui_fund_summary(fin_data)}</td>
            <td>{ui_chip_summary(fin_data)}</td>
            <td class="score">{total_score:.1f}</td>
        </tr>
        """

    html_content = f"""
    <!doctype html><html><head><meta charset="utf-8"><style>
        body {{ margin:0; width:1120px; background:#ffffff; color:#111827; font-family:"Noto Sans CJK TC","Microsoft JhengHei","Segoe UI",Arial,sans-serif; }}
        .wrap {{ padding:26px; background:#ffffff; }}
        .panel {{ border:1px solid #e5e7eb; border-radius:18px; overflow:hidden; background:#ffffff; box-shadow:0 18px 50px rgba(15,23,42,.10); }}
        .head {{ padding:24px 28px; background:linear-gradient(135deg,#f8fafc,#eef2ff); border-bottom:1px solid #e5e7eb; }}
        h1 {{ margin:0; font-size:30px; letter-spacing:.2px; color:#111827; }}
        .meta {{ margin-top:8px; color:#64748b; font-size:14px; }}
        table {{ width:100%; border-collapse:collapse; table-layout:fixed; }}
        th {{ padding:13px 12px; color:#475569; background:#f1f5f9; font-size:14px; text-align:center; font-weight:900; }}
        td {{ padding:15px 12px; border-top:1px solid #e5e7eb; text-align:center; font-size:14px; vertical-align:middle; }}
        tr:nth-child(even) td {{ background:#fcfcfd; }}
        .rank {{ width:44px; color:#94a3b8; font-weight:900; }} .stock {{ text-align:left; width:170px; }}
        .ticker {{ font-size:19px; font-weight:950; color:#111827; }} .cname {{ margin-top:3px; font-size:12px; color:#64748b; font-weight:800; }}
        .price {{ font-weight:900; color:#111827; }} .sub {{ margin-top:4px; color:#64748b; font-size:12px; }}
        .tag {{ display:inline-block; min-width:118px; padding:8px 11px; border-radius:999px; font-weight:900; }}
        .tag-red {{ background:#fee2e2; color:#991b1b; }} .tag-green {{ background:#dcfce7; color:#166534; }} .tag-purple {{ background:#ede9fe; color:#5b21b6; }} .tag-blue {{ background:#e0f2fe; color:#075985; }}
        .score {{ color:#b45309; font-weight:950; font-size:23px; }}
        .metric {{ border-radius:14px; padding:8px 9px; line-height:1.25; min-height:42px; display:flex; flex-direction:column; justify-content:center; }}
        .metric-label {{ font-size:11px; opacity:.72; font-weight:900; }} .metric-val {{ font-size:12px; font-weight:950; }}
        .tone-good {{ background:#fee2e2 !important; color:#991b1b !important; }} .tone-bad {{ background:#dcfce7 !important; color:#166534 !important; }} .tone-neutral {{ background:#f1f5f9 !important; color:#475569 !important; }}
    </style></head><body><div class="wrap"><div class="panel" id="capture-area">
        <div class="head"><h1>{html.escape(title)}</h1><div class="meta">Top 10 | strategy combo shown | red = profit/net buy, green = loss/net sell</div></div>
        <table><tr><th>#</th><th>股票 / 公司</th><th>盤面</th><th>策略型態</th><th>基本面</th><th>籌碼面</th><th>模型</th></tr>{rows_html}</table>
    </div></div></body></html>
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1120, "height": 920}, device_scale_factor=2)
            page.set_content(html_content, wait_until="networkidle")
            page.locator("#capture-area").screenshot(path=output_path)
            browser.close()
            return output_path
    except Exception as e:
        log_exception('[SCAN-SUMMARY-IMAGE-ERROR]', e)
        return None


def create_strategy_bucket_dashboard(ranked, output_path, region='US', category='vcp', title='Strategy Top 10'):
    if not ranked:
        return None
    rows_html = ''
    for idx, item in enumerate(ranked[:10], start=1):
        tech_pack = item.get('tech_pack') or {}
        metrics = tech_pack.get('metrics') or {}
        profile = item.get('profile_info') or {}
        fin = item.get('fin_data') or {}
        latest = qpro_get_latest(tech_pack)
        tag_text = ' / '.join(_full_signal_tags(item, category)) or '-'
        yoy_cls = ui_tone_pct(fin.get('single_month_yoy'))
        rows_html += f"""
        <tr>
            <td class="rank">{idx}</td><td class="stock">{ui_stock_cell(item)}</td>
            <td class="industry">{html.escape(str(profile.get('industry', 'N/A'))[:34])}</td>
            <td class="num">{safe_num_str(safe_float(latest.get('Close')), 2)}</td>
            <td class="score-main">{safe_num_str(item.get('category_score'), 1)}</td><td class="num">{safe_num_str(item.get('total_score'), 1)}</td>
            <td>{ui_fund_summary(fin)}</td><td>{ui_chip_summary(fin)}</td>
            <td class="{yoy_cls} compact">{safe_pct_str(fin.get('single_month_yoy'))}</td>
            <td class="mini">{safe_num_str(metrics.get('vcp_score'), 2)}</td><td class="mini">{safe_num_str(metrics.get('bb_score'), 2)}</td><td class="mini">{safe_num_str(metrics.get('pattern_score'), 2)}</td>
            <td class="tags">{html.escape(tag_text)}</td>
        </tr>
        """
    html_content = f"""
    <!doctype html><html><head><meta charset="utf-8"><style>
        body {{ margin:0; width:1280px; background:#ffffff; color:#111827; font-family:"Noto Sans CJK TC","Microsoft JhengHei","Segoe UI",Arial,sans-serif; }}
        .wrap {{ padding:26px; background:#ffffff; }} .panel {{ border:1px solid #e5e7eb; border-radius:18px; overflow:hidden; background:#ffffff; box-shadow:0 18px 50px rgba(15,23,42,.10); }}
        .head {{ display:flex; justify-content:space-between; align-items:flex-end; padding:24px 28px; background:linear-gradient(135deg,#f8fafc,#eef2ff); border-bottom:1px solid #e5e7eb; }}
        h1 {{ margin:0; font-size:30px; letter-spacing:.2px; color:#111827; }} .meta {{ color:#64748b; font-size:13px; margin-top:7px; }}
        table {{ width:100%; border-collapse:collapse; table-layout:fixed; font-size:13px; }} th {{ padding:12px 8px; background:#f1f5f9; color:#475569; text-align:center; font-weight:900; }}
        td {{ padding:13px 8px; border-top:1px solid #e5e7eb; text-align:center; vertical-align:middle; }} tr:nth-child(even) td {{ background:#fcfcfd; }}
        .rank {{ width:38px; color:#94a3b8; font-weight:900; }} .stock {{ text-align:left; width:140px; }} .ticker {{ font-size:18px; font-weight:950; color:#111827; }} .cname {{ margin-top:3px; font-size:12px; color:#64748b; font-weight:800; }}
        .industry {{ color:#334155; }} .num {{ font-variant-numeric:tabular-nums; font-weight:800; }} .score-main {{ color:#b45309; font-weight:950; font-size:20px; }} .mini {{ font-size:12px; color:#475569; }} .tags {{ color:#334155; text-align:left; line-height:1.45; font-size:12px; }}
        .compact {{ border-radius:12px; font-weight:950; font-variant-numeric:tabular-nums; }} .metric {{ border-radius:14px; padding:8px 9px; line-height:1.25; min-height:42px; display:flex; flex-direction:column; justify-content:center; }}
        .metric-label {{ font-size:11px; opacity:.72; font-weight:900; }} .metric-val {{ font-size:12px; font-weight:950; }}
        .tone-good {{ background:#fee2e2 !important; color:#991b1b !important; }} .tone-bad {{ background:#dcfce7 !important; color:#166534 !important; }} .tone-neutral {{ background:#f1f5f9 !important; color:#475569 !important; }}
    </style></head><body><div class="wrap"><div class="panel" id="capture-area">
        <div class="head"><div><h1>{html.escape(title)}</h1><div class="meta">{html.escape(region)} | {html.escape(category.upper())} | {html.escape(_strategy_description(category))} | generated {now_str()}</div></div><div class="meta">Top 10 by category score</div></div>
        <table><tr><th>#</th><th>股票 / 公司</th><th>Industry</th><th>Close</th><th>Cat</th><th>Total</th><th>基本面</th><th>籌碼面</th><th>YoY</th><th>VCP</th><th>BB</th><th>Pattern</th><th>Tags</th></tr>{rows_html}</table>
    </div></div></body></html>
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=2)
            page.set_content(html_content, wait_until="networkidle")
            page.locator("#capture-area").screenshot(path=output_path)
            browser.close()
            return output_path
    except Exception as e:
        log_exception(f'[STRATEGY-DASHBOARD-ERROR] {category}', e)
        return None


def create_priority_recommendation_dashboard(ranked, output_path, region='TW', market_mode='offensive'):
    if not ranked:
        return None
    rows_html = ''
    for idx, item in enumerate(ranked[:10], start=1):
        tech_pack = item.get('tech_pack') or {}
        profile = item.get('profile_info') or {}
        fin = item.get('fin_data') or {}
        latest = qpro_get_latest(tech_pack)
        primary = (item.get('strategy_hits') or [item.get('primary_strategy', 'minervini')])[0]
        tag_text = ' / '.join(_full_signal_tags(item, primary)) or '-'
        rows_html += f"""
        <tr>
            <td class="rank">{idx}</td><td class="stock">{ui_stock_cell(item)}</td>
            <td class="strategy">{html.escape(str(item.get('primary_strategy_name') or _strategy_combo_label(item)))}</td>
            <td class="win">{safe_num_str(item.get('estimated_win_rate_pct'), 1)}%</td><td class="priority">{safe_num_str(item.get('priority_score'), 1)}</td><td class="num">{safe_num_str(item.get('total_score'), 1)}</td>
            <td class="num">{safe_num_str(tech_pack.get('technical_score'), 1)}</td><td class="num">{safe_num_str(safe_float(latest.get('Close')), 2)}</td>
            <td>{ui_fund_summary(fin)}</td><td>{ui_chip_summary(fin)}</td><td class="industry">{html.escape(str(profile.get('industry', 'N/A'))[:30])}</td><td class="tags">{html.escape(tag_text)}</td>
        </tr>
        """
    html_content = f"""
    <!doctype html><html><head><meta charset="utf-8"><style>
        body {{ margin:0; width:1280px; background:#ffffff; color:#111827; font-family:"Noto Sans CJK TC","Microsoft JhengHei","Segoe UI",Arial,sans-serif; }} .wrap {{ padding:26px; background:#ffffff; }}
        .panel {{ border:1px solid #e5e7eb; border-radius:18px; overflow:hidden; background:#ffffff; box-shadow:0 18px 50px rgba(15,23,42,.12); }} .head {{ display:flex; justify-content:space-between; align-items:flex-end; padding:24px 28px; background:linear-gradient(135deg,#fff7ed,#fee2e2); border-bottom:1px solid #fecaca; }}
        h1 {{ margin:0; font-size:30px; color:#7f1d1d; letter-spacing:.2px; }} .meta {{ color:#7f1d1d; opacity:.78; font-size:13px; margin-top:7px; }} table {{ width:100%; border-collapse:collapse; table-layout:fixed; font-size:13px; }}
        th {{ padding:12px 8px; background:#f8fafc; color:#475569; text-align:center; font-weight:900; }} td {{ padding:13px 8px; border-top:1px solid #e5e7eb; text-align:center; vertical-align:middle; }} tr:nth-child(even) td {{ background:#fcfcfd; }}
        .rank {{ width:38px; color:#94a3b8; font-weight:900; }} .stock {{ text-align:left; width:145px; }} .ticker {{ font-size:18px; font-weight:950; color:#111827; }} .cname {{ margin-top:3px; font-size:12px; color:#64748b; font-weight:800; }} .strategy {{ font-weight:900; color:#334155; }} .win {{ color:#2563eb; font-weight:950; }} .priority {{ color:#b45309; font-size:20px; font-weight:950; }} .num {{ font-variant-numeric:tabular-nums; font-weight:800; }} .industry {{ color:#334155; }} .tags {{ color:#334155; text-align:left; line-height:1.45; font-size:12px; }}
        .metric {{ border-radius:14px; padding:8px 9px; line-height:1.25; min-height:42px; display:flex; flex-direction:column; justify-content:center; }} .metric-label {{ font-size:11px; opacity:.72; font-weight:900; }} .metric-val {{ font-size:12px; font-weight:950; }} .tone-good {{ background:#fee2e2 !important; color:#991b1b !important; }} .tone-bad {{ background:#dcfce7 !important; color:#166534 !important; }} .tone-neutral {{ background:#f1f5f9 !important; color:#475569 !important; }}
    </style></head><body><div class="wrap"><div class="panel" id="capture-area"><div class="head"><div><h1>{html.escape(region)} 五策略綜合優先推薦 Top 10</h1><div class="meta">VCP / 形態學突破 / 布林突破 / Minervini Trend / 族群連動 | Mode {html.escape(market_mode)} | {now_str()}</div></div><div class="meta">排序 = 分數 + 估勝率 + 多策略確認</div></div>
        <table><tr><th>#</th><th>股票 / 公司</th><th>主策略</th><th>估勝率</th><th>優先分</th><th>總分</th><th>技術</th><th>Close</th><th>基本面</th><th>籌碼面</th><th>Industry</th><th>Tags</th></tr>{rows_html}</table>
    </div></div></body></html>
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=2)
            page.set_content(html_content, wait_until="networkidle")
            page.locator("#capture-area").screenshot(path=output_path)
            browser.close()
            return output_path
    except Exception as e:
        log_exception('[PRIORITY-DASHBOARD-ERROR]', e)
        return None


def render_top_ranked_dashboard(ranked, output_path, region='TW', market_mode='neutral'):
    """Local override for quant.dashboard.render_top_ranked_dashboard.

    Adds TW company names and strategy combo tags so the top dashboard is
    consistent with the custom Telegram charts.
    """
    if not ranked:
        return None
    rows_html = ''
    for idx, item in enumerate(ranked[:10], start=1):
        tech_pack = item.get('tech_pack') or {}
        fin = item.get('fin_data') or {}
        profile = item.get('profile_info') or {}
        latest = qpro_get_latest(tech_pack)
        metrics = tech_pack.get('metrics') or {}
        tags = ' / '.join(_full_signal_tags(item, item.get('primary_strategy')))
        rows_html += f"""
        <tr>
          <td class="rank">{idx}</td><td class="stock">{ui_stock_cell(item)}</td><td>{html.escape(str(profile.get('industry','N/A'))[:28])}</td>
          <td class="score">{safe_num_str(item.get('total_score'),1)}</td><td>{safe_num_str(tech_pack.get('technical_score'),1)}</td><td class="{ui_tone_pct(fin.get('single_month_yoy'))}">{safe_pct_str(fin.get('single_month_yoy'))}</td>
          <td>{ui_chip_summary(fin)}</td><td>{safe_num_str((item.get('sector_info') or {}).get('sector_strength_score'),1)}</td><td class="tags">{html.escape(tags)}</td>
        </tr>
        """
    html_content = f"""
    <!doctype html><html><head><meta charset="utf-8"><style>
      body {{ margin:0; width:1180px; background:#ffffff; color:#111827; font-family:"Noto Sans CJK TC","Microsoft JhengHei","Segoe UI",Arial,sans-serif; }} .wrap{{padding:24px;background:#fff}} .panel{{border:1px solid #e5e7eb;border-radius:18px;overflow:hidden;background:#fff;box-shadow:0 18px 50px rgba(15,23,42,.10)}} .head{{display:flex;justify-content:space-between;align-items:flex-end;padding:22px 26px;background:#f8fafc;border-bottom:1px solid #e5e7eb}} h1{{margin:0;font-size:28px}} .meta{{color:#64748b;font-size:13px}} table{{width:100%;border-collapse:collapse;table-layout:fixed;font-size:13px}} th{{padding:12px 8px;background:#f1f5f9;color:#475569;text-align:center;font-weight:900}} td{{padding:12px 8px;border-top:1px solid #e5e7eb;text-align:center;vertical-align:middle}} tr:nth-child(even) td{{background:#fcfcfd}} .rank{{width:36px;color:#94a3b8;font-weight:900}} .stock{{text-align:left;width:150px}} .ticker{{font-size:17px;font-weight:950}} .cname{{margin-top:3px;font-size:12px;color:#64748b;font-weight:800}} .score{{font-weight:950;color:#991b1b}} .tags{{text-align:left;line-height:1.45}} .metric{{border-radius:12px;padding:6px 7px;line-height:1.22;min-height:36px;display:flex;flex-direction:column;justify-content:center}} .metric-label{{font-size:10px;font-weight:900;opacity:.75}} .metric-val{{font-size:11px;font-weight:950}} .tone-good{{background:#fee2e2!important;color:#991b1b!important}} .tone-bad{{background:#dcfce7!important;color:#166534!important}} .tone-neutral{{background:#f1f5f9!important;color:#475569!important}}
    </style></head><body><div class="wrap"><div class="panel" id="capture-area"><div class="head"><div><h1>{html.escape(region)} Top 10 Quant Dashboard</h1><div class="meta">公司名稱 + 策略組合版 | red = profit/net buy, green = loss/net sell</div></div><div class="meta">Mode: {html.escape(str(market_mode))} | CTA / VCP / BB / Minervini / Sector</div></div><table><tr><th>#</th><th>股票 / 公司</th><th>Industry</th><th>Total</th><th>Tech</th><th>YoY</th><th>籌碼面</th><th>Sector</th><th>Strategies / Tags</th></tr>{rows_html}</table></div></div></body></html>
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1180, "height": 760}, device_scale_factor=2)
            page.set_content(html_content, wait_until="networkidle")
            page.locator("#capture-area").screenshot(path=output_path)
            browser.close()
            return output_path
    except Exception as e:
        log_exception('[TOP-RANKED-DASHBOARD-ERROR]', e)
        return None


def qpro_export_latest_ranked(region, ranked):
    region = 'US' if str(region).upper() == 'US' else 'TW'
    os.makedirs(DATA_DIR, exist_ok=True)
    updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rows = []

    for idx, item in enumerate(list(ranked or [])[:50], start=1):
        if not isinstance(item, dict):
            continue

        ticker = str(item.get('ticker') or '').strip()
        if not ticker:
            continue

        tech_pack = item.get('tech_pack') or {}
        fin_data = item.get('fin_data') or {}
        profile = item.get('profile_info') or {}
        latest = qpro_get_latest(tech_pack) or {}
        is_us = is_us_ticker(ticker)

        tech_score = safe_float(item.get('tech_score'))
        if tech_score is None:
            tech_score = safe_float(tech_pack.get('technical_score'))

        fund_score = safe_float(item.get('fund_score'))
        if fund_score is None:
            try:
                fund_score = calc_fundamental_score(fin_data, is_us)
            except Exception:
                fund_score = None

        chip_score = safe_float(item.get('chip_score'))
        if chip_score is None:
            try:
                chip_score = calc_chip_score(fin_data, is_us)
            except Exception:
                chip_score = None

        rows.append({
            'rank': idx,
            'ticker': ticker,
            'company': ui_stock_name(item),
            'industry': profile.get('industry') or '',
            'close': safe_float(latest.get('Close')),
            'total_score': safe_float(item.get('total_score')),
            'tech_score': tech_score,
            'fund_score': fund_score,
            'chip_score': chip_score,
            'strategy': item.get('primary_strategy_name') or item.get('primary_strategy') or '',
            'priority_score': safe_float(item.get('priority_score')),
            'estimated_win_rate_pct': safe_float(item.get('estimated_win_rate_pct')),
            'is_priority': item.get('priority_score') is not None,
            'tags': _full_signal_tags(item, item.get('primary_strategy')),
            'updated_at': updated_at,
        })

    path = os.path.join(DATA_DIR, f'latest_ranked_{region.lower()}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'region': region,
            'updated_at': updated_at,
            'rows': rows,
        }, f, ensure_ascii=False, indent=2)
    return path


def qpro_export_latest_macro(region, macro_data=None, market_mode=None):
    region = 'US' if str(region).upper() == 'US' else 'TW'
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = macro_data if isinstance(macro_data, dict) else LAST_MACRO_DATA.get(region)
    if not payload:
        payload = get_macro_dashboard_data(region)
        LAST_MACRO_DATA[region] = payload

    exported = dict(payload)
    exported['region'] = region
    exported['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if market_mode:
        exported['latest_mode'] = market_mode
    exported['report_image'] = f'{region}_macro_dashboard.png'

    path = os.path.join(DATA_DIR, f'latest_macro_{region.lower()}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)
    return path

# ==========================================
# Main jobs and scheduler
# ==========================================
def run_market_scan_job(chat_id, requested_by_user=False, region='TW', full_scan=False):
    market_mode, macro_score = check_market_status(region)
    if market_mode == 'offensive':
        mode_msg = "🟢 **波段多方輪動：啟動 [攻擊型飆股引擎]**"
    elif market_mode == 'defensive':
        mode_msg = "🔴 **波段風險升高：啟動 [RS防守避險引擎 + ETF推薦]**"
    else:
        mode_msg = "🟡 **市場震盪中性：啟動 [精選強勢股 + 降低倉位]**"
    
    safe_send_message(chat_id, f'🔍 **{region} 市場波段雷達啟動中...**\n{mode_msg}', parse_mode='Markdown')
    macro_img_path = os.path.join(REPORT_DIR, f'{region}_macro_dashboard.png')
    dashboard_generated = create_macro_dashboard_image(market_mode, macro_score, macro_img_path, region)
    try:
        qpro_export_latest_macro(region, market_mode=market_mode)
    except Exception as e:
        log_exception('[LATEST-MACRO-EXPORT-ERROR]', e)
    if dashboard_generated and os.path.exists(dashboard_generated):
        safe_send_photo(chat_id, dashboard_generated)
        time.sleep(2)
    
    if region == 'TW': update_my_tw_coverage(chat_id)
    top_ranked = scan_and_rank_market(chat_id, requested_by_user, market_mode, region, full_scan=full_scan)
    
    if not top_ranked:
        safe_send_message(chat_id, '☕ **掃描完畢**\n本次資料源沒有可用 K 線標的，請稍後重試。')
        return

    full_ranked = LAST_SCAN_RANKED.get(region, top_ranked)
    priority_ranked = select_priority_recommendations(full_ranked, limit=FINAL_TOP_N)
    if priority_ranked:
        top_ranked = priority_ranked

    try:
        export_ranked = qpro_unique_items_by_code(list(top_ranked) + list(full_ranked), limit=50)
        qpro_export_latest_ranked(region, export_ranked)
        qpro_export_latest_macro(region, market_mode=market_mode)
    except Exception as e:
        log_exception('[LATEST-DATA-EXPORT-ERROR]', e)

    summary_img_path = os.path.join(REPORT_DIR, f'{region}_scan_summary.png')
    summary_img = create_scan_summary_image(top_ranked, summary_img_path, region)
    if summary_img and os.path.exists(summary_img):
        safe_send_photo(chat_id, summary_img, caption=f'🏆 {region} Top 10 強勢型態統整')
        time.sleep(2)

    dashboard_path = os.path.join(REPORT_DIR, f'{region}_top10_dashboard.png')
    try:
        dashboard = render_top_ranked_dashboard(top_ranked, dashboard_path, region=region, market_mode=market_mode)
        if dashboard and os.path.exists(dashboard):
            safe_send_photo(chat_id, dashboard)
            time.sleep(2)
    except Exception as e:
        log_exception('[DASHBOARD-ERROR]', e)

    try:
        priority_img_path = os.path.join(REPORT_DIR, f'{region}_priority_top10.png')
        priority_img = create_priority_recommendation_dashboard(top_ranked, priority_img_path, region=region, market_mode=market_mode)
        if priority_img and os.path.exists(priority_img):
            safe_send_photo(chat_id, priority_img, caption=f'⭐ {region} 五策略綜合優先推薦 Top 10')
            time.sleep(2)
    except Exception as e:
        log_exception('[PRIORITY-DASHBOARD-ERROR]', e)

    try:
        for title, img_path in create_strategy_bucket_images(full_ranked, region=region, market_mode=market_mode):
            safe_send_photo(chat_id, img_path, caption=f'{region} {title}')
            time.sleep(1.5)
    except Exception as e:
        log_exception('[STRATEGY-DASHBOARD-ERROR]', e)

    try:
        create_industry_map_images(full_ranked, region=region)
    except Exception as e:
        log_exception('[INDUSTRY-DASHBOARD-BATCH-ERROR]', e)

    try:
        create_industry_intel_exports(region=region)
    except Exception as e:
        log_exception('[INDUSTRY-INTEL-BATCH-ERROR]', e)

    try:
        write_web_dashboard(region, market_mode, top_ranked, full_ranked)
    except Exception as e:
        log_exception('[WEB-DASHBOARD-WRITE-ERROR]', e)
        
    summary = [f'🏆 **{region} 五策略優先推薦 Top 10**']
    for i, item in enumerate(top_ranked, start=1):
        icon = '🛡️' if ('00' in item["ticker"] or item["ticker"] in get_defensive_etf_pool('US')) else '🚀'
        win = safe_float(item.get('estimated_win_rate_pct'))
        priority = safe_float(item.get('priority_score'))
        strategy = item.get('primary_strategy_name', '綜合')
        summary.append(f'{i}. {icon} `{item["ticker"]}` | `{strategy}` | 勝率 `{safe_num_str(win, 1)}%` | 優先分 `{safe_num_str(priority, 1)}` | 總分 `{item["total_score"]:.1f}`')
    safe_send_message(chat_id, '\n'.join(summary), parse_mode='Markdown')
    time.sleep(2)
    
    for i, item in enumerate(top_ranked, start=1):
        try:
            report_tech_pack = dict(item['tech_pack'])
            report_tech_pack['sector_tags'] = item.get('sector_tags', [])
            report_tech_pack['sector_info'] = item.get('sector_info', {})
            report_tech_pack['primary_strategy'] = item.get('primary_strategy')
            report, img_path, strategy_img_path = build_stock_report(item['ticker'], report_tech_pack, item['fin_data'], item['profile_info'], rank=i)
            if img_path and os.path.exists(img_path): safe_send_photo(chat_id, img_path); time.sleep(1)
            if strategy_img_path and os.path.exists(strategy_img_path): safe_send_photo(chat_id, strategy_img_path); time.sleep(1)
            safe_send_message(chat_id, report, parse_mode='Markdown')
            time.sleep(3.5)
        except Exception as e: log_exception(f"發送 {item['ticker']} 失敗", e)
            
    safe_send_message(chat_id, '✅ **掃描完畢**', parse_mode='Markdown')

# ==========================================
# QPRO_FIX_20260513_2255_SCAN_LOCK
# QPRO_FIX_20260513_2325_SCAN_FAST_FULL
# QPRO_FIX_20260513_2318_RETAIL_RATIO_GUARD: MTX retail proxy denominator/range validation; purge bogus history values.
# 防止 /scan、排程、或重複 bot instance 在同一個 Python process 內重複啟動掃描。
# 注意：如果 Docker 同時跑了兩個 container，仍需用 docker ps 檢查並關掉舊 container。
# ==========================================
SCAN_LOCK = threading.Lock()
SCAN_RUNNING_BY_REGION = {'TW': False, 'US': False}

def start_scan_thread(chat_id, requested_by_user=False, region='TW', full_scan=False):
    region = 'US' if str(region).upper() == 'US' else 'TW'

    with SCAN_LOCK:
        if SCAN_RUNNING_BY_REGION.get(region, False):
            safe_send_message(
                chat_id,
                f'⏳ {region} 掃描已在執行中，這次請求已略過，避免重複發送相同報告。',
                parse_mode=None
            )
            return False
        SCAN_RUNNING_BY_REGION[region] = True

    def _runner():
        try:
            run_market_scan_job(chat_id, requested_by_user=requested_by_user, region=region, full_scan=full_scan)
        except Exception as e:
            log_exception(f'[SCAN-THREAD-ERROR] {region}', e)
            safe_send_message(chat_id, f'⚠️ {region} 掃描失敗：{str(e)[:160]}', parse_mode=None)
        finally:
            with SCAN_LOCK:
                SCAN_RUNNING_BY_REGION[region] = False

    threading.Thread(target=_runner, daemon=True).start()
    return True

def run_weekly_optimization():
    log("🧬 啟動週末回歸測試與策略進化...")
    try: subprocess.Popen(['python3', 'evolve_nsga2.py'])
    except Exception as e: log(f"啟動最佳化失敗: {e}")



# ==========================================
# QPRO_FIX_20260513_2238_TW_DATASOURCE_PRIORITY
# 台股資料優先順序修正：TWSE/TPEx/TAIFEX + FinMind + Goodinfo 優先，yfinance 僅最後 fallback
# ==========================================
TW_PRICE_SOURCE_ORDER = [
    'twse_official',
    'tpex_official',
    'finmind',
    'goodinfo_cached',
    'yfinance_fallback',
]
TW_FUND_SOURCE_ORDER = [
    'finmind',
    'goodinfo',
    'my_tw_coverage',
    'yfinance_us_only',
]
TW_CHIP_SOURCE_ORDER = [
    'twse_tpex_official',
    'taifex_official_macro',
    'finmind',
    'goodinfo',
]


def qpro_is_tw_ticker(ticker):
    t = str(ticker).strip().upper()
    return bool(re.match(r'^\d{4}(\.TW|\.TWO)?$', t))


def qpro_tw_stock_id(ticker):
    return qpro_tw_code(ticker)


def qpro_normalize_price_df(df, source='unknown'):
    """Normalize price dataframe to Open/High/Low/Close/Volume with DatetimeIndex."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    x = df.copy()

    # Flatten columns first.
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [' '.join(str(i).strip() for i in c if str(i).strip() not in ('', 'nan', 'None')) for c in x.columns]
    x.columns = [str(c).strip() for c in x.columns]

    # Locate date.
    date_col = None
    for c in x.columns:
        if str(c).lower() in ('date', 'datetime', 'trade_date', 'trading_date') or str(c) in ('日期', '交易日期'):
            date_col = c
            break
    if date_col is not None:
        x['date'] = pd.to_datetime(x[date_col], errors='coerce')
        x = x.dropna(subset=['date']).set_index('date')
    elif not isinstance(x.index, pd.DatetimeIndex):
        try:
            x.index = pd.to_datetime(x.index, errors='coerce')
            x = x[~pd.isna(x.index)]
        except Exception:
            return pd.DataFrame()

    aliases = {
        'Open': ['Open', 'open', '開盤價', '開盤', '開盤價(元)'],
        'High': ['High', 'max', '最高價', '最高', '最高價(元)'],
        'Low': ['Low', 'min', '最低價', '最低', '最低價(元)'],
        'Close': ['Close', 'close', '收盤價', '收盤', '收盤價(元)'],
        'Volume': ['Volume', 'Trading_Volume', 'trading_volume', '成交股數', '成交量', '成交股數(股)', '成交張數'],
    }
    out = pd.DataFrame(index=x.index)
    for std, cand in aliases.items():
        col = _pick_first_column(x, cand)
        if col is not None:
            out[std] = pd.to_numeric(
                x[col].astype(str).str.replace(',', '', regex=False).str.replace('--', '', regex=False),
                errors='coerce'
            )
        else:
            out[std] = np.nan

    # Goodinfo / some sources may return 成交張數; if volume is too small, keep as shares proxy still workable.
    out = out.sort_index()
    out = out[~out.index.duplicated(keep='last')]
    out = out.dropna(subset=['Close'])
    if out.empty:
        return pd.DataFrame()

    # Fill missing OHLC from Close when source lacks full OHLC. This keeps indicator code alive but logs as lower quality.
    for c in ['Open', 'High', 'Low']:
        if c not in out.columns or out[c].isna().all():
            out[c] = out['Close']
    if 'Volume' not in out.columns or out['Volume'].isna().all():
        out['Volume'] = 0
    out['source'] = source
    return out[['Open', 'High', 'Low', 'Close', 'Volume', 'source']]


def fetch_tw_price_finmind(ticker, years=5):
    """Taiwan daily K from FinMind. Primary TW price source for long lookback."""
    if dl is None:
        return pd.DataFrame()
    stock_id = qpro_tw_stock_id(ticker)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.Timedelta(days=365 * years + 20)).strftime('%Y-%m-%d')
    methods = ['taiwan_stock_daily', 'taiwan_stock_daily_adj']
    for method in methods:
        try:
            fn = getattr(dl, method, None)
            if fn is None:
                continue
            df = fn(stock_id=stock_id, start_date=start_date, end_date=end_date)
            out = qpro_normalize_price_df(df, source=f'FinMind-{method}')
            if len(out) >= 250:
                return out
        except Exception as e:
            log(f'[TW-PRICE-FINMIND-WARN] {ticker} {method}: {e}')
    return pd.DataFrame()


def _qpro_parse_roc_date(value):
    s = str(value).strip()
    # TWSE date format: 113/05/13
    m = re.match(r'^(\d{2,3})/(\d{1,2})/(\d{1,2})$', s)
    if m:
        y = int(m.group(1)) + 1911
        return pd.Timestamp(year=y, month=int(m.group(2)), day=int(m.group(3)))
    return pd.to_datetime(s, errors='coerce')


def fetch_tw_price_twse_official(ticker, months=18):
    """TWSE monthly STOCK_DAY fallback. Listed stocks only."""
    stock_id = qpro_tw_stock_id(ticker)
    rows = []
    today = pd.Timestamp.today().normalize()
    for i in range(months):
        d = today - pd.DateOffset(months=i)
        date_yyyymmdd = d.strftime('%Y%m01')
        try:
            url = 'https://www.twse.com.tw/exchangeReport/STOCK_DAY'
            params = {'response': 'json', 'date': date_yyyymmdd, 'stockNo': stock_id}
            r = requests.get(url, params=params, headers=_official_headers(), timeout=15)
            if r.status_code != 200:
                continue
            js = r.json()
            fields = js.get('fields') or []
            data = js.get('data') or []
            if not fields or not data:
                continue
            for row in data:
                item = {fields[j]: row[j] if j < len(row) else None for j in range(len(fields))}
                dt = _qpro_parse_roc_date(item.get('日期'))
                if pd.isna(dt):
                    continue
                rows.append({
                    'date': dt,
                    'Open': _clean_num(item.get('開盤價')),
                    'High': _clean_num(item.get('最高價')),
                    'Low': _clean_num(item.get('最低價')),
                    'Close': _clean_num(item.get('收盤價')),
                    'Volume': _clean_num(item.get('成交股數')),
                })
            time.sleep(0.05)
        except Exception as e:
            log(f'[TW-PRICE-TWSE-WARN] {ticker}: {e}')
            continue
    if not rows:
        return pd.DataFrame()
    return qpro_normalize_price_df(pd.DataFrame(rows), source='TWSE-STOCK_DAY')


def fetch_tw_price_tpex_official(ticker, months=18):
    """TPEx daily K fallback. Endpoint formats have changed over time, so this is best effort."""
    stock_id = qpro_tw_stock_id(ticker)
    rows = []
    today = pd.Timestamp.today().normalize()
    # Try current JSON endpoint first.
    for i in range(months):
        d = today - pd.DateOffset(months=i)
        roc = f'{d.year - 1911}/{d.month:02d}'
        candidates = [
            ('https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock', {'code': stock_id, 'date': roc, 'response': 'json'}),
            ('https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php', {'l': 'zh-tw', 'd': roc, 'stkno': stock_id}),
        ]
        for url, params in candidates:
            try:
                r = requests.get(url, params=params, headers=_official_headers(), timeout=15)
                if r.status_code != 200 or not r.text:
                    continue
                try:
                    js = r.json()
                except Exception:
                    js = None
                if js:
                    fields = js.get('fields') or js.get('aaDataField') or []
                    data = js.get('tables', [{}])[0].get('data') if isinstance(js.get('tables'), list) else None
                    data = data or js.get('data') or js.get('aaData') or []
                    if fields and data:
                        for row in data:
                            item = {fields[j]: row[j] if j < len(row) else None for j in range(len(fields))}
                            dt_val = item.get('日期') or item.get('Date') or row[0]
                            dt = _qpro_parse_roc_date(dt_val)
                            if pd.isna(dt):
                                continue
                            rows.append({
                                'date': dt,
                                'Open': _clean_num(item.get('開盤') or item.get('開盤價') or (row[4] if len(row) > 4 else None)),
                                'High': _clean_num(item.get('最高') or item.get('最高價') or (row[5] if len(row) > 5 else None)),
                                'Low': _clean_num(item.get('最低') or item.get('最低價') or (row[6] if len(row) > 6 else None)),
                                'Close': _clean_num(item.get('收盤') or item.get('收盤價') or (row[2] if len(row) > 2 else None)),
                                'Volume': _clean_num(item.get('成交股數') or item.get('成交仟股') or item.get('成交量') or (row[1] if len(row) > 1 else None)),
                            })
                    else:
                        # Some TPEx responses have table dictionaries with fields in title/data.
                        text = r.text
                        tables = _read_html_tables_from_text(text)
                        for tb in tables:
                            out = qpro_normalize_price_df(tb, source='TPEx-HTML')
                            if not out.empty:
                                return out
                time.sleep(0.05)
            except Exception as e:
                log(f'[TW-PRICE-TPEX-WARN] {ticker}: {e}')
                continue
    if not rows:
        return pd.DataFrame()
    return qpro_normalize_price_df(pd.DataFrame(rows), source='TPEx-OFFICIAL')


def fetch_tw_price_goodinfo_cached(ticker):
    """Goodinfo is kept as a light fallback only; most pages do not provide 250-day OHLC cleanly."""
    stock_id = qpro_tw_stock_id(ticker)
    try:
        html1, html2 = fetch_goodinfo_data(stock_id)
        tables = []
        for html_text in [html1, html2]:
            if html_text:
                tables.extend(get_goodinfo_tables(html_text))
        best = pd.DataFrame()
        for tb in tables:
            joined = ' '.join(str(c) for c in tb.columns)
            if ('收盤' in joined or 'Close' in joined) and ('開盤' in joined or '最高' in joined or '最低' in joined):
                out = qpro_normalize_price_df(tb, source='Goodinfo-Cached')
                if len(out) > len(best):
                    best = out
        return best if len(best) >= 60 else pd.DataFrame()
    except Exception as e:
        log(f'[TW-PRICE-GOODINFO-WARN] {ticker}: {e}')
        return pd.DataFrame()


def qpro_download_tw_stock_df(ticker):
    """TW price pipeline: TWSE/TPEx official -> FinMind -> Goodinfo -> yfinance fallback.

    Important: yfinance is kept as LAST fallback only.  /scan speed is controlled by
    qpro_prepare_scan_pool(), not by changing data priority.
    """
    ticker = normalize_ticker(ticker)
    if not qpro_is_tw_ticker(ticker):
        return ticker, pd.DataFrame()

    source_attempts = []

    # 1) Official first.  If ticker is listed .TW try TWSE first; if .TWO try TPEx first.
    if ticker.endswith('.TW'):
        official_candidates = [fetch_tw_price_twse_official, fetch_tw_price_tpex_official]
    elif ticker.endswith('.TWO'):
        official_candidates = [fetch_tw_price_tpex_official, fetch_tw_price_twse_official]
    else:
        official_candidates = [fetch_tw_price_twse_official, fetch_tw_price_tpex_official]

    for fn in official_candidates:
        df = fn(ticker)
        if len(df) >= 250:
            log(f'[TW-PRICE] {ticker} source={fn.__name__} OFFICIAL_FIRST rows={len(df)}')
            return ticker, df.drop(columns=['source'], errors='ignore')
        source_attempts.append(f'{fn.__name__}:{len(df)}')

    # 2) FinMind second.
    df = fetch_tw_price_finmind(ticker)
    if len(df) >= 250:
        log(f'[TW-PRICE] {ticker} source=FinMind rows={len(df)} attempts={source_attempts}')
        return ticker, df.drop(columns=['source'], errors='ignore')
    source_attempts.append(f'FinMind:{len(df)}')

    # 3) Goodinfo light fallback.
    df = fetch_tw_price_goodinfo_cached(ticker)
    if len(df) >= 250:
        log(f'[TW-PRICE] {ticker} source=Goodinfo rows={len(df)} attempts={source_attempts}')
        return ticker, df.drop(columns=['source'], errors='ignore')
    source_attempts.append(f'Goodinfo:{len(df)}')

    # Final fallback only. This keeps the bot usable when local official APIs are blocked.
    try:
        yf_ticker = ticker
        df = yf.download(yf_ticker, period='5y', progress=False, auto_adjust=True, threads=False)
        if (df is None or df.empty) and ticker.endswith('.TW'):
            alt = ticker.replace('.TW', '.TWO')
            df = yf.download(alt, period='5y', progress=False, auto_adjust=True, threads=False)
            if df is not None and not df.empty:
                yf_ticker = alt
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df is not None and not df.empty:
            df = qpro_normalize_price_df(df, source='yfinance-fallback')
            if len(df) >= 250:
                log(f'[TW-PRICE] {ticker} source=yfinance LAST_FALLBACK rows={len(df)} attempts={source_attempts}')
                return yf_ticker, df.drop(columns=['source'], errors='ignore')
    except Exception as e:
        log(f'[TW-PRICE-YF-FALLBACK-WARN] {ticker}: {e}')

    log(f'[TW-PRICE-WARN] {ticker} no usable K-line. attempts={source_attempts}')
    return ticker, pd.DataFrame()


# Override original download_stock_df so TW scanning does not use yfinance first.
def download_stock_df(ticker):
    ticker = normalize_ticker(ticker)
    if qpro_is_tw_ticker(ticker):
        return qpro_download_tw_stock_df(ticker)

    # US remains yfinance, because it is the intended US data source here.
    try:
        df = yf.download(ticker, period='5y', progress=False, auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return ticker, df if df is not None else pd.DataFrame()
    except Exception as e:
        log(f'[US-PRICE-YF-WARN] {ticker}: {e}')
        return ticker, pd.DataFrame()


def get_company_profile(ticker_num, ticker_full=None, yf_info=None):
    """Override profile priority: TW official/name-map + My-TW-Coverage + Goodinfo; yfinance only for US."""
    is_us = is_us_ticker(ticker_full) if ticker_full else False
    if is_us:
        if yf_info:
            industry = yf_info.get('industry', 'N/A')
            desc = yf_info.get('longBusinessSummary', '查無美股業務描述')
            safe_desc = clip_text(str(desc).replace('*', '').replace('_', ''), 200)
            return {
                'profile': safe_desc,
                'industry': industry,
                'raw_text': None,
                'company_name': yf_info.get('shortName') or yf_info.get('longName') or ticker_num,
            }
        return {'profile': '無法取得美股資料', 'industry': 'N/A', 'raw_text': None, 'company_name': ticker_num}

    ticker_full = normalize_ticker(ticker_full or ticker_num)
    code = qpro_tw_stock_id(ticker_full)
    company_name = None
    industry = 'N/A'
    raw_text = None
    desc = None

    try:
        mapping = qpro_fetch_tw_stock_name_map(force=False)
        company_name = qpro_clean_company_name(mapping.get(ticker_full) or mapping.get(code), ticker_full)
    except Exception:
        company_name = None

    # My-TW-Coverage is still useful for description/industry.
    try:
        if MY_TW_COVERAGE_PATH and os.path.isdir(MY_TW_COVERAGE_PATH):
            target_file = None
            for root, dirs, files in os.walk(MY_TW_COVERAGE_PATH):
                for file in files:
                    if file.startswith(str(code)) and file.endswith('.md'):
                        target_file = os.path.join(root, file)
                        break
                if target_file:
                    break
            if target_file:
                with open(target_file, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                if not company_name:
                    company_name = ui_company_name_from_raw_text(raw_text, ticker_full)
                for line in raw_text.splitlines():
                    s = line.strip()
                    if '產業' in s and ('：' in s or ':' in s):
                        industry = s.split('：', 1)[-1].strip() if '：' in s else s.split(':', 1)[-1].strip()
                        industry = clip_text(industry.replace('*', ''), 60)
                        break
                for line in raw_text.splitlines():
                    s = line.strip()
                    if len(s) > 20 and not s.startswith('#') and not s.startswith('|'):
                        desc = clip_text(s, 180)
                        break
    except Exception as e:
        log(f'[TW-PROFILE-MYTW-WARN] {ticker_full}: {e}')

    # Light Goodinfo fallback for company name/industry if local profile is missing.
    if not company_name or industry == 'N/A':
        try:
            html1, html2 = fetch_goodinfo_data(code)
            text_blob = re.sub(r'\s+', ' ', (html1 or '') + ' ' + (html2 or ''))
            if not company_name:
                m = re.search(rf'{re.escape(code)}\s*([\u4e00-\u9fffA-Za-z0-9\-]+)', text_blob)
                if m:
                    company_name = qpro_clean_company_name(m.group(1), ticker_full)
            if industry == 'N/A':
                m = re.search(r'產業(?:類別)?[:：\s]*([\u4e00-\u9fffA-Za-z &/\-]+)', text_blob)
                if m:
                    industry = clip_text(m.group(1), 60)
        except Exception as e:
            log(f'[TW-PROFILE-GOODINFO-WARN] {ticker_full}: {e}')

    company_name = qpro_clean_company_name(company_name, ticker_full) or code
    safe_desc = (desc or '查無業務描述').replace('*', '').replace('_', '')
    return {'profile': safe_desc, 'industry': industry, 'raw_text': raw_text, 'company_name': company_name}


# QPRO_FIX_20260513_2325_SCAN_FAST_FULL
def qpro_prepare_scan_pool(pool, region='TW', full_scan=False):
    """Prepare scan pool.

    /scan      => TW quick scan: prioritize liquid static/watch names, then cap by TW_SCAN_POOL_LIMIT.
    /scan_full => TW full scan: no cap unless TW_FULL_SCAN_POOL_LIMIT is set.
    """
    region = 'US' if str(region).upper() == 'US' else 'TW'
    pool = [normalize_ticker(x) for x in (pool or []) if x]

    if region != 'TW':
        return sorted(set(pool))

    # Put liquid/high-attention names first, then append official full pool.
    priority = []
    try:
        priority.extend(qpro_static_tw_scan_pool())
    except Exception:
        pass
    try:
        priority.extend(get_defensive_etf_pool('TW'))
    except Exception:
        pass

    ordered = []
    seen = set()
    for t in priority + sorted(set(pool)):
        tt = normalize_ticker(t)
        if tt and tt not in seen:
            ordered.append(tt)
            seen.add(tt)

    if full_scan:
        if TW_FULL_SCAN_POOL_LIMIT and TW_FULL_SCAN_POOL_LIMIT > 0:
            return ordered[:TW_FULL_SCAN_POOL_LIMIT]
        return ordered

    limit = max(50, int(TW_SCAN_POOL_LIMIT or 500))
    return ordered[:limit]


# Override scan loop so TW does not call yf.Ticker(...).info for every Taiwanese stock.
def scan_and_rank_market(chat_id=None, requested_by_user=False, market_mode='offensive', region='TW', full_scan=False):
    if region == 'TW':
        pool = get_tw_stock_pool(market_mode)
    else:
        pool = get_us_defensive_etf_pool() if market_mode == 'defensive' else get_us_stock_pool()

    raw_pool_size = len(set(pool))
    pool = qpro_prepare_scan_pool(pool, region=region, full_scan=full_scan)
    if TEST_MODE:
        pool = pool[:15]

    if requested_by_user:
        mode_text = '全市場完整掃描' if (region == 'TW' and full_scan) else ('快速掃描' if region == 'TW' else '標準掃描')
        safe_send_message(
            chat_id,
            f'📚 {region} 掃描池建立完成：原始 `{raw_pool_size}` 檔，本次 `{len(pool)}` 檔（{mode_text}），開始技術面初篩。\n'
            f'台股資料順序：TWSE/TPEx官方 → FinMind → Goodinfo → yfinance最後備援。\n'
            f'提示：`/scan` = 快掃；`/scan_full` = 全市場完整掃描。',
            parse_mode='Markdown'
        )

    precomputed_prescreen = []
    completed = 0
    max_workers = 10 if region == 'US' else 4
    scan_pool = list(pool)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_scan, ticker, market_mode) for ticker in scan_pool]
        for future in as_completed(futures):
            completed += 1
            if requested_by_user and (completed == 1 or completed % SCAN_PROGRESS_STEP == 0 or completed == len(scan_pool)):
                safe_send_message(chat_id, f'📡 {region} 技術面初篩中：`{completed}`/`{len(scan_pool)}` 檔；有效 `{len(precomputed_prescreen)}` 檔。', parse_mode='Markdown')
            try:
                res = future.result()
            except Exception:
                res = None
            if res:
                precomputed_prescreen.append(res)

    prescreen = precomputed_prescreen

    if not prescreen and region == 'TW':
        retry_pool = qpro_static_tw_scan_pool()[:80]
        safe_send_message(chat_id, f'⚠️ TW 初篩有效名單為 0，啟動官方/FinMind/yfinance最後備援重試 `{len(retry_pool)}` 檔。', parse_mode=None)
        for idx, ticker in enumerate(retry_pool, start=1):
            try:
                if requested_by_user and (idx == 1 or idx % 20 == 0 or idx == len(retry_pool)):
                    safe_send_message(chat_id, f'🔁 TW 備援初篩：`{idx}`/`{len(retry_pool)}` 檔；有效 `{len(prescreen)}` 檔。', parse_mode='Markdown')
                tkr, df = download_stock_df(ticker)
                if df is None or df.empty or len(df) < 250:
                    continue
                tech_pack = evaluate_technical(df, market_mode)
                prescreen.append({'ticker': tkr, 'df': df, 'tech_pack': tech_pack})
            except Exception as e:
                log(f'[TW-RETRY-SCAN-WARN] {ticker}: {e}')
                continue

    prescreen.sort(key=lambda x: x['tech_pack']['technical_score'], reverse=True)
    deep_limit = SCAN_DEEP_LIMIT_US if region == 'US' else SCAN_DEEP_LIMIT_TW
    prescreen = prescreen[:max(FINAL_TOP_N, deep_limit)]

    if requested_by_user:
        safe_send_message(
            chat_id,
            f'✅ 第一階段資料有效名單完成，共 `{len(prescreen)}` 檔進入第二階段深度評分。',
            parse_mode='Markdown'
        )

    if not prescreen:
        log(f'[SCAN-WARN] {region} prescreen empty after primary and fallback scans. pool_size={len(pool)}')
        return []

    ranked = []
    for idx, item in enumerate(prescreen, start=1):
        ticker = item['ticker']
        try:
            is_us = is_us_ticker(ticker)
            yf_info = {}
            if is_us:
                try:
                    yf_info = yf.Ticker(ticker).info
                except Exception:
                    yf_info = {}
            ticker_num = ticker.split('.')[0]

            profile_info = get_company_profile(ticker_num, ticker_full=ticker, yf_info=yf_info)
            fin_data = merge_financial_snapshot(ticker, profile_info.get('raw_text'), yf_info=yf_info)

            if region == 'TW' and not is_us:
                if requested_by_user:
                    safe_send_message(chat_id, f'🐢 TWSE/FinMind/Goodinfo 籌碼精查 `{ticker}` ({idx}/{len(prescreen)})...')
                chip_data = get_tw_chip_data(ticker)
                fin_data = merge_finmind_chip_into_snapshot(fin_data, chip_data)
                time.sleep(0.8)

            t_score = item['tech_pack']['technical_score']
            f_score = calc_fundamental_score(fin_data, is_us)
            c_score = calc_chip_score(fin_data, is_us)
            ranked.append({
                'ticker': ticker,
                'tech_pack': item['tech_pack'],
                'fin_data': fin_data,
                'profile_info': profile_info,
                'total_score': final_total_score(t_score, f_score, c_score, is_us),
            })

        except Exception as e:
            log_exception(f'[RANK-ERROR] {ticker}', e)
            continue

    ranked.sort(key=lambda x: x['total_score'], reverse=True)
    ranked = annotate_sector_strength(ranked)
    LAST_SCAN_RANKED[region] = ranked
    return ranked[:FINAL_TOP_N]



# ==========================================
# QPRO_FIX_20260515_2255_TWO_CODE_COMPANY_CHIP final overrides
# Fix .TWO normalization and make all TW data lookups use pure numeric stock_id.
# ==========================================
def _tw_numeric_stock_id(ticker):
    return qpro_tw_code(ticker)


def qpro_tw_stock_id(ticker):
    return qpro_tw_code(ticker)


def qpro_clean_company_name(name, ticker=''):
    if name is None:
        return None
    s = html.unescape(str(name)).strip()
    if not s:
        return None
    s = s.replace('`', '').replace('*', '').replace('#', '').strip()
    s = re.sub(r'^[\-–—•\s]+', '', s).strip()
    m = re.match(r'^\[([^\]]+)\]\([^\)]*\)$', s)
    if m:
        s = m.group(1).strip()
    m = re.match(r'^\[([^\]]+)\]$', s)
    if m:
        s = m.group(1).strip()
    s = s.strip('[]()（）【】「」『』')
    s = re.sub(r'\s+', ' ', s).strip()
    code = qpro_tw_code(ticker)
    if code:
        s = re.sub(rf'^{re.escape(code)}\s+', '', s).strip()
    bad_values = {'N/A', 'NA', '-', '--', 'None', 'null', ''}
    if s in bad_values:
        return None
    if re.fullmatch(r'\d{4}[A-Z]?|\d{4}\.(TW|TWO)', s.upper()):
        return None
    if code and s == code:
        return None
    return s[:32]


def ui_company_name_from_raw_text(raw_text, ticker=''):
    if not raw_text:
        return None
    code = qpro_tw_code(ticker)
    for line in str(raw_text).splitlines()[:100]:
        s = line.strip()
        if not s or s.startswith('|'):
            continue
        cleaned_line = s.replace('*', '').replace('#', '').strip()
        for key in ['公司名稱', '股票名稱', '名稱', '公司']:
            if key in cleaned_line and ('：' in cleaned_line or ':' in cleaned_line):
                name = cleaned_line.split('：', 1)[-1].strip() if '：' in cleaned_line else cleaned_line.split(':', 1)[-1].strip()
                name = qpro_clean_company_name(name, ticker)
                if name:
                    return clip_text(name, 24)
        if code:
            m = re.match(rf'^[\-–—•\s]*\[?{re.escape(code)}\]?\s+(.+)$', cleaned_line)
            if m:
                name = qpro_clean_company_name(m.group(1), ticker)
                if name:
                    return clip_text(name, 24)
        m = re.match(r'^[\-–—•\s]*\[([^\]]+)\]', cleaned_line)
        if m:
            name = qpro_clean_company_name(m.group(1), ticker)
            if name:
                return clip_text(name, 24)
    return None


def ui_stock_name(item):
    ticker = str(item.get('ticker', '')).strip().upper()
    code = qpro_tw_code(ticker)
    profile = item.get('profile_info') or {}
    if ticker.endswith(('.TW', '.TWO')) or code:
        mapping = qpro_fetch_tw_stock_name_map(force=False)
        for key in [ticker, code + qpro_tw_exchange_suffix(ticker), code]:
            name = qpro_clean_company_name(mapping.get(key), ticker)
            if name:
                return clip_text(name, 24)
    for key in ['company_name', 'stock_name', 'stockName', 'shortName', 'longName', 'name', '公司名稱']:
        name = qpro_clean_company_name(profile.get(key), ticker)
        if name:
            return clip_text(name, 24)
    name = ui_company_name_from_raw_text(profile.get('raw_text'), ticker)
    if name:
        return clip_text(name, 24)
    return ticker


def get_tw_finmind_financial_snapshot(ticker_full):
    stock_id = qpro_tw_code(ticker_full)
    result = {
        'single_month_revenue': None,
        'single_month_mom': None,
        'single_month_yoy': None,
        'eps_latest_quarter': None,
        'eps_ttm': None,
        'source': [],
    }
    if not stock_id.isdigit() or dl is None:
        return result
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.Timedelta(days=540)).strftime('%Y-%m-%d')
    try:
        revenue_df = dl.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date, end_date=end_date)
        revenue_df = _as_date_col(revenue_df)
        if revenue_df is not None and not revenue_df.empty:
            yoy_col = _pick_first_column(revenue_df, ['YoY', 'yoy', 'revenue_year_growth', '去年同月增減(%)', '去年同月增減'])
            mom_col = _pick_first_column(revenue_df, ['MoM', 'mom', 'revenue_month_growth', '上月比較增減(%)', '上月比較增減'])
            rev_col = _pick_first_column(revenue_df, ['revenue', '營收', '當月營收'])
            work = revenue_df.sort_values('date').copy()
            last = work.iloc[-1]
            if rev_col is not None:
                result['single_month_revenue'] = safe_float(last.get(rev_col))
                work['revenue_value'] = pd.to_numeric(work[rev_col], errors='coerce')
            else:
                work['revenue_value'] = np.nan
            if yoy_col is not None:
                result['single_month_yoy'] = safe_float(last.get(yoy_col))
            if mom_col is not None:
                result['single_month_mom'] = safe_float(last.get(mom_col))
            if result['single_month_revenue'] is not None and (result['single_month_yoy'] is None or result['single_month_mom'] is None):
                work = work.dropna(subset=['date', 'revenue_value'])
                if not work.empty:
                    latest = work.iloc[-1]
                    if result['single_month_mom'] is None and len(work) >= 2:
                        prev = safe_float(work.iloc[-2]['revenue_value'])
                        if prev not in (None, 0):
                            result['single_month_mom'] = (safe_float(latest['revenue_value']) - prev) / prev * 100
                    if result['single_month_yoy'] is None:
                        latest_date = pd.to_datetime(latest['date'])
                        prior_rows = work[
                            (pd.to_datetime(work['date']).dt.year == latest_date.year - 1)
                            & (pd.to_datetime(work['date']).dt.month == latest_date.month)
                        ]
                        if not prior_rows.empty:
                            prior = safe_float(prior_rows.iloc[-1]['revenue_value'])
                            if prior not in (None, 0):
                                result['single_month_yoy'] = (safe_float(latest['revenue_value']) - prior) / prior * 100
            result['source'].append('FinMind-Revenue')
    except Exception as e:
        log(f'[FinMind-FUND-WARN] revenue {ticker_full}/{stock_id}: {e}')
    try:
        fs_df = dl.taiwan_stock_financial_statement(stock_id=stock_id, start_date=start_date, end_date=end_date)
        fs_df = _as_date_col(fs_df)
        if fs_df is not None and not fs_df.empty:
            name_col = _pick_first_column(fs_df, ['type', 'name', '財報種類', 'item', 'label'])
            val_col = _pick_first_column(fs_df, ['value', 'EPS', 'eps', 'amount', '數值'])
            if name_col is not None and val_col is not None:
                eps_rows = fs_df[fs_df[name_col].astype(str).str.contains('EPS|每股盈餘|基本每股盈餘', case=False, na=False)].copy()
                if not eps_rows.empty:
                    eps_rows['eps_value'] = pd.to_numeric(eps_rows[val_col], errors='coerce')
                    eps_rows = eps_rows.dropna(subset=['eps_value']).sort_values('date')
                    if not eps_rows.empty:
                        result['eps_latest_quarter'] = float(eps_rows.iloc[-1]['eps_value'])
                        if len(eps_rows) >= 4:
                            result['eps_ttm'] = float(eps_rows.tail(4)['eps_value'].sum())
                        result['source'].append('FinMind-EPS')
    except Exception as e:
        log(f'[FinMind-FUND-WARN] eps {ticker_full}/{stock_id}: {e}')
    return result


def get_tw_chip_data(ticker, days=10):
    """TW chip-data priority with correct .TWO handling.

    All providers receive pure numeric stock_id, so 6425.TWO -> 6425, not 6425O.
    .TW: Goodinfo -> TWSE T86 -> FinMind.
    .TWO: Goodinfo -> FinMind -> TWSE fallback.  TPEx official chip parser is
    not stable across endpoint versions, so FinMind/Goodinfo are safer for OTC.
    """
    ticker = normalize_ticker(ticker)
    stock_id = qpro_tw_code(ticker)
    errors = []
    if not stock_id:
        return _init_tw_chip_result('台股代碼解析失敗')

    gi = fetch_goodinfo_chip_data_fallback(stock_id)
    if _tw_chip_has_data(gi):
        return gi
    errors.append(gi.get('chips_summary', 'Goodinfo 無資料'))

    if ticker.endswith('.TWO'):
        fm = fetch_finmind_chip_data_only(stock_id, days=days)
        if _tw_chip_has_data(fm):
            return fm
        errors.append(fm.get('chips_summary', 'FinMind 無資料'))
        twse = fetch_twse_t86_chip_data(stock_id)
        if _tw_chip_has_data(twse):
            return twse
        errors.append(twse.get('chips_summary', 'TWSE T86 無資料'))
    else:
        twse = fetch_twse_t86_chip_data(stock_id)
        if _tw_chip_has_data(twse):
            return twse
        errors.append(twse.get('chips_summary', 'TWSE T86 無資料'))
        fm = fetch_finmind_chip_data_only(stock_id, days=days)
        if _tw_chip_has_data(fm):
            return fm
        errors.append(fm.get('chips_summary', 'FinMind 無資料'))

    out = _init_tw_chip_result('籌碼資料三源皆不足或解析失敗')
    out['chips_summary'] = ' | '.join([str(e) for e in errors if e])[:260]
    return out




# ==========================================
# QPRO_FIX_20260521_0205_DEDUPE_COMPANY_QUERY
# 1) TW/TWO canonical-code de-dup for scan pool, ranked list, strategy buckets.
# 2) User can input Taiwan company name directly, e.g. 台積電 / 立端 / 鴻海.
# 3) Guard absurd YoY values from bad unit/zero denominator parsing.
# ==========================================
def qpro_canonical_stock_key(ticker):
    s = str(ticker or '').strip().upper()
    code = qpro_tw_code(s)
    if code:
        return 'TW:' + code
    return 'US:' + normalize_ticker(s)


def qpro_preferred_tw_ticker(ticker, name_map=None):
    code = qpro_tw_code(ticker)
    if not code:
        return normalize_ticker(ticker)
    name_map = name_map if name_map is not None else qpro_fetch_tw_stock_name_map(force=False)
    s = str(ticker or '').strip().upper()
    # Prefer the official suffix from ISIN/name map when it is unambiguous.
    has_tw = bool(name_map.get(code + '.TW'))
    has_two = bool(name_map.get(code + '.TWO'))
    if has_tw and not has_two:
        return code + '.TW'
    if has_two and not has_tw:
        return code + '.TWO'
    # Ambiguous or no map: keep user/source suffix; otherwise default listed.
    if s.endswith('.TWO'):
        return code + '.TWO'
    return code + '.TW'


def qpro_unique_tickers_by_code(tickers):
    """De-duplicate Taiwan tickers by pure 4-digit code.

    Fixes cases such as 6245.TW and 6245.TWO both entering the scan pool,
    or the same stock entering from TWSE/TPEx/FinMind/yfinance fallback paths.
    """
    out = []
    seen = set()
    try:
        name_map = qpro_fetch_tw_stock_name_map(force=False)
    except Exception:
        name_map = {}
    for t in tickers or []:
        if not t:
            continue
        tt = normalize_ticker(t)
        code = qpro_tw_code(tt)
        if code:
            tt = qpro_preferred_tw_ticker(tt, name_map)
        key = qpro_canonical_stock_key(tt)
        if key in seen:
            continue
        seen.add(key)
        out.append(tt)
    return out


def qpro_rank_value(item, category=None):
    if not isinstance(item, dict):
        return 0.0
    if category:
        try:
            return safe_float(_strategy_category_score(item, category), 0.0) or 0.0
        except Exception:
            return 0.0
    return (
        (safe_float(item.get('priority_score'), 0.0) or 0.0) * 1.5 +
        (safe_float(item.get('total_score'), 0.0) or 0.0) +
        (safe_float((item.get('tech_pack') or {}).get('technical_score'), 0.0) or 0.0) * 0.2
    )


def qpro_unique_items_by_code(items, category=None, limit=None):
    """Keep one row per real stock code, preserving the strongest row."""
    sorted_items = sorted(list(items or []), key=lambda x: qpro_rank_value(x, category), reverse=True)
    out = []
    seen = set()
    for item in sorted_items:
        ticker = item.get('ticker') if isinstance(item, dict) else None
        key = qpro_canonical_stock_key(ticker)
        if not ticker or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if limit and len(out) >= limit:
            break
    return out


def qpro_unique_scan_records(records, limit=None):
    def _record_df_len(record):
        df = (record or {}).get('df')
        if df is None:
            return 0
        try:
            return len(df)
        except Exception:
            return 0

    def _score(r):
        tp = (r or {}).get('tech_pack') or {}
        return (safe_float(tp.get('technical_score'), 0.0) or 0.0, _record_df_len(r))
    sorted_records = sorted(list(records or []), key=_score, reverse=True)
    out = []
    seen = set()
    for r in sorted_records:
        t = (r or {}).get('ticker')
        key = qpro_canonical_stock_key(t)
        if not t or key in seen:
            continue
        seen.add(key)
        out.append(r)
        if limit and len(out) >= limit:
            break
    return out


# Save the underlying financial merge and wrap it with sanity guards.
_qpro_original_merge_financial_snapshot = merge_financial_snapshot

def qpro_sane_pct(value, field_name='pct', ticker=''):
    v = safe_float(value)
    if v is None or not np.isfinite(v):
        return None
    # Monthly revenue YoY/MoM can be large for turnaround names, but billions % is parser/unit error.
    if abs(v) > 1000:
        try:
            log(f'[FUND-GUARD] {ticker} {field_name} impossible pct={v}; set to N/A')
        except Exception:
            pass
        return None
    return v


def merge_financial_snapshot(ticker_full, md_text, yf_info=None):
    data = _qpro_original_merge_financial_snapshot(ticker_full, md_text, yf_info=yf_info)
    if not isinstance(data, dict):
        return data
    for fld in ['single_month_yoy', 'single_month_mom', 'profit_margin_pct', 'earnings_growth_pct']:
        data[fld] = qpro_sane_pct(data.get(fld), fld, ticker_full)
    return data


# Override scan pool preparation with canonical TW-code de-dup.
def qpro_prepare_scan_pool(pool, region='TW', full_scan=False):
    region = 'US' if str(region).upper() == 'US' else 'TW'
    pool = [normalize_ticker(x) for x in (pool or []) if x]
    if region != 'TW':
        return sorted(set(pool))

    priority = []
    try:
        priority.extend(qpro_static_tw_scan_pool())
    except Exception:
        pass
    try:
        priority.extend(get_defensive_etf_pool('TW'))
    except Exception:
        pass

    ordered = qpro_unique_tickers_by_code(priority + sorted(set(pool)))
    if full_scan:
        if TW_FULL_SCAN_POOL_LIMIT and TW_FULL_SCAN_POOL_LIMIT > 0:
            return ordered[:TW_FULL_SCAN_POOL_LIMIT]
        return ordered
    limit = max(50, int(TW_SCAN_POOL_LIMIT or 500))
    return ordered[:limit]


# Override strategy-bucket ranking with de-dup before Top 10 rendering.
def _rank_strategy_bucket(ranked, category, limit=10):
    base = qpro_unique_items_by_code(ranked, category=category, limit=None)
    scored = []
    for item in base:
        score = _strategy_category_score(item, category)
        copied = dict(item)
        copied['category_score'] = score
        copied['primary_strategy'] = category
        copied['primary_strategy_name'] = _strategy_combo_label(copied, max_items=3)
        scored.append(copied)
    scored.sort(key=lambda x: (safe_float(x.get('category_score'), 0.0) or 0.0, safe_float(x.get('total_score'), 0.0) or 0.0), reverse=True)
    return qpro_unique_items_by_code(scored, category=category, limit=limit)


# Override priority selection with canonical-code candidate map.
def select_priority_recommendations(ranked, limit=10):
    ranked = qpro_unique_items_by_code(ranked, limit=None)
    candidate_map = {}
    for category in ['vcp', 'pattern', 'bollinger', 'minervini', 'sector']:
        for rank_idx, item in enumerate(_rank_strategy_bucket(ranked, category, limit=limit), start=1):
            key = qpro_canonical_stock_key(item.get('ticker'))
            if key not in candidate_map:
                copied = dict(item)
                copied['source_strategies'] = []
                copied['source_bucket_ranks'] = {}
                candidate_map[key] = copied
            if category not in candidate_map[key]['source_strategies']:
                candidate_map[key]['source_strategies'].append(category)
            candidate_map[key]['source_bucket_ranks'][category] = rank_idx

    source_items = list(candidate_map.values()) or ranked
    priority = []
    for item in source_items:
        copied = dict(item)
        strategy_scores = _all_strategy_scores(item)
        hits = _item_hit_strategy_keys(copied)
        primary_strategy = hits[0] if hits else max(strategy_scores, key=strategy_scores.get)
        best_strategy_score = max(strategy_scores.get(k, 0.0) for k in hits) if hits else max(strategy_scores.values())
        win_rate_pct = estimate_item_win_rate(item)
        total = safe_float(item.get('total_score'), 0.0) or 0.0
        strategy_average = sum(min(v, 140.0) for v in strategy_scores.values()) / max(1, len(strategy_scores))
        multi_bonus = min(len(hits), 4) * 2.2
        priority_score = total * 0.42 + win_rate_pct * 0.28 + min(best_strategy_score, 150.0) * 0.22 + strategy_average * 0.08 + multi_bonus
        copied['strategy_scores'] = strategy_scores
        copied['primary_strategy'] = primary_strategy
        copied['primary_strategy_name'] = _strategy_combo_label(copied, max_items=3)
        copied['strategy_hits'] = hits
        copied['estimated_win_rate_pct'] = win_rate_pct
        copied['priority_score'] = priority_score
        priority.append(copied)
    priority.sort(key=lambda x: x.get('priority_score', 0.0), reverse=True)
    return qpro_unique_items_by_code(priority, limit=limit)


# Override scan loop: de-dup after pool, prescreen, ranked, and after sector annotation.
def scan_and_rank_market(chat_id=None, requested_by_user=False, market_mode='offensive', region='TW', full_scan=False):
    if region == 'TW':
        pool = get_tw_stock_pool(market_mode)
    else:
        pool = get_us_defensive_etf_pool() if market_mode == 'defensive' else get_us_stock_pool()

    raw_pool_size = len(set(pool))
    pool = qpro_prepare_scan_pool(pool, region=region, full_scan=full_scan)
    if TEST_MODE:
        pool = pool[:15]

    if requested_by_user:
        mode_text = '全市場完整掃描' if (region == 'TW' and full_scan) else ('快速掃描' if region == 'TW' else '標準掃描')
        safe_send_message(
            chat_id,
            f'📚 {region} 掃描池建立完成：原始 `{raw_pool_size}` 檔，本次 `{len(pool)}` 檔（{mode_text}，已依台股代碼去重），開始技術面初篩。\n'
            f'台股資料順序：TWSE/TPEx官方 → FinMind → Goodinfo → yfinance最後備援。\n'
            f'提示：`/scan` = 快掃；`/scan_full` = 全市場完整掃描。',
            parse_mode='Markdown'
        )

    precomputed_prescreen = []
    completed = 0
    max_workers = 10 if region == 'US' else 4
    scan_pool = list(pool)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_scan, ticker, market_mode) for ticker in scan_pool]
        for future in as_completed(futures):
            completed += 1
            if requested_by_user and (completed == 1 or completed % SCAN_PROGRESS_STEP == 0 or completed == len(scan_pool)):
                safe_send_message(chat_id, f'📡 {region} 技術面初篩中：`{completed}`/`{len(scan_pool)}` 檔；有效 `{len(precomputed_prescreen)}` 檔。', parse_mode='Markdown')
            try:
                res = future.result()
            except Exception:
                res = None
            if res:
                precomputed_prescreen.append(res)

    prescreen = qpro_unique_scan_records(precomputed_prescreen)

    if not prescreen and region == 'TW':
        retry_pool = qpro_static_tw_scan_pool()[:80]
        safe_send_message(chat_id, f'⚠️ TW 初篩有效名單為 0，啟動官方/FinMind/yfinance最後備援重試 `{len(retry_pool)}` 檔。', parse_mode=None)
        for idx, ticker in enumerate(retry_pool, start=1):
            try:
                if requested_by_user and (idx == 1 or idx % 20 == 0 or idx == len(retry_pool)):
                    safe_send_message(chat_id, f'🔁 TW 備援初篩：`{idx}`/`{len(retry_pool)}` 檔；有效 `{len(prescreen)}` 檔。', parse_mode='Markdown')
                tkr, df = download_stock_df(ticker)
                if df is None or df.empty or len(df) < 250:
                    continue
                tech_pack = evaluate_technical(df, market_mode)
                prescreen.append({'ticker': tkr, 'df': df, 'tech_pack': tech_pack})
                prescreen = qpro_unique_scan_records(prescreen)
            except Exception as e:
                log(f'[TW-RETRY-SCAN-WARN] {ticker}: {e}')
                continue

    prescreen.sort(key=lambda x: x['tech_pack']['technical_score'], reverse=True)
    prescreen = qpro_unique_scan_records(prescreen)
    deep_limit = SCAN_DEEP_LIMIT_US if region == 'US' else SCAN_DEEP_LIMIT_TW
    prescreen = prescreen[:max(FINAL_TOP_N, deep_limit)]

    if requested_by_user:
        safe_send_message(chat_id, f'✅ 第一階段資料有效名單完成，共 `{len(prescreen)}` 檔進入第二階段深度評分。', parse_mode='Markdown')

    if not prescreen:
        log(f'[SCAN-WARN] {region} prescreen empty after primary and fallback scans. pool_size={len(pool)}')
        return []

    ranked = []
    for idx, item in enumerate(prescreen, start=1):
        ticker = item['ticker']
        try:
            is_us = is_us_ticker(ticker)
            yf_info = {}
            if is_us:
                try:
                    yf_info = yf.Ticker(ticker).info
                except Exception:
                    yf_info = {}
            ticker_num = qpro_tw_code(ticker) or ticker.split('.')[0]
            profile_info = get_company_profile(ticker_num, ticker_full=ticker, yf_info=yf_info)
            fin_data = merge_financial_snapshot(ticker, profile_info.get('raw_text'), yf_info=yf_info)
            if region == 'TW' and not is_us:
                if requested_by_user:
                    safe_send_message(chat_id, f'🐢 TWSE/FinMind/Goodinfo 籌碼精查 `{ticker}` ({idx}/{len(prescreen)})...')
                chip_data = get_tw_chip_data(ticker)
                fin_data = merge_finmind_chip_into_snapshot(fin_data, chip_data)
                time.sleep(0.8)
            t_score = item['tech_pack']['technical_score']
            f_score = calc_fundamental_score(fin_data, is_us)
            c_score = calc_chip_score(fin_data, is_us)
            ranked.append({
                'ticker': ticker,
                'tech_pack': item['tech_pack'],
                'fin_data': fin_data,
                'profile_info': profile_info,
                'total_score': final_total_score(t_score, f_score, c_score, is_us),
            })
        except Exception as e:
            log_exception(f'[RANK-ERROR] {ticker}', e)
            continue

    ranked.sort(key=lambda x: x['total_score'], reverse=True)
    ranked = qpro_unique_items_by_code(ranked, limit=None)
    ranked = annotate_sector_strength(ranked)
    ranked = qpro_unique_items_by_code(ranked, limit=None)
    LAST_SCAN_RANKED[region] = ranked
    return ranked[:FINAL_TOP_N]


def qpro_norm_company_query(text):
    s = str(text or '').strip()
    s = s.replace('股份有限公司', '').replace('有限公司', '').replace('公司', '')
    s = re.sub(r'[\s\-_．.・·•股份]', '', s)
    return s.upper()


def qpro_resolve_tw_company_query(text):
    """Resolve Chinese TW company name to Yahoo-style ticker.

    Examples: 台積電 -> 2330.TW, 鴻海 -> 2317.TW, 立端 -> 6245.TW/TWO by ISIN map.
    """
    query = qpro_norm_company_query(text)
    if not query:
        return None
    try:
        mapping = qpro_fetch_tw_stock_name_map(force=False)
    except Exception:
        mapping = {}
    candidates = []
    for key, name in mapping.items():
        if not re.match(r'^\d{4}(\.TW|\.TWO)?$', str(key).upper()):
            continue
        name_clean = qpro_clean_company_name(name, str(key))
        if not name_clean:
            continue
        n = qpro_norm_company_query(name_clean)
        if not n:
            continue
        score = 0
        if query == n:
            score = 100
        elif query in n:
            score = 80 - max(0, len(n) - len(query))
        elif n in query:
            score = 65 - max(0, len(query) - len(n))
        if score > 0:
            ticker = str(key).upper()
            if re.fullmatch(r'\d{4}', ticker):
                ticker = qpro_preferred_tw_ticker(ticker, mapping)
            else:
                ticker = qpro_preferred_tw_ticker(ticker, mapping)
            candidates.append((score, ticker, name_clean))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], 1 if x[1].endswith('.TW') else 0), reverse=True)
    return candidates[0][1]


def qpro_resolve_user_stock_input(text):
    raw = str(text or '').strip()
    if not raw:
        return None
    cleaned = raw.upper().replace('多', '').replace('空', '').strip()
    cleaned = cleaned.replace(' ', '')
    if re.fullmatch(r'[0-9]{4}(\.TW|\.TWO)?', cleaned):
        return normalize_ticker(cleaned)
    if re.fullmatch(r'[0-9]{4}', cleaned):
        return normalize_ticker(cleaned)
    if re.fullmatch(r'[A-Z]{1,5}([.-][A-Z])?', cleaned):
        return normalize_ticker(cleaned)
    return qpro_resolve_tw_company_query(raw)


# ==========================================
# Telegram handlers and polling
# ==========================================
if bot:
    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        safe_reply_to(message, '👋 歡迎使用 **Stock Minervini Pro** (跨國對沖基金版)\n\n🟢 `/scan`：台股快速掃描（預設前500檔）\n🔎 `/scan_full`：台股全市場完整掃描\n🇺🇸 `/scan_us`：美股 S&P500 掃描\n🟢 `/update`：手動同步台股資料\n🟢 直接輸入代碼或台股公司名稱 (例如 2330、台積電、AAPL)：單檔分析', parse_mode='Markdown')

    @bot.message_handler(commands=['update'])
    def handle_update(message): update_my_tw_coverage(message.chat.id)

    @bot.message_handler(commands=['scan'])
    def handle_scan(message):
        safe_send_message(message.chat.id, '🚀 啟動台股快速掃描並生成四維趨勢圖...（/scan_full 可跑全市場）', parse_mode='Markdown')
        start_scan_thread(message.chat.id, True, region='TW', full_scan=False)

    @bot.message_handler(commands=['scan_full'])
    def handle_scan_full(message):
        safe_send_message(message.chat.id, '🔎 啟動台股全市場完整掃描，會比較久（可能 20～45 分鐘）...', parse_mode='Markdown')
        start_scan_thread(message.chat.id, True, region='TW', full_scan=True)

    @bot.message_handler(commands=['scan_us'])
    def handle_scan_us(message):
        safe_send_message(message.chat.id, '🇺🇸 啟動美股多策略掃描：VCP / 布林突破 / 形態學 / 族群連動...', parse_mode='Markdown')
        start_scan_thread(message.chat.id, True, region='US')

    @bot.message_handler(func=lambda message: not message.text.startswith('/'))
    def handle_stock(message):
        raw_text = str(message.text or '').strip()
        ticker = qpro_resolve_user_stock_input(raw_text)
        if not ticker:
            return

        safe_reply_to(message, f'⏳ 正在產生 `{ticker}` 報告...')
        region = 'US' if is_us_ticker(normalize_ticker(ticker)) else 'TW'
        current_mode, _ = check_market_status(region)
        report, img_path, strategy_img_path = analyze_stock(ticker, current_mode, silent=True)

        if img_path and os.path.exists(img_path):
            safe_send_photo(message.chat.id, img_path)
        if strategy_img_path and os.path.exists(strategy_img_path):
            safe_send_photo(message.chat.id, strategy_img_path)
        if report:
            safe_send_message(message.chat.id, report, parse_mode='Markdown')

def schedule_loop():
    schedule.every().day.at('16:30').do(start_scan_thread, chat_id=CHAT_ID, requested_by_user=False, region='TW', full_scan=True)
    schedule.every().day.at('05:00').do(start_scan_thread, chat_id=CHAT_ID, requested_by_user=False, region='US')
    schedule.every().saturday.at("02:00").do(run_weekly_optimization)
    
    while True: schedule.run_pending(); time.sleep(1)



# QPRO_FIX_20260515_0015_MTX_RETAIL_OFFICIAL_TABLE
# Fix MTX retail proxy numerator parser:
#   Old token parser sometimes captured 契約金額 instead of 未平倉多空淨額口數,
#   producing impossible values such as inst_net=7,287,249 and retail=-18,000%.
#   This override uses TAIFEX futContractsDateExcel/read_html first and only accepts
#   plausible MTX open-interest net contract counts.

def _qpro_norm_text(value):
    try:
        return re.sub(r'\s+', '', str(value).replace('\n', '').replace('\r', '').strip())
    except Exception:
        return ''


def _qpro_table_product_keywords(commodity_id='MTX'):
    cid = str(commodity_id).upper()
    if cid == 'MTX':
        return ['小型臺指期貨', '小型台指期貨', '小型臺指', '小型台指', 'MTX']
    if cid == 'TX':
        return ['臺股期貨', '台股期貨', '臺股', '台股', 'TX']
    return [cid]


def _qpro_find_inst_oi_net_lot_col(df):
    """Find column for 未平倉餘額 / 多空淨額 / 口數 in TAIFEX table."""
    if df is None or df.empty:
        return None
    best = None
    for c in df.columns:
        cs = _qpro_norm_text(c)
        # Exact target column in the TAIFEX table after flattening MultiIndex:
        # 未平倉餘額 多空淨額 口數
        has_oi = ('未平倉' in cs) or ('未沖銷' in cs) or ('OpenInterest' in cs)
        has_net = ('多空淨額' in cs) or ('多空淨' in cs) or ('淨額' in cs) or ('Net' in cs)
        has_lot = ('口數' in cs) or ('契約數' in cs) or ('Contracts' in cs)
        is_amount = ('契約金額' in cs) or ('金額' in cs) or ('Amount' in cs) or ('千元' in cs)
        if has_oi and has_net and has_lot and not is_amount:
            return c
        if has_oi and has_net and not is_amount:
            best = c
    return best


def _qpro_row_is_contract_and_investor(row, commodity_id='MTX'):
    row_text = _qpro_norm_text(' '.join(str(x) for x in getattr(row, 'values', [])))
    product_ok = any(k in row_text for k in _qpro_table_product_keywords(commodity_id))
    investor_ok = any(k in row_text for k in ['自營商', '投信', '外資', '外資及陸資'])
    return product_ok and investor_ok


def _qpro_extract_inst_net_from_table(df, commodity_id='MTX'):
    """Extract sum of dealer + trust + foreign MTX OI net lots from one TAIFEX table.

    Returns only contract counts.  It rejects contract amount columns by plausibility:
    a valid MTX institutional net OI should not be millions of contracts.
    """
    if df is None or df.empty:
        return None
    try:
        work = _flatten_table_columns(df)
    except Exception:
        work = df.copy()

    target_col = _qpro_find_inst_oi_net_lot_col(work)
    vals = []

    for _, row in work.iterrows():
        if not _qpro_row_is_contract_and_investor(row, commodity_id):
            continue

        v = None
        if target_col is not None:
            v = _clean_num(row.get(target_col))

        if v is None:
            nums = [_clean_num(x) for x in getattr(row, 'values', [])]
            nums = [x for x in nums if x is not None]
            # TAIFEX row numeric order usually contains alternating 口數 / 契約金額.
            # Keep only plausible contract-count values and choose the last one,
            # which corresponds to 未平倉多空淨額口數 in the row layout.
            plausible_contract_counts = [x for x in nums if abs(float(x)) <= 500000]
            if plausible_contract_counts:
                v = plausible_contract_counts[-1]

        if v is None:
            continue

        # Reject impossible contract counts caused by reading 契約金額.
        if abs(float(v)) > 500000:
            continue
        vals.append(float(v))

    if not vals:
        return None

    total = sum(vals)
    if abs(total) > 500000:
        return None
    return total


def _taifex_contract_institutional_net_oi_one_day(date_ts, commodity_id='MTX'):
    """Official TAIFEX three-institution net OI for MTX/TX, contracts only.

    Priority:
      1. TAIFEX futContractsDateExcel HTML table, exact 未平倉餘額/多空淨額/口數 column.
      2. TAIFEX futContractsDate HTML table.
      3. Last-resort text parser, but with strict plausibility guard.
    """
    date_slash = pd.to_datetime(date_ts).strftime('%Y/%m/%d')
    urls = [
        'https://www.taifex.com.tw/cht/3/futContractsDateExcel',
        'https://www.taifex.com.tw/cht/3/futContractsDate',
        'https://www.taifex.com.tw/cht/3/futContractsDateView',
    ]
    params_candidates = [
        {'queryDate': date_slash, 'commodityId': str(commodity_id).upper()},
        {'queryDate': date_slash, 'commodity_id': str(commodity_id).upper()},
        {'date': date_slash, 'commodityId': str(commodity_id).upper()},
        {'queryDate': date_slash},
    ]

    for url in urls:
        for params in params_candidates:
            try:
                text = _fetch_text(url, params=params, timeout=20)
                if not text:
                    continue
                tables = _read_html_tables_from_text(text)
                for tbl in tables:
                    val = _qpro_extract_inst_net_from_table(tbl, commodity_id)
                    if val is not None:
                        log(f'[MTX-RETAIL-PARSER] {pd.to_datetime(date_ts).strftime("%Y-%m-%d")} {commodity_id} inst_net_oi={val:,.0f} source=TAIFEX_TABLE')
                        return val
            except Exception as e:
                try:
                    log(f'[MTX-RETAIL-PARSER-WARN] table parser failed {date_slash} {commodity_id}: {e}')
                except Exception:
                    pass
                continue

    # Last fallback to old text parser, but never accept amount-like values.
    paths = ['/cht/3/futContractsDateExcel', '/cht/3/futContractsDate']
    for text in _qpro_fetch_taifex_texts(paths, date_ts=date_ts, commodity_id=commodity_id, timeout=20):
        val = _qpro_extract_contract_inst_net_oi_from_text(text, commodity_id)
        val_f = safe_float(val)
        if val_f is not None and abs(val_f) <= 500000:
            log(f'[MTX-RETAIL-PARSER] {pd.to_datetime(date_ts).strftime("%Y-%m-%d")} {commodity_id} inst_net_oi={val_f:,.0f} source=TAIFEX_TEXT_GUARDED')
            return val_f
        if val_f is not None:
            log(f'[MTX-RETAIL-PARSER-WARN] reject impossible {commodity_id} inst_net_oi={val_f:,.0f}; likely contract amount, not lots')
    return None


def _qpro_escape_markdown_text(text):
    """Telegram Markdown fallback helper: avoid entity parse errors."""
    if text is None:
        return ''
    return str(text).replace('_', '\\_').replace('*', '\\*').replace('`', "'")

if __name__ == '__main__':
    log('🤖 Stock Minervini Pro (Cross-Border Edition) 啟動中...')
    threading.Thread(target=schedule_loop, daemon=True).start()
    if bot: bot.infinity_polling(timeout=60, long_polling_timeout=30)
