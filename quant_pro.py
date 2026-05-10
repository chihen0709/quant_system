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
from io import StringIO
from datetime import datetime

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

try:
    from FinMind.data import DataLoader
    dl = DataLoader()
except Exception as e:
    DataLoader = None
    dl = None
    print(f"[FinMind-WARN] FinMind unavailable: {e}")

# ==========================================
# Paths and settings
# ==========================================
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

os.makedirs(REPORT_DIR, exist_ok=True)
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


def fetch_tw_retail_sentiment_feature(start_date=None, end_date=None):
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
        if x >= 60:
            return -0.20
        if x <= 40:
            return 0.20
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
    """Real-data TW dimension: foreign spot net buy/sell in 億. No mock fallback."""
    start_date, end_date = (start_date, end_date) if start_date and end_date else _date_range(90)

    candidates = []

    # FinMind commonly exposes institutional investor flows by stock_id.
    # Some environments support MI_INDEX for market-level aggregates; if not,
    # we simply return empty and let the dashboard show N/A.
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

    # Prefer 外資 / Foreign rows when an investor-name column exists.
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

    # Try to normalize to 億. Different sources may return shares, dollars, or lots.
    # If the absolute value is huge, assume it is TWD and divide by 1e8.
    # Otherwise keep as-is and label still remains a market-flow proxy.
    if not out.empty and out['value'].abs().median() > 1_000_000:
        out['value'] = out['value'] / 100_000_000

    out['score'] = out['value'].apply(lambda x: _score_by_threshold(x, 50, -50, 0.25, -0.25))
    return out[['date', 'value', 'score']]


def fetch_real_macro_data(days=5):
    """Build a Taiwan Macro Wave dataframe from real sources only.

    Columns:
      Date, Spot, Future, PCR, Retail, Top10, Score, RealCount

    No mock fallback is used. Missing data remains None/N/A and the score is
    calculated only from dimensions that actually returned real data.
    """
    log('[DATA] 啟動真實大盤籌碼波段引擎...')

    index_df = _download_macro_index_df(region='TW', period='6mo')
    if index_df.empty:
        return pd.DataFrame(columns=['Date', 'Spot', 'Future', 'PCR', 'Retail', 'Top10', 'Score', 'RealCount'])

    last_rows = index_df.tail(days).copy()
    trade_dates = list(last_rows.index)
    start_date, end_date = _date_range(160)

    spot_df = fetch_tw_foreign_spot_feature(start_date, end_date)
    future_df = fetch_tw_foreign_futures_feature(start_date, end_date)
    pcr_df = fetch_tw_pcr_feature(start_date, end_date)
    retail_df = fetch_tw_retail_sentiment_feature(start_date, end_date)
    top10_df = fetch_tw_top10_traders_feature(start_date, end_date)

    spot_values = _latest_by_trade_dates(spot_df, trade_dates, display_func=lambda x: _format_signed_number(x, 1))
    future_values = _latest_by_trade_dates(future_df, trade_dates, display_func=lambda x: _format_signed_number(x, 0))
    pcr_values = _latest_by_trade_dates(pcr_df, trade_dates, display_func=lambda x: _format_ratio(x, 1))
    retail_values = _latest_by_trade_dates(retail_df, trade_dates, display_func=lambda x: _format_ratio(x, 1))
    top10_values = _latest_by_trade_dates(top10_df, trade_dates, display_func=lambda x: _format_signed_number(x, 0))

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
            'IndexRet': safe_float(last_rows.iloc[i].get('RET')),
            'IndexRetDisplay': 'N/A' if safe_float(last_rows.iloc[i].get('RET')) is None else f"{safe_float(last_rows.iloc[i].get('RET')):+.2f}%",
            'Score': round(final_score, 3),
            'RealCount': real_count,
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
    spread = spread.dropna(subset=['value']).reset_index().rename(columns={'index': 'date'})
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
        return {'rows': rows, 'latest_score': 0.0, 'latest_mode': 'offensive', 'region': region}

    last_rows = index_df.tail(3).copy().iloc[::-1]
    trade_dates = list(last_rows.index)

    if region == 'TW':
        dims = get_tw_four_dimensional_data(trade_dates)
        field_names = [('Foreign_Fut', '外資期貨'), ('PCR_Ratio', 'PCR'), ('Retail_Sentiment', '散戶多空'), ('Top_10_Traders', '十大交易人')]
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

        fields = []
        for key, label_name in field_names:
            item = dim.get(key, {'display': 'N/A', 'value': None, 'score': 0.0})
            fields.append({'key': key, 'label': label_name, 'display': item.get('display', 'N/A'), 'value': item.get('value'), 'score': safe_float(item.get('score'), 0.0) or 0.0})

        ret = safe_float(index_row.get('RET'))
        rows.append({
            'label': label,
            'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
            'index_ret': ret,
            'index_ret_display': 'N/A' if ret is None else f'{ret:+.2f}%',
            'score': total_score,
            'sentiment': _score_label(total_score),
            'fields': fields,
            'real_count': real_count,
            'data_status': '四維資料' if real_count > 0 else '價格代理',
        })

    latest_score = safe_float(rows[0]['score'], 0.0) or 0.0
    latest_mode = 'defensive' if latest_score < -0.20 else 'offensive'
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
        score = safe_float(macro_data.get('latest_score'), 0.0) or 0.0
        mode = macro_data.get('latest_mode', 'offensive')

        if region == 'TW' and os.path.exists(MACRO_MODEL_PATH):
            try:
                rf_features = _tw_rf_features_from_macro_data(macro_data)
                if rf_features is not None:
                    rf_model = joblib.load(MACRO_MODEL_PATH)
                    is_bull = rf_model.predict(rf_features)[0]
                    return ('offensive', max(score, 0.35)) if is_bull == 1 else ('defensive', min(score, -0.35))
                log('[MARKET-WARN] TW four-dimensional data incomplete. RF model skipped; using real index/proxy score.')
            except Exception as e:
                log(f'[MARKET-WARN] RF model skipped: {e}')
        return mode, score
    except Exception as e:
        log_exception('[MARKET-ERROR]', e)
        return 'offensive', 0.0


def create_macro_dashboard_image(market_mode, macro_score, output_path, region='TW'):
    """Render Macro dashboard.

    TW: dark Macro Wave Engine using real spot/futures/PCR/retail/top10 data when available.
    US: dark four-dimensional proxy dashboard using index, VIX, breadth, and risk appetite.

    This function intentionally does not use mock data. Missing data is rendered as N/A.
    """
    region = 'US' if str(region).upper() == 'US' else 'TW'

    if region == 'TW':
        macro_df = macro_score if isinstance(macro_score, pd.DataFrame) else fetch_real_macro_data(days=5)
        if macro_df is None or macro_df.empty:
            macro_data = get_macro_dashboard_data(region)
            macro_score = safe_float(macro_data.get('latest_score'), 0.0) or 0.0
            market_mode = macro_data.get('latest_mode', market_mode)
            macro_df = pd.DataFrame([{
                'Date': r.get('date', r.get('label', 'N/A')),
                'IndexRetDisplay': r.get('index_ret_display', 'N/A'),
                'SpotDisplay': 'N/A',
                'FutureDisplay': 'N/A',
                'PCRDisplay': 'N/A',
                'RetailDisplay': 'N/A',
                'Top10Display': 'N/A',
                'Score': safe_float(r.get('score'), 0.0) or 0.0,
                'RealCount': 0,
            } for r in macro_data.get('rows', [])])
        else:
            latest_score = safe_float(macro_df.iloc[-1].get('Score'), 0.0) or 0.0
            macro_score = latest_score
            market_mode = 'defensive' if latest_score < -0.20 else 'offensive'

        mode_text = '🟢 波段多方輪動 (Wave Up)' if market_mode == 'offensive' else '🔴 波段防守避險 (Wave Down)'
        mode_color = '#10b981' if market_mode == 'offensive' else '#ef4444'
        dates = macro_df['Date'].astype(str).tolist() if 'Date' in macro_df.columns else []
        scores = [round(safe_float(x, 0.0) or 0.0, 3) for x in macro_df.get('Score', pd.Series(dtype=float)).tolist()]
        latest = macro_df.iloc[-1].to_dict() if not macro_df.empty else {}

        table_rows = ''
        for _, row in macro_df.tail(5).iterrows():
            score = safe_float(row.get('Score'), 0.0) or 0.0
            score_color = 'pos' if score >= 0 else 'neg'
            idx_ret = row.get('IndexRetDisplay', 'N/A')
            idx_class = _td_class_by_value(idx_ret)
            real_count = int(safe_float(row.get('RealCount'), 0) or 0)
            table_rows += f"""
                        <tr>
                            <td>{row.get('Date', 'N/A')}</td>
                            <td class='{idx_class}'>{idx_ret}</td>
                            <td>{row.get('SpotDisplay', 'N/A')}</td>
                            <td>{row.get('FutureDisplay', 'N/A')}</td>
                            <td>{row.get('PCRDisplay', 'N/A')}</td>
                            <td>{row.get('RetailDisplay', 'N/A')}</td>
                            <td>{row.get('Top10Display', 'N/A')}</td>
                            <td class='{score_color}' style='font-weight:bold;'>{score:.2f}<br><span class='small'>{real_count}/5 real</span></td>
                        </tr>
            """

        def radar_value(raw_score):
            return max(0, min(100, 50 + (safe_float(raw_score, 0.0) or 0.0) * 100))

        # Build radar from latest actual values using the same threshold rules.
        radar_scores = []
        radar_scores.append(_score_by_threshold(latest.get('Future'), 5000, -5000))
        radar_scores.append(_score_by_threshold(latest.get('Spot'), 50, -50, 0.25, -0.25))
        pcr = safe_float(latest.get('PCR'))
        radar_scores.append(0.20 if pcr is not None and pcr >= 130 else (-0.20 if pcr is not None and pcr <= 80 else 0.0))
        retail = safe_float(latest.get('Retail'))
        radar_scores.append(-0.20 if retail is not None and retail >= 60 else (0.20 if retail is not None and retail <= 40 else 0.0))
        radar_scores.append(_score_by_threshold(latest.get('Top10'), 1000, -1000))
        radar_values = [radar_value(s) for s in radar_scores]

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #0f172a; margin: 0; padding: 20px; width: 1250px; color: #f8fafc; }}
                .dashboard {{ background: #1e293b; border-radius: 16px; padding: 30px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155; }}
                .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 20px; margin-bottom: 25px; }}
                .title {{ font-size: 28px; font-weight: 800; color: #f8fafc; margin: 0; letter-spacing: 1px; }}
                .subtitle {{ font-size: 14px; color: #94a3b8; margin-top: 8px; }}
                .status-badge {{ background: {mode_color}20; color: {mode_color}; padding: 12px 24px; border-radius: 12px; font-weight: 800; font-size: 20px; border: 2px solid {mode_color}; letter-spacing: 1px; }}
                .content-grid {{ display: grid; grid-template-columns: 1.9fr 1fr; gap: 30px; }}
                table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; text-align: center; border-radius: 12px; overflow: hidden; }}
                th {{ background: #334155; padding: 13px; color: #cbd5e1; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }}
                td {{ padding: 13px; background: #1e293b; border-bottom: 1px solid #334155; color: #f1f5f9; }}
                .small {{ font-size: 10px; color: #94a3b8; }}
                .neg {{ color: #ef4444; }} .pos {{ color: #10b981; }}
                .summary-box {{ background: #0f172a; border-radius: 12px; padding: 20px; border-left: 5px solid {mode_color}; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); }}
                .chart-container {{ height: 260px; width: 100%; margin-top: 25px; background: #0f172a; padding: 15px; border-radius: 12px; box-sizing: border-box; }}
                .note {{ margin-top: 12px; color: #94a3b8; font-size: 12px; line-height: 1.5; }}
            </style>
        </head>
        <body>
            <div class="dashboard" id="capture-area">
                <div class="header">
                    <div>
                        <h1 class="title">🌊 台股籌碼波段引擎 (Macro Wave Engine)</h1>
                        <div class="subtitle">No-mock 籌碼五維波段分析 | 產生時間: {now_str()}</div>
                    </div>
                    <div class="status-badge">系統狀態: {mode_text} | Score {macro_score:.2f}</div>
                </div>
                <div class="content-grid">
                    <div>
                        <table>
                            <tr><th>交易日期</th><th>指數動能</th><th>外資現貨</th><th>外資期貨</th><th>PCR</th><th>散戶多空</th><th>十大交易人</th><th>波段分數</th></tr>
                            {table_rows}
                        </table>
                        <div class="note">註：本面板不使用 mock 固定資料；資料源缺欄位會顯示 N/A，分數只使用實際取得的維度與真實指數代理計算。</div>
                        <div class="chart-container"><canvas id="trendChart"></canvas></div>
                    </div>
                    <div>
                        <div class="summary-box">
                            <h3 style="margin-top:0; color:#e2e8f0; font-size: 18px; border-bottom: 1px solid #334155; padding-bottom: 10px;">🧭 波段行動指南</h3>
                            <p style="font-size:15px; color:#cbd5e1; line-height:1.8;">
                                <b>中期趨勢定調：</b><br>
                                系統以外資現貨、外資期貨、PCR、散戶多空、十大交易人與指數動能整合判斷。<br><br>
                                <b>資產配置建議：</b><br>
                                {'市場處於偏多波段。適合偏多觀察，優先鎖定強勢突破與中大型權值股。' if market_mode == 'offensive' else '波段風險升高。建議提高現金水位，或轉向高股息、防禦型 ETF 觀察。'}
                            </p>
                        </div>
                        <div class="chart-container" style="height: 220px;"><canvas id="radarChart"></canvas></div>
                    </div>
                </div>
            </div>
            <script>
                Chart.defaults.color = '#94a3b8';
                Chart.defaults.font.family = 'Segoe UI';
                const ctxLine = document.getElementById('trendChart').getContext('2d');
                new Chart(ctxLine, {{
                    type: 'bar',
                    data: {{
                        labels: {json.dumps(dates, ensure_ascii=False)},
                        datasets: [{{
                            label: '波段動能分數',
                            data: {json.dumps(scores)},
                            backgroundColor: ctx => ctx.raw >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)',
                            borderRadius: 6
                        }}]
                    }},
                    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: '近五日波段動能演進', color: '#e2e8f0' }} }}, scales: {{ y: {{ grid: {{ color: '#334155' }} }}, x: {{ grid: {{ display: false }} }} }} }}
                }});
                const ctxRadar = document.getElementById('radarChart').getContext('2d');
                new Chart(ctxRadar, {{
                    type: 'radar',
                    data: {{
                        labels: ['期貨動能', '現貨買盤', 'PCR', '散戶反指標', '十大交易人'],
                        datasets: [{{ data: {json.dumps(radar_values)}, backgroundColor: '{mode_color}40', borderColor: '{mode_color}', borderWidth: 2, pointBackgroundColor: '{mode_color}' }}]
                    }},
                    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ r: {{ min: 0, max: 100, grid: {{ color: '#334155' }}, angleLines: {{ color: '#334155' }}, ticks: {{ display: false }} }} }} }}
                }});
            </script>
        </body>
        </html>
        """

    else:
        macro_data = get_macro_dashboard_data(region)
        rows = macro_data.get('rows', [])
        if rows:
            macro_score = safe_float(macro_data.get('latest_score'), macro_score) or macro_score
            market_mode = macro_data.get('latest_mode', market_mode)

        mode_text = '🟢 美股風險偏好回升 (Risk On)' if market_mode == 'offensive' else '🔴 美股風險降溫 (Risk Off)'
        mode_color = '#10b981' if market_mode == 'offensive' else '#ef4444'
        chart_labels = []
        chart_scores = []
        table_rows = ''
        for r in rows[:5]:
            field_map = {f['key']: f for f in r.get('fields', [])}
            score = safe_float(r.get('score'), 0.0) or 0.0
            score_class = 'pos' if score >= 0 else 'neg'
            idx_ret = r.get('index_ret_display', 'N/A')
            idx_class = _td_class_by_value(idx_ret)
            table_rows += f"""
                        <tr>
                            <td>{r.get('label', 'N/A')}<br><span class='small'>{r.get('date', '')}</span></td>
                            <td class='{idx_class}'>{idx_ret}</td>
                            <td>{field_map.get('VIX_Risk', {}).get('display', 'N/A')}</td>
                            <td>{field_map.get('Breadth_Proxy', {}).get('display', 'N/A')}</td>
                            <td>{field_map.get('Risk_Appetite', {}).get('display', 'N/A')}</td>
                            <td class='{score_class}' style='font-weight:bold;'>{score:.2f}</td>
                        </tr>
            """
            chart_labels.append(r.get('label', 'N/A'))
            chart_scores.append(round(score, 3))

        radar_values = [50, 50, 50, 50]
        if rows:
            latest_fields = rows[0].get('fields', [])
            radar_values = [max(0, min(100, 50 + (safe_float(f.get('score'), 0.0) or 0.0) * 100)) for f in latest_fields]
            while len(radar_values) < 4:
                radar_values.append(50)
            radar_values = radar_values[:4]

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #0f172a; margin: 0; padding: 20px; width: 1150px; color: #f8fafc; }}
                .dashboard {{ background: #1e293b; border-radius: 16px; padding: 30px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155; }}
                .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 20px; margin-bottom: 25px; }}
                .title {{ font-size: 28px; font-weight: 800; color: #f8fafc; margin: 0; letter-spacing: 1px; }}
                .subtitle {{ font-size: 14px; color: #94a3b8; margin-top: 8px; }}
                .status-badge {{ background: {mode_color}20; color: {mode_color}; padding: 12px 24px; border-radius: 12px; font-weight: 800; font-size: 20px; border: 2px solid {mode_color}; }}
                .content-grid {{ display: grid; grid-template-columns: 1.8fr 1fr; gap: 30px; }}
                table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 14px; text-align: center; border-radius: 12px; overflow: hidden; }}
                th {{ background: #334155; padding: 14px; color: #cbd5e1; font-weight: 700; }}
                td {{ padding: 14px; background: #1e293b; border-bottom: 1px solid #334155; color: #f1f5f9; }}
                .small {{ font-size: 10px; color: #94a3b8; }}
                .neg {{ color: #ef4444; }} .pos {{ color: #10b981; }}
                .summary-box {{ background: #0f172a; border-radius: 12px; padding: 20px; border-left: 5px solid {mode_color}; }}
                .chart-container {{ height: 260px; width: 100%; margin-top: 25px; background: #0f172a; padding: 15px; border-radius: 12px; box-sizing: border-box; }}
            </style>
        </head>
        <body>
            <div class="dashboard" id="capture-area">
                <div class="header">
                    <div>
                        <h1 class="title">🇺🇸 美股四維風險偏好引擎</h1>
                        <div class="subtitle">S&P 500 / VIX / Breadth / HYG-TLT proxy | 產生時間: {now_str()}</div>
                    </div>
                    <div class="status-badge">系統狀態: {mode_text} | Score {macro_score:.2f}</div>
                </div>
                <div class="content-grid">
                    <div>
                        <table>
                            <tr><th>日期</th><th>S&P 動能</th><th>VIX</th><th>市場廣度</th><th>風險偏好</th><th>分數</th></tr>
                            {table_rows}
                        </table>
                        <div class="chart-container"><canvas id="trendChart"></canvas></div>
                    </div>
                    <div>
                        <div class="summary-box">
                            <h3 style="margin-top:0; color:#e2e8f0;">🧭 美股行動指南</h3>
                            <p style="font-size:15px; color:#cbd5e1; line-height:1.8;">
                                {'風險偏好偏強，適合觀察大型成長股與突破型標的。' if market_mode == 'offensive' else '風險偏好偏弱，建議提高防守、觀察美債或低波動資產。'}
                            </p>
                        </div>
                        <div class="chart-container" style="height: 220px;"><canvas id="radarChart"></canvas></div>
                    </div>
                </div>
            </div>
            <script>
                Chart.defaults.color = '#94a3b8';
                const ctxLine = document.getElementById('trendChart').getContext('2d');
                new Chart(ctxLine, {{ type: 'bar', data: {{ labels: {json.dumps(list(reversed(chart_labels or ['T-2','T-1','今日'])), ensure_ascii=False)}, datasets: [{{ data: {json.dumps(list(reversed(chart_scores or [0,0,0])))}, backgroundColor: ctx => ctx.raw >= 0 ? 'rgba(16,185,129,0.8)' : 'rgba(239,68,68,0.8)', borderRadius: 6 }}] }}, options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }} }} }});
                const ctxRadar = document.getElementById('radarChart').getContext('2d');
                new Chart(ctxRadar, {{ type: 'radar', data: {{ labels: ['指數', 'VIX', '廣度', '風險偏好'], datasets: [{{ data: {json.dumps(radar_values)}, backgroundColor: '{mode_color}40', borderColor: '{mode_color}', pointBackgroundColor: '{mode_color}' }}] }}, options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ r: {{ min: 0, max: 100, grid: {{ color: '#334155' }}, angleLines: {{ color: '#334155' }}, ticks: {{ display: false }} }} }} }} }});
            </script>
        </body>
        </html>
        """

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html_content)
            page.wait_for_timeout(2000)
            element = page.locator('#capture-area')
            element.screenshot(path=output_path, omit_background=True)
            browser.close()
            return output_path
    except Exception as e:
        log_exception('[PLOT-ERROR] 大盤儀表板生成失敗', e)
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
def clip_text(text, limit=180): return '' if not text else (str(text).strip() if len(str(text).strip()) <= limit else str(text).strip()[:limit] + '...')

def normalize_ticker(ticker):
    ticker = str(ticker).strip().upper()
    if ticker.isdigit() and len(ticker) == 4:
        return ticker + '.TW'
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
    return not ticker.endswith(('.TW', '.TWO'))

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
            return {'profile': safe_desc, 'industry': industry, 'raw_text': None}
        return {'profile': '無法取得美股資料', 'industry': 'N/A', 'raw_text': None}

    try:
        if not MY_TW_COVERAGE_PATH or not os.path.isdir(MY_TW_COVERAGE_PATH):
            return {'profile': '未設定 My-TW-Coverage，本地公司資料略過', 'industry': 'N/A', 'raw_text': None}

        target_file = None
        for root, dirs, files in os.walk(MY_TW_COVERAGE_PATH):
            for file in files:
                if file.startswith(str(ticker_num)) and file.endswith('.md'):
                    target_file = os.path.join(root, file)
                    break
            if target_file: break

        if not target_file:
            return {'profile': '查無資料', 'industry': 'N/A', 'raw_text': None}

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
        return {'profile': safe_desc, 'industry': industry, 'raw_text': content}
    except Exception:
        return {'profile': '讀取失敗', 'industry': 'N/A', 'raw_text': None}

def fetch_goodinfo_data(ticker_num):
    url_main = f'https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={ticker_num}'
    url_chip = f'https://goodinfo.tw/tw/ShowBuySaleChart.asp?STOCK_ID={ticker_num}&CHT_CAT=DATE'
    main_html, chip_html = "", ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
            page = context.new_page()
            try: page.goto(url_main, wait_until='domcontentloaded', timeout=15000)
            except Exception: pass
            page.wait_for_timeout(2000)
            try: main_html = page.content()
            except Exception: pass
            
            try: page.goto(url_chip, wait_until='domcontentloaded', timeout=15000)
            except Exception: pass
            page.wait_for_timeout(2000)
            try: chip_html = page.content()
            except Exception: pass
            browser.close()
    except Exception: pass
    return main_html, chip_html

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

def merge_financial_snapshot(ticker_full, md_text, yf_info=None):
    is_us = is_us_ticker(ticker_full)
    if is_us:
        teps = safe_float(yf_info.get('trailingEps')) if yf_info else None
        rg = safe_float(yf_info.get('revenueGrowth')) if yf_info else None
        if rg is not None: rg = rg * 100
        return {
            'single_month_revenue': None, 'single_month_mom': None, 'single_month_yoy': rg,
            'eps_latest_quarter': None, 'eps_ttm': teps,
            'chips_summary': '美股無日籌碼結構，依賴技術與動能',
            'foreign_2d': None, 'foreign_3d': None, 'foreign_5d': None, 'foreign_10d': None,
            'trust_2d': None, 'trust_3d': None, 'trust_5d': None, 'trust_10d': None,
            'dealer_2d': None, 'dealer_3d': None, 'dealer_5d': None, 'dealer_10d': None,
            'total_2d': None, 'total_3d': None, 'total_5d': None, 'total_10d': None,
            'sources': ['Yahoo Finance']
        }
        
    ticker_num = ticker_full.split('.')[0]
    from_md = parse_financials_from_mytwcoverage(md_text)
    main_html, chip_html = fetch_goodinfo_data(ticker_num)
    main_tables = get_goodinfo_tables(main_html) if main_html else []
    chip_tables = get_goodinfo_tables(chip_html) if chip_html else []
    
    rev_text = parse_goodinfo_revenue_from_text(main_html)
    rev_table = parse_goodinfo_revenue_table(main_tables)
    eps = parse_goodinfo_eps_table(main_tables)
    chip = parse_goodinfo_chip_table(chip_tables)

    merged = {
        'single_month_revenue': rev_table['single_month_revenue'] or rev_text['single_month_revenue'],
        'single_month_mom': from_md['single_month_mom'] or rev_table['single_month_mom'] or rev_text['single_month_mom'],
        'single_month_yoy': from_md['single_month_yoy'] or rev_table['single_month_yoy'] or rev_text['single_month_yoy'],
        'eps_latest_quarter': from_md['eps_latest_quarter'] or eps['eps_latest_quarter'],
        'eps_ttm': from_md['eps_ttm'] or eps['eps_ttm'],
        'chips_summary': chip['chips_summary'],
        'foreign_2d': chip['foreign_2d'], 'foreign_3d': chip['foreign_3d'], 'foreign_5d': chip['foreign_5d'], 'foreign_10d': chip['foreign_10d'],
        'trust_2d': chip['trust_2d'], 'trust_3d': chip['trust_3d'], 'trust_5d': chip['trust_5d'], 'trust_10d': chip['trust_10d'],
        'dealer_2d': chip['dealer_2d'], 'dealer_3d': chip['dealer_3d'], 'dealer_5d': chip['dealer_5d'], 'dealer_10d': chip['dealer_10d'],
        'total_2d': chip['total_2d'], 'total_3d': chip['total_3d'], 'total_5d': chip['total_5d'], 'total_10d': chip['total_10d'],
        'sources': list(set(from_md['source'] + rev_table['source'] + eps['source'] + chip['source']))
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
    return str(ticker).replace('.TW', '').replace('.TWO', '').strip()

def get_tw_chip_data(ticker, days=10):
    result = {
        'chips_summary': 'FinMind 籌碼資料不足或查無資料',
        'foreign_2d': None, 'foreign_3d': None, 'foreign_5d': None, 'foreign_10d': None,
        'trust_2d': None, 'trust_3d': None, 'trust_5d': None, 'trust_10d': None,
        'dealer_2d': None, 'dealer_3d': None, 'dealer_5d': None, 'dealer_10d': None,
        'total_2d': None, 'total_3d': None, 'total_5d': None, 'total_10d': None,
        'source': []
    }

    if dl is None:
        result['chips_summary'] = 'FinMind 未安裝或初始化失敗，請檢查 requirements.txt'
        return result

    stock_id = _tw_numeric_stock_id(ticker)
    if not stock_id.isdigit(): return result

    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=45)).strftime('%Y-%m-%d')
        df = dl.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )

        if df is None or df.empty:
            result['chips_summary'] = f'FinMind 回傳 0 筆資料 (區間 {start_date} ~ {end_date})'
            return result

        df = df.copy()
        if 'date' in df.columns: df = df.sort_values('date')

        for col in ['buy', 'sell']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if 'net_buy' not in df.columns:
            if 'buy' in df.columns and 'sell' in df.columns: df['net_buy'] = df['buy'] - df['sell']
            elif 'buy_sell' in df.columns: df['net_buy'] = pd.to_numeric(df['buy_sell'], errors='coerce').fillna(0)
            else: return result

        def actor_mask(actor):
            if 'name' not in df.columns: return pd.Series(False, index=df.index)
            return df['name'].astype(str).str.contains(actor, na=False)

        def sum_last(actor, n):
            tmp = df[actor_mask(actor)].tail(n)
            if tmp.empty: return None
            return float(tmp['net_buy'].sum())

        actor_map = {'foreign': '外資', 'trust': '投信', 'dealer': '自營'}
        for key, cname in actor_map.items():
            for n in [2, 3, 5, 10]:
                result[f'{key}_{n}d'] = sum_last(cname, n)

        for n in [2, 3, 5, 10]:
            vals = [result.get(f'{actor}_{n}d') for actor in ['foreign', 'trust', 'dealer']]
            valid_vals = [v for v in vals if v is not None]
            result[f'total_{n}d'] = sum(valid_vals) if valid_vals else None

        t5 = result.get('total_5d')
        if t5 is not None:
            if t5 > 0: result['chips_summary'] = f'FinMind：近 5 日三大法人合計買超 {t5:.0f} 張'
            elif t5 < 0: result['chips_summary'] = f'FinMind：近 5 日三大法人合計賣超 {abs(t5):.0f} 張'
            else: result['chips_summary'] = 'FinMind：近 5 日三大法人中性'
            result['source'].append('FinMind-Chips')

    except Exception as e:
        result['chips_summary'] = f'FinMind 籌碼讀取失敗: {e}'

    return result

def merge_finmind_chip_into_snapshot(fin_data, chip_data):
    if not chip_data: return fin_data
    merged = dict(fin_data)
    
    if 'chips_summary' in chip_data:
        merged['chips_summary'] = chip_data['chips_summary']

    chip_keys = [
        'foreign_2d', 'foreign_3d', 'foreign_5d', 'foreign_10d',
        'trust_2d', 'trust_3d', 'trust_5d', 'trust_10d',
        'dealer_2d', 'dealer_3d', 'dealer_5d', 'dealer_10d',
        'total_2d', 'total_3d', 'total_5d', 'total_10d'
    ]

    has_fm_data = any(chip_data.get(k) is not None for k in chip_keys)
    if has_fm_data:
        for k in chip_keys:
            if k in chip_data: merged[k] = chip_data[k]
        sources = list(merged.get('sources', []))
        for s in chip_data.get('source', []):
            if s not in sources: sources.append(s)
        merged['sources'] = sources
    return merged

# ==========================================
# Stock pools and technical filters
# ==========================================
def get_us_stock_pool():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers, timeout=15)
        
        table = pd.read_html(StringIO(response.text))
        return table[0]['Symbol'].tolist()
        
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
    except Exception as e:
        log(f"[FEATURE-WARN] VCP/BB feature generation failed: {e}")
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
    
    score = 0
    if market_mode == 'offensive':
        score = sum([16*c1, 10*c2, 10*c3, 12*c4, 8*c5, 8*c6, 6*c7, 8*wk_up, 5*mo_up, 9*wk_macd_pos, 6*mo_macd_pos])
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
    vcp_score = safe_float(latest.get('vcp_score'), 0.0) or 0.0
    bb_breakout = bool(latest.get('bb_breakout', 0))
    bb_width_pctile = safe_float(latest.get('bb_width_pctile'))

    return {
        'df': df, 'weekly': weekly, 'monthly': monthly, 'technical_score': technical_score, 'latest': latest, 'mode': market_mode,
        'conditions': {'trend_stack': c1, 'off_bottom': c2, 'near_high': c3, 'momentum': c4, 'liquidity': c5, 'short_mid_ma_stack': c6, 'above_ma240': c7, 'weekly_up': wk_up, 'monthly_up': mo_up, 'weekly_macd_positive': wk_macd_pos, 'monthly_macd_positive': mo_macd_pos, 'vcp_setup': vcp_score >= 0.65, 'bb_breakout': bb_breakout},
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
            'atr': safe_float(latest.get('ATR'))
        }
    }

def calc_fundamental_score(f, is_us=False):
    score = 0
    syoy = f.get('single_month_yoy')
    smom = f.get('single_month_mom')
    eq = f.get('eps_latest_quarter')
    ettm = f.get('eps_ttm')

    if syoy is not None: score += 25 if syoy >= 30 else (18 if syoy >= 15 else (10 if syoy >= 5 else (-10 if syoy < 0 else 0)))
    if smom is not None: score += 16 if smom >= 20 else (10 if smom >= 5 else (5 if smom >= 0 else -6))
    if eq is not None: score += 18 if eq >= 20 else (14 if eq >= 10 else (8 if eq > 0 else -8))
    if ettm is not None: score += 20 if ettm >= 40 else (14 if ettm >= 20 else (8 if ettm > 0 else -8))
    if is_us and score < 30 and (syoy is not None or ettm is not None): score += 20 
    return max(0, min(score, 100))

def calc_chip_score(f, is_us=False):
    if is_us: return 0
    score = 0
    t2, t5, t10, f5 = f.get('total_2d'), f.get('total_5d'), f.get('total_10d'), f.get('foreign_5d')
    if t2 is not None: score += 12 if t2 > 0 else -6
    if t5 is not None: score += 24 if t5 > 5000 else (18 if t5 > 1000 else (10 if t5 > 0 else (-16 if t5 < -5000 else -8)))
    if t10 is not None: score += 14 if t10 > 0 else -8
    if f5 is not None: score += 10 if f5 > 0 else -5
    return max(0, min(score, 100))

def final_total_score(t, f, c, is_us=False):
    tech_w = max(0.0, float(SYS_PARAMS.get('tech_weight', WEIGHT_TECH)))
    fund_w = max(0.0, float(SYS_PARAMS.get('fund_weight', WEIGHT_FUND)))
    chip_w = 0.0 if is_us else max(0.0, float(SYS_PARAMS.get('chip_weight', WEIGHT_CHIP)))
    total_w = tech_w + fund_w + chip_w
    if total_w <= 0:
        return t * (0.70 if is_us else WEIGHT_TECH) + f * (0.30 if is_us else WEIGHT_FUND) + (0 if is_us else c * WEIGHT_CHIP)
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
    """生成高階資金控管與分批進場計畫"""
    grade = 'A' if total_score >= 80 else ('B' if total_score >= 65 else 'C')
    regime = '趨勢多頭' if total_score >= 65 else '震盪/偏空'
    win_rate = min(0.85, 0.40 + (total_score / 200))
    
    warnings_list = []
    if rsi is not None and rsi > 70: warnings_list.append('RSI_HOT')
    if bias20 is not None and bias20 > 15: warnings_list.append('BETA_HIGH')
    if volume is not None and avg_vol is not None and volume < avg_vol * 0.7: warnings_list.append('VOL_SHRINK')
    warnings_str = "['" + "', '".join(warnings_list) + "']" if warnings_list else "['SAFE']"
    deduction = len(warnings_list) * 3.5

    pos_pct = 0.20 if grade == 'A' else (0.125 if grade == 'B' else 0.05)
    pos_amt = capital * pos_pct
    action = 'ENTER (分批建倉)' if grade in ['A', 'B'] else 'WATCH (觀望)'
    
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

    plan_text = f"""```text
==================================================
   📊 {ticker} 評估結果
==================================================
總分        : {total_score:.0f}/100   ({grade})
Regime      : {regime}
勝率 p      : {win_rate:.3f}
警示燈      : {warnings_str}
扣分        : {deduction:.1f}%
--------------------------------------------------
建議倉位    : {pos_pct*100:.1f}%
建議金額    : {pos_amt:,.0f} 元
動作        : {action}
==================================================
   📍 {ticker} 進出計畫 (基於 ATR 動態波幅)
==================================================
現價        : {close:.2f}
ATR(14)     : {atr:.2f}
--------------------------------------------------
📥 進場 (分 3 批)
   第1批  價位  {p1:7.2f}    50%   金額  {amt1:8,.0f} 元
   第2批  價位  {p2:7.2f}    30%   金額  {amt2:8,.0f} 元
   第3批  價位  {p3:7.2f}    20%   金額  {amt3:8,.0f} 元
   若三批全成交，平均成本 ≈ {avg_cost:.2f}
--------------------------------------------------
🛑 停損        : {sl_price:.2f}  ({sl_pct:+.1f}%)
💰 停利第1段   : {tp1_price:.2f}  ({tp1_pct:+.1f}%) 賣一半鎖利
💰 停利第2段   : 從持有期最高點回落 8% 即出場
==================================================
```"""
    return plan_text

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
        report += f'🧮 總分：`{total_score:.1f}` | 技術：`{tech_score:.1f}` | 基本：`{fund_score:.1f}`\n'
    else:
        report += f'💰 最新收盤：`{latest["Close"]:.2f}` _(資料日期: {m["latest_date"]})_\n'
        report += f'🧮 總分：`{total_score:.1f}` | 技術：`{tech_score:.1f}` | 基本：`{fund_score:.1f}` | 籌碼：`{chip_score:.1f}`\n'
        
    report += '------------------------\n'
    report += f'🏢 **產業:** {profile_info["industry"]}\n_{profile_info["profile"]}_\n'
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

    report += '\n📈 **技術數據面板：**\n'
    report += f'🔹 RSI(日)：`{safe_num_str(m["rsi"], 1)}`\n'
    report += f'🔹 MACD Hist 日/週/月：`{safe_num_str(m["macd_osc_d"], 3)}` / `{safe_num_str(m["macd_osc_w"], 3)}` / `{safe_num_str(m["macd_osc_m"], 3)}`\n'
    report += f'🔹 5/20/60/240MA乖離率：`{safe_pct_str(m["bias5"])}` / `{safe_pct_str(m["bias20"])}` / `{safe_pct_str(m["bias60"])}` / `{safe_pct_str(m["bias240"])}`\n'

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
        report += '🏦 **籌碼面：** 美股無台股三大法人結構，評分已自動調高技術面比重。\n'
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

    return report, img_path, None

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

def scan_and_rank_market(chat_id=None, requested_by_user=False, market_mode='offensive', region='TW'):
    if region == 'TW':
        pool = get_tw_stock_pool(market_mode)
    else:
        pool = get_us_defensive_etf_pool() if market_mode == 'defensive' else get_us_stock_pool()
        
    if TEST_MODE:
        pool = pool[:15]

    prescreen = []
    for idx, ticker in enumerate(pool, start=1):
        try:
            if idx == 1 or idx % SCAN_PROGRESS_STEP == 0:
                if requested_by_user:
                    safe_send_message(chat_id, f'⏳ {region} 技術初篩：已處理 `{idx}`/`{len(pool)}` 檔...')

            tkr, df = download_stock_df(ticker)
            if df.empty or len(df) < 250:
                continue

            tech_pack = evaluate_technical(df, market_mode)
            if tech_pack['technical_score'] >= 50:
                prescreen.append({'ticker': tkr, 'df': df, 'tech_pack': tech_pack})

        except Exception:
            continue
        
    prescreen.sort(key=lambda x: x['tech_pack']['technical_score'], reverse=True)
    prescreen = prescreen[:TECHNICAL_PRESCREEN_LIMIT]

    if requested_by_user:
        safe_send_message(
            chat_id,
            f'✅ 第一階段技術面海選完成，共 `{len(prescreen)}` 檔進入第二階段深度評分。'
        )

    ranked = []
    for idx, item in enumerate(prescreen, start=1):
        ticker = item['ticker']
        try:
            is_us = is_us_ticker(ticker)
            yf_info = yf.Ticker(ticker).info
            ticker_num = ticker.split('.')[0]
            
            profile_info = get_company_profile(ticker_num, ticker_full=ticker, yf_info=yf_info)
            fin_data = merge_financial_snapshot(ticker, profile_info['raw_text'], yf_info=yf_info)

            if region == 'TW' and not is_us:
                if requested_by_user:
                    safe_send_message(chat_id, f'🐢 FinMind 籌碼精查 `{ticker}` ({idx}/{len(prescreen)})...')
                chip_data = get_tw_chip_data(ticker)
                fin_data = merge_finmind_chip_into_snapshot(fin_data, chip_data)
                time.sleep(3)
            
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
    return ranked[:FINAL_TOP_N]

# ==========================================
# Main jobs and scheduler
# ==========================================
def run_market_scan_job(chat_id, requested_by_user=False, region='TW'):
    market_mode, macro_score = check_market_status(region)
    mode_msg = "🟢 **波段多方輪動：啟動 [攻擊型飆股引擎]**" if market_mode == 'offensive' else "🔴 **波段風險升高：啟動 [RS防守避險引擎 + ETF推薦]**"
    
    safe_send_message(chat_id, f'🔍 **{region} 市場波段雷達啟動中...**\n{mode_msg}', parse_mode='Markdown')
    macro_img_path = os.path.join(REPORT_DIR, f'{region}_macro_dashboard.png')
    dashboard_generated = create_macro_dashboard_image(market_mode, macro_score, macro_img_path, region)
    if dashboard_generated and os.path.exists(dashboard_generated):
        safe_send_photo(chat_id, dashboard_generated)
        time.sleep(2)
    
    if region == 'TW': update_my_tw_coverage(chat_id)
    top_ranked = scan_and_rank_market(chat_id, requested_by_user, market_mode, region)
    
    if not top_ranked:
        safe_send_message(chat_id, '☕ **掃描完畢**\n本次無達標股票。')
        return
        
    summary = [f'🏆 **{region} 今日 Top 10 觀察清單**']
    for i, item in enumerate(top_ranked, start=1):
        icon = '🛡️' if ('00' in item["ticker"] or item["ticker"] in get_defensive_etf_pool('US')) else '🚀'
        summary.append(f'{i}. {icon} `{item["ticker"]}` | 總分 `{item["total_score"]:.1f}`')
    safe_send_message(chat_id, '\n'.join(summary), parse_mode='Markdown')
    time.sleep(2)
    
    for i, item in enumerate(top_ranked, start=1):
        try:
            report, img_path, strategy_img_path = build_stock_report(item['ticker'], item['tech_pack'], item['fin_data'], item['profile_info'], rank=i)
            if img_path and os.path.exists(img_path): safe_send_photo(chat_id, img_path); time.sleep(1)
            if strategy_img_path and os.path.exists(strategy_img_path): safe_send_photo(chat_id, strategy_img_path); time.sleep(1)
            safe_send_message(chat_id, report, parse_mode='Markdown')
            time.sleep(3.5)
        except Exception as e: log_exception(f"發送 {item['ticker']} 失敗", e)
            
    safe_send_message(chat_id, '✅ **掃描完畢**', parse_mode='Markdown')

def start_scan_thread(chat_id, requested_by_user, region='TW'):
    threading.Thread(target=run_market_scan_job, args=(chat_id, requested_by_user, region), daemon=True).start()

def run_weekly_optimization():
    log("🧬 啟動週末回歸測試與策略進化...")
    try: subprocess.Popen(['python3', 'evolve_nsga2.py'])
    except Exception as e: log(f"啟動最佳化失敗: {e}")

# ==========================================
# Telegram handlers and polling
# ==========================================
if bot:
    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        safe_reply_to(message, '👋 歡迎使用 **Stock Minervini Pro** (跨國對沖基金版)\n\n🟢 `/scan`：台股全市場掃描\n🇺🇸 `/scan_us`：美股 S&P500 掃描\n🟢 `/update`：手動同步台股資料\n🟢 直接輸入代碼 (例如 2330 或 AAPL)：單檔分析', parse_mode='Markdown')

    @bot.message_handler(commands=['update'])
    def handle_update(message): update_my_tw_coverage(message.chat.id)

    @bot.message_handler(commands=['scan'])
    def handle_scan(message):
        safe_send_message(message.chat.id, '🚀 啟動台股掃描並生成四維趨勢圖...', parse_mode='Markdown')
        start_scan_thread(message.chat.id, True, region='TW')

    @bot.message_handler(commands=['scan_us'])
    def handle_scan_us(message):
        safe_send_message(message.chat.id, '🇺🇸 啟動美股標普 500 掃描...', parse_mode='Markdown')
        start_scan_thread(message.chat.id, True, region='US')

    @bot.message_handler(func=lambda message: not message.text.startswith('/'))
    def handle_stock(message):
        
        raw_text = message.text.strip().upper().replace('多', '').replace('空', '')
        
        if not re.match(r'^([0-9]{4}|[A-Z]{1,5})$', raw_text):
            return
            
        ticker = raw_text
        safe_reply_to(message, f'⏳ 正在產生 `{ticker}` 報告...')
        
        region = 'US' if is_us_ticker(normalize_ticker(ticker)) else 'TW'
        current_mode, _ = check_market_status(region)
        report, img_path, strategy_img_path = analyze_stock(ticker, current_mode, silent=True)
        
        if img_path and os.path.exists(img_path): safe_send_photo(message.chat.id, img_path)
        if strategy_img_path and os.path.exists(strategy_img_path): safe_send_photo(message.chat.id, strategy_img_path)
        
        if report: 
            safe_send_message(message.chat.id, report, parse_mode='Markdown')

def schedule_loop():
    schedule.every().day.at('16:30').do(start_scan_thread, chat_id=CHAT_ID, requested_by_user=False, region='TW')
    schedule.every().day.at('05:00').do(start_scan_thread, chat_id=CHAT_ID, requested_by_user=False, region='US')
    schedule.every().saturday.at("02:00").do(run_weekly_optimization)
    
    while True: schedule.run_pending(); time.sleep(1)

if __name__ == '__main__':
    log('🤖 Stock Minervini Pro (Cross-Border Edition) 啟動中...')
    threading.Thread(target=schedule_loop, daemon=True).start()
    if bot: bot.infinity_polling(timeout=60, long_polling_timeout=30)
