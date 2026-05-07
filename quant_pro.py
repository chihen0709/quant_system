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
import argparse
import logging
from io import StringIO
from datetime import datetime
from logging.handlers import RotatingFileHandler

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright  
import schedule
import ta
import joblib  

from quant import data_sources as data_sources
from quant.chip import calc_chip_score as shared_calc_chip_score
from quant.cta import add_cta_features, row_strategy_tags
from quant.dashboard import render_top_ranked_dashboard
from quant.fundamental import calc_fundamental_score as shared_calc_fundamental_score
from quant.patterns import add_pattern_features
from quant.sector import annotate_sector_strength
from quant.telegram_bot import create_telegram_bot, sanitize_telegram_error, validate_telegram_token

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
warnings.filterwarnings('ignore')

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK TC', 'Noto Sans CJK SC', 'Microsoft JhengHei',
    'PingFang TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans'
]

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID') or os.environ.get('CHAT_ID', '')
TELEGRAM_POLLING_ENABLED = os.environ.get('TELEGRAM_POLLING_ENABLED', '1').strip().lower() not in ('0', 'false', 'no', 'off')
USE_MACRO_MODEL = os.environ.get('USE_MACRO_MODEL', '0').strip().lower() in ('1', 'true', 'yes', 'on')

TEST_MODE = False
HOLD_DAYS = 20

REPORT_DIR = 'reports'
MODEL_DIR = 'models'
CONFIG_DIR = 'config'
LOG_DIR = os.environ.get('LOG_DIR', 'logs')
LOG_FILE = os.environ.get('LOG_FILE', os.path.join(LOG_DIR, 'quant_system.log'))

MY_TW_COVERAGE_PATH = os.environ.get('MY_TW_COVERAGE_PATH', './My-TW-Coverage')

MACRO_MODEL_PATH = os.path.join(MODEL_DIR, 'macro_rf_model.pkl')
PARAMS_FILE_PATH = os.environ.get('SCORE_CONFIG_PATH', os.path.join(CONFIG_DIR, 'best_params.json'))

GIT_PULL_TIMEOUT = 30
FINANCIAL_UPDATE_TIMEOUT = 300
TELEGRAM_RETRY = 2
SCAN_PROGRESS_STEP = 100 

TECHNICAL_PRESCREEN_LIMIT = int(os.environ.get('TECHNICAL_PRESCREEN_LIMIT', '60'))
FINAL_TOP_N = 10
WEIGHT_TECH = 0.50
WEIGHT_FUND = 0.35
WEIGHT_CHIP = 0.15

US_MIN_AVG_VOLUME = int(os.environ.get('US_MIN_AVG_VOLUME', '3000000'))
US_POOL_MAX_CANDIDATES = int(os.environ.get('US_POOL_MAX_CANDIDATES', '240'))
US_POOL_MAX_RESULTS = int(os.environ.get('US_POOL_MAX_RESULTS', '180'))
US_POOL_CACHE_TTL_HOURS = float(os.environ.get('US_POOL_CACHE_TTL_HOURS', '12'))
US_POOL_CACHE_PATH = os.path.join(CONFIG_DIR, 'us_stock_pool_cache.json')

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOGGER = logging.getLogger('quant_system')
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    log_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding='utf-8')
    log_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(message)s'))
    LOGGER.addHandler(log_handler)

def now_str(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    print(f'[{now_str()}] {msg}', flush=True)
    try:
        LOGGER.info(str(msg))
    except Exception:
        pass

def log_exception(prefix, exc):
    log(f'{prefix}: {sanitize_telegram_error(exc)}')
    tb = traceback.format_exc()
    print(tb, flush=True)
    try:
        LOGGER.error('%s\n%s', prefix, tb)
    except Exception:
        pass

if not TELEGRAM_TOKEN or not CHAT_ID:
    log("[Telegram] Bot disabled: TELEGRAM_TOKEN or TELEGRAM_CHAT_ID is missing.")
elif not validate_telegram_token(TELEGRAM_TOKEN):
    log("[Telegram] Bot disabled: TELEGRAM_TOKEN format is invalid.")
    TELEGRAM_TOKEN = ''

bot = create_telegram_bot(TELEGRAM_TOKEN, log=log) if TELEGRAM_TOKEN and CHAT_ID else None

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

def start_telegram_polling():
    if not bot:
        log('[Telegram] Bot disabled because TELEGRAM_TOKEN/CHAT_ID is missing or invalid.')
        return
    if not TELEGRAM_POLLING_ENABLED:
        log('[Telegram] Polling disabled by TELEGRAM_POLLING_ENABLED=0. Research/backtest commands can still run.')
        return

    try:
        bot.delete_webhook(drop_pending_updates=False)
        bot.get_updates(timeout=1, limit=1)
    except Exception as e:
        error_code = getattr(e, 'error_code', None)
        description = str(e)
        if error_code == 409 or 'Conflict' in description or 'getUpdates' in description:
            log('[Telegram-409] Another process is already polling this bot token. Stop the other bot/container, or set TELEGRAM_POLLING_ENABLED=0 on this machine.')
            return
        log(f'[Telegram-WARN] Polling preflight failed: {e}')

    log('[Telegram] Polling started.')
    bot.infinity_polling(timeout=60, long_polling_timeout=30)

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
def check_market_status(region='TW'):
    log(f'[MARKET] 正在評估 {region} 大盤系統風險...')
    if USE_MACRO_MODEL and region == 'TW' and os.path.exists(MACRO_MODEL_PATH):
        try:
            rf_model = joblib.load(MACRO_MODEL_PATH)
            mock_today_data = pd.DataFrame([[5000, 110, 2, 1000]], columns=['Foreign_Fut', 'PCR_Ratio', 'Retail_Sentiment', 'Top_10_Traders'])
            is_bull = rf_model.predict(mock_today_data)[0]
            if is_bull == 1: return 'offensive', 0.65
            else: return 'defensive', -0.45
        except Exception as e: log(f"[MARKET-WARN] 載入 RF 模型失敗: {e}")

    try:
        index_ticker = '^TWII' if region == 'TW' else '^GSPC'
        df = data_sources.download_yfinance(index_ticker, period='18mo', auto_adjust=True)
        if df.empty: return 'defensive', -0.1
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df['MA20'] = ta.trend.sma_indicator(df['Close'], 20)
        df['MA50'] = ta.trend.sma_indicator(df['Close'], 50)
        df['MA200'] = ta.trend.sma_indicator(df['Close'], 200)
        latest = df.iloc[-1]

        close = safe_float(latest.get('Close'), 0) or 0
        ma20 = safe_float(latest.get('MA20'))
        ma50 = safe_float(latest.get('MA50'))
        ma200 = safe_float(latest.get('MA200'))
        ma200_prev = safe_float(df['MA200'].iloc[-21]) if len(df) > 220 else None
        ret20 = close / df['Close'].iloc[-21] - 1 if len(df) > 21 and df['Close'].iloc[-21] else 0
        high60 = df['Close'].tail(60).max()
        drawdown60 = close / high60 - 1 if high60 else 0

        score = 0.0
        score += 0.18 if ma20 is not None and close > ma20 else -0.18
        score += 0.22 if ma50 is not None and close > ma50 else -0.22
        score += 0.28 if ma200 is not None and close > ma200 else -0.28
        score += 0.12 if ma50 is not None and ma200 is not None and ma50 > ma200 else -0.12
        score += 0.10 if ma200 is not None and ma200_prev is not None and ma200 > ma200_prev else -0.10
        score += 0.10 if ret20 > 0 else -0.10
        if drawdown60 < -0.10: score -= 0.18
        elif drawdown60 > -0.04: score += 0.08

        market_mode = 'offensive' if score >= 0.25 else 'defensive'
        macro_score = max(-1.0, min(1.0, score))
        log(f'[MARKET] {index_ticker} score={macro_score:.2f}, 20d={ret20*100:.1f}%, 60dDD={drawdown60*100:.1f}% => {market_mode}')
        return market_mode, macro_score
    except Exception as e:
        log_exception('[MARKET-ERROR]', e)
        return 'defensive', -0.1

def create_macro_dashboard_image(market_mode, macro_score, output_path, region='TW'):
    mode_text = "🟢 多方輪動 (Offensive)" if market_mode == 'offensive' else "🔴 崩盤避險 (Defensive)"
    mode_color = "#16a34a" if market_mode == 'offensive' else "#dc2626"
    title_text = "台股籌碼四維趨勢報告" if region == 'TW' else "美股市場廣度趨勢報告"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #f3f4f6; margin: 0; padding: 20px; width: 1100px; }}
            .dashboard {{ background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            .header {{ display: flex; justify-content: space-between; border-bottom: 2px solid #e5e7eb; padding-bottom: 15px; margin-bottom: 20px; }}
            .title {{ font-size: 24px; font-weight: bold; color: #1f2937; margin: 0; }}
            .subtitle {{ font-size: 14px; color: #6b7280; margin-top: 5px; }}
            .status-badge {{ background: {mode_color}20; color: {mode_color}; padding: 8px 16px; border-radius: 8px; font-weight: bold; font-size: 18px; border: 1px solid {mode_color}; }}
            .content-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: center; }}
            th {{ background: #f8fafc; padding: 10px; border-bottom: 2px solid #e5e7eb; color: #4b5563; }}
            td {{ padding: 10px; border-bottom: 1px solid #e5e7eb; color: #1f2937; }}
            .neg {{ color: #dc2626; }}
            .pos {{ color: #16a34a; }}
            .chart-container {{ height: 300px; width: 100%; margin-top: 20px; }}
            .summary-box {{ background: #f8fafc; border-radius: 8px; padding: 15px; border-left: 4px solid {mode_color}; }}
        </style>
    </head>
    <body>
        <div class="dashboard" id="capture-area">
            <div class="header">
                <div>
                    <h1 class="title">📊 {title_text}</h1>
                    <div class="subtitle">AI 隨機森林預測引擎 | 產生時間: {now_str()}</div>
                </div>
                <div class="status-badge">
                    狀態: {mode_text} (綜合評分: {macro_score:.2f})
                </div>
            </div>
            <div class="content-grid">
                <div>
                    <table>
                        <tr><th>日期</th><th>大盤動能</th><th>VIX 恐慌</th><th>散戶多空</th><th>法人現貨</th><th>綜合分數</th></tr>
                        <tr><td>今日</td><td class="pos">+2.45%</td><td class="pos">14.5</td><td class="pos">偏空</td><td class="neg">觀望</td><td style="font-weight:bold; color:{mode_color}">{macro_score:.2f}</td></tr>
                        <tr><td>T-1</td><td class="neg">-1.50%</td><td>15.2</td><td class="neg">偏多</td><td class="neg">賣超</td><td>-0.12</td></tr>
                        <tr><td>T-2</td><td class="neg">-2.20%</td><td class="neg">18.4</td><td class="neg">極多</td><td class="neg">大賣</td><td class="neg">-0.45</td></tr>
                    </table>
                    <div class="chart-container"><canvas id="trendChart"></canvas></div>
                </div>
                <div>
                    <div class="summary-box">
                        <h3 style="margin-top:0; color:#1f2937;">📝 系統判定與行動指南</h3>
                        <p style="font-size:14px; color:#4b5563; line-height:1.6;">
                            <b>模型解析：</b><br>
                            根據模型推算，目前廣度與心理指標呈現 <b>{mode_text.split(' ')[1]}</b>。<br><br>
                            <b>自動因應動作：</b><br>
                            {"已開啟『防守避險引擎』，雷達優先掃描避險 ETF (如美債、反向或低波)，嚴格限縮乖離率。" if market_mode == 'defensive' else "處於『攻擊引擎』，資金偏多操作，雷達專注強勢突破與動能發散股。"}
                        </p>
                    </div>
                    <div style="height: 220px; width: 100%; margin-top: 20px;"><canvas id="radarChart"></canvas></div>
                </div>
            </div>
        </div>
        <script>
            const ctxLine = document.getElementById('trendChart').getContext('2d');
            new Chart(ctxLine, {{
                type: 'bar',
                data: {{
                    labels: ['T-2', 'T-1', '今日'],
                    datasets: [{{
                        label: '綜合分數',
                        data: [-0.45, -0.12, {macro_score}],
                        backgroundColor: ctx => ctx.raw > 0 ? 'rgba(22, 163, 74, 0.5)' : 'rgba(220, 38, 38, 0.5)',
                        borderColor: ctx => ctx.raw > 0 ? 'rgb(22, 163, 74)' : 'rgb(220, 38, 38)',
                        borderWidth: 1
                    }}]
                }},
                options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }} }}
            }});
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            new Chart(ctxRadar, {{
                type: 'radar',
                data: {{
                    labels: ['廣度', '波動率', '散戶', '大戶'],
                    datasets: [{{ label: '目前位階', data: [60, 55, 30, 45], backgroundColor: '{mode_color}30', borderColor: '{mode_color}', pointBackgroundColor: '{mode_color}' }}]
                }},
                options: {{ responsive: true, maintainAspectRatio: false, scales: {{ r: {{ min: 0, max: 100 }} }} }}
            }});
        </script>
    </body>
    </html>
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html_content)
            page.wait_for_timeout(1500)
            element = page.locator("#capture-area")
            element.screenshot(path=output_path, omit_background=True)
            browser.close()
            return output_path
    except Exception as e:
        log_exception(f'[PLOT-ERROR] 大盤儀表板生成失敗', e)
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
    if not ticker.endswith(('.TW', '.TWO')):
        ticker = ticker.replace('.', '-')
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

def configured_scan_universe(region):
    raw = os.environ.get(f'SCAN_UNIVERSE_{region.upper()}', '').strip()
    if not raw:
        return []
    tickers = []
    for item in re.split(r'[,;\s]+', raw):
        if not item.strip():
            continue
        ticker = normalize_ticker(item)
        if region.upper() == 'US':
            ticker = ticker.replace('.US', '').replace('.', '-')
        tickers.append(ticker)
    return list(dict.fromkeys(tickers))

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
    try:
        return data_sources.fetch_goodinfo_pages(ticker_num)
    except Exception as e:
        log(f"[Goodinfo-WARN] fetch failed for {ticker_num}: {e}")
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

def merge_financial_snapshot(ticker_full, md_text, yf_info=None):
    is_us = is_us_ticker(ticker_full)
    if is_us:
        teps = safe_float(yf_info.get('trailingEps')) if yf_info else None
        rg = safe_float(yf_info.get('revenueGrowth')) if yf_info else None
        if rg is not None: rg = rg * 100
        institutional = safe_float(yf_info.get('heldPercentInstitutions')) if yf_info else None
        short_float = safe_float(yf_info.get('shortPercentOfFloat')) if yf_info else None
        profit_margin = safe_float(yf_info.get('profitMargins')) if yf_info else None
        earnings_growth = safe_float(yf_info.get('earningsGrowth')) if yf_info else None
        avg_vol = safe_float(yf_info.get('averageVolume')) if yf_info else None
        avg_vol10 = safe_float(yf_info.get('averageVolume10days')) if yf_info else None
        latest_vol = safe_float(yf_info.get('volume')) if yf_info else None
        short_ratio = safe_float(yf_info.get('shortRatio')) if yf_info else None
        market_cap = safe_float(yf_info.get('marketCap')) if yf_info else None

        institutional_pct = institutional * 100 if institutional is not None else None
        short_float_pct = short_float * 100 if short_float is not None else None
        profit_margin_pct = profit_margin * 100 if profit_margin is not None else None
        earnings_growth_pct = earnings_growth * 100 if earnings_growth is not None else None

        chip_summary_parts = []
        if institutional_pct is not None: chip_summary_parts.append(f'機構持股 {institutional_pct:.1f}%')
        if short_float_pct is not None: chip_summary_parts.append(f'空單/流通股 {short_float_pct:.1f}%')
        if avg_vol is not None: chip_summary_parts.append(f'三月均量 {avg_vol/1_000_000:.1f}M')
        chips_summary = '；'.join(chip_summary_parts) if chip_summary_parts else '美股籌碼資料不足，暫以流動性與空單壓力評估'

        return {
            'single_month_revenue': None, 'single_month_mom': None, 'single_month_yoy': rg,
            'eps_latest_quarter': None, 'eps_ttm': teps,
            'profit_margin_pct': profit_margin_pct, 'earnings_growth_pct': earnings_growth_pct,
            'market_cap': market_cap,
            'chips_summary': chips_summary,
            'institutional_ownership_pct': institutional_pct,
            'short_percent_float': short_float_pct,
            'short_ratio': short_ratio,
            'avg_volume_3m': avg_vol,
            'avg_volume_10d': avg_vol10,
            'latest_volume': latest_vol,
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
        df = data_sources.fetch_finmind_institutional_investors(
            dl,
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
        result['chips_summary'] = f'FinMind 籌碼讀取失敗，已略過本次籌碼資料'
        log(f"[FinMind-WARN] {stock_id}: {e}")

    return result

def merge_finmind_chip_into_snapshot(fin_data, chip_data):
    if not chip_data: return fin_data
    merged = dict(fin_data)
    
    # 🌟 關鍵修復：強制讓系統把 FinMind 的狀態文字印出來
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
def _load_cached_us_pool():
    try:
        if not os.path.exists(US_POOL_CACHE_PATH): return None
        with open(US_POOL_CACHE_PATH, 'r', encoding='utf-8') as f: payload = json.load(f)
        age_hours = (time.time() - payload.get('created_at', 0)) / 3600
        if age_hours <= US_POOL_CACHE_TTL_HOURS and payload.get('tickers'):
            log(f"[US-POOL] 使用快取股票池 {len(payload['tickers'])} 檔，快取年齡 {age_hours:.1f}h")
            return payload['tickers']
    except Exception as e:
        log(f"[US-POOL-WARN] 讀取快取失敗: {e}")
    return None

def _save_cached_us_pool(tickers):
    try:
        with open(US_POOL_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump({'created_at': time.time(), 'tickers': tickers}, f, indent=2)
    except Exception as e:
        log(f"[US-POOL-WARN] 寫入快取失敗: {e}")

def _parse_market_cap(value):
    if value is None: return 0
    text = str(value).replace('$', '').replace(',', '').strip()
    if not text or text in ('N/A', 'nan', '--'): return 0
    multiplier = 1
    suffix = text[-1].upper()
    if suffix == 'T': multiplier = 1_000_000_000_000; text = text[:-1]
    elif suffix == 'B': multiplier = 1_000_000_000; text = text[:-1]
    elif suffix == 'M': multiplier = 1_000_000; text = text[:-1]
    return safe_float(text, 0) * multiplier

def _normalize_us_symbol(symbol):
    s = str(symbol).strip().upper()
    if not s or '^' in s or '$' in s: return None
    s = s.replace('/', '-')
    if len(s) > 8 or any(x in s for x in ['.W', '-WT', '-WS', '-U', '-R']): return None
    return s

def get_nasdaq_screener_symbols(limit=US_POOL_MAX_CANDIDATES):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json,text/plain,*/*',
        'Origin': 'https://www.nasdaq.com',
        'Referer': 'https://www.nasdaq.com/market-activity/stocks/screener',
    }
    symbols = []
    for exchange in ['nasdaq', 'nyse', 'amex']:
        try:
            url = f'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange={exchange}'
            data = data_sources.get_json(
                url,
                headers=headers,
                namespace='nasdaq_screener',
                cache_key=exchange,
                ttl_hours=24,
            )
            rows = data.get('data', {}).get('table', {}).get('rows', [])
            rows.sort(key=lambda row: _parse_market_cap(row.get('marketCap')), reverse=True)
            for row in rows:
                symbol = _normalize_us_symbol(row.get('symbol'))
                if symbol: symbols.append(symbol)
                if len(symbols) >= limit: break
        except Exception as e:
            log(f"[US-POOL-WARN] Nasdaq screener {exchange} failed: {e}")
        if len(symbols) >= limit: break
    return list(dict.fromkeys(symbols))[:limit]

def passes_us_liquidity_and_fundamental_filter(ticker, min_avg_volume=US_MIN_AVG_VOLUME):
    try:
        df = data_sources.download_yfinance(ticker, period='3mo', auto_adjust=True)
        if df.empty or len(df) < 25: return False
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        avg5 = float(df['Volume'].tail(5).mean())
        avg20 = float(df['Volume'].tail(20).mean())
        if avg5 < min_avg_volume or avg20 < min_avg_volume:
            return False

        info = data_sources.get_yahoo_info(ticker)
        quote_type = str(info.get('quoteType', '')).upper()
        if quote_type and quote_type not in ('EQUITY', 'ETF'):
            return False
        if quote_type == 'ETF':
            return avg20 >= min_avg_volume * 2

        market_cap = safe_float(info.get('marketCap'), 0) or 0
        eps = safe_float(info.get('trailingEps')) or safe_float(info.get('forwardEps'))
        revenue_growth = safe_float(info.get('revenueGrowth'))
        earnings_growth = safe_float(info.get('earningsGrowth'))
        profit_margin = safe_float(info.get('profitMargins'))

        if market_cap < 1_000_000_000: return False
        profitable = eps is not None and eps > 0
        quality = (
            (revenue_growth is not None and revenue_growth > 0.03) or
            (earnings_growth is not None and earnings_growth > 0) or
            (profit_margin is not None and profit_margin > 0.05)
        )
        return profitable and quality
    except Exception:
        return False

def get_us_stock_pool():
    configured = configured_scan_universe('US')
    if configured:
        log(f"[US-POOL] 使用 SCAN_UNIVERSE_US 設定 {len(configured)} 檔")
        return configured

    sqlite_cached = data_sources.cache.get('stock_pool', 'US')
    if sqlite_cached:
        log(f"[US-POOL] 使用 SQLite 快取股票池 {len(sqlite_cached)} 檔")
        return sqlite_cached

    cached = _load_cached_us_pool()
    if cached:
        data_sources.cache.set('stock_pool', 'US', cached, US_POOL_CACHE_TTL_HOURS)
        return cached

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        html = data_sources.get_text(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            headers=headers,
            namespace='wikipedia',
            cache_key='sp500_components',
            ttl_hours=24,
        )
        table = pd.read_html(StringIO(html))
        sp500 = [_normalize_us_symbol(s) for s in table[0]['Symbol'].tolist()]
        sp500 = [s for s in sp500 if s]
    except Exception as e:
        log_exception("[US-POOL-ERROR]", e)
        sp500 = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META', 'AMD', 'BRK-B', 'JPM']

    candidates = list(dict.fromkeys(sp500 + get_nasdaq_screener_symbols()))
    candidates = candidates[:US_POOL_MAX_CANDIDATES]
    log(f"[US-POOL] 候選 {len(candidates)} 檔，套用 5/20日均量>{US_MIN_AVG_VOLUME/1_000_000:.1f}M + 基本面初篩...")
    filtered = []
    for idx, ticker in enumerate(candidates, start=1):
        if passes_us_liquidity_and_fundamental_filter(ticker):
            filtered.append(ticker)
        if idx % 25 == 0:
            log(f"[US-POOL] 已檢查 {idx}/{len(candidates)}，通過 {len(filtered)} 檔")
        if len(filtered) >= US_POOL_MAX_RESULTS:
            break

    if not filtered:
        filtered = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META', 'AMD', 'BRK-B', 'JPM']
    data_sources.cache.set('stock_pool', 'US', filtered, US_POOL_CACHE_TTL_HOURS)
    _save_cached_us_pool(filtered)
    return filtered

def get_tw_stock_pool(mode='offensive'):
    configured = configured_scan_universe('TW')
    if configured:
        log(f"[TW-POOL] 使用 SCAN_UNIVERSE_TW 設定 {len(configured)} 檔")
        return configured

    sqlite_key = f"TW:{mode}"
    sqlite_cached = data_sources.cache.get('stock_pool', sqlite_key)
    if sqlite_cached:
        log(f"[TW-POOL] 使用 SQLite 快取股票池 {len(sqlite_cached)} 檔")
        return sqlite_cached

    tickers = []
    if mode == 'defensive': tickers.extend(get_defensive_etf_pool('TW'))
    for m in [2, 4]:
        try:
            html = data_sources.get_text(
                f'https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}',
                namespace='twse_isin',
                cache_key=f'mode_{m}',
                ttl_hours=24,
            )
            df = pd.read_html(StringIO(html))[0]
            df.columns = df.iloc[0]
            valid_codes = df.iloc[1:][df.iloc[1:]['CFICode'] == 'ESVUFR']['有價證券代號及名稱'].str.extract(r'^([0-9]{4})\b')[0].dropna()
            tickers.extend((valid_codes + ('.TW' if m == 2 else '.TWO')).tolist())
        except Exception: pass
    unique_tickers = list(set(tickers))
    data_sources.cache.set('stock_pool', sqlite_key, unique_tickers, 24)
    return unique_tickers

def download_stock_df(ticker):
    ticker = normalize_ticker(ticker)
    df = data_sources.download_yfinance(ticker, period='5y', auto_adjust=True)
    if df.empty and ticker.endswith('.TW'):
        alt = ticker.replace('.TW', '.TWO')
        df = data_sources.download_yfinance(alt, period='5y', auto_adjust=True)
        if not df.empty: ticker = alt
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    return ticker, df

def compute_indicators(df):
    df = df.copy()
    for win in [5, 20, 50, 60, 150, 200, 240]: df[f'MA{win}'] = ta.trend.sma_indicator(df['Close'], win)
    df['High52W'] = df['High'].rolling(250).max()
    df['Low52W'] = df['Low'].rolling(250).min()
    df['RSI'] = ta.momentum.rsi(df['Close'], 14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Osc'] = macd.macd(), macd.macd_signal(), macd.macd_diff()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BBMid'] = bb.bollinger_mavg()
    df['BBUpper'] = bb.bollinger_hband()
    df['BBLower'] = bb.bollinger_lband()
    df['BBWidth'] = np.where(df['BBMid'] != 0, (df['BBUpper'] - df['BBLower']) / df['BBMid'], np.nan)
    for ma in [5, 20, 60, 240]:
        df[f'BIAS{ma}'] = np.where(df[f'MA{ma}'] != 0, (df['Close'] - df[f'MA{ma}']) / df[f'MA{ma}'] * 100, np.nan)
    df['volume_avg20'] = df['Volume'].rolling(20, min_periods=10).mean()
    df = add_cta_features(df)
    df = add_pattern_features(df)
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
    c_engulfing_5d = bool(latest.get('engulfing_5d', 0))
    c_box_breakout = bool(latest.get('close_box_breakout', 0))
    c_bb_squeeze_breakout = bool(latest.get('bb_squeeze_breakout', 0))
    c_bb_momentum_breakout = bool(latest.get('bb_momentum_breakout', 0))
    triangle_score = safe_float(latest.get('triangle_contraction_score'), 0.0) or 0.0
    inverse_hs_score = safe_float(latest.get('inverse_head_shoulders_score'), 0.0) or 0.0
    c_triangle = triangle_score >= 0.65
    c_inverse_hs = inverse_hs_score >= 0.65

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
        if c_box_breakout: score += 15
        if c_engulfing_5d: score += 10
        if c_bb_squeeze_breakout: score += 12
        if c_bb_momentum_breakout: score += 8
        if c_triangle: score += 5
        if c_inverse_hs: score += 5
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
    bb_score = safe_float(latest.get('bb_score'), 0.0) or 0.0
    bb_width_pctile = safe_float(latest.get('bb_width_pctile'))
    strategy_tags = row_strategy_tags(latest)
    if vcp_score >= 0.65: strategy_tags.append('VCP量縮收斂')
    if bb_breakout: strategy_tags.append('BB帶量突破')
    if c_triangle: strategy_tags.append('收斂三角')
    if c_inverse_hs: strategy_tags.append('頭肩底雛形')

    return {
        'df': df, 'weekly': weekly, 'monthly': monthly, 'technical_score': technical_score, 'latest': latest, 'mode': market_mode,
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
            'vcp_score': vcp_score, 'vcp_pivot': safe_float(latest.get('vcp_pivot')), 'bb_width_pctile': bb_width_pctile,
            'bb_width': safe_float(latest.get('BBWidth')), 'bb_breakout': bb_breakout, 'bb_score': bb_score,
            'box_width': safe_float(latest.get('Box_Width')), 'close_box_high': safe_float(latest.get('Close_Max_20_Prior')),
            'cta_score': safe_float(latest.get('cta_score'), 0.0), 'triangle_score': triangle_score,
            'inverse_head_shoulders_score': inverse_hs_score
        }
    }

def calc_fundamental_score(f, is_us=False):
    return shared_calc_fundamental_score(f, is_us)

def calc_chip_score(f, is_us=False):
    return shared_calc_chip_score(f, is_us)

def _scale_unit_score(value, default=0.0):
    score = safe_float(value, default)
    if score is None:
        score = default
    score = float(score)
    if score <= 1.5:
        score *= 100.0
    return max(0.0, min(score, 100.0))

def technical_model_scores(tech_pack):
    metrics = (tech_pack or {}).get('metrics', {})
    latest = (tech_pack or {}).get('latest')
    bb_score = metrics.get('bb_score')
    if bb_score is None and latest is not None:
        bb_score = latest.get('bb_score', latest.get('bb_breakout', 0.0))

    cta_score = metrics.get('cta_score', 0.0)
    triangle_score = metrics.get('triangle_score', 0.0)
    inverse_hs_score = metrics.get('inverse_head_shoulders_score', 0.0)
    pattern_score = max(_scale_unit_score(triangle_score), _scale_unit_score(inverse_hs_score))

    return {
        'minervini': _scale_unit_score((tech_pack or {}).get('technical_score', 0.0), 0.0),
        'vcp': _scale_unit_score(metrics.get('vcp_score', 0.0), 0.0),
        'bb': _scale_unit_score(bb_score, 0.0),
        'cta': _scale_unit_score(cta_score, 0.0),
        'pattern': pattern_score,
    }

def candidate_model_average(tech_pack):
    scores = technical_model_scores(tech_pack)
    return sum(scores.values()) / max(1, len(scores))

def final_total_score(t, f, c, is_us=False, tech_pack=None):
    model_scores = technical_model_scores(tech_pack)
    tech_w = max(0.0, float(SYS_PARAMS.get('tech_weight', WEIGHT_TECH)))
    fund_w = max(0.0, float(SYS_PARAMS.get('fund_weight', WEIGHT_FUND)))
    chip_w = max(0.0, float(SYS_PARAMS.get('chip_weight', WEIGHT_CHIP)))
    vcp_w = max(0.0, float(SYS_PARAMS.get('vcp_weight', 0.15)))
    bb_w = max(0.0, float(SYS_PARAMS.get('bb_weight', 0.10)))
    cta_w = max(0.0, float(SYS_PARAMS.get('cta_weight', 0.08)))
    pattern_w = max(0.0, float(SYS_PARAMS.get('pattern_weight', 0.07)))
    total_w = tech_w + fund_w + chip_w + vcp_w + bb_w + cta_w + pattern_w
    if total_w <= 0:
        return t * WEIGHT_TECH + f * WEIGHT_FUND + c * WEIGHT_CHIP
    return (
        t * tech_w +
        f * fund_w +
        c * chip_w +
        model_scores['vcp'] * vcp_w +
        model_scores['bb'] * bb_w +
        model_scores['cta'] * cta_w +
        model_scores['pattern'] * pattern_w
    ) / total_w

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

def create_strategy_card_image(ticker, close_price, ma5, ma20, high52w, hard_stop, output_path):
    entry_a_low, entry_a_high = ma20 * 0.98, ma20 * 1.02
    stop_a = entry_a_low * (1 - hard_stop)
    entry_b_low, entry_b_high = (ma5 + ma20) / 2, ma5 * 1.01
    stop_b = entry_b_low * (1 - hard_stop)
    entry_c_low, entry_c_high = high52w * 0.98, high52w * 1.02
    stop_c = entry_c_low * (1 - hard_stop)
    
    target_1 = high52w if high52w > close_price else close_price * 1.15
    target_2 = target_1 * 1.15
    def calc_rr(entry, stop, target): return (target - entry) / (entry - stop) if entry > stop else 0

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"><style>
        body {{ font-family: sans-serif; padding: 20px; }}
        .card-container {{ width: 380px; background: white; border-radius: 16px; padding: 20px; border: 1px solid #e5e7eb; }}
        .header {{ border-bottom: 1px solid #e5e7eb; padding-bottom: 12px; margin-bottom: 16px; }}
        .title {{ font-size: 20px; font-weight: bold; margin: 0; }}
        .price-info {{ font-size: 16px; color: #dc2626; font-weight: 600; margin: 5px 0 0 0; }}
        .box {{ border-radius: 12px; padding: 12px; margin-bottom: 12px; border-left: 6px solid; }}
        .box-a {{ background: #fffbeb; border-color: #facc15; }}
        .box-a .tag {{ color: #ca8a04; font-weight: bold; font-size: 16px; }}
        .box-b {{ background: #f0fdf4; border-color: #4ade80; }}
        .box-b .tag {{ color: #16a34a; font-weight: bold; font-size: 16px; }}
        .box-c {{ background: #eff6ff; border-color: #60a5fa; }}
        .box-c .tag {{ color: #2563eb; font-weight: bold; font-size: 16px; }}
        .price-range {{ font-size: 18px; font-weight: bold; margin-left: 8px; }}
        .warning {{ font-size: 13px; color: #dc2626; font-weight: bold; margin-top: 6px; }}
        .target-box {{ border-radius: 10px; padding: 12px; margin-bottom: 10px; }}
        .t1 {{ background: #dcfce7; color: #166534; }}
        .t2 {{ background: #f3e8ff; color: #6b21a8; }}
        .target-price {{ font-size: 22px; font-weight: bold; margin-left: 8px; }}
    </style></head>
    <body>
        <div class="card-container" id="capture-area">
            <div class="header">
                <h2 class="title">📊 {ticker} 操作計畫</h2>
                <p class="price-info">最新收盤: {close_price:.2f}</p>
            </div>
            <div style="font-weight: bold; margin-bottom: 8px;">📍 進場區</div>
            <div class="box box-a">
                <div><span class="tag">A 低接</span><span class="price-range">{entry_a_low:.1f} - {entry_a_high:.1f}</span></div>
                <div class="warning">❗️ 停損 {stop_a:.1f} (-{hard_stop*100:.1f}%) | 風報 1:{calc_rr(entry_a_high, stop_a, target_1):.1f}</div>
            </div>
            <div class="box box-b">
                <div><span class="tag">B 回穩</span><span class="price-range">{entry_b_low:.1f} - {entry_b_high:.1f}</span></div>
                <div class="warning">❗️ 停損 {stop_b:.1f} (-{hard_stop*100:.1f}%) | 風報 1:{calc_rr(entry_b_high, stop_b, target_1):.1f}</div>
            </div>
            <div class="box box-c">
                <div><span class="tag">C 突破</span><span class="price-range">{entry_c_low:.1f} - {entry_c_high:.1f}</span></div>
                <div class="warning">❗️ 停損 {stop_c:.1f} (-{hard_stop*100:.1f}%) | 風報 1:{calc_rr(entry_c_high, stop_c, target_2):.1f}</div>
            </div>
            <div style="font-weight: bold; margin: 20px 0 10px 0;">🎯 壓力區 (非固定止盈)</div>
            <div class="target-box t1"><strong>壓力 1</strong><span class="target-price">{target_1:.1f}</span></div>
            <div class="target-box t2"><strong>壓力 2</strong><span class="target-price">{target_2:.1f}</span></div>
        </div>
    </body>
    </html>
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html_content)
            element = page.locator("#capture-area")
            element.screenshot(path=output_path, omit_background=True)
            browser.close()
            return output_path
    except Exception as e: log_exception('[PLOT-ERROR]', e); return None

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
    opt_hard_stop = SYS_PARAMS.get('hard_stop', 0.08)
    opt_ma_val = safe_float(m.get('opt_ma')) or close_val

    strategy_card_path = os.path.join(REPORT_DIR, f'{ticker}_strategy.png')
    create_strategy_card_image(ticker, close_val, safe_float(m.get("ma5")) or close_val, opt_ma_val, safe_float(latest.get("High52W")) or (close_val * 1.1), opt_hard_stop, strategy_card_path)

    report = ''
    if rank is not None: report += f'🏆 **排名 #{rank}**\n'
    
    is_etf = ('00' in ticker) or (ticker in get_defensive_etf_pool('US'))
    mode_text = '🛡️ ETF 防守避風港' if is_etf else ('🔥 攻擊型飆股' if mode == 'offensive' else '🛡️ RS相對強勢')

    report += f'📊 **【量化診斷：{ticker}】** ({mode_text})\n'
    
    if is_us:
        report += f'💰 最新收盤：`{latest["Close"]:.2f}` _({m["latest_date"]})_\n'
        report += f'🧮 總分：`{total_score:.1f}` | 技術：`{tech_score:.1f}` | 基本：`{fund_score:.1f}` | 籌碼：`{chip_score:.1f}`\n'
    else:
        report += f'💰 最新收盤：`{latest["Close"]:.2f}` _(資料日期: {m["latest_date"]})_\n'
        report += f'🧮 總分：`{total_score:.1f}` | 技術：`{tech_score:.1f}` | 基本：`{fund_score:.1f}` | 籌碼：`{chip_score:.1f}`\n'
        
    report += '------------------------\n'
    report += f'🏢 **產業:** {profile_info["industry"]}\n_{profile_info["profile"]}_\n'
    all_tags = list(dict.fromkeys(list(tech_pack.get('strategy_tags', [])) + list(tech_pack.get('sector_tags', []))))
    if all_tags:
        report += f'🏷️ **策略標籤:** `{" / ".join(all_tags[:6])}`\n'
    sector_info = tech_pack.get('sector_info') or {}
    if sector_info:
        report += f'🌐 **族群強度:** `{safe_num_str(sector_info.get("sector_strength_score"), 1)}` | 同族樣本 `{sector_info.get("count", 0)}` | 平均分 `{safe_num_str(sector_info.get("avg_score"), 1)}`\n'
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
        if c.get('c_box_breakout'):
            report += '🔥 **【進階型態】觸發無雜訊 20 日收盤箱型帶量突破**\n'
        if c.get('c_engulfing_5d'):
            report += '🔥 **【進階型態】出現強勢五日陣吞噬**\n'
        if c.get('c_bb_squeeze_breakout'):
            report += '🔥 **【布林型態】布林壓縮後帶量突破上軌**\n'
        if c.get('c_bb_momentum_breakout'):
            report += '🔥 **【布林動能】突破上軌且 RSI/MACD 同步轉強**\n'
        if c.get('c_triangle_contraction'):
            report += '📐 **【型態學】偵測到收斂三角雛形**\n'
        if c.get('c_inverse_head_shoulders'):
            report += '📐 **【型態學】偵測到頭肩底雛形**\n'

    report += '\n📈 **技術數據面板：**\n'
    report += f'🔹 RSI(日)：`{safe_num_str(m["rsi"], 1)}`\n'
    report += f'🔹 MACD Hist 日/週/月：`{safe_num_str(m["macd_osc_d"], 3)}` / `{safe_num_str(m["macd_osc_w"], 3)}` / `{safe_num_str(m["macd_osc_m"], 3)}`\n'
    report += f'🔹 箱型寬度 / BB寬度：`{safe_pct_str((m.get("box_width") or 0) * 100 if m.get("box_width") is not None else None)}` / `{safe_pct_str((m.get("bb_width") or 0) * 100 if m.get("bb_width") is not None else None)}`\n'
    report += f'🔹 VCP / CTA / 三角 / 頭肩底分數：`{safe_num_str(m.get("vcp_score"), 2)}` / `{safe_num_str(m.get("cta_score"), 2)}` / `{safe_num_str(m.get("triangle_score"), 2)}` / `{safe_num_str(m.get("inverse_head_shoulders_score"), 2)}`\n'
    report += f'🔹 5/20/60/240MA乖離率：`{safe_pct_str(m["bias5"])}` / `{safe_pct_str(m["bias20"])}` / `{safe_pct_str(m["bias60"])}` / `{safe_pct_str(m["bias240"])}`\n'

    if not is_etf:
        report += '------------------------\n'
        if is_us:
            report += '💹 **基本面 (Yahoo Finance)：**\n'
            report += f'🔸 近四季 EPS (TTM)：`{safe_num_str(fin_data.get("eps_ttm"))}`\n'
            report += f'🔸 營收成長 (Y/Y)：`{safe_pct_str(fin_data.get("single_month_yoy"))}`\n'
            report += f'🔸 獲利率 / 盈餘成長：`{safe_pct_str(fin_data.get("profit_margin_pct"))}` / `{safe_pct_str(fin_data.get("earnings_growth_pct"))}`\n'
        else:
            report += '💹 **基本面：**\n'
            report += f'🔸 最新一季 EPS：`{safe_num_str(fin_data.get("eps_latest_quarter"))}`\n🔸 近四季 EPS：`{safe_num_str(fin_data.get("eps_ttm"))}`\n'
            report += f'🔸 單月營收 Y/Y：`{safe_pct_str(fin_data.get("single_month_yoy"))}`\n🔸 月營收 M/M：`{safe_pct_str(fin_data.get("single_month_mom"))}`\n'

    report += '------------------------\n'
    if is_us:
        report += '🏦 **美股籌碼面：**\n'
        report += f'🔸 籌碼摘要：`{fin_data.get("chips_summary", "N/A")}`\n'
        report += f'🔸 機構持股 / 空單占流通股：`{safe_pct_str(fin_data.get("institutional_ownership_pct"))}` / `{safe_pct_str(fin_data.get("short_percent_float"))}`\n'
        report += f'🔸 Short Ratio：`{safe_num_str(fin_data.get("short_ratio"), 2)}`\n'
        report += f'🔸 成交量 最新/10日/3月均量：`{safe_num_str((fin_data.get("latest_volume") or 0) / 1_000_000, 1)}M` / `{safe_num_str((fin_data.get("avg_volume_10d") or 0) / 1_000_000, 1)}M` / `{safe_num_str((fin_data.get("avg_volume_3m") or 0) / 1_000_000, 1)}M`\n'
        srcs = ', '.join(fin_data.get('sources', [])) if fin_data.get('sources') else 'N/A'
        report += f'🔸 資料來源：`{srcs}`\n'
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

    return report, img_path, strategy_card_path

def analyze_stock(ticker, market_mode='offensive', silent=False):
    try:
        ticker, df = download_stock_df(ticker)
        if df.empty or len(df) < 250: return (None, None, None) if silent else ('❌ 找不到資料', None, None)
        tech_pack = evaluate_technical(df, market_mode)
        
        is_us = is_us_ticker(ticker)
        yf_info = data_sources.get_yahoo_info(ticker) if not TEST_MODE else {}
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
            prescreen.append({
                'ticker': tkr,
                'df': df,
                'tech_pack': tech_pack,
                'model_average': candidate_model_average(tech_pack),
            })

        except Exception as e:
            log_exception(f'[PRESCREEN-ERROR] {ticker}', e)
            continue
        
    prescreen.sort(
        key=lambda x: (x.get('model_average', 0.0), x['tech_pack']['technical_score']),
        reverse=True,
    )
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
            yf_info = data_sources.get_yahoo_info(ticker)
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
            model_scores = technical_model_scores(item['tech_pack'])
            ranked.append({
                'ticker': ticker,
                'tech_pack': item['tech_pack'],
                'fin_data': fin_data,
                'profile_info': profile_info,
                'model_scores': model_scores,
                'model_average': (
                    model_scores['minervini'] + model_scores['vcp'] + model_scores['bb'] +
                    model_scores['cta'] + model_scores['pattern'] + f_score + c_score
                ) / 7.0,
                'technical_score': t_score,
                'fundamental_score': f_score,
                'chip_score': c_score,
                'total_score': final_total_score(t_score, f_score, c_score, is_us, item['tech_pack'])
            })

        except Exception as e:
            log_exception(f'[RANK-ERROR] {ticker}', e)
            continue

    ranked.sort(key=lambda x: x['total_score'], reverse=True)
    ranked = annotate_sector_strength(ranked)
    return ranked

def top_ranked_by_model(ranked, model_key, limit=FINAL_TOP_N, require_signal=False):
    def model_value(item):
        if model_key == 'sector':
            return safe_float((item.get('sector_info') or {}).get('sector_strength_score'), 0.0) or 0.0
        return safe_float((item.get('model_scores') or {}).get(model_key), 0.0) or 0.0

    filtered = []
    for item in ranked:
        value = model_value(item)
        if require_signal and value <= 0:
            continue
        filtered.append(item)
    filtered.sort(key=lambda item: (model_value(item), item.get('total_score', 0.0)), reverse=True)
    return filtered[:limit]

def format_ranked_summary(title, ranked, score_label='總分', model_key=None):
    lines = [title]
    if not ranked:
        lines.append('本次沒有可排序標的。')
        return '\n'.join(lines)
    for i, item in enumerate(ranked[:FINAL_TOP_N], start=1):
        tags = ' / '.join((item.get('tech_pack', {}).get('strategy_tags') or [])[:2] + (item.get('sector_tags') or [])[:2])
        tag_text = f' | `{tags}`' if tags else ''
        if model_key == 'sector':
            score = safe_float((item.get('sector_info') or {}).get('sector_strength_score'), 0.0) or 0.0
        elif model_key:
            score = safe_float((item.get('model_scores') or {}).get(model_key), 0.0) or 0.0
        else:
            score = safe_float(item.get('total_score'), 0.0) or 0.0
        lines.append(f'{i}. `{item["ticker"]}` | {score_label} `{score:.1f}` | 加權 `{item.get("total_score", 0.0):.1f}` | 平均 `{item.get("model_average", 0.0):.1f}`{tag_text}')
    return '\n'.join(lines)

# ==========================================
# Main jobs and scheduler
# ==========================================
def run_market_scan_job(chat_id, requested_by_user=False, region='TW'):
    market_mode, macro_score = check_market_status(region)
    mode_msg = "🟢 **多方輪動：啟動 [攻擊型飆股引擎]**" if market_mode == 'offensive' else "🔴 **崩盤風險：啟動 [RS防守避險引擎 + ETF推薦]**"
    
    safe_send_message(chat_id, f'🔍 **{region} 市場量化雷達啟動中...**\n{mode_msg}', parse_mode='Markdown')
    macro_img_path = os.path.join(REPORT_DIR, f'{region}_macro_dashboard.png')
    dashboard_generated = create_macro_dashboard_image(market_mode, macro_score, macro_img_path, region)
    if dashboard_generated and os.path.exists(dashboard_generated):
        safe_send_photo(chat_id, dashboard_generated)
        time.sleep(2)
    
    if region == 'TW': update_my_tw_coverage(chat_id)
    all_ranked = scan_and_rank_market(chat_id, requested_by_user, market_mode, region)
    top_ranked = all_ranked[:FINAL_TOP_N]
    
    if not top_ranked:
        safe_send_message(chat_id, '☕ **掃描完畢**\n本次無達標股票。')
        return

    dashboard_path = os.path.join(REPORT_DIR, f'{region}_top10_dashboard.png')
    try:
        dashboard = render_top_ranked_dashboard(top_ranked, dashboard_path, region=region, market_mode=market_mode)
        if dashboard and os.path.exists(dashboard):
            safe_send_photo(chat_id, dashboard)
            time.sleep(2)
    except Exception as e:
        log_exception('[DASHBOARD-ERROR]', e)
        
    safe_send_message(
        chat_id,
        format_ranked_summary(f'🏆 **{region} 加權平均 Top 10 觀察清單**', top_ranked),
        parse_mode='Markdown',
    )
    time.sleep(2)

    if region == 'TW':
        tw_sections = [
            ('🧬 **TW VCP 形態 Top 10**', top_ranked_by_model(all_ranked, 'vcp'), 'VCP', 'vcp'),
            ('🔥 **TW 布林突破 Top 10**', top_ranked_by_model(all_ranked, 'bb'), '布林', 'bb'),
            ('🌐 **TW 族群連動 Top 10**', top_ranked_by_model(all_ranked, 'sector'), '族群', 'sector'),
        ]
        for title, items, label, key in tw_sections:
            safe_send_message(chat_id, format_ranked_summary(title, items, label, key), parse_mode='Markdown')
            time.sleep(1)
    
    for i, item in enumerate(top_ranked, start=1):
        try:
            tech_pack = dict(item['tech_pack'])
            tech_pack['sector_info'] = item.get('sector_info')
            tech_pack['sector_tags'] = item.get('sector_tags', [])
            report, img_path, strategy_img_path = build_stock_report(item['ticker'], tech_pack, item['fin_data'], item['profile_info'], rank=i)
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
        raw_text = (message.text or '').strip().upper().replace('多', '').replace('空', '')

        # Quietly ignore normal chat/noise. Only accept TW 4-digit symbols or
        # compact US tickers such as AAPL/NVDA. Commands are handled above.
        if not re.match(r'^([0-9]{4}|[A-Z]{1,5})$', raw_text):
            return

        ticker = raw_text
        safe_reply_to(message, f'⏳ 正在產生 `{ticker}` 報告...')
        
        region = 'US' if is_us_ticker(normalize_ticker(ticker)) else 'TW'
        current_mode, _ = check_market_status(region)
        report, img_path, strategy_img_path = analyze_stock(ticker, current_mode, silent=True)
        
        if img_path and os.path.exists(img_path): safe_send_photo(message.chat.id, img_path)
        if strategy_img_path and os.path.exists(strategy_img_path): safe_send_photo(message.chat.id, strategy_img_path)
        if report: safe_send_message(message.chat.id, report, parse_mode='Markdown')

def schedule_loop():
    schedule.every().day.at('16:30').do(start_scan_thread, chat_id=CHAT_ID, requested_by_user=False, region='TW')
    schedule.every().day.at('05:00').do(start_scan_thread, chat_id=CHAT_ID, requested_by_user=False, region='US')
    schedule.every().saturday.at("02:00").do(run_weekly_optimization)
    
    while True: schedule.run_pending(); time.sleep(1)

def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description='Stock Minervini Pro scanner and Telegram bot.')
    parser.add_argument('--scan', choices=['tw', 'us'], help='Run a one-shot market scan without Telegram polling.')
    parser.add_argument('--ticker', help='Run a one-shot single ticker report without Telegram polling.')
    parser.add_argument('--market-mode', choices=['auto', 'offensive', 'defensive'], default='auto')
    parser.add_argument('--no-bot', action='store_true', help='Do not start Telegram polling or scheduler.')
    parser.add_argument('--test-mode', action='store_true', help='Limit scan universe for fast smoke tests.')
    return parser.parse_args(argv)

def run_cli_command(args):
    global TEST_MODE
    if args.test_mode:
        TEST_MODE = True

    if args.ticker:
        ticker = normalize_ticker(args.ticker)
        region = 'US' if is_us_ticker(ticker) else 'TW'
        market_mode = args.market_mode
        if market_mode == 'auto':
            market_mode, macro_score = check_market_status(region)
        else:
            macro_score = 0.0
        print(f"[CLI] ticker={ticker} region={region} mode={market_mode} macro_score={macro_score:.2f}", flush=True)
        report, img_path, strategy_img_path = analyze_stock(ticker, market_mode)
        if report:
            print(report, flush=True)
        if img_path:
            print(f"[CLI] chart={img_path}", flush=True)
        if strategy_img_path:
            print(f"[CLI] strategy_card={strategy_img_path}", flush=True)
        return True

    if args.scan:
        region = args.scan.upper()
        market_mode = args.market_mode
        if market_mode == 'auto':
            market_mode, macro_score = check_market_status(region)
        else:
            macro_score = 0.0
        print(f"[CLI] scan={region} mode={market_mode} macro_score={macro_score:.2f}", flush=True)
        ranked = scan_and_rank_market(None, False, market_mode, region)
        if not ranked:
            print("[CLI] no qualified stocks", flush=True)
            return True
        for idx, item in enumerate(ranked[:FINAL_TOP_N], start=1):
            print(f"{idx}. {item['ticker']} total_score={item['total_score']:.1f}", flush=True)
        return True

    return args.no_bot

if __name__ == '__main__':
    cli_args = parse_cli_args()
    if run_cli_command(cli_args):
        log('CLI/no-bot command finished; Telegram polling not started.')
        sys.exit(0)

    log('🤖 Stock Minervini Pro (Cross-Border Edition) 啟動中...')
    threading.Thread(target=schedule_loop, daemon=True).start()
    start_telegram_polling()
