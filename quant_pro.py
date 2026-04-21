import os
import re
import time
import math
import json
import warnings
import threading
import subprocess
import traceback
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

# ==========================================
# ⚙️ 系統設定與路徑
# ==========================================
warnings.filterwarnings('ignore')

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK TC', 'Noto Sans CJK SC', 'Microsoft JhengHei',
    'PingFang TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans'
]

TELEGRAM_TOKEN = '8778080771:AAHvSIm9yYvOP2CKnVPnfhZLuP5Awmq9nHM'
CHAT_ID = '7255083299'

TEST_MODE = False
HOLD_DAYS = 20

REPORT_DIR = 'reports'
MODEL_DIR = 'models'
CONFIG_DIR = 'config'
MY_TW_COVERAGE_PATH = '/home/randal/My-TW-Coverage'

MACRO_MODEL_PATH = os.path.join(MODEL_DIR, 'macro_rf_model.pkl')
PARAMS_FILE_PATH = os.path.join(CONFIG_DIR, 'best_params.json')

GIT_PULL_TIMEOUT = 30
FINANCIAL_UPDATE_TIMEOUT = 300
TELEGRAM_RETRY = 2
SCAN_PROGRESS_STEP = 150

TECHNICAL_PRESCREEN_LIMIT = 20
FINAL_TOP_N = 10
WEIGHT_TECH = 0.50
WEIGHT_FUND = 0.35
WEIGHT_CHIP = 0.15

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

if TELEGRAM_TOKEN and TELEGRAM_TOKEN != '您的_BOT_TOKEN_貼在這裡':
    bot = telebot.TeleBot(TELEGRAM_TOKEN)
else:
    bot = None

# ==========================================
# 🧠 模組串接：載入最佳化參數
# ==========================================
def load_best_params():
    default_params = {'hard_stop': 0.08, 'ma_period': 20}
    if os.path.exists(PARAMS_FILE_PATH):
        try:
            with open(PARAMS_FILE_PATH, 'r') as f:
                params = json.load(f)
                return params
        except Exception as e:
            print(f"[CONFIG-WARN] 讀取參數失敗，使用預設值: {e}")
    return default_params

SYS_PARAMS = load_best_params()

# ==========================================
# 🧰 大盤監測器 (雙引擎核心 + 隨機森林)
# ==========================================
def check_market_status():
    print('[MARKET] 正在評估大盤系統風險...')
    
    if os.path.exists(MACRO_MODEL_PATH):
        try:
            rf_model = joblib.load(MACRO_MODEL_PATH)
            mock_today_data = pd.DataFrame([[5000, 110, 2, 1000]], columns=['Foreign_Fut', 'PCR_Ratio', 'Retail_Sentiment', 'Top_10_Traders'])
            is_bull = rf_model.predict(mock_today_data)[0]
            if is_bull == 1:
                log("[MARKET] 隨機森林預測：多頭 (Offensive)")
                return 'offensive', 0.65 # 模擬分數
            else:
                log("[MARKET] 隨機森林預測：空頭崩盤風險 (Defensive)")
                return 'defensive', -0.45
        except Exception as e:
            log(f"[MARKET-WARN] 載入 RF 模型失敗，退回技術面判斷: {e}")

    try:
        df = yf.download('^TWII', period='6mo', progress=False, auto_adjust=True)
        if df.empty: return 'offensive', 0.1
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df['MA20'] = ta.trend.sma_indicator(df['Close'], 20)
        latest = df.iloc[-1]
        
        if not pd.isna(latest['MA20']) and latest['Close'] < latest['MA20']:
            log(f'[MARKET] ^TWII Close < 20MA -> DEFENSIVE MODE')
            return 'defensive', -0.3
        else:
            log(f'[MARKET] ^TWII Close >= 20MA -> OFFENSIVE MODE')
            return 'offensive', 0.5
    except Exception as e:
        log_exception('[MARKET-ERROR]', e)
        return 'offensive', 0.1

# ==========================================
# 🎨 繪製大盤四維趨勢儀表板 (HTML to Image)
# ==========================================
def create_macro_dashboard_image(market_mode, macro_score, output_path):
    """使用 HTML 與 Chart.js 產生大盤儀表板"""
    
    mode_text = "🟢 多方輪動 (Offensive)" if market_mode == 'offensive' else "🔴 崩盤避險 (Defensive)"
    mode_color = "#16a34a" if market_mode == 'offensive' else "#dc2626"
    
    # 這裡放入模擬的近期大盤數據 (未來對接真實資料庫)
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
                    <h1 class="title">📊 台股籌碼四維趨勢報告</h1>
                    <div class="subtitle">AI 隨機森林預測引擎 | 產生時間: {now_str()}</div>
                </div>
                <div class="status-badge">
                    系統狀態: {mode_text} (分數: {macro_score:.2f})
                </div>
            </div>
            
            <div class="content-grid">
                <div>
                    <table>
                        <tr>
                            <th>日期</th><th>外資期貨(口)</th><th>選擇權 PCR</th><th>散戶多空比</th><th>四大特法</th><th>四維分數</th>
                        </tr>
                        <tr><td>T-0 (今日)</td><td class="pos">+2,450</td><td class="pos">110.5%</td><td class="pos">-4.2%</td><td class="neg">-1,200</td><td style="font-weight:bold; color:{mode_color}">{macro_score:.2f}</td></tr>
                        <tr><td>T-1</td><td class="neg">-1,500</td><td>105.2%</td><td class="neg">+2.1%</td><td class="neg">-3,400</td><td>-0.12</td></tr>
                        <tr><td>T-2</td><td class="neg">-4,200</td><td class="neg">95.4%</td><td class="neg">+5.8%</td><td class="neg">-5,100</td><td class="neg">-0.45</td></tr>
                        <tr><td>T-3</td><td class="neg">-3,100</td><td class="neg">98.1%</td><td class="pos">-1.2%</td><td class="pos">+1,500</td><td>-0.20</td></tr>
                        <tr><td>T-4</td><td class="pos">+1,200</td><td>102.3%</td><td class="neg">+1.5%</td><td class="pos">+2,800</td><td class="pos">+0.15</td></tr>
                    </table>
                    
                    <div class="chart-container">
                        <canvas id="trendChart"></canvas>
                    </div>
                </div>
                
                <div>
                    <div class="summary-box">
                        <h3 style="margin-top:0; color:#1f2937;">📝 系統判定與行動指南</h3>
                        <p style="font-size:14px; color:#4b5563; line-height:1.6;">
                            <b>模型解析：</b><br>
                            根據隨機森林模型運算，目前籌碼與散戶心理指標呈現 <b>{mode_text.split(' ')[1]}</b> 狀態。<br><br>
                            <b>自動因應動作：</b><br>
                            {"系統已自動開啟『防守避險引擎』，選股雷達將優先掃描高息低波與美債 ETF，並嚴格限縮個股乖離率條件。" if market_mode == 'defensive' else "系統處於『攻擊引擎』，資金可偏多操作，選股雷達將專注於強勢突破與動能發散股。"}
                        </p>
                    </div>
                    
                    <div style="height: 220px; width: 100%; margin-top: 20px;">
                        <canvas id="radarChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // 繪製趨勢圖
            const ctxLine = document.getElementById('trendChart').getContext('2d');
            new Chart(ctxLine, {{
                type: 'bar',
                data: {{
                    labels: ['T-4', 'T-3', 'T-2', 'T-1', '今日'],
                    datasets: [{{
                        label: '四維綜合分數',
                        data: [0.15, -0.20, -0.45, -0.12, {macro_score}],
                        backgroundColor: context => context.raw > 0 ? 'rgba(22, 163, 74, 0.5)' : 'rgba(220, 38, 38, 0.5)',
                        borderColor: context => context.raw > 0 ? 'rgb(22, 163, 74)' : 'rgb(220, 38, 38)',
                        borderWidth: 1
                    }}]
                }},
                options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }} }}
            }});

            // 繪製雷達圖
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            new Chart(ctxRadar, {{
                type: 'radar',
                data: {{
                    labels: ['外資期貨', '選擇權PCR', '散戶心理', '大戶現貨'],
                    datasets: [{{
                        label: '目前籌碼位階',
                        data: [60, 55, 30, 45], // 模擬數據
                        backgroundColor: '{mode_color}30',
                        borderColor: '{mode_color}',
                        pointBackgroundColor: '{mode_color}'
                    }}]
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
            # 等待 1 秒讓 Chart.js 動畫渲染完畢
            page.wait_for_timeout(1000)
            element = page.locator("#capture-area")
            element.screenshot(path=output_path, omit_background=True)
            browser.close()
            return output_path
    except Exception as e:
        log_exception(f'[PLOT-ERROR] Failed to create macro dashboard', e)
        return None

# ==========================================
# 🛡️ 防守引擎：ETF 避險池
# ==========================================
def get_defensive_etf_pool():
    return [
        '0050.TW',    # 台灣50
        '0056.TW',    # 元大高股息
        '00713.TW',   # 台灣高息低波
        '00878.TW',   # 國泰永續高股息
        '00679B.TWO', # 元大美債20年
        '00687B.TWO', # 國泰20年美債
        '00632R.TW',  # 台灣50反1
    ]

# ==========================================
# 🧰 基礎工具與爬蟲 (維持不變)
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
    return ticker + '.TW' if ticker.isdigit() and len(ticker) == 4 else ticker

def safe_send_message(chat_id, text, parse_mode=None):
    if not bot or not chat_id: return False
    for _ in range(TELEGRAM_RETRY):
        try: bot.send_message(chat_id, text, parse_mode=parse_mode); return True
        except Exception: time.sleep(1)
    return False

def safe_reply_to(message, text, parse_mode=None):
    if not bot: return False
    try: bot.reply_to(message, text, parse_mode=parse_mode); return True
    except Exception: return False

def safe_send_photo(chat_id, photo_path, caption=None, parse_mode=None):
    if not bot or not chat_id or not os.path.exists(photo_path): return False
    for _ in range(TELEGRAM_RETRY):
        try:
            with open(photo_path, 'rb') as f: bot.send_photo(chat_id, f, caption=caption, parse_mode=parse_mode)
            return True
        except Exception: time.sleep(1)
    return False

def run_cmd(cmd, cwd, timeout_sec, step_name):
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_sec)
        return {'ok': result.returncode == 0, 'timeout': False, 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired as e: return {'ok': False, 'timeout': True, 'stdout': getattr(e, 'stdout', ''), 'stderr': getattr(e, 'stderr', '')}
    except Exception as e: return {'ok': False, 'timeout': False, 'stdout': '', 'stderr': str(e)}

def update_my_tw_coverage(chat_id=None):
    safe_send_message(chat_id, '🔄 正在同步更新本地 My-TW-Coverage 與財報資料...')
    if not os.path.isdir(MY_TW_COVERAGE_PATH): return
    run_cmd(['git', 'pull'], MY_TW_COVERAGE_PATH, GIT_PULL_TIMEOUT, 'git pull')
    run_cmd(['python3', 'scripts/update_financials.py'], MY_TW_COVERAGE_PATH, FINANCIAL_UPDATE_TIMEOUT, 'update_financials.py')

def get_company_profile(ticker_num):
    try:
        target_file = None
        for root, dirs, files in os.walk(MY_TW_COVERAGE_PATH):
            for file in files:
                if file.startswith(str(ticker_num)) and file.endswith('.md'):
                    target_file = os.path.join(root, file)
                    break
            if target_file: break

        if not target_file: return {'profile': '⚠️ 未收錄於 My-TW-Coverage', 'industry': 'N/A', 'raw_text': None}

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
        return {'profile': desc or '查無業務描述', 'industry': industry, 'raw_text': content}
    except Exception: return {'profile': '讀取失敗', 'industry': 'N/A', 'raw_text': None}

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
            if m:
                result[key] = safe_float(m.group(1))
                break
    if any(v is not None for k, v in result.items() if k != 'source'): result['source'].append('My-TW-Coverage')
    return result

def fetch_goodinfo_data(ticker_num):
    url_main = f'https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={ticker_num}'
    url_chip = f'https://goodinfo.tw/tw/ShowBuySaleChart.asp?STOCK_ID={ticker_num}&CHT_CAT=DATE'
    main_html, chip_html = "", ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36')
            page = context.new_page()
            
            try: page.goto(url_main, wait_until='domcontentloaded', timeout=15000)
            except Exception: pass
            page.wait_for_timeout(3000)
            try: main_html = page.content()
            except Exception: page.wait_for_timeout(2000); main_html = page.content()

            try: page.goto(url_chip, wait_until='domcontentloaded', timeout=15000)
            except Exception: pass
            page.wait_for_timeout(3000)
            try: chip_html = page.content()
            except Exception: page.wait_for_timeout(2000); chip_html = page.content()
            
            browser.close()
    except Exception: pass
    return main_html, chip_html

def normalize_goodinfo_table(df):
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [' '.join([str(c).strip() for c in col if str(c).strip() not in ('', 'nan')]) for col in x.columns]
    else: x.columns = [str(c).strip() for c in x.columns]
    x = x.dropna(how='all').dropna(axis=1, how='all')
    x.columns = [str(c).replace('\n', ' ').replace('\r', ' ').strip() for c in x.columns]
    return x

def get_goodinfo_tables(html, ticker_num):
    tables = []
    try:
        raw = pd.read_html(StringIO(html))
        for df in raw: tables.append(normalize_goodinfo_table(df))
    except Exception: pass
    return tables

def parse_goodinfo_revenue_from_text(html, ticker_num):
    text = re.sub(r'\s+', ' ', html.replace(',', ''))
    out = {'single_month_revenue': None, 'single_month_mom': None, 'single_month_yoy': None, 'ytd_revenue': None, 'ytd_yoy': None, 'source': []}
    patterns = {
        'single_month_revenue': [r'3月份單月.*?營收金額.*?(\d+(?:\.\d+)?)', r'單月營收.*?(\d+(?:\.\d+)?)', r'營收金額.*?(\d+(?:\.\d+)?)'],
        'single_month_mom': [r'月增率.*?([\-+]?\d+(?:\.\d+)?)'],
        'single_month_yoy': [r'年增率.*?([\-+]?\d+(?:\.\d+)?)'],
        'ytd_revenue': [r'1\-3月累計.*?(\d+(?:\.\d+)?)', r'累計.*?營收.*?(\d+(?:\.\d+)?)'],
        'ytd_yoy': [r'1\-3月累計.*?年增率.*?([\-+]?\d+(?:\.\d+)?)', r'累計.*?年增率.*?([\-+]?\d+(?:\.\d+)?)']
    }
    for key, plist in patterns.items():
        for p in plist:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                out[key] = safe_float(m.group(1))
                break
    if any(v is not None for k, v in out.items() if k != 'source'): out['source'].append('Goodinfo-TextRevenue')
    return out

def parse_goodinfo_revenue_table(tables, ticker_num):
    result = {'single_month_revenue': None, 'single_month_mom': None, 'single_month_yoy': None, 'ytd_revenue': None, 'ytd_yoy': None, 'source': []}
    for idx, df in enumerate(tables):
        cols = [str(c).replace(' ', '').replace('\n', '') for c in df.columns]
        joined = ' | '.join(cols)
        if not (('營收' in joined) and ('月增' in joined or 'MoM' in joined) and ('年增' in joined or 'YoY' in joined)): continue
        try:
            work = df.copy().dropna(how='all')
            if work.empty: continue
            for ridx, row in work.iterrows():
                rev_col = mom_col = yoy_col = ytd_rev_col = ytd_yoy_col = None
                for c in work.columns:
                    cs = str(c).replace(' ', '').replace('\n', '')
                    if rev_col is None and ('單月營收' in cs or '營收(億)' in cs or '月營收' in cs) and '增' not in cs and '累計' not in cs: rev_col = c
                    if mom_col is None and ('月增' in cs or 'MoM' in cs or 'M/M' in cs) and '累計' not in cs: mom_col = c
                    if yoy_col is None and ('年增' in cs or 'YoY' in cs or 'Y/Y' in cs) and '累計' not in cs: yoy_col = c
                    if ytd_rev_col is None and '累計' in cs and '營收' in cs and '增' not in cs: ytd_rev_col = c
                    if ytd_yoy_col is None and '累計' in cs and '年增' in cs: ytd_yoy_col = c

                single_rev = safe_float(row[rev_col]) if rev_col is not None else None
                single_mom = safe_float(row[mom_col]) if mom_col is not None else None
                single_yoy = safe_float(row[yoy_col]) if yoy_col is not None else None
                ytd_rev = safe_float(row[ytd_rev_col]) if ytd_rev_col is not None else None
                ytd_yoy = safe_float(row[ytd_yoy_col]) if ytd_yoy_col is not None else None

                if any(v is not None for v in [single_rev, single_mom, single_yoy]):
                    result.update({'single_month_revenue': single_rev, 'single_month_mom': single_mom, 'single_month_yoy': single_yoy, 'ytd_revenue': ytd_rev, 'ytd_yoy': ytd_yoy})
                    result['source'].append('Goodinfo-TableRevenue')
                    return result
        except Exception: pass
    return result

def parse_goodinfo_eps_table(tables, ticker_num):
    result = {'eps_latest_quarter': None, 'eps_ttm': None, 'eps_quarters': [], 'source': []}
    for idx, df in enumerate(tables):
        cols = [str(c).replace(' ', '').replace('\n', '') for c in df.columns]
        joined = ' | '.join(cols)
        if not ('EPS' in joined or '每股盈餘' in joined): continue
        if 'PER' in joined or 'PBR' in joined or '最高價' in joined: continue
        try:
            work = df.copy().dropna(how='all')
            if work.empty: continue
            eps_col = None
            for c in work.columns:
                cs = str(c).replace(' ', '').replace('\n', '')
                if eps_col is None and ('EPS' in cs or '每股盈餘' in cs) and '成長' not in cs and '平均' not in cs: eps_col = c
            if eps_col is None: continue

            vals = []
            latest_q = None
            for _, row in work.iterrows():
                v = safe_float(row[eps_col])
                if v is not None:
                    vals.append(v)
                    if latest_q is None: latest_q = v
                if len(vals) >= 4: break

            if vals:
                result['eps_latest_quarter'] = latest_q
                result['eps_quarters'] = vals
                result['eps_ttm'] = round(sum(vals[:4]), 4)
                result['source'].append('Goodinfo-EPS')
                return result
        except Exception: pass
    return result

def parse_goodinfo_chip_table(tables, ticker_num):
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
    for idx, df in enumerate(tables):
        cols_str = " ".join([str(c) for c in df.columns])
        if not ('2日' in cols_str and '5日' in cols_str and '10日' in cols_str): continue
        try:
            work = df.copy().dropna(how='all')
            if work.empty: continue
            current_actor = None
            for ridx, row in work.iterrows():
                row_str = str(row.name) + " " + " ".join([str(x) for x in row.values])
                row_str_no_space = row_str.replace(" ", "")
                
                if '外資' in row_str_no_space: current_actor = 'foreign'
                elif '投信' in row_str_no_space: current_actor = 'trust'
                elif '自營' in row_str_no_space: current_actor = 'dealer'
                elif '總計' in row_str_no_space: 
                    current_actor = None
                    continue
                
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
        if t5 > 0: result['chips_summary'] = f'近 5 日三大法人偏多，合計買超約 {t5:.0f} 張'
        elif t5 < 0: result['chips_summary'] = f'近 5 日三大法人偏空，合計賣超約 {abs(t5):.0f} 張'
        else: result['chips_summary'] = '近 5 日三大法人大致中性'
        result['source'].append('Goodinfo-Chips')

    if not found_data:
        for k in result.keys():
            if isinstance(result[k], (int, float)) and result[k] == 0: result[k] = None

    return result

def get_goodinfo_snapshot(ticker_num):
    result = {
        'single_month_revenue': None, 'single_month_mom': None, 'single_month_yoy': None,
        'ytd_revenue': None, 'ytd_yoy': None, 'eps_latest_quarter': None, 'eps_ttm': None,
        'eps_quarters': [], 'chips_summary': '⚠️ Goodinfo 無法取得',
        'foreign_2d': None, 'foreign_3d': None, 'foreign_5d': None, 'foreign_10d': None,
        'trust_2d': None, 'trust_3d': None, 'trust_5d': None, 'trust_10d': None,
        'dealer_2d': None, 'dealer_3d': None, 'dealer_5d': None, 'dealer_10d': None,
        'total_2d': None, 'total_3d': None, 'total_5d': None, 'total_10d': None,
        'source': []
    }
    try:
        main_html, chip_html = fetch_goodinfo_data(ticker_num)
        main_tables = get_goodinfo_tables(main_html, ticker_num) if main_html else []
        chip_tables = get_goodinfo_tables(chip_html, f"{ticker_num}_chip") if chip_html else []

        rev_text = parse_goodinfo_revenue_from_text(main_html, ticker_num)
        rev_table = parse_goodinfo_revenue_table(main_tables, ticker_num)
        eps = parse_goodinfo_eps_table(main_tables, ticker_num)
        chip = parse_goodinfo_chip_table(chip_tables, ticker_num)

        result['single_month_revenue'] = rev_table['single_month_revenue'] if rev_table['single_month_revenue'] is not None else rev_text['single_month_revenue']
        result['single_month_mom'] = rev_table['single_month_mom'] if rev_table['single_month_mom'] is not None else rev_text['single_month_mom']
        result['single_month_yoy'] = rev_table['single_month_yoy'] if rev_table['single_month_yoy'] is not None else rev_text['single_month_yoy']
        result['ytd_revenue'] = rev_table['ytd_revenue'] if rev_table['ytd_revenue'] is not None else rev_text['ytd_revenue']
        result['ytd_yoy'] = rev_table['ytd_yoy'] if rev_table['ytd_yoy'] is not None else rev_text['ytd_yoy']
        result['eps_latest_quarter'] = eps['eps_latest_quarter']
        result['eps_ttm'] = eps['eps_ttm']
        result['eps_quarters'] = eps['eps_quarters']

        for k, v in chip.items():
            if k != 'source': result[k] = v
        result['source'] = rev_table['source'] + rev_text['source'] + eps['source'] + chip['source']
        return result
    except Exception: return result

def merge_financial_snapshot(ticker_num, md_text, yf_info=None):
    from_md = parse_financials_from_mytwcoverage(md_text)
    from_goodinfo = get_goodinfo_snapshot(ticker_num)
    merged = {
        'single_month_revenue': from_goodinfo['single_month_revenue'],
        'single_month_mom': from_md['single_month_mom'] if from_md['single_month_mom'] is not None else from_goodinfo['single_month_mom'],
        'single_month_yoy': from_md['single_month_yoy'] if from_md['single_month_yoy'] is not None else from_goodinfo['single_month_yoy'],
        'ytd_revenue': from_goodinfo['ytd_revenue'],
        'ytd_yoy': from_goodinfo['ytd_yoy'],
        'eps_latest_quarter': from_md['eps_latest_quarter'] if from_md['eps_latest_quarter'] is not None else from_goodinfo['eps_latest_quarter'],
        'eps_ttm': from_md['eps_ttm'] if from_md['eps_ttm'] is not None else from_goodinfo['eps_ttm'],
        'chips_summary': from_goodinfo['chips_summary'],
        'foreign_2d': from_goodinfo['foreign_2d'], 'foreign_3d': from_goodinfo['foreign_3d'], 'foreign_5d': from_goodinfo['foreign_5d'], 'foreign_10d': from_goodinfo['foreign_10d'],
        'trust_2d': from_goodinfo['trust_2d'], 'trust_3d': from_goodinfo['trust_3d'], 'trust_5d': from_goodinfo['trust_5d'], 'trust_10d': from_goodinfo['trust_10d'],
        'dealer_2d': from_goodinfo['dealer_2d'], 'dealer_3d': from_goodinfo['dealer_3d'], 'dealer_5d': from_goodinfo['dealer_5d'], 'dealer_10d': from_goodinfo['dealer_10d'],
        'total_2d': from_goodinfo['total_2d'], 'total_3d': from_goodinfo['total_3d'], 'total_5d': from_goodinfo['total_5d'], 'total_10d': from_goodinfo['total_10d'],
        'sources': from_md['source'] + from_goodinfo['source']
    }
    if merged['eps_ttm'] is None and yf_info:
        teps = safe_float(yf_info.get('trailingEps'))
        if teps is not None:
            merged['eps_ttm'] = teps
            merged['sources'].append('yfinance-trailingEps')
    if merged['single_month_yoy'] is None and yf_info:
        rg = yf_info.get('revenueGrowth')
        if rg is not None:
            merged['single_month_yoy'] = float(rg) * 100
            merged['sources'].append('yfinance-revenueGrowth')
    return merged

# ==========================================
# 股票池與技術面評分 
# ==========================================
def get_tw_stock_pool(mode='offensive'):
    tickers = []
    if mode == 'defensive':
        tickers.extend(get_defensive_etf_pool())
        
    for m in [2, 4]:
        try:
            res = requests.get(f'https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}', timeout=15)
            df = pd.read_html(StringIO(res.text))[0]
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            df = df[df['CFICode'] == 'ESVUFR']
            valid_codes = df['有價證券代號及名稱'].str.extract(r'^([0-9]{4})\b')[0].dropna()
            suffix = '.TW' if m == 2 else '.TWO'
            tickers.extend((valid_codes + suffix).tolist())
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

def resample_ohlcv(df, rule):
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].resample(rule).agg(agg).dropna()

def compute_indicators(df):
    df = df.copy()
    for win in [5, 20, 50, 60, 150, 200, 240]:
        df[f'MA{win}'] = ta.trend.sma_indicator(df['Close'], win)
    df['High52W'] = df['High'].rolling(250).max()
    df['Low52W'] = df['Low'].rolling(250).min()
    df['BBMid'] = ta.volatility.bollinger_mavg(df['Close'], 20)
    df['RSI'] = ta.momentum.rsi(df['Close'], 14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Osc'] = macd.macd_diff()
    for ma in [5, 20, 60, 240]:
        df[f'BIAS{ma}'] = np.where(df[f'MA{ma}'] != 0, (df['Close'] - df[f'MA{ma}']) / df[f'MA{ma}'] * 100, np.nan)
    return df

def evaluate_technical(df, market_mode='offensive'):
    df = compute_indicators(df)
    latest = df.iloc[-1]
    latest_date_str = df.index[-1].strftime('%Y-%m-%d')

    c1 = bool(latest['Close'] > latest['MA50'] > latest['MA150'] > latest['MA200'])
    c2 = bool(latest['Close'] > latest['Low52W'] * 1.30) if not pd.isna(latest['Low52W']) else False
    c3 = bool(latest['Close'] > latest['High52W'] * 0.75) if not pd.isna(latest['High52W']) else False
    c4 = bool((latest['Close'] >= latest['BBMid']) and (latest['RSI'] > 60) and (latest['MACD_Osc'] > 0)) if not pd.isna(latest['BBMid']) else False
    c5 = bool(latest['Volume'] > 1_500_000)
    c6 = bool(latest['Close'] > latest['MA5'] > latest['MA20'] > latest['MA60']) if not pd.isna(latest['MA60']) else False
    c7 = bool(latest['Close'] > latest['MA240']) if not pd.isna(latest['MA240']) else False

    weekly = compute_indicators(resample_ohlcv(df, 'W-FRI'))
    monthly = compute_indicators(resample_ohlcv(df, 'ME'))
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
        if bias60 is not None:
            if 0 <= bias60 <= 15: score += 4
            elif bias60 > 30: score -= 4
    else:
        score += 25 if c7 else 0 
        if latest['High52W']:
            drop_from_high = (latest['High52W'] - latest['Close']) / latest['High52W']
            if drop_from_high <= 0.10: score += 25
            elif drop_from_high <= 0.20: score += 15
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

    return {
        'df': df, 'weekly': weekly, 'monthly': monthly, 'technical_score': max(0, min(score, 100)), 'latest': latest, 'mode': market_mode,
        'conditions': {'trend_stack': c1, 'off_bottom': c2, 'near_high': c3, 'momentum': c4, 'liquidity': c5, 'short_mid_ma_stack': c6, 'above_ma240': c7, 'weekly_up': wk_up, 'monthly_up': mo_up, 'weekly_macd_positive': wk_macd_pos, 'monthly_macd_positive': mo_macd_pos},
        'metrics': {
            'latest_date': latest_date_str, 'rsi': safe_float(latest['RSI']), 'macd_osc_d': safe_float(latest['MACD_Osc']),
            'macd_osc_w': safe_float(weekly.iloc[-1].get('MACD_Osc')) if len(weekly) else None,
            'macd_osc_m': safe_float(monthly.iloc[-1].get('MACD_Osc')) if len(monthly) else None,
            'close': safe_float(latest['Close']), 'volume': safe_float(latest['Volume']),
            'dist_high_pct': ((latest['High52W'] - latest['Close']) / latest['High52W']) * 100 if latest['High52W'] else None,
            'dist_ma20_pct': ((latest['Close'] / latest['MA20']) - 1) * 100 if not pd.isna(latest['MA20']) and latest['MA20'] else None,
            'ma5': safe_float(latest['MA5']), 'ma20': safe_float(latest['MA20']), 'opt_ma': safe_float(opt_ma_val),
            'ma50': safe_float(latest['MA50']), 'ma60': safe_float(latest['MA60']), 'ma240': safe_float(latest['MA240']),
            'bias5': safe_float(latest.get('BIAS5')), 'bias20': bias20, 'bias60': bias60, 'bias240': safe_float(latest.get('BIAS240'))
        }
    }

def calc_fundamental_score(fin_data):
    score = 0
    single_yoy = fin_data.get('single_month_yoy')
    single_mom = fin_data.get('single_month_mom')
    ytd_yoy = fin_data.get('ytd_yoy')
    eps_q = fin_data.get('eps_latest_quarter')
    eps_ttm = fin_data.get('eps_ttm')

    if single_yoy is not None:
        if single_yoy >= 30: score += 25
        elif single_yoy >= 15: score += 18
        elif single_yoy >= 5: score += 10
        elif single_yoy < 0: score -= 10
    if single_mom is not None:
        if single_mom >= 20: score += 16
        elif single_mom >= 5: score += 10
        elif single_mom >= 0: score += 5
        else: score -= 6
    if ytd_yoy is not None:
        if ytd_yoy >= 20: score += 14
        elif ytd_yoy >= 5: score += 8
        elif ytd_yoy < 0: score -= 5
    if eps_q is not None:
        if eps_q >= 20: score += 18
        elif eps_q >= 10: score += 14
        elif eps_q > 0: score += 8
        else: score -= 8
    if eps_ttm is not None:
        if eps_ttm >= 40: score += 20
        elif eps_ttm >= 20: score += 14
        elif eps_ttm > 0: score += 8
        else: score -= 8
    return max(0, min(score, 100))

def calc_chip_score(fin_data):
    score = 0
    t2, t3, t5, t10 = fin_data.get('total_2d'), fin_data.get('total_3d'), fin_data.get('total_5d'), fin_data.get('total_10d')
    f5 = fin_data.get('foreign_5d')

    if t2 is not None: score += 12 if t2 > 0 else -6
    if t3 is not None: score += 12 if t3 > 0 else -6
    if t5 is not None:
        if t5 > 5000: score += 24
        elif t5 > 1000: score += 18
        elif t5 > 0: score += 10
        elif t5 < -5000: score -= 16
        else: score -= 8
    if t10 is not None: score += 14 if t10 > 0 else -8
    if f5 is not None: score += 10 if f5 > 0 else -5
    return max(0, min(score, 100))

def final_total_score(technical_score, fundamental_score, chip_score):
    w_t = SYS_PARAMS.get('tech_weight', WEIGHT_TECH)
    w_f = SYS_PARAMS.get('fund_weight', WEIGHT_FUND)
    w_c = SYS_PARAMS.get('chip_weight', WEIGHT_CHIP)
    return technical_score * w_t + fundamental_score * w_f + chip_score * w_c

# ==========================================
# 報告圖表與卡片產生
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

    fig.suptitle(f'{ticker} Quant Report ({ranking_info["mode"].upper()} MODE) | Total {ranking_info["total_score"]:.1f} | Tech {ranking_info["technical_score"]:.1f} | Fund {ranking_info["fundamental_score"]:.1f} | Chip {ranking_info["chip_score"]:.1f}', fontsize=16, fontweight='bold')
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
    
    def calc_rr(entry, stop, target):
        risk = entry - stop
        return (target - entry) / risk if risk > 0 else 0

    rr_a = calc_rr(entry_a_high, stop_a, target_1)
    rr_b = calc_rr(entry_b_high, stop_b, target_1)
    rr_c = calc_rr(entry_c_high, stop_c, target_2) 

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: transparent; margin: 0; padding: 20px; }}
            .card-container {{ width: 380px; background: white; border-radius: 16px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border: 1px solid #e5e7eb; }}
            .header {{ display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #e5e7eb; padding-bottom: 12px; margin-bottom: 16px; }}
            .title {{ font-size: 20px; font-weight: bold; color: #1f2937; margin: 0; }}
            .price-info {{ font-size: 16px; color: #dc2626; font-weight: 600; margin: 5px 0 0 0; }}
            .box {{ border-radius: 12px; padding: 12px; margin-bottom: 12px; border-left: 6px solid; }}
            
            .box-a {{ background: #fffbeb; border-color: #facc15; border-top: 1px solid #fef08a; border-right: 1px solid #fef08a; border-bottom: 1px solid #fef08a; }}
            .box-a .tag {{ color: #ca8a04; font-weight: bold; font-size: 16px; }}
            
            .box-b {{ background: #f0fdf4; border-color: #4ade80; border-top: 1px solid #bbf7d0; border-right: 1px solid #bbf7d0; border-bottom: 1px solid #bbf7d0; }}
            .box-b .tag {{ color: #16a34a; font-weight: bold; font-size: 16px; }}
            
            .box-c {{ background: #eff6ff; border-color: #60a5fa; border-top: 1px solid #bfdbfe; border-right: 1px solid #bfdbfe; border-bottom: 1px solid #bfdbfe; }}
            .box-c .tag {{ color: #2563eb; font-weight: bold; font-size: 16px; }}
            
            .price-range {{ font-size: 18px; font-weight: bold; color: #374151; margin-left: 8px; }}
            .desc {{ font-size: 13px; color: #6b7280; margin: 4px 0; }}
            .warning {{ font-size: 13px; color: #dc2626; font-weight: bold; margin-top: 6px; display: flex; align-items: center; }}
            .rr-ratio {{ font-size: 13px; color: #b45309; font-weight: bold; float: right; }}
            
            .target-title {{ font-size: 16px; font-weight: bold; margin: 20px 0 10px 0; display: flex; align-items: center; }}
            .target-box {{ border-radius: 10px; padding: 12px; margin-bottom: 10px; }}
            .t1 {{ background: #dcfce7; color: #166534; }}
            .t2 {{ background: #f3e8ff; color: #6b21a8; }}
            .target-price {{ font-size: 22px; font-weight: bold; margin-left: 8px; }}
        </style>
    </head>
    <body>
        <div class="card-container" id="capture-area">
            <div class="header">
                <div>
                    <h2 class="title">📊 {ticker} 操作計畫</h2>
                    <p class="price-info">最新收盤: {close_price:.2f}</p>
                </div>
            </div>
            
            <div style="font-weight: bold; color: #4b5563; margin-bottom: 8px;">📍 進場區</div>
            
            <div class="box box-a">
                <div><span class="tag">A 低接</span><span class="price-range">{entry_a_low:.1f} - {entry_a_high:.1f}</span></div>
                <div class="desc">箱底/防守線 附近，止跌後確認進場</div>
                <div class="warning">❗️ 停損 {stop_a:.1f} (約 -{hard_stop*100:.1f}%) <span class="rr-ratio">風報 1:{rr_a:.1f}</span></div>
            </div>
            
            <div class="box box-b">
                <div><span class="tag">B 回穩</span><span class="price-range">{entry_b_low:.1f} - {entry_b_high:.1f}</span></div>
                <div class="desc">中間支撐，5MA/防守均線之間確認</div>
                <div class="warning">❗️ 停損 {stop_b:.1f} (約 -{hard_stop*100:.1f}%) <span class="rr-ratio">風報 1:{rr_b:.1f}</span></div>
            </div>
            
            <div class="box box-c">
                <div><span class="tag">C 突破</span><span class="price-range">{entry_c_low:.1f} - {entry_c_high:.1f}</span></div>
                <div class="desc">突破短期高點確認入場 (僅小倉)</div>
                <div class="warning">❗️ 停損 {stop_c:.1f} (約 -{hard_stop*100:.1f}%) <span class="rr-ratio">風報 1:{rr_c:.1f}</span></div>
            </div>
            
            <div class="target-title">🎯 壓力區 (非固定止盈)</div>
            
            <div class="target-box t1">
                <div><strong>壓力 1</strong><span class="target-price">{target_1:.1f}</span><span style="float: right; font-size: 13px; margin-top: 6px;">主要出場參考</span></div>
                <div style="font-size: 12px; margin-top: 4px; opacity: 0.8;">接近壓力 1 留意短線賣壓，可分批減碼</div>
            </div>
            
            <div class="target-box t2">
                <div><strong>壓力 2</strong><span class="target-price">{target_2:.1f}</span><span style="float: right; font-size: 13px; margin-top: 6px;">波段目標</span></div>
                <div style="font-size: 12px; margin-top: 4px; opacity: 0.8;">突破壓力且放量 ➔ 續抱 (移動停利)</div>
            </div>
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
    except Exception as e:
        log_exception(f'[PLOT-ERROR] Failed to create strategy card', e)
        return None

def build_stock_report(ticker, tech_pack, fin_data, profile_info, rank=None):
    latest, c, m = tech_pack['latest'], tech_pack['conditions'], tech_pack['metrics']
    mode = tech_pack['mode']
    tech_score = tech_pack['technical_score']
    fund_score = calc_fundamental_score(fin_data)
    chip_score = calc_chip_score(fin_data)
    total_score = final_total_score(tech_score, fund_score, chip_score)

    img_path = save_exquisite_plot(tech_pack['df'], tech_pack['weekly'], tech_pack['monthly'], ticker, {
        'technical_score': tech_score, 'fundamental_score': fund_score, 'chip_score': chip_score, 'total_score': total_score, 'mode': mode
    })

    close_val = latest["Close"]
    opt_hard_stop = SYS_PARAMS.get('hard_stop', 0.08)
    opt_ma_val = safe_float(m.get('opt_ma')) or close_val

    strategy_card_path = os.path.join(REPORT_DIR, f'{ticker}_strategy.png')
    create_strategy_card_image(
        ticker=ticker, close_price=close_val, ma5=safe_float(m.get("ma5")) or close_val,
        ma20=opt_ma_val, high52w=safe_float(latest.get("High52W")) or (close_val * 1.1),
        hard_stop=opt_hard_stop, output_path=strategy_card_path
    )

    report = ''
    if rank is not None: report += f'🏆 **排名 #{rank}**\n'
    
    is_etf = '00' in ticker
    mode_text = '🛡️ ETF 防守避風港' if is_etf else ('🔥 攻擊型飆股' if mode == 'offensive' else '🛡️ RS相對強勢')
    
    report += f'📊 **【量化診斷：{ticker}】** ({mode_text})\n'
    report += f'💰 最新收盤：`{latest["Close"]:.2f}` _(資料日期: {m["latest_date"]})_\n'
    report += f'🧮 總分：`{total_score:.1f}` | 技術：`{tech_score:.1f}` | 基本：`{fund_score:.1f}` | 籌碼：`{chip_score:.1f}`\n'
    report += '------------------------\n'
    report += f'🏢 **公司 / 業務摘要：**\n**產業:** {profile_info["industry"]}\n_{profile_info["profile"]}_\n'
    report += '------------------------\n'
    
    if is_etf:
        report += '🛡️ **ETF 防守狀態：**\n'
        report += f'└ 距離 52W 高點：`{safe_pct_str(m["dist_high_pct"])}`\n'
        report += f'└ 站穩年線 (240MA)：{"✅" if c["above_ma240"] else "❌"}\n'
        report += f'⚠️ **【重要警告】買進前務必至投信官網確認「即時折溢價」，溢價大於 1% 請勿追高進場！**\n'
    else:
        report += '🔍 **技術面分析：**\n'
        report += f'{"✅" if c["trend_stack"] else "❌"} 長天期多頭排列 (價>50>150>200)\n'
        report += f'{"✅" if c["short_mid_ma_stack"] else "❌"} 短中期均線順多 (價>5>20>60)\n'
        report += f'{"✅" if c["above_ma240"] else "❌"} 站上 240MA\n'
        report += f'{"✅" if c["off_bottom"] else "❌"} 已脫離 52W 低點至少 30%\n'
        report += f'{"✅" if c["near_high"] else "❌"} 靠近 52W 高點 25% 內\n'
        report += f'{"✅" if c["momentum"] else "❌"} 日線動能：RSI>60 且 MACD>0\n'
        
    report += '\n📈 **技術數據面板：**\n'
    report += f'🔹 RSI(日)：`{safe_num_str(m["rsi"], 1)}`\n'
    report += f'🔹 MACD Hist 日/週/月：`{safe_num_str(m["macd_osc_d"], 3)}` / `{safe_num_str(m["macd_osc_w"], 3)}` / `{safe_num_str(m["macd_osc_m"], 3)}`\n'
    report += f'🔹 5/20/60/240MA乖離率：`{safe_pct_str(m["bias5"])}` / `{safe_pct_str(m["bias20"])}` / `{safe_pct_str(m["bias60"])}` / `{safe_pct_str(m["bias240"])}`\n'
    
    if not is_etf:
        report += '------------------------\n'
        report += '💹 **基本面：**\n'
        report += f'🔸 最新一季 EPS：`{safe_num_str(fin_data.get("eps_latest_quarter"))}`\n🔸 近四季 EPS：`{safe_num_str(fin_data.get("eps_ttm"))}`\n'
        report += f'🔸 單月營收 Y/Y：`{safe_pct_str(fin_data.get("single_month_yoy"))}`\n🔸 月營收 M/M：`{safe_pct_str(fin_data.get("single_month_mom"))}`\n'
        
    report += '------------------------\n'
    report += '🏦 **籌碼面：**\n'
    report += f'🔸 籌碼摘要：`{fin_data.get("chips_summary", "N/A")}`\n'
    report += f'🔸 外資 2/3/5/10日：`{safe_num_str(fin_data.get("foreign_2d"), 0)}` / `{safe_num_str(fin_data.get("foreign_3d"), 0)}` / `{safe_num_str(fin_data.get("foreign_5d"), 0)}` / `{safe_num_str(fin_data.get("foreign_10d"), 0)}`\n'
    report += f'🔸 投信 2/3/5/10日：`{safe_num_str(fin_data.get("trust_2d"), 0)}` / `{safe_num_str(fin_data.get("trust_3d"), 0)}` / `{safe_num_str(fin_data.get("trust_5d"), 0)}` / `{safe_num_str(fin_data.get("trust_10d"), 0)}`\n'
    report += f'🔸 自營 2/3/5/10日：`{safe_num_str(fin_data.get("dealer_2d"), 0)}` / `{safe_num_str(fin_data.get("dealer_3d"), 0)}` / `{safe_num_str(fin_data.get("dealer_5d"), 0)}` / `{safe_num_str(fin_data.get("dealer_10d"), 0)}`\n'
    report += f'🔸 三大法人合計：`{safe_num_str(fin_data.get("total_2d"), 0)}` / `{safe_num_str(fin_data.get("total_3d"), 0)}` / `{safe_num_str(fin_data.get("total_5d"), 0)}` / `{safe_num_str(fin_data.get("total_10d"), 0)}`\n'
    srcs = ', '.join(fin_data.get('sources', [])) if fin_data.get('sources') else 'N/A'
    report += f'🔸 資料來源：`{srcs}`\n'
    report += '------------------------\n'
    report += '🎯 **出場雷達 (動態策略)：**\n'
    
    bias20 = safe_float(m.get("bias20"))
    
    if bias20 is not None and bias20 >= 18:
        report += f'🚨 **高潮噴出警報**：短線正乖離達 `{bias20:.1f}%`，強烈建議了結部分部位。\n'
    elif bias20 is not None and bias20 >= 10:
        report += f'⚠️ **動能過熱**：留意獲利回吐，持股者上移停利點。\n'
    if close_price < opt_ma_val:
        report += f'📉 **短線轉弱**：跌破動態防守均線 `{SYS_PARAMS.get("ma_period", 20)}MA` ({opt_ma_val:.2f})，部位過大應考慮減碼。\n'
    else:
        report += f'🛡️ **趨勢健康 (Hold)**：防守底線設於 `{SYS_PARAMS.get("ma_period", 20)}MA` ({opt_ma_val:.2f})。\n'
        
    report += '\n'
    if total_score >= 80: report += '🚀 **結論：三方共振，屬高優先級觀察名單。**'
    elif total_score >= 65: report += '🟡 **結論：結構偏強，可列入次高優先級。**'
    else: report += '⚪ **結論：有部分條件符合，尚未達到最強勢組。**'

    return report, img_path, strategy_card_path, total_score, tech_score, fund_score, chip_score

def analyze_stock(ticker, market_mode='offensive', silent=False):
    try:
        ticker, df = download_stock_df(ticker)
        if df.empty or len(df) < 250: return (None, None, None) if silent else ('❌ 找不到資料', None, None)
        
        tech_pack = evaluate_technical(df, market_mode)
        ticker_num = ticker.split('.')[0]
        profile_info = get_company_profile(ticker_num)
        yf_info = yf.Ticker(ticker).info if not TEST_MODE else {}
        fin_data = merge_financial_snapshot(ticker_num, profile_info['raw_text'], yf_info=yf_info)
        report, img_path, strategy_img_path, *_ = build_stock_report(ticker, tech_pack, fin_data, profile_info)
        return report, img_path, strategy_img_path
    except Exception as e:
        log_exception(f'[ANALYZE-ERROR] {ticker}', e)
        return (None, None, None) if silent else (f'❌ 錯誤：{e}', None, None)

def scan_and_rank_market(chat_id=None, requested_by_user=False, market_mode='offensive'):
    pool = get_tw_stock_pool(market_mode)
    if TEST_MODE: pool = pool[:20]
    prescreen = []
    for idx, ticker in enumerate(pool, start=1):
        try:
            if idx == 1 or idx % SCAN_PROGRESS_STEP == 0:
                if requested_by_user: safe_send_message(chat_id, f'⏳ 技術初篩：已處理 `{idx}`/`{len(pool)}` 檔...', parse_mode='Markdown')
            tkr, df = download_stock_df(ticker)
            if df.empty or len(df) < 250: continue
            tech_pack = evaluate_technical(df, market_mode)
            if tech_pack['technical_score'] >= 50: prescreen.append({'ticker': tkr, 'df': df, 'tech_pack': tech_pack})
        except Exception: pass
        
    prescreen.sort(key=lambda x: x['tech_pack']['technical_score'], reverse=True)
    prescreen = prescreen[:TECHNICAL_PRESCREEN_LIMIT]
    if requested_by_user: safe_send_message(chat_id, f'✅ 初篩完成，共 `{len(prescreen)}` 檔進入深度評分。', parse_mode='Markdown')

    ranked = []
    for idx, item in enumerate(prescreen, start=1):
        ticker = item['ticker']
        try:
            ticker_num = ticker.split('.')[0]
            profile_info = get_company_profile(ticker_num)
            fin_data = merge_financial_snapshot(ticker_num, profile_info['raw_text'], yf_info=yf.Ticker(ticker).info)
            t_score = item['tech_pack']['technical_score']
            f_score = calc_fundamental_score(fin_data)
            c_score = calc_chip_score(fin_data)
            ranked.append({
                'ticker': ticker, 'tech_pack': item['tech_pack'], 'fin_data': fin_data, 'profile_info': profile_info,
                'total_score': final_total_score(t_score, f_score, c_score), 'tech': t_score, 'fund': f_score, 'chip': c_score
            })
        except Exception: pass
    ranked.sort(key=lambda x: (x['total_score'], x['tech']), reverse=True)
    return ranked[:FINAL_TOP_N]

# ==========================================
# 🤖 自動化排程：週末回歸測試
# ==========================================
def run_weekly_optimization():
    log("🧬 [SYSTEM] 啟動週末回歸測試與策略進化...")
    if bot:
        safe_send_message(CHAT_ID, "🧬 **系統通知：啟動週末回歸測試 (Regression Test) 與機器學習尋優。**", parse_mode='Markdown')
    try:
        subprocess.Popen(['python3', 'optimize_bayesian.py'])
        log("[SYSTEM] 已成功派發最佳化任務至背景執行。")
    except Exception as e:
        log(f"[SYSTEM-ERROR] 啟動最佳化任務失敗: {e}")

# ==========================================
# 主線掃描任務 (發送大盤圖表 + 個股)
# ==========================================
def run_market_scan_job(chat_id, requested_by_user=False):
    # 1. 執行大盤判斷
    market_mode, macro_score = check_market_status()
    mode_msg = "🟢 **大盤偏多：啟動 [攻擊型飆股引擎]**" if market_mode == 'offensive' else "🔴 **大盤崩盤/偏空：啟動 [RS防守避險引擎 + ETF推薦]**"
    ma_param = SYS_PARAMS.get('ma_period', 20)
    stop_param = SYS_PARAMS.get('hard_stop', 0.08) * 100
    
    # 2. 發布大盤儀表板圖片
    safe_send_message(chat_id, f'🔍 **全市場量化雷達啟動中...**\n{mode_msg}\n⚙️ 套用最佳化參數: `MA{ma_param}`, 停損: `-{stop_param:.1f}%`', parse_mode='Markdown')
    
    macro_img_path = os.path.join(REPORT_DIR, 'macro_dashboard.png')
    dashboard_generated = create_macro_dashboard_image(market_mode, macro_score, macro_img_path)
    if dashboard_generated and os.path.exists(dashboard_generated):
        safe_send_photo(chat_id, dashboard_generated)
    
    update_my_tw_coverage(chat_id)
    
    # 3. 掃描個股
    top_ranked = scan_and_rank_market(chat_id, requested_by_user, market_mode)
    
    if not top_ranked:
        safe_send_message(chat_id, '☕ **掃描完畢**\n本次無達標股票。', parse_mode='Markdown')
        return
        
    summary = ['🏆 **今日 Top 10 觀察清單**']
    for i, item in enumerate(top_ranked, start=1):
        is_etf = '00' in item["ticker"]
        icon = '🛡️' if is_etf else '🚀'
        summary.append(f'{i}. {icon} `{item["ticker"]}` | 總分 `{item["total_score"]:.1f}`')
    safe_send_message(chat_id, '\n'.join(summary), parse_mode='Markdown')
    
    for i, item in enumerate(top_ranked, start=1):
        try:
            report, img_path, strategy_img_path, *_ = build_stock_report(item['ticker'], item['tech_pack'], item['fin_data'], item['profile_info'], rank=i)
            if img_path and os.path.exists(img_path): safe_send_photo(chat_id, img_path)
            if strategy_img_path and os.path.exists(strategy_img_path): safe_send_photo(chat_id, strategy_img_path)
            safe_send_message(chat_id, report, parse_mode='Markdown')
        except Exception: pass
    safe_send_message(chat_id, '✅ **掃描完畢**', parse_mode='Markdown')

def start_scan_thread(chat_id, requested_by_user):
    threading.Thread(target=run_market_scan_job, args=(chat_id, requested_by_user), daemon=True).start()

# ==========================================
# Telegram 監聽
# ==========================================
if bot:
    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        safe_reply_to(message, '👋 歡迎使用 **Stock Minervini Pro** (Quant Hedge Fund Edition)\n\n🟢 `/scan`：全市場掃描 (自動多空雙引擎切換 + 大盤圖表)\n🟢 `/update`：手動同步資料\n🟢 直接輸入股票代碼：單檔分析', parse_mode='Markdown')

    @bot.message_handler(commands=['update'])
    def handle_update(message):
        update_my_tw_coverage(message.chat.id)

    @bot.message_handler(commands=['scan'])
    def handle_scan(message):
        safe_send_message(message.chat.id, '🚀 啟動掃描並生成大盤四維趨勢圖...', parse_mode='Markdown')
        start_scan_thread(message.chat.id, True)

    @bot.message_handler(func=lambda message: not message.text.startswith('/'))
    def handle_stock(message):
        ticker = message.text.strip().upper().replace('多', '').replace('空', '')
        safe_reply_to(message, f'⏳ 正在產生 `{ticker}` 的診斷報告...', parse_mode='Markdown')
        
        current_mode, _ = check_market_status()
        report, img_path, strategy_img_path = analyze_stock(ticker, current_mode)
        
        if img_path and os.path.exists(img_path): safe_send_photo(message.chat.id, img_path)
        if strategy_img_path and os.path.exists(strategy_img_path): safe_send_photo(message.chat.id, strategy_img_path)
            
        if report: safe_send_message(message.chat.id, report, parse_mode='Markdown')
        else: safe_send_message(message.chat.id, '❌ 找不到資料', parse_mode='Markdown')

def schedule_loop():
    schedule.every().day.at('16:30').do(start_scan_thread, chat_id=CHAT_ID, requested_by_user=False)
    schedule.every().saturday.at("02:00").do(run_weekly_optimization)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    log('🤖 Stock Minervini Pro 啟動中...')
    threading.Thread(target=schedule_loop, daemon=True).start()
    if bot: bot.infinity_polling(timeout=60, long_polling_timeout=30)
