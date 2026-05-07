# Quant System

A fully automated quantitative stock technical analysis and strategy evolution tool.

> This project is currently for internal testing and after-market stock recommendation research only.  
> It does not execute orders and does not provide financial advice.  
> Investment funds involve risks. Please read the public prospectus carefully before investing.

## Features

- **Multi-Market Scanning**
  - Automatically scans both the Taiwan stock market and the US market.

- **Dual-Engine Strategy**
  - **Offensive Mode:** Focuses on strong breakouts and momentum stocks based on the Minervini Trend Template.
  - **Defensive Mode:** Switches to a defensive radar when the macro market is bearish, prioritizing high-yield, low-volatility stocks and US Treasury ETFs.

- **Machine Learning & AI**
  - Uses a Random Forest classifier to predict macro market trends.
  - Generates macro-trend dashboard signals for market regime judgment.

- **Strategy Optimization**
  - Runs Bayesian Optimization with Gaussian Process Regression.
  - Runs surrogate-assisted NSGA-II multi-objective evolution with real Backtrader fitness.
  - Writes balanced Pareto-front parameters to `config/best_params.json`.

- **Hybrid Research Pipeline**
  - Builds reusable OHLCV feature frames with Minervini trend, VCP, Bollinger squeeze/breakout, DL signals, and hybrid scores.
  - Provides a PyTorch CNN-BiLSTM-Attention training entrypoint for `dl_signal`.
  - Keeps ML inference outside the Backtrader `next()` loop by precomputing signals.

- **Telegram Integration**
  - Sends technical diagnostic reports, strategy entry/exit cards, and macro dashboards to Telegram.

- **Cross-Platform Deployment**
  - Supports Docker-based deployment for Windows, macOS, and Linux.
  - Reduces dependency issues by containerizing Python packages and Playwright browser setup.

## Quick Start

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd quant_system
```

### 2. Create `.env`

Create a `.env` file in the project root:

```bash
touch .env
```

Add your Telegram bot settings:

```env
TELEGRAM_TOKEN=YOUR_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_CHAT_ID
TELEGRAM_POLLING_ENABLED=1
FINMIND_TOKEN=YOUR_FINMIND_TOKEN
MY_TW_COVERAGE_PATH=/path/to/your/My-TW-Coverage
HTTP_TIMEOUT_SEC=20
HTTP_RETRIES=3
HTTP_BACKOFF_SEC=1.5
GOODINFO_MIN_INTERVAL_SEC=3
FINMIND_MIN_INTERVAL_SEC=2
EXTERNAL_CACHE_PATH=config/external_cache.sqlite3
EXTERNAL_CACHE_TTL_HOURS=12
SCORE_CONFIG_PATH=config/best_params.json
BOX_LOOKBACK=20
BOX_WIDTH_THRESHOLD=0.10
BOX_BREAKOUT_VOLUME_MULT=1.5
ENGULFING_LOOKBACK=5
ENGULFING_BODY_PCT=0.02
BB_WIDTH_THRESHOLD=0.10
BB_BREAKOUT_VOLUME_MULT=1.5
PATTERN_SWING_ORDER=5
SECTOR_STRONG_THRESHOLD=65
# Optional comma-separated universes:
# SCAN_UNIVERSE_TW=2330,2454
# SCAN_UNIVERSE_US=AAPL,NVDA,BRK-B
```

Example:

```env
TELEGRAM_TOKEN=YOUR_NUMERIC_BOT_ID:YOUR_BOT_SECRET
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID
TELEGRAM_POLLING_ENABLED=1
MY_TW_COVERAGE_PATH=/home/YOUR_USERNAME/My-TW-Coverage
```

Only one running process can poll a Telegram bot token. If this machine is only
for training, backtesting, or optimization, set:

```env
TELEGRAM_POLLING_ENABLED=0
```

This prevents Telegram `409 Conflict: terminated by other getUpdates request`
errors while keeping local research commands usable.

### 3. Run with Docker

```bash
COMPOSE_FILE=docker-compose.yml docker compose up -d --build
```

For NVIDIA GPU training inside Docker, first install NVIDIA Container Toolkit
in the VM/host, then build with the GPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker exec -it stock_minervini_bot python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If the command prints `True`, `python -m quant.dl ...` will train on CUDA.
If Docker reports `could not select device driver "" with capabilities: [[gpu]]`,
the host cannot provide an NVIDIA GPU runtime. Use the default CPU container
instead:

```bash
docker compose down
COMPOSE_FILE=docker-compose.yml docker compose up -d --build
```

Check logs:

```bash
docker logs -f stock_minervini_bot
```

### 4. Test Telegram Bot

Open Telegram and send one of the following commands to your bot:

```text
/scan
/scan_us
/update
/help
```

You can also send a stock ticker directly:

```text
2330
2454
AAPL
NVDA
```

## Configuration

Sensitive values such as Telegram bot tokens should be stored in environment variables instead of being hardcoded in Python files.

### Telegram Bot Token

Create a Telegram bot using BotFather:

```text
Telegram → Search BotFather → /newbot
```

BotFather will return a token with this format:

```text
<numeric_bot_id>:<bot_secret>
```

The token must contain a colon `:`. If the token does not contain a colon, it is incomplete or incorrect.

Add it to `.env`:

```env
TELEGRAM_TOKEN=YOUR_NUMERIC_BOT_ID:YOUR_BOT_SECRET
```

### Telegram Chat ID

Set your chat ID in `.env`:

```env
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID
```

If you do not know your chat ID, you can get it from Telegram helper bots such as `userinfobot`, or by checking the Telegram Bot API update response after sending a message to your bot.

### Local Database Path

This project can optionally use an external Markdown financial database named `My-TW-Coverage`.

Recommended workspace layout:

```text
Your_Workspace/
├── My-TW-Coverage/
└── quant_system/
    ├── .env
    ├── docker-compose.yml
    ├── quant_pro.py
    └── ...
```

If you do not use `My-TW-Coverage`, the system can still run, but some Taiwan stock company profile and local financial data may be unavailable.

Set the path in `.env`:

```env
MY_TW_COVERAGE_PATH=/home/randal/My-TW-Coverage
```

## Docker Usage

Docker is the recommended way to run this system on Windows, macOS, or Linux.

### Prerequisites

Install Docker Desktop and make sure the Docker engine is running.

Also make sure you have created the `.env` file described above.

### Build and Start

```bash
COMPOSE_FILE=docker-compose.yml docker compose up -d --build
```

### View Logs

```bash
docker logs -f stock_minervini_bot
```

### Stop

```bash
docker compose down
```

### Restart

```bash
docker compose restart
```

### Rebuild After Code Changes

```bash
docker compose down
COMPOSE_FILE=docker-compose.yml docker compose up -d --build
```

### Run Commands Inside Container

Run the main bot:

```bash
docker exec -it stock_minervini_bot python quant_pro.py
```

Train the macro model:

```bash
docker exec -it stock_minervini_bot python train_macro_model.py
```

Run Bayesian optimization:

```bash
docker exec -it stock_minervini_bot python optimize_bayesian.py
```

Run NSGA-II strategy evolution:

```bash
docker exec -it stock_minervini_bot python evolve_nsga2.py
```

Run the hybrid Backtrader pipeline:

```bash
docker exec -it stock_minervini_bot python -m backtests.run_backtest --tickers 2454.TW 2330.TW AAPL NVDA --start 2019-01-01 --folds 2
```

Train the first DL signal model:

```bash
docker exec -it stock_minervini_bot python -m quant.dl --market MIXED --start 2018-01-01 --epochs 12 --tickers 2454.TW 2330.TW AAPL NVDA
```

Run surrogate-assisted NSGA-II and update `config/best_params.json`:

```bash
docker exec -it stock_minervini_bot python evolve_nsga2.py --tickers 2454.TW 2330.TW AAPL NVDA --start 2019-01-01 --folds 2
```

Open a shell inside the container:

```bash
docker exec -it stock_minervini_bot bash
```

## Manual Linux Deployment

If you prefer to run this project directly on Debian or Ubuntu Linux without Docker, follow the steps below.

### Environment Setup

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Package Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dl-cpu.txt
```

Install Playwright browser dependency:

```bash
playwright install chromium
```

If the project uses FinMind:

```bash
pip install FinMind
```

### Environment Variables

Set Telegram environment variables:

```bash
export TELEGRAM_TOKEN="YOUR_BOT_TOKEN"
export TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
export MY_TW_COVERAGE_PATH="/home/randal/My-TW-Coverage"
```

To save them permanently:

```bash
echo 'export TELEGRAM_TOKEN="YOUR_BOT_TOKEN"' >> ~/.bashrc
echo 'export TELEGRAM_CHAT_ID="YOUR_CHAT_ID"' >> ~/.bashrc
echo 'export MY_TW_COVERAGE_PATH="/home/randal/My-TW-Coverage"' >> ~/.bashrc
source ~/.bashrc
```

Run the bot:

```bash
python quant_pro.py
```

## Command Usage

### Telegram Commands

| Command | Description |
|---|---|
| `/scan` | Run Taiwan stock market scan |
| `/scan_us` | Run US stock market scan |
| `/update` | Update local `My-TW-Coverage` financial database |
| `/help` | Show available commands |

### Direct Stock Analysis

You can send a ticker directly to the Telegram bot.

Taiwan stocks:

```text
2330
2454
2317
```

US stocks:

```text
AAPL
NVDA
MSFT
TSLA
```

The bot will generate:

```text
1. Technical diagnostic report
2. Fundamental data summary
3. Chip/fund-flow data if available
4. K-line chart image
5. Strategy entry/exit card
```

### Local Python Commands

Run main bot:

```bash
python quant_pro.py
```

Train macro model:

```bash
python train_macro_model.py
```

Run Bayesian optimization:

```bash
python optimize_bayesian.py
```

Run NSGA-II strategy evolution:

```bash
python evolve_nsga2.py
```

Run historical backtest:

```bash
python backtest_minervini_bt.py
```

Run the new hybrid feature-frame backtest:

```bash
python -m backtests.run_backtest --tickers 2454.TW 2330.TW AAPL NVDA --start 2019-01-01 --folds 2
```

Train a PyTorch DL signal model:

```bash
python -m quant.dl --market MIXED --start 2018-01-01 --epochs 12 --tickers 2454.TW 2330.TW AAPL NVDA
```

Run the surrogate-assisted NSGA-II optimizer:

```bash
python evolve_nsga2.py --tickers 2454.TW 2330.TW AAPL NVDA --start 2019-01-01 --folds 2
```

Check Python syntax:

```bash
python3 -m py_compile quant_pro.py
```

## Usage Flow

Typical workflow:

```text
1. Start bot with Docker or Python
2. Send /scan in Telegram
3. System evaluates macro market mode
4. System scans Taiwan or US stock universe
5. System ranks candidates by technical, fundamental, and chip scores
6. System sends reports, charts, and strategy cards to Telegram
```

Taiwan market scan flow:

```text
1. Use yfinance for fast technical pre-screening
2. Select strong technical candidates
3. Use Goodinfo / My-TW-Coverage / FinMind when available for deeper checks
4. Rank final candidates
5. Send Top N reports to Telegram
```

US market scan flow:

```text
1. Use S&P 500 universe or fallback large-cap list
2. Use yfinance for price, technical, and fundamental data
3. Rank candidates by technical and fundamental scores
4. Send reports to Telegram
```

## Process Management

### Run in Background Without Docker

To keep the bot running after closing the terminal:

```bash
nohup python quant_pro.py > system_log.txt 2>&1 &
```

### Check Background Process

```bash
ps aux | grep quant_pro.py
```

### Stop Background Process

Replace `<PID>` with the actual process ID:

```bash
kill <PID>
```

## Automated Scheduling

When `quant_pro.py` or the Docker container runs continuously, the system follows the internal schedule below:

| Time | Task |
|---|---|
| Every day at 16:30 CST | Run Taiwan stock market scan and send Telegram notifications |
| Every day at 05:00 CST | Run US stock market scan and send Telegram notifications |
| Every Saturday at 02:00 CST | Run weekly optimization and strategy evolution |

The weekly optimization process writes the latest optimized parameters to:

```text
config/best_params.json
```

## Future Implementation Roadmap

This section describes the future implementation direction of this project.

The current system focuses on after-market stock screening, technical diagnosis, Telegram reporting, and strategy optimization experiments. Future versions may gradually evolve toward a hybrid quantitative research framework that combines Minervini-style trend logic, Volatility Contraction Pattern recognition, Bollinger Band breakout detection, machine learning, deep learning, and surrogate-assisted optimization.

### 1. Minervini Trend Template as the Hard Filter

The first layer of the future system will keep the Minervini Trend Template as the strict entry filter.

The system should only evaluate aggressive long candidates when the stock satisfies a Stage 2 uptrend structure:

- Price is above the 150-day and 200-day moving averages.
- The 150-day moving average is above the 200-day moving average.
- The 200-day moving average is trending upward.
- The 50-day moving average is above the 150-day and 200-day moving averages.
- Price is above the 50-day moving average.
- Relative strength is stronger than the broader market.

This hard filter prevents the system from buying weak stocks in long-term downtrends.

Future AI, VCP, Bollinger Band, or breakout signals should only be evaluated after this trend filter passes.


### 2. Risk Management Upgrade

Future versions should strengthen risk control before any production-like usage.

Possible risk modules:

- Fixed stop loss between 7% and 8%.
- Risk per trade limited to 1% to 2% of capital.
- Maximum number of active positions.
- Maximum sector concentration.
- Maximum drawdown guard.
- Market regime filter.
- ETF premium / discount warning.
- No-trade mode during bearish macro regime.

Example risk rule:

```python
risk_amount = account_equity * risk_per_trade
stop_distance = entry_price * stop_loss_pct
position_size = risk_amount / stop_distance
```

### 9. Future Architecture Refactor

The current `quant_pro.py` may eventually be refactored into smaller modules.

Recommended future structure:

```text
quant_system/
├── quant/
│   ├── data_sources.py
│   ├── technical.py
│   ├── vcp.py
│   ├── bollinger.py
│   ├── fundamental.py
│   ├── chip.py
│   ├── scoring.py
│   ├── risk.py
│   ├── report.py
│   ├── telegram_bot.py
│   └── scheduler.py
├── backtests/
│   ├── datafeed.py
│   ├── strategies.py
│   └── run_backtest.py
├── optimizers/
│   ├── bayesian.py
│   ├── genetic.py
│   └── surrogate_evolution.py
├── models/
├── config/
├── reports/
├── quant_pro.py
└── README.md
```

### 3. Future Research Direction

Future research may focus on:

- Financially grounded loss functions.
- Alpha factor discovery.
- Time-series Transformer models.
- CNN-based chart pattern recognition.
- VCP detection quality evaluation.
- Hybrid Minervini + Bollinger + machine learning scoring.
- Surrogate-assisted strategy optimization.
- Portfolio-level risk control.
- Backtrader-based walk-forward validation.
- Telegram-based research dashboard.

These features are research directions only. They should be implemented gradually and validated with historical data before being used for any real decision-making.

## Project Structure

```text
quant_system/
├── models/                   # Random Forest macro models
├── config/                   # Optimized parameters from weekly evolution
├── reports/                  # Generated charts and strategy cards
├── .env                      # Environment variables, not tracked by Git
├── .gitignore
├── docker-compose.yml        # Docker deployment configuration
├── Dockerfile                # Docker environment blueprint
├── requirements.txt          # Python dependencies
├── README.md
├── backtest_minervini_bt.py  # Backtrader historical backtesting engine
├── evolve_nsga2.py           # NSGA-II multi-objective strategy evolution module
├── optimize_bayesian.py      # Bayesian optimization parameter tuning module
├── quant_pro.py              # Main console and Telegram bot orchestrator
└── train_macro_model.py      # Macro market trend training module
```

## Recommended `.gitignore`

Make sure your `.gitignore` includes sensitive and generated files:

```gitignore
# Python
__pycache__/
*.pyc
venv/
.env

# Reports and runtime output
reports/
logs/
*.log
*.sqlite3
*.sqlite3-shm
*.sqlite3-wal

# Model and config outputs
models/*.pkl
models/*.pt
models/*.pth

# OS / editor
.DS_Store
.vscode/
.idea/
```

## TODO

### High Priority

- [x] Move all sensitive values to `.env`
- [x] Make sure `.env` is included in `.gitignore`
- [x] Add validation for Telegram token format
- [x] Add safer error messages when Telegram bot initialization fails
- [x] Add retry and timeout handling for all external data sources
- [x] Add rate limit protection for Goodinfo / FinMind requests
- [x] Add unit tests for ticker normalization and scoring logic
- [x] Add VCP detection module
- [x] Add Bollinger Band squeeze and breakout module
- [x] Add Minervini trend template as a reusable hard filter

### Medium Priority

- [x] Refactor `quant_pro.py` into smaller modules
  - `data_sources.py`
  - `technical.py`
  - `vcp.py`
  - `bollinger.py`
  - `fundamental.py`
  - `chip.py`
  - `scoring.py`
  - `risk.py`
  - `report.py`
  - `telegram_bot.py`
- [x] Add CLI arguments, for example:
  - `python quant_pro.py --scan tw`
  - `python quant_pro.py --scan us`
  - `python quant_pro.py --ticker 2330`
- [x] Add configurable scan universe
- [x] Add configurable score weights
- [x] Add SQLite or CSV cache for external API results
- [x] Add logging rotation for long-running deployment
- [x] Add Docker healthcheck
- [x] Add Backtrader custom data feed for hybrid signals
- [x] Add GA + GPR + EI optimizer integration

### Low Priority

- [ ] Add web dashboard
- [ ] Add backtest report export
- [ ] Add more chart styles
- [ ] Add portfolio watchlist mode
- [ ] Add Telegram inline keyboard support
- [ ] Add multi-user Telegram permission control
- [ ] Add CNN / LSTM / Transformer research experiments
- [ ] Add walk-forward validation workflow

## Notes

- This project is currently for internal testing and after-market stock recommendation research only.
- This project does not place orders, execute trades, or manage real investment accounts.
- This project does not provide investment advice.
- Investment funds involve risks. Please read the public prospectus carefully before investing.
- The current implementation focuses on technical analysis, macro model training, Telegram reporting, and dynamic optimization experiments.
- Future AI, deep learning, VCP, Bollinger Band, Backtrader, and GA + GPR + EI features are research directions and should be validated before practical use.
- Make sure Telegram bot settings are correctly configured before running the system.
- If you use `My-TW-Coverage`, make sure the local database path is correctly configured or mounted into Docker.

## Disclaimer

All analysis results are generated automatically by the program and are for research, learning, internal testing, and after-market stock recommendation reference only.

This system does not execute orders, does not provide automated trading services, and does not guarantee any investment return.

Investment involves risks. Fund investments involve specific risks, fees, and product characteristics. Please read the public prospectus carefully before investing.

Please evaluate investment risks independently before making any trading decisions.
