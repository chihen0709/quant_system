# Quant System

A fully automated quantitative stock technical analysis and strategy evolution tool.

> This project is for self-study and quantitative stock analysis practice.  
> It does not constitute financial advice.

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
  - Runs NSGA-II multi-objective genetic algorithms to optimize trading parameters.

- **Telegram Integration**
  - Sends technical diagnostic reports, strategy entry/exit cards, and macro dashboards to Telegram.

## Configuration

Before running the system, open `quant_pro.py` and modify the following global variables.

### Telegram Bot Settings

```python
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
```

### Local Database Path

The system relies on an external Markdown financial database folder.  
Please make sure the path is correct.

```python
MY_TW_COVERAGE_PATH = "/path/to/your/My-TW-Coverage"
```

## Quick Start

### Environment Setup

For Debian / Ubuntu Linux:

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

Install required Python packages:

```bash
pip install yfinance pandas numpy ta pyTelegramBotAPI playwright scikit-learn scikit-optimize deap backtrader matplotlib schedule
```

Install Playwright browser dependency:

```bash
playwright install chromium
```

## Usage

### 1. Train Macro Model

Train the Random Forest classifier for macro market decision-making:

```bash
python train_macro_model.py
```

### 2. Run Bayesian Optimization

Find the current market's optimal parameters using Bayesian Optimization:

```bash
python optimize_bayesian.py
```

### 3. Run Strategy Evolution

Run NSGA-II multi-objective strategy evolution:

```bash
python evolve_nsga2.py
```

### 4. Run Quant Analysis

Start the main console, Telegram bot, and scheduling loop:

```bash
python quant_pro.py
```

## Process Management & Scheduling

### Automated Scheduling

When `quant_pro.py` runs continuously, the system follows the internal schedule below:

| Time | Task |
|---|---|
| Every day at 16:30 CST | Run Taiwan stock market scan and send Telegram notifications |
| Every day at 05:00 CST | Run US stock market scan and send Telegram notifications |
| Every Saturday at 02:00 CST | Run weekly optimization and strategy evolution |

The weekly optimization process writes the latest optimized parameters to:

```text
config/best_params.json
```

### Run in Background

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

## Project Structure

```text
quant_system/
├── models/                  # Random Forest macro models
├── config/                  # Optimized parameters from weekly evolution
├── reports/                 # Generated charts and strategy cards
├── README.md
├── backtest_minervini_bt.py  # Backtrader historical backtesting engine
├── evolve_nsga2.py           # NSGA-II multi-objective strategy evolution module
├── optimize_bayesian.py      # Bayesian optimization parameter tuning module
├── quant_pro.py              # Main console and Telegram bot orchestrator
└── train_macro_model.py      # Macro market trend training module
```

## Notes

- This project is for self-study and quantitative stock analysis practice.
- This project does not provide investment advice.
- The current implementation focuses on technical analysis, macro model training, and dynamic optimization experiments.
- Before running any scripts, make sure the Python virtual environment is activated.
- Make sure Telegram bot settings and local database paths are configured before running `quant_pro.py`.

## Disclaimer

All analysis results are generated automatically by the program and are for research and learning purposes only.  
Please evaluate investment risks independently before making any trading decisions.
