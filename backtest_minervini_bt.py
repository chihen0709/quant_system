import datetime
import math
import yfinance as yf
import pandas as pd
import backtrader as bt
import warnings


warnings.filterwarnings('ignore')

# ==========================================
# 1. Minervini core strategy
# ==========================================
class MinerviniStrategy(bt.Strategy):
    # Strategy parameters for future Bayesian tuning.
    params = (
        ('hard_stop', 0.08),       # 8% hard stop.
        ('exit_ma_period', 20),    # Exit below the 20MA.
        ('printlog', True),        # Print trade logs.
    )

    def log(self, txt, dt=None):
        """Log trade events."""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'[{dt.isoformat()}] {txt}')

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low

        # Moving average indicators.
        self.ma20 = bt.indicators.SMA(self.datas[0], period=self.params.exit_ma_period)
        self.ma50 = bt.indicators.SMA(self.datas[0], period=50)
        self.ma150 = bt.indicators.SMA(self.datas[0], period=150)
        self.ma200 = bt.indicators.SMA(self.datas[0], period=200)

        # 52-week high and low, about 250 trading days.
        self.high52w = bt.indicators.Highest(self.datahigh, period=250)
        self.low52w = bt.indicators.Lowest(self.datalow, period=250)

        # Momentum indicators.
        self.rsi = bt.indicators.RSI_SMA(self.dataclose, period=14)
        self.macd = bt.indicators.MACD(self.dataclose)

        # Track orders and entry price.
        self.order = None
        self.buyprice = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🟢 買入執行: 價格 {order.executed.price:.2f}, 成本 {order.executed.value:.2f}, 手續費 {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
            elif order.issell():
                self.log(f'🔴 賣出執行: 價格 {order.executed.price:.2f}, 手續費 {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('⚠️ 訂單取消/保證金不足/拒絕')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        pnl_str = f"毛利 {trade.pnl:.2f}, 淨利 {trade.pnlcomm:.2f}"
        if trade.pnlcomm > 0:
            self.log(f'💰 交易獲利結算 ➔ {pnl_str}')
        else:
            self.log(f'🩸 交易虧損結算 ➔ {pnl_str}')

    def next(self):
        # Skip while an order is pending.
        if self.order:
            return

        # ==================================
        # Entry rules based on the Minervini Trend Template.
        # ==================================
        if not self.position:
            # Rule 1: price > 50MA > 150MA > 200MA.
            cond1 = self.dataclose[0] > self.ma50[0] > self.ma150[0] > self.ma200[0]
            # Rule 2: 200MA is above its level one month ago.
            cond2 = self.ma200[0] > self.ma200[-20]
            # Rule 3: price is at least 30% above the 52-week low.
            cond3 = self.dataclose[0] > (self.low52w[0] * 1.30)
            # Rule 4: price is within 25% of the 52-week high.
            cond4 = self.dataclose[0] > (self.high52w[0] * 0.75)
            # Rule 5: RSI > 60 and MACD histogram > 0.
            cond5 = self.rsi[0] > 60 and (self.macd.macd[0] - self.macd.signal[0]) > 0
            # Rule 6: price is above the 20MA.
            cond6 = self.dataclose[0] > self.ma20[0]

            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
                self.log(f'🚀 觸發 Minervini 攻擊型買進訊號！')
                # Use most available cash, leaving room for fees.
                size = math.floor((self.broker.get_cash() * 0.95) / self.dataclose[0]) 
                self.order = self.buy(size=size)

        # ==================================
        # Exit rules for trailing protection and hard stop.
        # ==================================
        else:
            current_pnl_pct = (self.dataclose[0] - self.buyprice) / self.buyprice

            # Exit A: close below the 20MA defense line.
            if self.dataclose[0] < self.ma20[0]:
                self.log(f'🛡️ 跌破 {self.params.exit_ma_period}MA 防守線，執行獲利了結/減碼賣出')
                self.order = self.sell(size=self.position.size)
                
            # Exit B: hard stop is hit.
            elif current_pnl_pct <= -self.params.hard_stop:
                self.log(f'💀 觸發硬停損 (-{self.params.hard_stop*100:.1f}%)，斷尾求生賣出')
                self.order = self.sell(size=self.position.size)


# ==========================================
# 2. Run the Cerebro backtest engine
# ==========================================
def run_backtest(ticker, start_date, end_date):
    print(f"📥 正在從 Yahoo Finance 下載 {ticker} 歷史資料...")
    
    # Download price history and flatten MultiIndex columns.
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    if df.empty:
        print("❌ 找不到資料，請檢查股票代碼。")
        return

    # Convert the DataFrame for Backtrader.
    data = bt.feeds.PandasData(dataname=df)

    # Create the Cerebro engine.
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    
    # Add the strategy.
    cerebro.addstrategy(MinerviniStrategy, hard_stop=0.08, exit_ma_period=20)

    # Initial capital: 1,000,000 TWD.
    INITIAL_CASH = 1000000.0
    cerebro.broker.setcash(INITIAL_CASH)
    # Approximate Taiwan stock trading fee.
    cerebro.broker.setcommission(commission=0.002)

    # Add performance analyzers.
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print(f'💰 初始資產淨值: {cerebro.broker.getvalue():,.2f}')
    print("=" * 50)
    print("⏳ 開始執行歷史回測...")
    
    # Run the backtest.
    results = cerebro.run()
    strat = results[0]

    # Collect performance metrics.
    final_value = cerebro.broker.getvalue()
    total_return = ((final_value / INITIAL_CASH) - 1) * 100
    
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    if sharpe_ratio is None: sharpe_ratio = 0
        
    mdd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
    
    trade_info = strat.analyzers.trades.get_analysis()
    total_trades = trade_info.get('total', {}).get('closed', 0)
    won_trades = trade_info.get('won', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    print("=" * 50)
    print(f"📊 **【{ticker} 回測績效報告】** ({start_date} ~ {end_date})")
    print(f"🔹 最終資產淨值: {final_value:,.2f} (總報酬率: {total_return:.2f}%)")
    print(f"🔹 總交易次數: {total_trades} 次")
    print(f"🔹 交易勝率: {win_rate:.2f}%")
    print(f"🔹 最大資金回撤 (MDD): {mdd:.2f}%")
    print(f"🔹 夏普比率 (Sharpe Ratio): {sharpe_ratio:.2f}")
    print("=" * 50)

    # Plot the backtest chart.
    try:
        print("📈 正在繪製回測圖表...")
        # Use candlesticks for price bars.
        cerebro.plot(style='candlestick', barup='red', bardown='green', volume=True)
    except Exception as e:
        print(f"⚠️ 繪圖失敗 (可能是遠端/無UI環境導致): {e}")

if __name__ == '__main__':
    # Backtest 2454.TW over the sample period.
    run_backtest('2454.TW', start_date='2019-01-01', end_date='2024-01-01')
