import yfinance as yf
import pandas as pd
import backtrader as bt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import warnings

warnings.filterwarnings('ignore')

# Simple momentum strategy for fast optimizer backtests.
class QuickMomentumStrategy(bt.Strategy):
    params = (('hard_stop', 0.08), ('ma_period', 20))

    def __init__(self):
        self.ma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.ma_period)
        self.order = None
        self.buyprice = None

    def next(self):
        if self.order: return
        
        if not self.position:
            # Enter on a moving-average breakout.
            if self.datas[0].close[0] > self.ma[0] and self.datas[0].close[-1] <= self.ma[-1]:
                self.order = self.buy()
                self.buyprice = self.datas[0].close[0]
        else:
            pnl_pct = (self.datas[0].close[0] - self.buyprice) / self.buyprice
            # Exit below the moving average or hard stop.
            if self.datas[0].close[0] < self.ma[0] or pnl_pct <= -self.params.hard_stop:
                self.order = self.sell()

# 1. Download sample data once to speed up optimization.
print("📥 準備回測資料 (2454 聯發科)...")
df = yf.download('2454.TW', start='2021-01-01', end='2025-01-01', auto_adjust=True, progress=False)
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
data = bt.feeds.PandasData(dataname=df)

# 2. Define the search space.
space = [
    Real(0.03, 0.15, name='hard_stop'),     # Stop-loss range: 3% to 15%.
    Integer(10, 60, name='ma_period'),      # MA range: 10MA to 60MA.
]

# 3. Define the objective function.
@use_named_args(space)
def objective(**params):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(data)
    cerebro.addstrategy(QuickMomentumStrategy, hard_stop=params['hard_stop'], ma_period=params['ma_period'])
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(commission=0.002) # Approximate trading cost.
    
    cerebro.run()
    
    final_value = cerebro.broker.getvalue()
    return_rate = (final_value / 1000000.0) - 1
    
    # gp_minimize minimizes, so return negative return.
    print(f"🔄 測試參數: 停損 {params['hard_stop']*100:.1f}%, 均線 {params['ma_period']}MA ➔ 報酬率: {return_rate*100:.1f}%")
    return -return_rate

def run_bayesian_opt():
    print("\n🎯 啟動 GPR + EI 貝氏最佳化 (AI 尋優引擎)...")
    print("這將會跑 20 次回測，請稍候...")
    
    res = gp_minimize(
        objective, 
        space, 
        n_calls=20,         # Try 20 parameter sets.
        n_random_starts=5,  # Use 5 random starts before guided search.
        random_state=42
    )
    
    print("\n" + "="*40)
    print("🏆 尋優完成！找到量化聖杯參數：")
    print(f"📍 最佳停損設定: -{res.x[0]*100:.2f}%")
    print(f"📍 最佳防守均線: {res.x[1]} MA")
    print(f"💰 該參數預期歷史報酬率: {-res.fun*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_bayesian_opt()
