import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

def generate_mock_macro_data(days=1500):
    """
    產生高仿真的歷史大盤籌碼資料 (假設我們已經寫爬蟲抓了過去 6 年的資料)
    """
    np.random.seed(42)
    print("📥 正在載入歷史大盤籌碼資料...")
    
    # 模擬 4 個維度的特徵
    foreign_fut = np.random.normal(0, 10000, days)   # 外資期貨淨未平倉口數
    pcr_ratio = np.random.normal(100, 20, days)      # 選擇權 PCR (通常 100 是多空分水嶺)
    retail_sent = np.random.normal(0, 15, days)      # 散戶小台多空比 (%)
    top_10 = np.random.normal(0, 5000, days)         # 前十大交易人淨口數
    
    # 模擬目標值 (未來 5 天大盤是否上漲：1=漲, 0=跌)
    # 這裡刻意加入一些非線性邏輯讓模型去學：
    # 當外資多單大於 0 且 PCR > 105 時，上漲機率高；當散戶做多(>5%)時，下跌機率高(反指標)
    y = np.where((foreign_fut > 0) & (pcr_ratio > 105) & (retail_sent < 5), 1, 0)
    # 加入隨機雜訊，模擬真實市場的不確定性
    noise = np.random.randint(0, 2, days)
    y = np.where(np.random.rand(days) > 0.8, noise, y)
    
    df = pd.DataFrame({
        'Foreign_Fut': foreign_fut,
        'PCR_Ratio': pcr_ratio,
        'Retail_Sentiment': retail_sent,
        'Top_10_Traders': top_10,
        'Target_Up_5d': y
    })
    return df

def train_macro_model():
    df = generate_mock_macro_data()
    
    # 定義特徵 X 與目標 Y
    X = df[['Foreign_Fut', 'PCR_Ratio', 'Retail_Sentiment', 'Top_10_Traders']]
    y = df['Target_Up_5d']
    
    # 切割訓練集(前 80%) 與 測試集(後 20%) - 時間序列不能洗牌
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("\n🧠 開始訓練隨機森林模型 (Random Forest)...")
    # 建立模型：300棵樹，最大深度5(避免過擬合)
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # 預測與評估
    y_pred = rf_model.predict(X_test)
    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print("-" * 40)
    print(f"✅ 訓練完成！")
    print(f"📊 準確率 (Accuracy): {acc*100:.2f}%")
    print(f"📊 模型辨識力 (AUC): {auc:.3f} (大於 0.6 即有實戰價值)")
    print("-" * 40)
    
    # 將模型存檔 (這就是兵工廠造出來的武器)
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/macro_rf_model.pkl')
    print("💾 模型已成功儲存至 'models/macro_rf_model.pkl'")

if __name__ == "__main__":
    train_macro_model()

