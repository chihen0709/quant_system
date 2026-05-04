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
    Generate mock market-positioning history.
    """
    np.random.seed(42)
    print("📥 正在載入歷史大盤籌碼資料...")
    
    # Simulate four feature groups.
    foreign_fut = np.random.normal(0, 10000, days)   # Foreign futures net OI.
    pcr_ratio = np.random.normal(100, 20, days)      # Options put-call ratio.
    retail_sent = np.random.normal(0, 15, days)      # Retail sentiment ratio.
    top_10 = np.random.normal(0, 5000, days)         # Top 10 traders net position.
    
    # Target: whether the market rises over the next 5 days.
    # Add simple nonlinear rules for the model to learn.
    y = np.where((foreign_fut > 0) & (pcr_ratio > 105) & (retail_sent < 5), 1, 0)
    # Add noise to mimic market uncertainty.
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
    
    # Define features and target.
    X = df[['Foreign_Fut', 'PCR_Ratio', 'Retail_Sentiment', 'Top_10_Traders']]
    y = df['Target_Up_5d']
    
    # Split by time, without shuffling.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("\n🧠 開始訓練隨機森林模型 (Random Forest)...")
    # Use a shallow random forest to reduce overfitting.
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # Predict and evaluate.
    y_pred = rf_model.predict(X_test)
    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print("-" * 40)
    print(f"✅ 訓練完成！")
    print(f"📊 準確率 (Accuracy): {acc*100:.2f}%")
    print(f"📊 模型辨識力 (AUC): {auc:.3f} (大於 0.6 即有實戰價值)")
    print("-" * 40)
    
    # Save the trained model.
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/macro_rf_model.pkl')
    print("💾 模型已成功儲存至 'models/macro_rf_model.pkl'")

if __name__ == "__main__":
    train_macro_model()
