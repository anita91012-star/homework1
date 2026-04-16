import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 讀取資料
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 資料修剪 (Data Clipping) - 教材提到的技巧，移除居住面積異常大的極端值
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)

# 3. 準備目標值：對房價取 Log (Log Scaling)
# 這樣模型預測的是 log(price)，能有效處理偏態數據
y = np.log1p(train_df['SalePrice'])

# 4. 特徵工程：合併訓練與測試集一起處理（確保特徵數量對齊）
all_data = pd.concat((train_df.drop(['SalePrice'], axis=1), test_df), axis=0)
all_data_id = all_data['Id']
all_data = all_data.drop(['Id'], axis=1)

# A. 處理缺失值 (教材：補齊空值)
# 找出所有數值欄位並補中位數
numeric_cols = all_data.select_dtypes(include=['number']).columns
all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].median())

# B. 類別變數轉數字 (教材：One-Hot Encoding)
all_data = pd.get_dummies(all_data)

# 5. 拆回訓練集與測試集
X = all_data[:len(train_df)]
X_test = all_data[len(train_df):]

# 6. 切分驗證集 (Validation Set)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 訓練模型 (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# 8. 評估模型性能 (計算 Log 空間下的 RMSE)
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"你的驗證集 Log RMSE 為: {rmse:.5f}")

# 9. 預測測試集並「轉回」原始價格 (Inverse Log)
# 使用 np.expm1 將 log(price) 轉回真正的價格
final_log_preds = model.predict(X_test)
final_preds = np.expm1(final_log_preds)

# 10. 產出提交檔案
submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': final_preds})
submission.to_csv('submission_optimized.csv', index=False)
print("優化後的 submission 檔案已存檔！請重新上傳 Kaggle。")