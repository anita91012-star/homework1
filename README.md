# homework1_梁芷芸
# AI 新秀 - Homework 1: 房價預測 (Regression)

## 實作成果
* **Kaggle 分數 (Log RMSE):** 0.16715
* **模型:** 線性迴歸 (Linear Regression)

## 特徵工程技巧
1. **資料清理**: 處理缺失值，針對數值欄位補中位數，類別欄位補 'None'。
2. **對數轉換 (Log Scaling)**: 針對房價 (SalePrice) 取 Log，有效解決數據偏態問題。
3. **異常值處理 (Data Clipping)**: 移除居住面積 (GrLivArea) 超過 4000 且價格異常的樣本。
4. **類別轉換 (Encoding)**: 使用 One-Hot Encoding 將文字特徵轉為數值。



# AI 新秀 - Homework 2: Titanic

### Titanic Survival Prediction
* **Kaggle Score:** 0.77511
* **技術重點:** 使用 RandomForest 分類器處理乘客特徵，並透過 One-Hot Encoding 轉換性別資料。
