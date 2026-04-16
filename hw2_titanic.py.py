import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. 讀取資料
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 2. 選擇簡單但有效的特徵
features = ["Pclass", "Sex", "SibSp", "Parch"]

# 合併訓練與測試集一起處理 One-Hot Encoding，確保特徵數量對齊
all_data = pd.concat((train_data[features], test_data[features]), axis=0)
all_data = pd.get_dummies(all_data)

# 拆回訓練集與測試集
X = all_data[:len(train_data)]
X_test = all_data[len(train_data):]
y = train_data["Survived"]

# 3. 建立隨機森林模型 (Random Forest)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# 4. 預測並存檔
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('hw2_gender_submission.csv', index=False)

print("Titanic prediction complete. Score saved to hw2_gender_submission.csv")