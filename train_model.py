import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score

# 特征文件就在本地的 date 文件夹中
data = np.load(r"D:\.kaggle\商店销售\date\features_v6.npz", allow_pickle=True)
X_num_train = data['X_num_train']
X_cat_train = data['X_cat_train']
X_text_train = data['X_text_train']
y_train = data['y_train']
X_num_test = data['X_num_test']
X_cat_test = data['X_cat_test']
X_text_test = data['X_text_test']
y_test = data['y_test']

X_train = np.concatenate([X_num_train, X_cat_train, X_text_train], axis=1)
X_test  = np.concatenate([X_num_test, X_cat_test, X_text_test], axis=1)

model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10,
                          random_state=42, verbose=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"验证 R²: {r2_score(y_test, preds):.4f}")

# 保存到商店销售文件夹根目录
out_path = Path(r"D:\.kaggle\商店销售\lgb_model.txt")
model.booster_.save_model(str(out_path))
print(f"✅ 模型已保存至 {out_path}")