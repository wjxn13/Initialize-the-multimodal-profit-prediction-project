"""
利用现有预处理文件重新构建特征并训练 LightGBM
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 路径
RAW_DATA = Path(r"D:\.kaggle\商店销售\Global Superstore.csv")
DATE_DIR = Path(r"D:\.kaggle\商店销售\date")
SCALER_FILE = DATE_DIR / "scalers_v6.pkl"
ENCODER_FILE = DATE_DIR / "encoders_v6.pkl"
EMBEDDINGS_FILE = DATE_DIR / "text_embeddings_v6.npy"
PRODUCT_MAP_FILE = DATE_DIR / "product_map_v6.pkl"
OUTPUT_MODEL = Path(r"D:\.kaggle\商店销售\lgb_model.txt")
OUTPUT_FEATURES = DATE_DIR / "features_v6.npz"

# 加载预处理器
with open(SCALER_FILE, 'rb') as f:
    scaler = pickle.load(f)['scaler']
    num_cols = scaler.feature_names_in_  # 从 sklearn 1.0+ 可获取特征名
    # 如果上面失败，手动指定
    if not hasattr(scaler, 'feature_names_in_'):
        num_cols = ['Sales', 'Quantity', 'Discount', 'Shipping Cost',
                    'order_item_count', 'price_per_item', 'shipping_cost_ratio', 'discount_flag',
                    'month', 'quarter', 'year_norm', 'week_sin', 'week_cos',
                    'subcat_avg_profit', 'subcat_avg_sales', 'market_avg_profit']

with open(ENCODER_FILE, 'rb') as f:
    encoder = pickle.load(f)['encoder']
    cat_cols = encoder.feature_names_in_  # 同样从 sklearn 获取
    if not hasattr(encoder, 'feature_names_in_'):
        cat_cols = ['Market', 'Region', 'Category', 'Sub-Category', 'Segment', 'Ship Mode', 'Order Priority']

product_map = pickle.load(open(PRODUCT_MAP_FILE, 'rb'))
embeddings = np.load(EMBEDDINGS_FILE)

# 加载原始数据并清洗（与之前训练一致）
df = pd.read_csv(RAW_DATA)
drop_cols = ['ji_lu-shu', 'Order Date', 'Ship Date', 'Row ID',
             'Customer ID', 'Customer Name', 'Product ID', 'City', 'State', 'Country']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
bad_mask = (df['Sales'] <= 0) | (df['Quantity'] <= 0)
df = df[~bad_mask]
df.dropna(inplace=True)

iso = IsolationForest(contamination=0.01, random_state=42)
outlier_labels = iso.fit_predict(df[['Profit']])
df = df[outlier_labels == 1]

q_low, q_high = df['Profit'].quantile(0.005), df['Profit'].quantile(0.995)
df['Profit'] = df['Profit'].clip(q_low, q_high)
df['Product Name'] = df['Product Name'].astype(str)

# 划分训练/测试（年份）
train = df[df['Year'] < 2014].copy()
test = df[df['Year'] == 2014].copy()

# 订单商品数
order_count = df.groupby('Order ID').size().rename('order_item_count')
train = train.merge(order_count, on='Order ID', how='left')
test = test.merge(order_count, on='Order ID', how='left')
train.drop(columns=['Order ID'], inplace=True)
test.drop(columns=['Order ID'], inplace=True)

# 统计特征（从训练集计算）
subcat_stats = train.groupby('Sub-Category').agg(
    subcat_avg_profit=('Profit', 'mean'),
    subcat_avg_sales=('Sales', 'mean')
).reset_index()
train = train.merge(subcat_stats, on='Sub-Category', how='left')
test = test.merge(subcat_stats, on='Sub-Category', how='left')

market_stats = train.groupby('Market').agg(
    market_avg_profit=('Profit', 'mean')
).reset_index()
train = train.merge(market_stats, on='Market', how='left')
test = test.merge(market_stats, on='Market', how='left')

# 衍生特征
def add_derived_features(data):
    data = data.copy()
    data['price_per_item'] = data['Sales'] / (data['Quantity'] + 1e-8)
    data['shipping_cost_ratio'] = data['Shipping Cost'] / (data['Sales'] + 1e-8)
    data['discount_flag'] = (data['Discount'] > 0).astype(int)
    data['month'] = (data['weeknum'] // 4.33 + 1).clip(1, 12).astype(int)
    data['quarter'] = ((data['month'] - 1) // 3 + 1).astype(int)
    data['year_norm'] = data['Year'] - 2011
    data['week_sin'] = np.sin(2 * np.pi * data['weeknum'] / 53.0)
    data['week_cos'] = np.cos(2 * np.pi * data['weeknum'] / 53.0)
    return data

train = add_derived_features(train)
test = add_derived_features(test)

# 数值特征
X_num_train = scaler.transform(train[num_cols])
X_num_test = scaler.transform(test[num_cols])

# 类别特征
X_cat_train = encoder.transform(train[cat_cols])
X_cat_test = encoder.transform(test[cat_cols])

# 文本嵌入
train_idx = train['Product Name'].map(product_map).values
test_idx = test['Product Name'].map(product_map).values
X_text_train = embeddings[train_idx]
X_text_test = embeddings[test_idx]

# 目标
y_train = train['Profit'].values
y_test = test['Profit'].values

# 保存特征缓存（避免下次再处理）
np.savez_compressed(OUTPUT_FEATURES,
                    X_num_train=X_num_train, X_num_test=X_num_test,
                    X_cat_train=X_cat_train, X_cat_test=X_cat_test,
                    X_text_train=X_text_train, X_text_test=X_text_test,
                    y_train=y_train, y_test=y_test)

# 拼接并训练 LightGBM
X_train_all = np.concatenate([X_num_train, X_cat_train, X_text_train], axis=1)
X_test_all = np.concatenate([X_num_test, X_cat_test, X_text_test], axis=1)

model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10,
                          random_state=42, verbose=-1)
model.fit(X_train_all, y_train)
preds = model.predict(X_test_all)
r2 = r2_score(y_test, preds)
print(f"验证 R²: {r2:.4f}")

model.booster_.save_model(str(OUTPUT_MODEL))
print(f"✅ 模型已保存至 {OUTPUT_MODEL}")