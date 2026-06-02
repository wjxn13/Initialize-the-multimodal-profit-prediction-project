"""
多模态利润预测 - 最终优化版 (v6)
新增：统计特征、Optuna 自动调参、模型集成、文本 PCA 降维
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
import logging, sys, os, json, pickle, warnings, gc
from datetime import datetime
from pathlib import Path
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
class Config:
    DATA_DIR = Path(r"D:\.kaggle\商店销售\date")
    RAW_DATA = Path(r"D:\.kaggle\商店销售\Global Superstore.csv")
    MODEL_DIR = Path(r"D:\.kaggle\商店销售\models\all-MiniLM-L6-v2")

    # v6 缓存文件
    CLEANED_DATA = DATA_DIR / "cleaned_data_v6.pkl"
    TEXT_EMBEDDINGS = DATA_DIR / "text_embeddings_v6.npy"
    TEXT_EMBEDDINGS_PCA = DATA_DIR / "text_embeddings_pca_v6.npy"
    PRODUCT_MAPPING = DATA_DIR / "product_map_v6.pkl"
    FEATURES = DATA_DIR / "features_v6.npz"
    SCALERS = DATA_DIR / "scalers_v6.pkl"
    ENCODERS = DATA_DIR / "encoders_v6.pkl"
    MODEL_WEIGHTS = DATA_DIR / "best_model_v6.pt"
    LGB_MODEL = Path(r"D:\.kaggle\lgb_model.txt")      # 英文路径避免乱码
    TRAIN_HISTORY = DATA_DIR / "history_v6.json"
    SHAP_PNG = DATA_DIR / "shap_summary_v6.png"

    # 深度学习超参
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-4
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 384
    ATTN_HEADS = 4
    DROPOUT = 0.3

    # 特征工程参数
    TEXT_PCA_DIM = 128                 # 文本嵌入降维维度
    USE_OPTUNA = False                 # 是否启用 Optuna 超参搜索（耗时较长）
    OPTUNA_TRIALS = 30                 # Optuna 尝试次数

    # 集成权重（可通过验证集调整，此处取经验值）
    ENSEMBLE_WEIGHT_LGB = 0.65
    ENSEMBLE_WEIGHT_DL = 0.35

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True
    SEED = 42

Config.DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 日志 ====================
def setup_logging():
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    log_file = Config.DATA_DIR / f"pipeline_v6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logging.info("=" * 60)
    logging.info("🚀 多模态利润预测 - 最终优化版 (v6)")
    logging.info(f"📁 日志: {log_file}  设备: {Config.DEVICE}")
    return logger

logger = setup_logging()
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
if Config.DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(Config.SEED)

# ==================== 数据处理函数 ====================
def load_and_clean_data():
    if Config.CLEANED_DATA.exists():
        logger.info("✅ 加载已清洗数据")
        return pd.read_pickle(Config.CLEANED_DATA)

    df = pd.read_csv(Config.RAW_DATA)
    logger.info(f"原始: {df.shape}")

    drop_cols = ['ji_lu-shu', 'Order Date', 'Ship Date', 'Row ID',
                 'Customer ID', 'Customer Name', 'Product ID', 'City', 'State', 'Country']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    bad_mask = (df['Sales'] <= 0) | (df['Quantity'] <= 0)
    df = df[~bad_mask]
    df.dropna(inplace=True)

    iso = IsolationForest(contamination=0.01, random_state=Config.SEED)
    outlier_labels = iso.fit_predict(df[['Profit']])
    df = df[outlier_labels == 1]

    q_low, q_high = df['Profit'].quantile(0.005), df['Profit'].quantile(0.995)
    df['Profit'] = df['Profit'].clip(q_low, q_high)

    df['Product Name'] = df['Product Name'].astype(str)
    df.to_pickle(Config.CLEANED_DATA)
    logger.info(f"✅ 清洗完成: {df.shape}")
    return df

def extract_text_embeddings(df):
    if Config.TEXT_EMBEDDINGS.exists() and Config.PRODUCT_MAPPING.exists():
        logger.info("✅ 加载已有文本嵌入")
        return np.load(Config.TEXT_EMBEDDINGS), pickle.load(open(Config.PRODUCT_MAPPING, 'rb'))

    unique_products = df['Product Name'].unique()
    logger.info(f"唯一产品数: {len(unique_products)}")

    if Config.MODEL_DIR.exists():
        model = SentenceTransformer(str(Config.MODEL_DIR), device=Config.DEVICE)
    else:
        return extract_text_embeddings_offline(df)

    embeddings = model.encode(unique_products.tolist(), batch_size=64,
                              show_progress_bar=True, normalize_embeddings=True)
    product_map = dict(zip(unique_products, range(len(unique_products))))
    np.save(Config.TEXT_EMBEDDINGS, embeddings)
    pickle.dump(product_map, open(Config.PRODUCT_MAPPING, 'wb'))
    logger.info(f"✅ 文本嵌入保存: {embeddings.shape}")
    return embeddings, product_map

def extract_text_embeddings_offline(df):
    logger.info("🧠 使用 TF-IDF+SVD 离线方案")
    unique_products = df['Product Name'].unique()
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    tfidf_matrix = tfidf.fit_transform(unique_products)
    svd = TruncatedSVD(n_components=384, random_state=Config.SEED)
    embeddings = normalize(svd.fit_transform(tfidf_matrix))
    product_map = dict(zip(unique_products, range(len(unique_products))))
    np.save(Config.TEXT_EMBEDDINGS, embeddings)
    pickle.dump(product_map, open(Config.PRODUCT_MAPPING, 'wb'))
    return embeddings, product_map

def apply_text_pca(embeddings, product_map, df):
    """对文本嵌入进行 PCA 降维（使用全部产品嵌入拟合，避免数据泄露）"""
    if Config.TEXT_EMBEDDINGS_PCA.exists():
        logger.info("✅ 加载已有 PCA 文本嵌入")
        return np.load(Config.TEXT_EMBEDDINGS_PCA), product_map

    logger.info(f"🔧 文本嵌入 PCA 降维: {embeddings.shape[1]} -> {Config.TEXT_PCA_DIM}")
    pca = PCA(n_components=Config.TEXT_PCA_DIM, random_state=Config.SEED)
    embeddings_pca = pca.fit_transform(embeddings)
    np.save(Config.TEXT_EMBEDDINGS_PCA, embeddings_pca)
    logger.info(f"✅ PCA 文本嵌入保存: {embeddings_pca.shape}")
    return embeddings_pca, product_map

# ==================== 特征工程（增加统计特征） ====================
def build_features(df, embeddings, product_map):
    if Config.FEATURES.exists() and Config.SCALERS.exists() and Config.ENCODERS.exists():
        logger.info("✅ 加载已有特征")
        data = np.load(Config.FEATURES, allow_pickle=True)
        with open(Config.SCALERS, 'rb') as f: scalers = pickle.load(f)
        with open(Config.ENCODERS, 'rb') as f: encoders = pickle.load(f)
        return (data['X_num_train'], data['X_num_test'],
                data['X_cat_train'], data['X_cat_test'],
                data['X_text_train'], data['X_text_test'],
                data['y_train'], data['y_test'],
                scalers, encoders)

    train = df[df['Year'] < 2014].copy()
    test = df[df['Year'] == 2014].copy()
    logger.info(f"训练: {len(train)}  测试: {len(test)}")

    # 订单商品数
    order_count = df.groupby('Order ID').size().rename('order_item_count')
    train = train.merge(order_count, on='Order ID', how='left')
    test = test.merge(order_count, on='Order ID', how='left')
    train.drop(columns=['Order ID'], inplace=True)
    test.drop(columns=['Order ID'], inplace=True)

    # ----- 新增统计特征（仅用训练集计算）-----
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
    logger.info("✅ 添加统计特征: subcat_avg_profit, subcat_avg_sales, market_avg_profit")

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

    num_cols = ['Sales', 'Quantity', 'Discount', 'Shipping Cost',
                'order_item_count', 'price_per_item', 'shipping_cost_ratio', 'discount_flag',
                'month', 'quarter', 'year_norm', 'week_sin', 'week_cos',
                'subcat_avg_profit', 'subcat_avg_sales', 'market_avg_profit']
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(train[num_cols])
    X_num_test = scaler.transform(test[num_cols])

    cat_cols = ['Market', 'Region', 'Category', 'Sub-Category', 'Segment', 'Ship Mode', 'Order Priority']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat_train = encoder.fit_transform(train[cat_cols])
    X_cat_test = encoder.transform(test[cat_cols])
    cat_dims = [len(encoder.categories_[i]) + 1 for i in range(len(cat_cols))]

    train_idx = train['Product Name'].map(product_map).values
    test_idx = test['Product Name'].map(product_map).values
    X_text_train = embeddings[train_idx]
    X_text_test = embeddings[test_idx]

    y_train = train['Profit'].values
    y_test = test['Profit'].values

    np.savez_compressed(Config.FEATURES,
                        X_num_train=X_num_train, X_num_test=X_num_test,
                        X_cat_train=X_cat_train, X_cat_test=X_cat_test,
                        X_text_train=X_text_train, X_text_test=X_text_test,
                        y_train=y_train, y_test=y_test)
    scalers = {'scaler': scaler, 'num_cols': num_cols}
    encoders = {'encoder': encoder, 'cat_cols': cat_cols, 'cat_dims': cat_dims}
    pickle.dump(scalers, open(Config.SCALERS, 'wb'))
    pickle.dump(encoders, open(Config.ENCODERS, 'wb'))
    logger.info(f"✅ 特征保存: 数值 {len(num_cols)} 类, 类别 {len(cat_cols)} 类")
    return (X_num_train, X_num_test, X_cat_train, X_cat_test,
            X_text_train, X_text_test, y_train, y_test, scalers, encoders)

# ==================== LightGBM 超参数调优（可选） ====================
def tune_lightgbm(X_train, y_train):
    logger.info("🔧 启动 Optuna 超参数搜索...")
    import optuna
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'max_depth': trial.suggest_int('max_depth', 6, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': Config.SEED,
            'n_jobs': -1,
            'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=Config.OPTUNA_TRIALS, show_progress_bar=True)
    logger.info(f"最佳参数: {study.best_params}")
    logger.info(f"最佳交叉验证 R²: {study.best_value:.4f}")
    return study.best_params

# ==================== 深度模型（增强自注意力 MLP） ====================
class GatedSelfAttentionMLP(nn.Module):
    def __init__(self, num_dim, cat_dims, text_dim,
                 embedding_dim=32, hidden_dim=384, num_heads=4, dropout=0.3):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(d, embedding_dim) for d in cat_dims])
        cat_out_dim = len(cat_dims) * embedding_dim
        self.text_proj = nn.Linear(text_dim, hidden_dim // 2)
        self.num_proj = nn.Linear(num_dim, hidden_dim // 2)

        combined_dim = hidden_dim // 2 + cat_out_dim + hidden_dim // 2
        self.fc_proj = nn.Linear(combined_dim, hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Softmax(dim=-1)
        )

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_num, x_cat, text_emb):
        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_vec = torch.cat(cat_embeds, dim=1)

        num_feat = self.num_proj(x_num)
        text_feat = self.text_proj(text_emb)

        combined = torch.cat([num_feat, cat_vec, text_feat], dim=1)
        combined = self.fc_proj(combined)

        gate_weights = self.gate(combined)
        weighted = combined * gate_weights.sum(dim=1, keepdim=True)

        weighted = weighted.unsqueeze(1)
        attn_out, _ = self.attn(weighted, weighted, weighted)
        attn_out = self.attn_norm(weighted + attn_out).squeeze(1)

        return self.head(attn_out).squeeze(-1)

class MultiModalDataset(Dataset):
    def __init__(self, X_num, X_cat, X_text, y):
        self.X_num = torch.FloatTensor(X_num)
        self.X_cat = torch.LongTensor(X_cat.astype(np.int64))
        self.X_text = torch.FloatTensor(X_text)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return {'num': self.X_num[idx], 'cat': self.X_cat[idx], 'text': self.X_text[idx], 'y': self.y[idx]}

def train_deep_model(X_num_train, X_cat_train, X_text_train, y_train,
                     X_num_test, X_cat_test, X_text_test, y_test, encoders):
    logger.info("🚀 开始训练深度模型 (增强自注意力 MLP)")
    train_ds = MultiModalDataset(X_num_train, X_cat_train, X_text_train, y_train)
    test_ds = MultiModalDataset(X_num_test, X_cat_test, X_text_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE*2, shuffle=False, pin_memory=True)

    cat_dims = encoders['cat_dims']
    model = GatedSelfAttentionMLP(
        num_dim=X_num_train.shape[1],
        cat_dims=cat_dims,
        text_dim=X_text_train.shape[1],
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_heads=Config.ATTN_HEADS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)

    criterion = nn.HuberLoss(delta=10.0)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP and Config.DEVICE.type == 'cuda')

    best_val_loss = float('inf')
    patience = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            num = batch['num'].to(Config.DEVICE)
            cat = batch['cat'].to(Config.DEVICE)
            text = batch['text'].to(Config.DEVICE)
            y = batch['y'].to(Config.DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                pred = model(num, cat, text)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_loss, preds, targets = 0, [], []
        with torch.no_grad():
            for batch in test_loader:
                num = batch['num'].to(Config.DEVICE)
                cat = batch['cat'].to(Config.DEVICE)
                text = batch['text'].to(Config.DEVICE)
                y = batch['y'].to(Config.DEVICE)
                pred = model(num, cat, text)
                loss = criterion(pred, y)
                val_loss += loss.item()
                preds.extend(pred.cpu().numpy())
                targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        val_mae = mean_absolute_error(targets, preds)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(val_mae)

        logger.info(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), Config.MODEL_WEIGHTS)
            logger.info("   ✅ 最佳模型保存")
        else:
            patience += 1
            if patience >= 15:
                logger.info("⏹️ 早停触发")
                break

    json.dump(history, open(Config.TRAIN_HISTORY, 'w'))
    logger.info("深度模型训练完成")
    return model, history

# ==================== 主流程 ====================
def main():
    try:
        # 1. 数据准备
        df = load_and_clean_data()
        embeddings_full, product_map = extract_text_embeddings(df)
        # PCA 降维（可选，若不需要可注释掉，后续代码使用原始 embeddings_full）
        # embeddings, product_map = apply_text_pca(embeddings_full, product_map, df)
        embeddings = embeddings_full   # 直接使用原始嵌入，如需 PCA 替换此行

        (X_num_train, X_num_test, X_cat_train, X_cat_test,
         X_text_train, X_text_test, y_train, y_test,
         scalers, encoders) = build_features(df, embeddings, product_map)

        num_cols = scalers['num_cols']
        cat_cols = encoders['cat_cols']
        text_feature_names = [f"txt_{i}" for i in range(X_text_train.shape[1])]
        all_feature_names = num_cols + cat_cols + text_feature_names

        # 拼接完整训练/测试表格
        X_train_tab = np.concatenate([X_num_train, X_cat_train, X_text_train], axis=1)
        X_test_tab = np.concatenate([X_num_test, X_cat_test, X_text_test], axis=1)

        # 2. LightGBM 模型（支持 Optuna 调参）
        if Config.USE_OPTUNA:
            best_params = tune_lightgbm(X_train_tab, y_train)
            lgb_model = lgb.LGBMRegressor(**best_params, random_state=Config.SEED, verbose=-1)
        else:
            lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10,
                                          random_state=Config.SEED, verbose=-1)
        lgb_model.fit(X_train_tab, y_train)
        lgb_preds = lgb_model.predict(X_test_tab)
        lgb_mae = mean_absolute_error(y_test, lgb_preds)
        lgb_r2 = r2_score(y_test, lgb_preds)
        logger.info(f"📈 LightGBM 模型 -> MAE={lgb_mae:.4f}  R²={lgb_r2:.4f}")

        # 保存模型（英文路径）
        try:
            lgb_model.booster_.save_model(str(Config.LGB_MODEL))
            logger.info(f"💾 LightGBM 模型已保存至 {Config.LGB_MODEL}")
        except Exception as e:
            logger.warning(f"模型保存失败: {e}")

        # 3. SHAP 分析
        logger.info("🔍 计算 SHAP 值...")
        X_sample = X_test_tab[:1000]
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_sample)
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=all_feature_names, show=False)
        plt.tight_layout()
        plt.savefig(Config.SHAP_PNG, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 SHAP 图已保存至 {Config.SHAP_PNG}")

        # 4. 消融实验（无文本）
        X_train_no_text = np.concatenate([X_num_train, X_cat_train], axis=1)
        X_test_no_text = np.concatenate([X_num_test, X_cat_test], axis=1)
        lgb_no_text = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10,
                                        random_state=Config.SEED, verbose=-1)
        lgb_no_text.fit(X_train_no_text, y_train)
        no_text_r2 = r2_score(y_test, lgb_no_text.predict(X_test_no_text))
        no_text_mae = mean_absolute_error(y_test, lgb_no_text.predict(X_test_no_text))
        logger.info(f"📈 无文本 LightGBM -> MAE={no_text_mae:.4f}  R²={no_text_r2:.4f}  (提升 R² +{lgb_r2 - no_text_r2:.4f})")

        # 5. 深度模型训练（若需要集成）
        logger.info("🔧 训练深度模型（用于集成）...")
        deep_model, _ = train_deep_model(X_num_train, X_cat_train, X_text_train, y_train,
                                         X_num_test, X_cat_test, X_text_test, y_test, encoders)
        deep_model.load_state_dict(torch.load(Config.MODEL_WEIGHTS))
        deep_model.eval()
        # 获取深度模型预测
        with torch.no_grad():
            deep_preds = []
            test_ds = MultiModalDataset(X_num_test, X_cat_test, X_text_test, y_test)
            test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
            for batch in test_loader:
                num = batch['num'].to(Config.DEVICE)
                cat = batch['cat'].to(Config.DEVICE)
                text = batch['text'].to(Config.DEVICE)
                deep_preds.extend(deep_model(num, cat, text).cpu().numpy())
        deep_preds = np.array(deep_preds)
        deep_mae = mean_absolute_error(y_test, deep_preds)
        deep_r2 = r2_score(y_test, deep_preds)
        logger.info(f"📈 深度模型 -> MAE={deep_mae:.4f}  R²={deep_r2:.4f}")

        # 6. 模型集成（加权平均）
        w_lgb = Config.ENSEMBLE_WEIGHT_LGB
        w_dl = Config.ENSEMBLE_WEIGHT_DL
        ensemble_preds = w_lgb * lgb_preds + w_dl * deep_preds
        ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
        ensemble_r2 = r2_score(y_test, ensemble_preds)
        logger.info(f"📈 集成模型 (LGB {w_lgb:.2f} + DL {w_dl:.2f}) -> MAE={ensemble_mae:.4f}  R²={ensemble_r2:.4f}")

        # 7. 最终评估对比
        logger.info("=" * 60)
        logger.info("📊 最终结果对比:")
        logger.info(f"   LightGBM (多模态):   MAE={lgb_mae:.4f}  R²={lgb_r2:.4f}")
        logger.info(f"   LightGBM (无文本):    MAE={no_text_mae:.4f}  R²={no_text_r2:.4f}")
        logger.info(f"   深度模型 (注意力):    MAE={deep_mae:.4f}  R²={deep_r2:.4f}")
        logger.info(f"   集成模型 (融合):      MAE={ensemble_mae:.4f}  R²={ensemble_r2:.4f}")
        logger.info("=" * 60)

        logger.info("🎉 全优化流程完成！")
    except Exception as e:
        logger.error(f"❌ 错误: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()