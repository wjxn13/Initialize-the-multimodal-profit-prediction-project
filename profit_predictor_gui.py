"""
多模态利润预测 - 交互式图形界面（适配 v6 特征）
Multimodal Profit Prediction - Interactive GUI (Bilingual, v6 Compatible)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pickle
from pathlib import Path
import lightgbm as lgb
import warnings
import logging
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# ---------- 日志配置 ----------
LOG_DIR = Path(r"D:\.kaggle\商店销售\date")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)



# 所有资源现在都在商店销售文件夹内
DATA_DIR = Path(r"D:\.kaggle\商店销售\date")
SCALER_FILE = DATA_DIR / "scalers_v6.pkl"
ENCODER_FILE = DATA_DIR / "encoders_v6.pkl"
EMBEDDINGS_FILE = DATA_DIR / "text_embeddings_v6.npy"
PRODUCT_MAP_FILE = DATA_DIR / "product_map_v6.pkl"
LGB_MODEL_FILE = Path(r"D:\.kaggle\商店销售\lgb_model.txt")
# ---------- 加载资源 ----------

def load_artifacts():
    logger.info("开始加载预处理资源和模型...")
    missing = []
    for f in [SCALER_FILE, ENCODER_FILE, EMBEDDINGS_FILE, PRODUCT_MAP_FILE, LGB_MODEL_FILE]:
        if not f.exists():
            missing.append(str(f))
    if missing:
        logger.error(f"文件缺失: {missing}")
        raise FileNotFoundError("以下文件缺失 Missing files：\n" + "\n".join(missing))

    logger.info(f"加载标准化器: {SCALER_FILE}")
    with open(SCALER_FILE, 'rb') as f:
        scalers = pickle.load(f)
    scaler = scalers['scaler']
    num_cols = scalers['num_cols']   # 现在包含 16 个特征
    logger.info(f"数值特征列 ({len(num_cols)}): {num_cols}")

    logger.info(f"加载编码器: {ENCODER_FILE}")
    with open(ENCODER_FILE, 'rb') as f:
        encoders = pickle.load(f)
    encoder = encoders['encoder']
    cat_cols = encoders['cat_cols']
    logger.info(f"类别特征列: {cat_cols}")

    logger.info(f"加载文本嵌入: {EMBEDDINGS_FILE}")
    product_map = pickle.load(open(PRODUCT_MAP_FILE, 'rb'))
    embeddings = np.load(EMBEDDINGS_FILE)
    mean_embedding = embeddings.mean(axis=0, keepdims=True)
    logger.info(f"文本嵌入形状: {embeddings.shape}, 产品数: {len(product_map)}")

    logger.info(f"加载 LightGBM 模型: {LGB_MODEL_FILE}")
    lgb_model = lgb.Booster(model_file=str(LGB_MODEL_FILE))
    logger.info("所有资源加载完毕。")
    return scaler, num_cols, encoder, cat_cols, product_map, embeddings, mean_embedding, lgb_model

# ---------- 构建特征（与训练时完全一致） ----------
def build_features(sales, quantity, discount, shipping_cost, year, weeknum,
                   market, region, category, sub_category, segment, ship_mode, order_priority,
                   product_name, scaler, encoder, cat_cols,
                   product_map, embeddings, mean_embedding, num_cols):
    # 基础衍生特征
    price_per_item = sales / (quantity + 1e-8)
    shipping_cost_ratio = shipping_cost / (sales + 1e-8)
    discount_flag = 1 if discount > 0 else 0
    month = max(1, min(12, int(weeknum // 4.33 + 1)))
    quarter = (month - 1) // 3 + 1
    year_norm = year - 2011
    week_sin = np.sin(2 * np.pi * weeknum / 53.0)
    week_cos = np.cos(2 * np.pi * weeknum / 53.0)
    order_item_count = 1  # 假设订单仅当前商品

    # 构造数值特征字典（顺序需与 num_cols 一致）
    num_values = {
        'Sales': sales,
        'Quantity': quantity,
        'Discount': discount,
        'Shipping Cost': shipping_cost,
        'order_item_count': order_item_count,
        'price_per_item': price_per_item,
        'shipping_cost_ratio': shipping_cost_ratio,
        'discount_flag': discount_flag,
        'month': month,
        'quarter': quarter,
        'year_norm': year_norm,
        'week_sin': week_sin,
        'week_cos': week_cos,
        # 统计特征：从训练集统计量中获取（均通过均值近似，因为单样本无全局信息）
        'subcat_avg_profit': 28.6,   # 使用训练集全局平均利润（可更精确）
        'subcat_avg_sales': 246.5,   # 全局平均销售额
        'market_avg_profit': 28.6
    }

    # 确保顺序与训练时一致
    num_feats = np.array([[num_values[col] for col in num_cols]])
    num_feats_scaled = scaler.transform(num_feats)

    # 类别特征
    cat_input = np.array([[market, region, category, sub_category, segment, ship_mode, order_priority]])
    cat_encoded = encoder.transform(cat_input)

    # 文本特征
    if product_name in product_map:
        text_emb = embeddings[product_map[product_name]]
        logger.debug(f"找到产品 '{product_name}' 的嵌入")
    else:
        logger.warning(f"产品 '{product_name}' 未知，使用平均嵌入")
        text_emb = mean_embedding[0]
    text_emb = text_emb.reshape(1, -1)

    X = np.concatenate([num_feats_scaled, cat_encoded, text_emb], axis=1)
    logger.debug(f"最终特征维度: {X.shape}")
    return X.astype(np.float32)

# ---------- GUI ----------
class ProfitPredictorApp:
    def __init__(self, root, scaler, num_cols, encoder, cat_cols,
                 product_map, embeddings, mean_embedding, lgb_model):
        self.root = root
        self.scaler = scaler
        self.num_cols = num_cols
        self.encoder = encoder
        self.cat_cols = cat_cols
        self.product_map = product_map
        self.embeddings = embeddings
        self.mean_embedding = mean_embedding
        self.lgb_model = lgb_model

        root.title("💸 多模态利润预测系统 Multimodal Profit Predictor")
        root.geometry("720x620")
        root.resizable(False, False)

        cat_options = {col: list(encoder.categories_[i]) for i, col in enumerate(cat_cols)}

        ttk.Label(root, text="请输入订单信息 / Please enter order details",
                  font=("Arial", 14, "bold")).pack(pady=10)

        main_frame = ttk.Frame(root)
        main_frame.pack(padx=20, pady=5, fill='both')

        # 左侧数值特征
        left_frame = ttk.LabelFrame(main_frame, text="数值特征 Numerical Features", padding=10)
        left_frame.grid(row=0, column=0, sticky='n', padx=5)
        self.num_entries = {}
        num_fields = [
            ("销售额 Sales ($)", "sales"),
            ("数量 Quantity", "quantity"),
            ("折扣 Discount (0~1)", "discount"),
            ("运费 Shipping Cost ($)", "shipping_cost"),
            ("年份 Year (e.g. 2014)", "year"),
            ("周数 Week Number (1~53)", "weeknum")
        ]
        for i, (label, key) in enumerate(num_fields):
            ttk.Label(left_frame, text=label).grid(row=i, column=0, sticky='w', pady=2)
            var = tk.StringVar()
            ttk.Entry(left_frame, textvariable=var, width=18).grid(row=i, column=1, padx=5, pady=2)
            self.num_entries[key] = var
        self.num_entries['sales'].set("200")
        self.num_entries['quantity'].set("3")
        self.num_entries['discount'].set("0.0")
        self.num_entries['shipping_cost'].set("10")
        self.num_entries['year'].set("2014")
        self.num_entries['weeknum'].set("25")

        # 右侧类别特征
        right_frame = ttk.LabelFrame(main_frame, text="类别特征 Categorical Features", padding=10)
        right_frame.grid(row=0, column=1, sticky='n', padx=5)
        self.cat_combos = {}
        cat_labels = {
            'Market': '市场 Market',
            'Region': '地区 Region',
            'Category': '大类 Category',
            'Sub-Category': '子类 Sub-Category',
            'Segment': '客户群 Segment',
            'Ship Mode': '运输方式 Ship Mode',
            'Order Priority': '优先级 Priority'
        }
        for i, col in enumerate(cat_cols):
            ttk.Label(right_frame, text=cat_labels.get(col, col)).grid(row=i, column=0, sticky='w', pady=2)
            combo = ttk.Combobox(right_frame, values=cat_options[col], state='readonly', width=20)
            combo.grid(row=i, column=1, padx=5, pady=2)
            combo.set(cat_options[col][0])
            self.cat_combos[col] = combo

        # 产品名称
        prod_frame = ttk.LabelFrame(root, text="产品名称 Product Name", padding=10)
        prod_frame.pack(padx=20, pady=10, fill='x')
        ttk.Label(prod_frame, text="输入完整产品名 Enter full product name (e.g. Xerox 225)：").pack(side='left')
        self.product_var = tk.StringVar(value="Xerox 225")
        ttk.Entry(prod_frame, textvariable=self.product_var, width=50).pack(side='left', padx=10)

        # 按钮与结果
        ttk.Button(root, text="🔮 预测利润 Predict Profit", command=self.predict).pack(pady=10)
        self.result_label = ttk.Label(root, text="点击按钮开始预测 / Click the button to predict",
                                      font=("Arial", 16))
        self.result_label.pack(pady=15)

    def predict(self):
        logger.info("用户点击了预测按钮。")
        try:
            sales = float(self.num_entries['sales'].get())
            quantity = int(self.num_entries['quantity'].get())
            discount = float(self.num_entries['discount'].get())
            shipping_cost = float(self.num_entries['shipping_cost'].get())
            year = int(self.num_entries['year'].get())
            weeknum = int(self.num_entries['weeknum'].get())

            market = self.cat_combos['Market'].get()
            region = self.cat_combos['Region'].get()
            category = self.cat_combos['Category'].get()
            sub_category = self.cat_combos['Sub-Category'].get()
            segment = self.cat_combos['Segment'].get()
            ship_mode = self.cat_combos['Ship Mode'].get()
            order_priority = self.cat_combos['Order Priority'].get()

            product_name = self.product_var.get().strip()
            if not product_name:
                messagebox.showwarning("输入错误 Input Error",
                                       "产品名称不能为空 / Product name cannot be empty")
                return

            X = build_features(sales, quantity, discount, shipping_cost,
                               year, weeknum, market, region, category, sub_category,
                               segment, ship_mode, order_priority,
                               product_name, self.scaler, self.encoder, self.cat_cols,
                               self.product_map, self.embeddings, self.mean_embedding,
                               self.num_cols)

            prediction = self.lgb_model.predict(X)[0]
            logger.info(f"预测完成，预测利润: ${prediction:.2f}")
            self.result_label.config(
                text=f"📈 预测利润 / Predicted Profit：${prediction:.2f}",
                foreground="green"
            )
        except Exception as e:
            logger.error(f"预测过程发生错误: {e}", exc_info=True)
            messagebox.showerror("预测失败 Prediction Failed", f"{str(e)}")

def main():
    logger.info("=" * 50)
    logger.info("应用程序启动。")
    logger.info(f"日志文件: {LOG_FILE}")
    try:
        (scaler, num_cols, encoder, cat_cols,
         product_map, embeddings, mean_embedding, lgb_model) = load_artifacts()
        root = tk.Tk()
        app = ProfitPredictorApp(root, scaler, num_cols, encoder, cat_cols,
                                 product_map, embeddings, mean_embedding, lgb_model)
        logger.info("进入主事件循环。")
        root.mainloop()
        logger.info("应用程序正常退出。")
    except FileNotFoundError as e:
        logger.error(f"文件缺失: {e}")
        messagebox.showerror("文件缺失 Missing Files", str(e))
    except Exception as e:
        logger.error(f"启动失败: {e}", exc_info=True)
        messagebox.showerror("启动失败 Launch Failed", str(e))

if __name__ == "__main__":
    main()