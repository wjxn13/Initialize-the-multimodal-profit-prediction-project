# 多模态利润预测系统 (Multimodal Profit Predictor)

基于 **LightGBM + MiniLM 文本嵌入** 的订单利润预测模型，支持中英双语交互式 GUI 界面。

> 模型 R² 达到 0.77，其中产品名文本嵌入贡献了 0.118 的绝对提升。

---

## 📁 项目结构

```
.
├── Global Superstore.csv          # 原始销售数据
├── lgb_model.txt                  # 训练好的 LightGBM 模型
├── multimodal_profit.py           # 完整训练与评估脚本（含特征工程、SHAP 分析）
├── profit_predictor_gui.py        # 中英双语交互式 GUI
├── rebuild_model.py               # 利用已有预处理文件快速重建模型
├── download_assets.py             # 下载依赖的大文件（见下文）
├── requirements.txt               # Python 依赖列表
├── README.md
├── date/                          # 预处理文件与缓存
│   ├── scalers_v6.pkl
│   ├── encoders_v6.pkl
│   ├── text_embeddings_v6.npy
│   ├── product_map_v6.pkl
│   ├── features_v6.npz            # ⚠️ 大文件，需从 Release 下载
│   └── shap_summary_v6.png
└── models/
    └── all-MiniLM-L6-v2/          # 文本嵌入模型
        ├── pytorch_model.bin      # ⚠️ 大文件，需从 Release 下载
        └── ...
```

---

## 🚀 快速开始

### 1. 克隆仓库
```bash   ”“bash
git clone https://github.com/wjxn13/Initialize-the-multimodal-profit-prediction-project.git
cd Initialize-the-multimodal-profit-prediction-project
```

### 2. 安装依赖
```bash   ”“bash
pip install -r requirements.txtPIP install -r requirements.txt
```

### 3. 下载大文件（必须）
模型权重和特征文件未包含在 Git 仓库中，请从 [Releases](https://github.com/wjxn13/Initialize-the-multimodal-profit-prediction-project/releases) 下载以下两个文件：
- `features_v6.npz` → 放入 `date/` 目录
- `pytorch_model.bin` → 放入 `models/all-MiniLM-L6-v2/` 目录

或直接运行自动下载脚本：
```bash   ”“bash
python download_assets.py
```

### 4. 启动 GUI
```bash   ”“bash
python profit_predictor_gui.py
```
输入订单信息（销售额、数量、产品名等），点击“预测利润”即可获得预测值。

---

## 🧠 模型与特征

- **文本模态**：使用 `sentence-transformers/all-MiniLM-L6-v2` 提取产品名的 384 维语义向量。
- **表格特征**：包含 16 个数值特征（销售额、折扣、运费占比、统计特征等）和 7 个类别特征（市场、地区、产品类别等）。
- **预测模型**：LightGBM 回归器（500 棵树，最大深度 10）。
- **性能指标**：
  - LightGBM（多模态）：**R² = 0.770**
  - LightGBM（仅表格）：R² = 0.651（文本嵌入提升 **0.118**）
  - 深度模型（自注意力 MLP）：R² = 0.660

详细训练过程与消融实验见 `multimodal_profit.py`。

---

## 📊 可解释性

运行 `multimodal_profit.py` 会生成 SHAP 特征重要性图（`date/shap_summary_v6.png`），展示各个特征对利润预测的贡献。重要特征包括：
- **Sales**（销售额）
- **Discount**（折扣）
- **Sub-Category**（产品子类）
- 部分文本嵌入维度

---

## 🔄 重新训练

若需要重新训练模型（例如更新数据或调参），执行：
```bash   ”“bash
python multimodal_profit.py
```
或仅重建特征并训练 LightGBM（速度更快）：
```bash   ”“bash
python rebuild_model.py
```

训练结束后会在 `date/` 目录下生成新的特征缓存，并更新 `lgb_model.txt`。

---

## 📦 生成 requirements.txt

```bash   ”“bash
pip freeze > requirements.txt
```
建议手动检查并只保留必要的包（如 `lightgbm`, `numpy`, `pandas`, `scikit-learn`, `sentence-transformers`, `shap`, `matplotlib`, `tkinter` 等）。

---

## 📝 许可证

本项目使用的销售数据集 `Global Superstore.csv` 为公开数据集，可用于学习与研究目的。  
其余代码遵循 MIT License。

---

## 🤝 贡献

欢迎提交 Issue 或 Pull Request 改进项目。

---

**Enjoy predicting profits with multimodal AI!**  
如有问题，请查看日志文件 `date/gui_*.log` 或提交 Issue。
