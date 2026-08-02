# 多模态利润预测系统 (Multimodal Profit Predictor)

基于 LightGBM + MiniLM 文本嵌入的订单利润预测模型，支持中英双语交互式 GUI 界面。融合数值特征 + 类别特征 + 产品名语义向量，预测 Global Superstore 订单利润。

> 诚实指标（v7，TEST R²=0.744）：详见「模型性能」。早期版本头条「R²=0.77」因深度模型在测试集上选早停而偏高，v7 已修正。

## 项目结构

```
.
├── Global Superstore.csv          # 原始销售数据
├── lgb_model.txt                  # 训练好的多模态 LightGBM 模型（GUI 加载）
├── multimodal_profit.py           # 完整训练与评估流水线（特征工程 / 训练 / 集成 / 评估）
├── rebuild_model.py               # 利用已有预处理文件快速重建模型
├── train_model.py                 # 轻量训练脚本（同 LGBM 逻辑）
├── profit_predictor_gui.py        # 中英双语交互式 GUI
├── date/                          # 预处理文件与缓存（重跑生成）
│   ├── scalers_v7.pkl
│   ├── encoders_v7.pkl
│   ├── text_embeddings_v6.npy
│   └── shap_summary_v7.png
└── models/
    └── all-MiniLM-L6-v2/          # 文本嵌入模型
        └── pytorch_model.bin      # 大文件，需从 Release 下载
```

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/wjxn13/Initialize-the-multimodal-profit-prediction-project.git
cd Initialize-the-multimodal-profit-prediction-project
```

### 2. 安装依赖
```bash
pip install lightgbm numpy pandas scikit-learn sentence-transformers torch shap matplotlib
```

### 3. 准备文本嵌入模型（必须）
产品名嵌入使用 `sentence-transformers/all-MiniLM-L6-v2`。若本地无模型，请从 [Releases](https://github.com/wjxn13/Initialize-the-multimodal-profit-prediction-project/releases) 下载 `pytorch_model.bin` 放入 `models/all-MiniLM-L6-v2/`；或让 `sentence_transformers` 自动联网下载。其余缓存（特征/编码器/缩放器）运行 `multimodal_profit.py` 会自动生成。

### 4. 启动 GUI
```bash
python profit_predictor_gui.py
```
输入订单信息（销售额、数量、产品名等），点击“预测利润”即可获得预测值。

## 模型与特征

- **文本模态**：使用 `sentence-transformers/all-MiniLM-L6-v2` 提取产品名的 384 维语义向量。
- **表格特征**：包含数值特征（销售额、折扣、运费占比、统计特征等）和类别特征（市场、地区、产品类别等）。
- **预测模型**：多模态 LightGBM 主模型 + 深度多模态融合模型（跨模态自注意力）的集成。

### v7 修复清单（相对 v6）

1. **消除选择泄漏**：深度模型早停/最佳 checkpoint 改在验证集(2013)上选，测试集(2014)严格只评估。
2. **学集成权重**：不再写死 0.65/0.35，改为在验证集上网格搜索最优加权。
3. **修复空转自注意力**：3 个模态投影成 3 个 token 做真正的跨模态互相注意（v6 序列长度=1，只能注意自己）。
4. **修复退化门控**：门控权重逐模态分别乘到对应段（v6 是 sum 成标量全局缩放，没起作用）。
5. **修复 GUI 模型错位**：流水线模型路径与 GUI 加载路径已对齐，GUI 真正使用 v7 多模态 LightGBM。

## 模型性能（v7，诚实指标）

测试集为严格留出、从未参与训练的 **2014 年**（训练 ≤2012 / 验证 2013 / 测试 2014）。

| 模型 | TEST R² | MAE |
|---|---|---|
| 多模态 LightGBM（主模型） | **0.7438** | 22.89 |
| 无文本 LightGBM（消融） | 0.6378 | 28.18 |
| 深度多模态模型（架构已修复） | 0.6055 | 29.64 |
| 集成（验证集上学权重） | 0.7438 | 22.89 |

- 文本模态贡献显著：去掉产品文本嵌入，LGBM 从 0.744 掉到 0.638（**+0.106**），「多模态」非噱头。
- 深度模型在小表格+文本问题上仍弱于调好的 GBM，故集成学得的权重退化为纯 LGBM（诚实结果，非失败）。

> 注：早期版本头条「R²=0.77」偏高，原因是深度模型当初在**测试集**上选早停、膨胀了指标。v7 消除该泄漏后诚实最优为 **0.744**。

## 可解释性

运行 `multimodal_profit.py` 会生成 SHAP 特征重要性图（`date/shap_summary_v7.png`），展示各特征对利润预测的贡献。重要特征包括 Sales（销售额）、Discount（折扣）、Sub-Category（产品子类）及部分文本嵌入维度。

## 重新训练

```bash
python multimodal_profit.py
```
训练结束后会在 `date/` 目录下生成新的特征缓存，并更新 `lgb_model.txt`。或仅重建特征并训练 LightGBM（速度更快）：
```bash
python rebuild_model.py
```

## 许可证

本项目使用的销售数据集 `Global Superstore.csv` 为公开数据集，可用于学习与研究目的。其余代码遵循 MIT License。

## 贡献

欢迎提交 Issue 或 Pull Request 改进项目。

---

Enjoy predicting profits with multimodal AI!
