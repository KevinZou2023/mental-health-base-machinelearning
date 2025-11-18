# 抑郁症预测集成模型

这是一个基于多个机器学习模型集成的抑郁症预测项目，采用了分层集成策略来提高预测准确率。

## 项目概述

本项目使用多种机器学习算法构建预测模型，通过逻辑回归和岭回归进行特征权重学习，最终通过加权集成获得最优预测结果。

### 数据集

- **训练集**: `dataset/train.csv` - 包含特征和目标变量（是否抑郁）
- **测试集**: `dataset/test.csv` - 用于生成最终预测
- **原始数据**: `dataset/final_depression_dataset_1.csv` - 用于训练时的数据增强
- **样本提交**: `dataset/sample_submission.csv` - 提交格式参考

## 模型架构

### 第一层（L1）- 基础模型
项目使用7个不同的基础模型：

1. **逻辑回归** (Logistic Regression with OneHot Encoding)
2. **CatBoost** - 梯度提升决策树，原生支持类别特征
3. **XGBoost** - 极限梯度提升
4. **LightGBM** (GBDT) - 基础梯度提升决策树
5. **LightGBM (GOSS)** - 基于梯度的单边采样
6. **LightGBM (DART)** - 使用随机drop的梯度提升
7. **Gradient Boosting** - scikit-learn 梯度提升分类器
8. **AdaBoost** - 自适应提升

### 第二层（L2）- 元学习器
- **L2 Logistic Regression**: 使用logit变换的逻辑回归集成
- **L2 Ridge**: 直接使用岭回归集成

### 第三层（L3）- 最终集成
- **L3 Weighted Ensemble**: 通过Optuna优化的加权集成

## 项目结构

```
.
├── s04e11-depression-prediction-ensemble.ipynb  # 主要笔记本：三层集成模型（7+2+1）
├── baseline.ipynb                               # 基线模型：逻辑回归 + Linear SVM
├── depression-prediction-ensemble.ipynb         # 另一个集成模型版本
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── final_depression_dataset_1.csv
├── oof_pred_probs/                              # 保存的训练集预测概率
├── test_pred_probs/                             # 保存的测试集预测概率
├── requirements.txt                             # 项目依赖
└── README.md                                    # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 笔记本说明

### 1. Baseline 模型 (`baseline.ipynb`)

这是一个快速的基线模型实现，使用两个简单的分类器：

**模型：**
- **Logistic Regression** - 逻辑回归
- **Linear SVM with CalibratedClassifierCV** - 线性支持向量机（带概率校准）

**特点：**
- 使用5折交叉验证
- 数据增强：结合原始数据集进行训练
- 完整的EDA可视化（4张图表）
- 模型评估可视化（混淆矩阵、特征重要性等）
- 自动生成提交文件

**输出：**
- `1_target_distribution.png` - 目标变量分布
- `2_missing_values_heatmap.png` - 缺失值热力图
- `3_categorical_vs_target.png` - 分类特征与目标的关系
- `4_numerical_vs_target.png` - 数值特征分布
- `5_oof_confusion_matrix.png` - 混淆矩阵
- `6_lr_feature_importance.png` - 特征重要性
- `submission_*.csv` - 最终提交文件

### 2. 主集成模型 (`s04e11-depression-prediction-ensemble.ipynb`)

这是核心的三层集成模型，包含7个基础模型和2个元学习器。

详见下面的"模型架构"部分。

## 使用方法

### 运行 Baseline 模型

打开 `baseline.ipynb` 并执行所有单元格：

1. **环境设置** - 导入库和配置
2. **EDA可视化** - 生成4张探索性分析图表
3. **数据准备** - 加载和处理数据
4. **预处理管道** - 构建特征处理流程
5. **模型定义** - 定义LR和SVM模型
6. **交叉验证训练** - 执行5折交叉验证
7. **模型评估** - 生成混淆矩阵和特征重要性图
8. **生成提交文件** - 输出CSV提交文件

### 运行完整的集成模型

打开 `s04e11-depression-prediction-ensemble.ipynb` 笔记本并按顺序执行所有单元格：

1. **导入和配置** - 加载必要的库和配置参数
2. **数据加载和预处理** - 读取和处理数据集
3. **训练基础模型** - 训练所有7个L1模型
4. **L2 逻辑回归集成** - 使用logit变换和Optuna优化
5. **L2 岭回归集成** - 使用直接预测和Optuna优化
6. **L3 加权集成** - 优化L2模型的权重
7. **结果展示** - 展示所有模型的性能对比

### 主要参数配置

在 `CFG` 类中可以配置：

```python
class CFG:
    train_path = 'dataset/train.csv'
    test_path = 'dataset/test.csv'
    sample_sub_path = 'dataset/sample_submission.csv'
    original_data_path = 'dataset/final_depression_dataset_1.csv'
    
    target = 'Depression'           # 目标列名
    n_folds = 5                     # 交叉验证折数
    seed = 42                       # 随机种子
```

## 核心特性

### 1. 多模型策略
- 支持7种不同类型的机器学习算法
- 每个模型都经过超参数优化

### 2. 两阶段集成
- **第一阶段**: 使用Logit变换和岭回归进行元学习
- **第二阶段**: 通过加权平均进一步优化

### 3. 自动化超参数优化
- 使用Optuna框架进行贝叶斯优化
- 同时优化模型参数和分类阈值

### 4. 数据增强
- 使用原始数据集扩展训练集
- 提高模型的泛化能力

### 5. 5折交叉验证
- 确保模型评估的稳定性
- 生成OOF（Out-of-Fold）预测用于集成

## 输出文件

项目会生成以下输出文件：

- `sub_*.csv` - 最终的提交文件（格式为 `sub_{model_name}_{score}.csv`）
- `oof_pred_probs/` - 训练集的预测概率（用于集成）
- `test_pred_probs/` - 测试集的预测概率（用于集成）

## 依赖版本

| 包名 | 版本 | 用途 |
|------|------|------|
| pandas | >=1.3.0 | 数据处理 |
| numpy | >=1.20.0 | 数值计算 |
| scikit-learn | >=1.0.0 | 机器学习基础算法 |
| matplotlib | >=3.4.0 | 数据可视化 |
| seaborn | >=0.11.0 | 统计数据可视化 |
| catboost | >=1.0.0 | CatBoost模型 |
| lightgbm | >=3.3.0 | LightGBM模型 |
| xgboost | >=1.5.0 | XGBoost模型 |
| scipy | >=1.7.0 | 科学计算 |
| optuna | >=3.0.0 | 超参数优化 |

## 性能指标

所有模型使用**准确率**作为主要评估指标，通过5折交叉验证计算。

## 注意事项

1. CatBoost 在第一次运行时会创建 `catboost_info/` 文件夹，项目结束时会自动删除
2. 超参数优化（Optuna）需要较长的计算时间，可根据需要调整试验次数
3. 确保数据集文件完整，特别是 `final_depression_dataset_1.csv`

## 参考资源

- [CatBoost文档](https://catboost.ai/)
- [LightGBM文档](https://lightgbm.readthedocs.io/)
- [XGBoost文档](https://xgboost.readthedocs.io/)
- [Optuna文档](https://optuna.readthedocs.io/)

## 许可证

MIT License
