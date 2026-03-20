# MindXLib

MindXLib是达摩院决策智能实验室数据决策团队在XAI(Explainable AI，可解释机器学习)领域深耕，展示算法成果的一个open toolkit。

## 安装

您可以使用pip安装MindXLib：

pip install mindxlib

## 快速开始

### 使用SHAP进行特征归因

```python
import xgboost
import shap # just for retrival of adult dataset
from mindxlib import ShapExplainer

# 加载adult数据集
X, y = shap.datasets.adult()

# 训练XGBoost分类器
model = xgboost.XGBClassifier()
model.fit(X, y)

# 初始化Tree SHAP解释器
explainer = ShapExplainer(model, method="tree")

# 生成解释
explanation = explainer.explain(X[:1000], baseline=X, mode="origin")

# 展示Age特征的散点图
explanation.show('scatter', feature='Age')
```

### 使用SSRL进行规则学习

```python
import pandas as pd
import numpy as np
from mindxlib import SSRL
from mindxlib.data import tic_tac_toe

# Load tic-tac-toe dataset
X, y = tic_tac_toe()

# 初始化并训练SSRL
explainer = SSRL(cc=10, lambda_1=1, distorted_step=10, 
                categorical_features=X.columns.tolist())
explainer.fit(X, y)

# 展示学习到的规则
explainer.show()

# 进行预测
predictions = explainer.predict(X)
acc = np.sum(predictions.values == y.values) / y.shape[0]
print(f'训练准确率: {acc:.2f}')

# 输出示例:
'''
IF 1==o AND 4==o AND 7==o, THEN negative
ELIF 3==o AND 4==o AND 5==o, THEN negative
ELIF 0==o AND 1==o AND 2==o, THEN negative
ELIF 6==o AND 7==o AND 8==o, THEN negative
ELIF 0==o AND 3==o AND 6==o, THEN negative
ELIF 2==o AND 5==o AND 8==o, THEN negative
ELIF 0!=x AND 4!=x AND 8!=x, THEN negative
ELIF 2!=x AND 4!=x AND 6!=x, THEN negative
ELSE positive
训练准确率: 0.98
'''
```

## 架构
目前算法包主支持的模型如下：

### 基于规则的方法
1. [RuleSet](docs/examples/ruleset.md) - 使用次模优化的基于规则的分类器，支持二分类
2. [RuleSetImb](docs/examples/ruleset_imb.md) - 针对不平衡数据优化的基于规则的分类器，支持二分类
3. [Diver](docs/examples/diver.md) - 通过组合优化进行规则发现，支持二分类
4. [DrillUp](docs/examples/drillup.md) - 用于判别规则的模式检测算法，支持二分类
5. [SSRL (Scalable Sparse Rule Lists)](docs/examples/rulelist.md) - 高效的决策规则列表学习，支持多分类

### 特征归因方法
1. [SHAP](docs/examples/shap.md) - 用于模型解释的SHapley加性解释，在shap基础上提供自定义基线
2. [LIME](docs/examples/lime.md) - 局部可解释的模型无关解释
3. [IG (Integrated Gradients)](docs/examples/ig.md) - 深度学习模型的路径归因方法
4. [GAM](docs/examples/gam.md) - 具有形状函数的广义加性模型

## 相关论文
1. [Efficient Decision Rule List Learning via Unified Sequence Submodular Optimization](https://dl.acm.org/doi/10.1145/3637528.3671827)
2. [SLIM: a Scalable Light-weight Root Cause Analysis for Imbalanced Data in Microservice](https://dl.acm.org/doi/pdf/10.1145/3639478.3643098)
3. [Interactive Generalized Additive Models for Electricity Load Forecasting](https://dl.acm.org/doi/10.1145/3580305.3599533)
4. [Learning Interpretable Decision Rule Sets: A Submodular Optimization Approach](https://arxiv.org/abs/2206.03718) 