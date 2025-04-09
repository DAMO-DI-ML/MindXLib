# MindXLib

MindXLib is an open toolkit ensemble of algorithmic achievements in XAI (Explainable AI) from the Data Decision Team at Alibaba DAMO Academy's Decision Intelligence Lab.

[中文文档](README_CN.md)

## Installation

You can install MindXLib using pip:

~~pip install mindxlib~~
```bash
pip install git+http://gitlab.alibaba-inc.com/MindXAI/MindXLib.git@develop-2025
```

Or install from source:

```bash
git clone http://gitlab.alibaba-inc.com/MindXAI/MindXLib.git
cd mindxlib
git checkout develop-2025  # Switch to development branch
pip install -e .
```

## Quick Start

### Feature Attribution with SHAP

```python
import xgboost
import shap # just for retrival of adult dataset
from mindxlib import ShapExplainer

# Load adult dataset
X, y = shap.datasets.adult()

# Train XGBoost classifier
model = xgboost.XGBClassifier()
model.fit(X, y)

# Initialize Tree SHAP explainer
explainer = ShapExplainer(model, method="tree")

# Generate explanations
explanation = explainer.explain(X[:1000], baseline=X, mode="origin")

# Show scatter plot for Age feature
explanation.show('scatter', feature='Age')
```

### Rule Learning with SSRL

```python
import pandas as pd
import numpy as np
from mindxlib import SSRL
from mindxlib.data import tic_tac_toe

# Load tic-tac-toe dataset
X, y = tic_tac_toe()

# Initialize and fit SSRL
explainer = SSRL(cc=10, lambda_1=1, distorted_step=10, 
                categorical_features=X.columns.tolist())
explainer.fit(X, y)

# Show learned rules
explainer.show()

# Make predictions
predictions = explainer.predict(X)
acc = np.sum(predictions.values == y.values) / y.shape[0]
print(f'Training accuracy: {acc:.2f}')

# Example output:
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
Training accuracy: 0.98
'''
```

## Architecture
The algorithm package currently supports the following models:

### Rule-based Methods
1. [RuleSet](docs/examples/ruleset.md) - Rule-based classifier using submodular optimization that supports binary classification
2. [RuleSetImb](docs/examples/ruleset_imb.md) - Rule-based classifier optimized for imbalanced data that supports binary classification
3. [Diver](docs/examples/diver.md) - Rule discovery through combinatorial optimization that supports binary classification
4. [DrillUp](docs/examples/drillup.md) - Pattern detection algorithm for discriminative rules that supports binary classification
5. [SSRL (Scalable Sparse Rule Lists)](docs/examples/rulelist.md) - Efficient decision rule list learning that supports **multi-class** classification

### Feature Attribution Methods
1. [SHAP](docs/examples/shap.md) - SHapley Additive exPlanations for model interpretation, providing customizable baselines and a user-friendly interface
2. [LIME](docs/examples/lime.md) - Local Interpretable Model-agnostic Explanations
3. [IG (Integrated Gradients)](docs/examples/ig.md) - Path attribution method for deep learning models
4. [GAM](docs/examples/gam.md) - Generalized Additive Models with shape functions

### Time Series Explanation
1. [FDTemp](docs/examples/fdtemp.md) - Functional Decomposition Temperature method for explaining temporal black-box models

## Related Papers
1. [Efficient Decision Rule List Learning via Unified Sequence Submodular Optimization](https://dl.acm.org/doi/10.1145/3637528.3671827)
2. [SLIM: a Scalable Light-weight Root Cause Analysis for Imbalanced Data in Microservice](https://dl.acm.org/doi/pdf/10.1145/3639478.3643098)
3. [Interactive Generalized Additive Models for Electricity Load Forecasting](https://dl.acm.org/doi/10.1145/3580305.3599533)
4. [Learning Interpretable Decision Rule Sets: A Submodular Optimization Approach](https://arxiv.org/abs/2206.03718)
5. [Explain temporal black-box models via functional decomposition](https://dl.acm.org/doi/10.5555/3692070.3694400)