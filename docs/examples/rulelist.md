# SSRL (Scalable Sparse Rule Lists) Tutorial

SSRL is an implementation of the paper ["Efficient Decision Rule List Learning via Unified Sequence Submodular Optimization"](https://dl.acm.org/doi/10.1145/3637528.3671827) (KDD 2024). It is an interpretable machine learning algorithm that learns rule lists from data. It aims to find a compact set of if-then rules that accurately predict the target variable while maintaining interpretability. SSRL supports both binary and multi-class classification tasks.


## Basic Usage

Here's a simple example of how to use SSRL:

```python
from mindxlib.explainers.rules.rulelist import SSRL
import pandas as pd
import numpy as np
```
### Create sample data

```python
# Binary classification example
X = pd.DataFrame({
'age': [25, 35, 45, 55, 22],
'income': [30000, 45000, 60000, 75000, 75000],
'education_years': [12, 14, 16, 18, 18]
})
y = pd.Series([0, 1, 1, 0, 0], name='label')

# Multi-class classification example (risk assessment)
X_multi = pd.DataFrame({
'age': [25, 35, 45, 55, 22, 28, 32, 42, 50, 38, 48, 52],
'income': [30000, 45000, 60000, 75000, 75000, 50000, 65000, 40000, 80000, 70000, 55000, 62000],
'education_years': [12, 14, 16, 18, 18, 15, 16, 14, 19, 17, 15, 16]
})
# Three classes: 0 (low risk), 1 (medium risk), 2 (high risk)
y_multi = pd.Series([0, 1, 2, 0, 1, 2, 1, 0, 2, 2, 1, 2], name='risk_level')
```

### Initialize SSRL

```python
explainer = SSRL(
    lambda_1=1.0,  # Regularization parameter for rule length
    distorted_step=10,
    cc=10,
    binarize_features=True,  # Automatically binarize features
    num_thresh=3,  # Number of thresholds for numeric features
    negation=True  # Allow negation in rules
)
```

### Fit the model

```python
# For binary classification
explainer.fit(X, y)

# For multi-class classification
explainer.fit(X_multi, y_multi)  # Automatically handles multiple classes

# You can also specify a default label
explainer.fit(X_multi, y_multi, default_label=1)  # Set medium risk as default
```

### Make predictions on new data
```python
test_data = pd.DataFrame({
    'age': [30, 50],
    'income': [50000, 70000],
    'education_years': [15, 17]
})
predictions = explainer.predict(test_data)
```

### Show the rules in pure text
```python
explainer.rules.show()
```

## Notes on Multi-class Classification

- SSRL automatically detects the number of unique classes in your target variable
- The rules generated will predict all possible class labels in your dataset
- You can specify a default_label parameter in fit() to set the default prediction when no rules match
- The model maintains interpretability even with multiple classes
