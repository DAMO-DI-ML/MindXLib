# SSRL (Scalable Sparse Rule Lists) Tutorial

SSRL is an interpretable machine learning algorithm that learns rule lists from data. It aims to find a compact set of if-then rules that accurately predict the target variable while maintaining interpretability.


## Basic Usage

Here's a simple example of how to use SSRL:

```python
from mindxlib.explainers.rules.rulelist import SSRL
import pandas as pd
import numpy as np
```
### Create sample data

```python
X = pd.DataFrame({
'age': [25, 35, 45, 55, 22],
'income': [30000, 45000, 60000, 75000, 75000],
'education_years': [12, 14, 16, 18, 18]
})
y = pd.Series([0, 1, 1, 0, 0], name='label')
```

### Initialize SSRL

```python
explainer = SSRL(
lambda_1=1.0, # Regularization parameter for rule length
binarize_features=True, # Automatically binarize features
num_thresh=3 # Number of thresholds for numeric features
)
```

### Fit the model

```python
explainer.fit(X, y)
```

### Make predictions on new data
X_test = pd.DataFrame({
'age': [30],
'income': [50000],
'education_years': [15]
})
predictions = explainer.predict(X_test)
```

### Show the pure text rules

```python
explainer.rules.show()
```
