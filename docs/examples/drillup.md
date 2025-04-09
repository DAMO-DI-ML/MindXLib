# DrillUp

DrillUp implements a pattern detection algorithm for discovering discriminative rule patterns from data. Primarily used for anomaly detection or class differentiation in classification tasks.

## Class Definition

```python
class mindxlib.explainer.DrillUp(RuleExplainer,
    label_col: str,
    label_val: Any,
    dim_list: Optional[List[str]] = None,
    min_dim_val_cnt: int = 5,
    sup_ratio: float = 0.01,
    out_num: int = 100,
    jcd_limit: float = 0.75,
    min_pat_len: int = 1,
    score_gap: float = 1.0,
    score_type: str = 'risk'
)
```

### Parameters

- **label_col** : `str`
  - Column name representing the label in the dataset

- **label_val** : `Any`
  - Target class value (e.g., `1` for anomalies)

- **dim_list** : `List[str]`, optional, default=None
  - Feature columns for rule mining
  - Uses all non-label columns if None

- **min_dim_val_cnt** : `int`, default=5
  - Minimum value count threshold
  - Values below this are grouped as 'other'

- **sup_ratio** : `float`, default=0.01
  - Minimum support ratio (relative to target class count)

- **out_num** : `int`, default=100
  - Number of top patterns to return

- **jcd_limit** : `float`, default=0.75
  - Jaccard similarity threshold for pattern uniqueness

- **min_pat_len** : `int`, default=1
  - Minimum pattern length to include

- **score_gap** : `float`, default=1.0
  - Score ratio threshold for pattern selection

- **score_type** : `str`, default='risk'
  - Scoring metric ('risk' or 'diffScore')

## Methods

### fit()

Learn patterns from training data.

```python
def fit(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    X_columns: Optional[List[str]] = None,
    y_column: Optional[str] = None,
    default_label: Optional[Any] = None
) -> DrillUp
```

**Parameters:**
- **X** : `pd.DataFrame` or `np.ndarray`
  - Feature matrix of shape (n_samples, n_features)

- **y** : `pd.Series` or `np.ndarray`
  - Label vector of shape (n_samples,)

- **X_columns** : `List[str]`, optional
  - Feature column names (required when X is ndarray)

- **y_column** : `str`, optional
  - Label column name (required when y is ndarray)

- **default_label** : `Any`, optional
  - Manually specified default rule label

**Returns:**
- Fitted `DrillUp` instance

### predict()

Make predictions on new data.

```python
def predict(X_test: Union[pd.DataFrame, np.ndarray]) -> pd.Series
```

**Parameters:**
- **X_test** : `pd.DataFrame` or `np.ndarray`
  - Test data of shape (n_samples, n_features)

**Returns:**
- `pd.Series` of predicted labels, shape (n_samples,)

## Examples

### Basic Usage

```python
from mindxlib import DrillUp
import pandas as pd
import numpy as np

# Create binary classification example data (anomaly detection)
X = pd.DataFrame({
    'age': ['young', 'middle-aged', 'old', 'middle-aged', 'old'],
    'income': ['high', 'medium', 'low', 'low', 'medium'],
})
y = pd.Series([0, 1, 1, 0, 1], name='is_anomaly')  # 1 indicates anomaly

# Initialize DrillUp
explainer = DrillUp(
    label_col='is_anomaly',
    label_val=1,  # We're interested in rules for anomalies
    min_dim_val_cnt=5,
    sup_ratio=0.01,
    out_num=100,
    jcd_limit=0.75,
    min_pat_len=1,
    dim_list=['age', 'income']  # Focus on these features
)

# Fit the Model
explainer.fit(X, y)

# Make predictions on new data
test_data = pd.DataFrame({
    'age': ['young', 'middle-aged', 'old'],
    'income': ['high', 'medium', 'low']
})
fraud_predictions = explainer.predict(test_data)

# Show the rules in pure text
explainer.rules.show()
```
