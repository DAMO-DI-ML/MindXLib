# Diver

Diver implements a rule-based explainer that discovers discriminative patterns through combinatorial optimization. Primarily used for anomaly detection or class differentiation in classification tasks.

## Class Definition

```python
class mindxlib.explainer.Diver(
    label_col: str,
    label_val: Any,
    pos_beta: float = 1.5,
    overlap_beta_: float = 0.2,
    complexity_cost: float = 0.00001,
    dim_list: Optional[List[str]] = None,
    sup_ratio: float = 0.01,
    write_model: bool = False,
    disable_log: bool = True,
    cache_ind: bool = False
)
```

### Parameters

- **label_col** : `str`
  - Column name for the label in the dataset

- **label_val** : `Any`
  - Target class value (e.g., `1` for anomalies)

- **pos_beta** : `float`, default=1.5
  - Weight coefficient for positive samples

- **overlap_beta_** : `float`, default=0.2
  - Penalty coefficient for rule overlap

- **complexity_cost** : `float`, default=0.00001
  - Penalty coefficient for rule length

- **dim_list** : `List[str]`, optional, default=None
  - Feature columns for rule mining
  - Uses all non-label columns if None

- **sup_ratio** : `float`, default=0.01
  - Minimum support ratio (relative to target class count)

- **write_model** : `bool`, default=False
  - Whether to save optimization models (for debugging)

- **disable_log** : `bool`, default=True
  - Disable logging output

- **cache_ind** : `bool`, default=False
  - Enable intermediate result caching

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
) -> Diver
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
- Fitted `Diver` instance

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
from mindxlib import Diver
import pandas as pd
import numpy as np

# Create binary classification example data (anomaly detection)
X = pd.DataFrame({
    'age': ['young', 'middle-aged', 'old', 'middle-aged', 'old'],
    'income': ['high', 'medium', 'low', 'low', 'medium'],
})
y = pd.Series([0, 1, 1, 0, 1], name='is_anomaly')  # 1 indicates anomaly

# Initialize Diver
explainer = Diver(
    label_col='is_anomaly',
    label_val=1,  # We're interested in rules for anomalies
    pos_beta=1.5,  # Higher weight on catching anomalies
    overlap_beta_=0.3,  # Moderate penalty for rule overlap
    complexity_cost=0.001,  # Small penalty for longer rules
    dim_list=['age', 'income'],  # Focus on these features
    sup_ratio=0.2  # Minimum 20% support in anomaly class
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
