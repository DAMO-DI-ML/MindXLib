# RuleSet

RuleSet implements a rule-based classifier using submodular optimization to discover discriminative patterns. The algorithm efficiently learns a compact set of rules that maximize coverage of the positive class while maintaining interpretability.

## Class Definition

```python
class mindxlib.explainer.RuleSet(
    max_num_rules: int = 16,
    time_limit: int = 60,
    beta_pos: float = 1.0,
    beta_neg: float = 1.0,
    beta_diverse: float = 0.1,
    beta_complex: float = 0.1,
    parallelism: int = 0,
    warmcache: int = 0,
    bestsubset: int = 0,
    exactdepth: int = 0,
    allowrandom: int = 0,
    verbose: bool = False,
    feature_prefix: str = 'f',
    num_thresh: int = 9,
    negation: bool = True,
    categorical_features: list = [],
    binarize_features: bool = True
)
```

### Parameters

- **max_num_rules** : `int`, default=16
  - Maximum number of rules to learn

- **time_limit** : `int`, default=60
  - Maximum optimization time in seconds

- **beta_pos** : `float`, default=1.0
  - Weight for positive class coverage

- **beta_neg** : `float`, default=1.0
  - Weight for negative class avoidance

- **beta_diverse** : `float`, default=0.1
  - Diversity penalty coefficient

- **beta_complex** : `float`, default=0.1
  - Complexity penalty coefficient

- **parallelism** : `int`, default=0
  - Enable parallel execution (0=off, 1=on)

- **warmcache** : `int`, default=0
  - Enable warm start caching (0=off, 1=on)

- **bestsubset** : `int`, default=0
  - Enable best subset selection (0=off, 1=on)

- **exactdepth** : `int`, default=0
  - Exact depth for rule mining (0=off, 1=on)

- **allowrandom** : `int`, default=0
  - Allow random restarts (0=off, 1=on)

- **feature_prefix** : `str`, default='f'
  - Prefix for feature names

- **num_thresh** : `int`, default=9
  - Number of thresholds for numerical feature binarization

- **negation** : `bool`, default=True
  - Allow negated features in rules

- **categorical_features** : `list`, default=[]
  - List of categorical feature names

- **binarize_features** : `bool`, default=True
  - Enable automatic feature binarization

- **verbose** : `bool`, default=False
  - Enable verbose output

## Methods

### fit()

Learn a rule set from training data.

```python
def fit(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    default_label: Optional[Any] = None
)
```

**Parameters:**
- **X** : `pd.DataFrame` or `np.ndarray`
  - Feature matrix of shape (n_samples, n_features)

- **y** : `pd.Series` or `np.ndarray`
  - Label vector of shape (n_samples,)

- **default_label** : `Any`, optional
  - Label for default rule (uses most frequent class if None)

**Returns:**
- `RuleExplanation` object containing:
  - Learned rules
  - Default rule

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
from mindxlib import RuleSet
import pandas as pd
import numpy as np

# Create binary classification example data (anomaly detection)
X = pd.DataFrame({
    'age': ['young', 'middle-aged', 'old', 'middle-aged', 'old'],
    'income': ['high', 'medium', 'low', 'low', 'medium'],
})
y = pd.Series([0, 1, 1, 0, 1], name='is_anomaly')  # 1 indicates anomaly

# Initialize RuleSet
explainer = RuleSet(
    max_num_rules=3, 
    time_limit=60, 
    verbose=True,
    feature_prefix='feature_',
    binarize_features=True,
    categorical_features=[],
    num_thresh=3,
    negation=True
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

## References

1. Yang, Fan, Kai He, Linxiao Yang, Hongxia Du, Jingbang Yang, Bo Yang, and Liang Sun. "[Learning Interpretable Decision Rule Sets: A Submodular Optimization Approach](https://arxiv.org/abs/2206.03718)." In Advances in Neural Information Processing Systems (NeurIPS), vol. 34, pp. 27890-27902. 2021. (Spotlight)