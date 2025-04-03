# SSRL (Scalable Sparse Rule Lists)

SSRL is an implementation of the paper ["Efficient Decision Rule List Learning via Unified Sequence Submodular Optimization"](https://dl.acm.org/doi/10.1145/3637528.3671827) (KDD 2024). It learns interpretable rule lists from data using a scalable optimization approach. The algorithm finds a compact set of if-then rules that accurately predict the target variable while maintaining interpretability.

## Class Definition

```python
class SSRL(RuleExplainer):
    def __init__(self,
                 model=None,                    # Optional model to explain (not used)
                 data=None,                     # Optional training data
                 feature_prefix='f',            # Prefix for feature names when using numpy arrays
                 lambda_1=1,                    # Regularization parameter for rule length
                 distorted_step=10,             # Number of distortion steps in optimization
                 cc=None,                       # Parameter for subproblem solver (default: 5*lambda_1)
                 use_multi_pool=False,          # Whether to use multiprocessing
                 binarize_features=True,        # Whether to automatically binarize features
                 categorical_features=[],        # List of categorical column indices
                 num_thresh=9,                  # Number of thresholds for numeric features
                 negation=True                  # Whether to include negated features
                 )
```

### Parameters

- **model** : object, optional (default=None)
  - Model to explain (not used in SSRL)

- **data** : DataFrame or array-like, optional (default=None)
  - Training data

- **feature_prefix** : str, default='f'
  - Prefix used for feature names when input is numpy array

- **lambda_1** : float, default=1.0
  - Regularization parameter controlling rule length
  - Larger values encourage shorter rules

- **distorted_step** : int, default=10
  - Number of steps in the distorted optimization process
  - Controls exploration of the rule space

- **cc** : float, optional (default=5*lambda_1)
  - Parameter for the subproblem solver
  - Affects optimization stability

- **use_multi_pool** : bool, default=False
  - Whether to use multiprocessing for optimization
  - Can speed up training on large datasets

- **binarize_features** : bool, default=True
  - Whether to automatically binarize numeric features
  - Required for rule learning

- **categorical_features** : list, default=[]
  - List of indices/names of categorical columns
  - Used for proper feature binarization

- **num_thresh** : int, default=9
  - Number of thresholds used for numeric feature binarization
  - More thresholds allow finer splits but increase complexity

- **negation** : bool, default=True
  - Whether to include negated conditions in rules
  - Allows rules like "NOT (age > 50)"

## Methods

### fit(X, y, default_label=None)

Learn a rule list from data.

```python
def fit(self,
        X,                      # Input features (DataFrame or ndarray)
        y,                      # Target labels (required)
        default_label=None      # Optional name for default rule
        )
```

**Parameters:**
- **X** : DataFrame or ndarray
  - Input features
  - Can be numeric or categorical

- **y** : Series, DataFrame or ndarray
  - Target labels
  - Can be binary or multi-class

- **default_label** : object, optional (default=None)
  - Label to use for default rule
  - If None, uses most frequent class

**Returns:**
- RuleExplanation object containing learned rules

### predict(X)

Make predictions using learned rules.

```python
def predict(self,
           X       # Input features (DataFrame or ndarray)
           )
```

**Parameters:**
- **X** : DataFrame or ndarray
  - Input features to make predictions for
  - Must have same columns/features as training data

**Returns:**
- ndarray or Series with shape (n_samples,)
  - Predicted labels for each input sample

## Examples

### Binary Classification

```python
from mindxlib import SSRL
import pandas as pd

# Create sample data
X = pd.DataFrame({
    'age': [25, 35, 45, 55, 22],
    'income': [30000, 45000, 60000, 75000, 75000],
    'education_years': [12, 14, 16, 18, 18]
})
y = pd.Series([0, 1, 1, 0, 0], name='label')

# Initialize and fit explainer
explainer = SSRL(lambda_1=1.0, binarize_features=True)
explainer.fit(X, y)

# Show learned rules
explainer.rules.show()

# Make predictions
test_data = pd.DataFrame({
    'age': [30],
    'income': [50000],
    'education_years': [15]
})
predictions = explainer.predict(test_data)
```

### Multi-class Classification

```python
# Create multi-class data
X_multi = pd.DataFrame({
    'age': [25, 35, 45, 55, 22, 28],
    'income': [30000, 45000, 60000, 75000, 75000, 50000],
    'education_years': [12, 14, 16, 18, 18, 15]
})
y_multi = pd.Series([0, 1, 2, 0, 1, 2], name='risk_level')

# Fit with specific default label
explainer = SSRL(lambda_1=0.5, num_thresh=5)
explainer.fit(X_multi, y_multi, default_label=1)

# Show rules and make predictions
explainer.rules.show()
predictions = explainer.predict(X_multi)
```

## Notes

- SSRL automatically handles both binary and multi-class classification
- The algorithm optimizes for both accuracy and interpretability
- Rules are presented in IF-THEN format for easy interpretation
- Supports both pandas DataFrames and numpy arrays as input
- Automatically binarizes numeric features for rule learning
- Can handle missing values and categorical features
- Provides probabilistic predictions through rule confidence scores

## References

1. Original paper: ["Efficient Decision Rule List Learning via Unified Sequence Submodular Optimization"](https://dl.acm.org/doi/10.1145/3637528.3671827) (KDD 2024)
