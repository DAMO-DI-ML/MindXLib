# Ruleset文档

# Diver  Tutorial

Implements a rule-based explainer Diver that discovers discriminative patterns through combinatorial optimization. Primarily used for anomaly detection or class differentiation in classification tasks.

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

## Init Parameters

**label\_col** (`str`) - Column name for the label in the dataset

**label\_val** (`int`) - Target class value (e.g., `1` for anomalies)

**pos\_beta** (`float`, optional) - Weight coefficient for positive samples. Default: `1.5`

**overlap\_beta\_** (`float`, optional) - Penalty coefficient for rule overlap. Default: `0.2`

**complexity\_cost** (`float`, optional) - Penalty coefficient for rule length. Default: `0.00001`

**dim\_list** (`List[str]`, optional) - Feature columns for rule mining. Uses all non-label columns if `None`. Default: `None`

**sup\_ratio** (`float`, optional) - Minimum support ratio (relative to target class count). Default: `0.01`

**write\_model** (`bool`, optional) - Whether to save optimization models (for debugging). Default: `False`

**disable\_log** (`bool`, optional) - Disable logging output. Default: `True`

**cache\_ind** (`bool`, optional) - Enable intermediate result caching. Default: `False`

## Methods

### fit(X, y, X\_columns=None, y\_column=None, default\_label=None)

```python
fit(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    X_columns: Optional[List[str]] = None,
    y_column: Optional[str] = None,
    default_label: Optional[Any] = None
) -> Diver
```

#### Parameters

**X** (`pd.DataFrame` or `np.ndarray`) - Feature matrix of shape (n\_samples, n\_features)

**y** (`pd.Series` or `np.ndarray`) - Label vector of shape (n\_samples,)

**X\_columns** (`List[str]`, optional) - Feature column names (required when X is ndarray)

**y\_column** (`str`, optional) - Label column name (required when y is ndarray)

**default\_label** (`Any`, optional) - Manually specified default rule label

#### Returns

Fitted `Diver` instance

### predict(X\_test)

```plaintext
predict(X_test: Union[pd.DataFrame, np.ndarray]) -> pd.Series
```

#### Parameters

**X\_test** (`pd.DataFrame` or `np.ndarray`) - Test data of shape (n\_samples, n\_features)

#### Returns

`pd.DataFrame` - Predicted labels of shape (n\_samples,)

## Basic Usage

Here's a simple example of how to use Diver:

```python
from mindxlib import Diver
import pandas as pd
import numpy as np
```

### Create binary classification example data (anomaly detection):

```python
X = pd.DataFrame({
        'age': ['young', 'middle-aged', 'old', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low', 'low', 'medium'],
    })
y = pd.Series([0, 1, 1, 0, 1], name='is_anomaly')  # 1 indicates anomaly
```

### Initialize Diver:

```python
explainer = Diver(
    label_col='is_anomaly',
    label_val=1,  # We're interested in rules for anomalies
    pos_beta=1.5,  # Higher weight on catching anomalies
    overlap_beta_=0.3,  # Moderate penalty for rule overlap
    complexity_cost=0.001,  # Small penalty for longer rules
    dim_list=['age', 'income'],  # Focus on these features
    sup_ratio=0.2  # Minimum 20% support in anomaly class
)
```

### Fit the Model:

```python
explainer.fit(X, y)
```

### Make predictions on new data

```python
test_data = pd.DataFrame({
        'age': ['young', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low']
    })
fraud_predictions = explainer.predict(test_data)
```

### Show the rules in pure text

```python
explainer.rules.show()
```

# DrillUp  Tutorial

Implements the  DrillUp algorithm for pattern detection, discovering discriminative rule patterns from data.  Primarily used for anomaly detection or class differentiation in classification tasks.

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
    score_type: str = 'risk')
```

## Init Parameters

**label\_col** (`str`) - Column name representing the label in the dataset

**label\_val** (`Any`) - Target class value (e.g., `1` for anomalies)

**dim\_list** (`List[str]`, optional) - Feature columns for rule mining. Uses all non-label columns if `None`. Default: `None`

**min\_dim\_val\_cnt** (`int`, optional) - Minimum value count threshold (values below this are grouped as 'other'). Default: `5`

**sup\_ratio** (`float`, optional) - Minimum support ratio (relative to target class count). Default: `0.01`

**out\_num** (`int`, optional) - Number of top patterns to return. Default: `100`

**jcd\_limit** (`float`, optional) - Jaccard similarity threshold for pattern uniqueness. Default: `0.75`

**min\_pat\_len** (`int`, optional) - Minimum pattern length to include. Default: `1`

**score\_gap** (`float`, optional) - Score ratio threshold for pattern selection. Default: `1.0`

**score\_type** (`str`, optional) - Scoring metric ('risk' or 'diffScore'). Default: `'risk'`

## Methods

### fit(X, y, X\_columns=None, y\_column=None, default\_label=None)

```plaintext
fit(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    X_columns: Optional[List[str]] = None,
    y_column: Optional[str] = None,
    default_label: Optional[Any] = None
) -> DrillUp
```

#### Parameters

**X** (`pd.DataFrame` or `np.ndarray`) - Feature matrix of shape (n\_samples, n\_features)

**y** (`pd.Series` or `np.ndarray`) - Label vector of shape (n\_samples,)

**X\_columns** (`List[str]`, optional) - Feature column names (required when X is ndarray)

**y\_column** (`str`, optional) - Label column name (required when y is ndarray)

**default\_label** (`Any`, optional) - Manually specified default rule label

#### Returns

Fitted `DrillUp` instance

### predict(X\_test)

```python
predict(X_test: Union[pd.DataFrame, np.ndarray]) -> pd.Series
```

#### Parameters

**X\_test** (`pd.DataFrame` or `np.ndarray`) - Test data of shape (n\_samples, n\_features)

#### Returns

`pd.DataFrame` - Predicted labels of shape (n\_samples,)

**Basic Usage**

Here's a simple example of how to use DrillUp:

```python
from mindxlib import DrillUp
import pandas as pd
import numpy as np
```

### Create binary classification example data (anomaly detection):

```python
X = pd.DataFrame({
        'age': ['young', 'middle-aged', 'old', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low', 'low', 'medium'],
    })
y = pd.Series([0, 1, 1, 0, 1], name='is_anomaly')  # 1 indicates anomaly
```

### Initialize Diver:

```python
explainer = DrillUp(
    label_col='is_anomaly',
    label_val=1,  # We're interested in rules for anomalies
    min_dim_val_cnt = 5,
    sup_ratio = 0.01,
    out_num = 100,
    jcd_limit = 0.75,
    min_pat_len = 1,
    dim_list=['age', 'income'],  # Focus on these features
)
```

### Fit the Model:

```python
explainer.fit(X, y)
```

### Make predictions on new data

```python
test_data = pd.DataFrame({
        'age': ['young', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low']
    })
fraud_predictions = explainer.predict(test_data)
```

### Show the rules in pure text

```python
explainer.rules.show()
```

# Ruleset\_imb  Tutorial

Implements a rule-based classifier for imbalanced data that learns a set of discriminative rules through combinatorial optimization.  The algorithm is particularly effective for binary classification tasks with class imbalance.

```python
class mindxlib.explainer.RuleSetImb(
    max_num_rules: int = 16,
    time_limit: int = 60,
    factor_g: float = 0.0,
    local_search_iter: int = 0,
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

## Init Parameters

**max\_num\_rules** (

**time\_limit** (`int`) - Maximum optimization time in seconds. Default: 60

**factor\_g** (`float`) - Growth factor for rule expansion. Default: 0.0

**local\_search\_iter** (`int`) - Number of local search iterations. Default: 0

**beta\_pos** (`float`) - Weight for positive class coverage. Default: 1.0

**beta\_neg** (`float`) - Weight for negative class avoidance. Default: 1.0

**beta\_diverse** (`float`) - Diversity penalty coefficient. Default: 0.1

**beta\_complex** (`float`) - Complexity penalty coefficient. Default: 0.1

**parallelism** (`int`) - Enable parallel execution (0/1). Default: 0

**warmcache** (`int`) - Enable warm start caching (0/1). Default: 0

**bestsubset** (`int`) - Enable best subset selection (0/1). Default: 0

**exactdepth** (`int`) - Exact depth for rule mining (0/1). Default: 0

**allowrandom** (`int`) - Allow random restarts (0/1). Default: 0

**verbose** (`bool`) - Enable verbose output. Default: False

**feature\_prefix** (`str`) - Prefix for feature names. Default: 'f'

**num\_thresh** (`int`) - Number of thresholds for numerical feature binarization. Default: 9

**negation** (`bool`) - Allow negated features in rules. Default: True

**categorical\_features** (`list`) - List of categorical feature names. Default: \[\]

**binarize\_features** (`bool`) - Enable automatic feature binarization. Default: True

## Methods

### fit( X, y,default\_label=None)

```plaintext
fit(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    default_label: Optional[Any] = None
)
```

Learn a rule set from training data.

#### Parameters

**X** (pd.DataFrame or np.ndarray) - Feature matrix of shape (n\_samples, n\_features)

**y** (pd.Series or np.ndarray) - Label vector of shape (n\_samples,)

**default\_label** (Any, optional) - Label for default rule (uses most frequent class if None)

#### Returns

**RuleExplanation** - Object containing learned rules and default rule

### predict(X\_test)

```plaintext
predict(X_test: Union[pd.DataFrame, np.ndarray]) -> pd.Series
```

#### Parameters

**X\_test** (pd.DataFrame or np.ndarray) - Test data of shape (n\_samples, n\_features).

#### Returns

**pd.DataFrame** - Predicted labels of shape (n\_samples,)

**Basic Usage**

Here's a simple example of how to use RuleSetImb:

```python
from mindxlib import RuleSetImb
import pandas as pd
import numpy as np
```

### Create binary classification example data (anomaly detection):

```python
X = pd.DataFrame({
        'age': ['young', 'middle-aged', 'old', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low', 'low', 'medium'],
    })
y = pd.Series([0, 1, 1, 0, 1], name='is_anomaly')  # 1 indicates anomaly
```

### Initialize RuleSetImb:

```python
explainer = RuleSetImb(
        max_num_rules=15, 
        time_limit=120, 
        verbose=True,
        feature_prefix='feature_',
        binarize_features=True,
        categorical_features=[],
        num_thresh=25,
        negation=True
    )
```

### Fit the Model:

```python
explainer.fit(X, y)
```

### Make predictions on new data

```python
test_data = pd.DataFrame({
        'age': ['young', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low']
    })
fraud_predictions = explainer.predict(test_data)
```

### Show the rules in pure text

```python
explainer.rules.show()
```

# Ruleset  Tutorial

Implements a rule-based classifier using submodular optimization to discover discriminative patterns. The algorithm efficiently learns a compact set of rules that maximize coverage of the positive class while maintaining interpretability.

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
Implements a rule-based classifier using submodular optimization to discover discriminative patterns. The algorithm efficiently learns a compact set of rules that maximize coverage of the positive class while maintaining interpretability.

```

## Init Parameters

**max\_num\_rules** (`int`): Maximum number of rules to learn (default: 16)

**time\_limit** (`int`): Maximum optimization time in seconds (default: 60)

**beta\_pos** (`float`): Weight for positive class coverage (default: 1.0)

**beta\_neg** (`float`): Weight for negative class avoidance (default: 1.0)

**beta\_diverse** (`float`): Diversity penalty coefficient (default: 0.1)

**beta\_complex** (`float`): Complexity penalty coefficient (default: 0.1)

**parallelism** (`int`): Enable parallel execution (0=off, 1=on) (default: 0)

**warmcache** (`int`): Enable warm start caching (0=off, 1=on) (default: 0)

**bestsubset** (`int`): Enable best subset selection (0=off, 1=on) (default: 0)

**exactdepth** (`int`): Exact depth for rule mining (0=off, 1=on) (default: 0)

**allowrandom** (`int`): Allow random restarts (0=off, 1=on) (default: 0)

**feature\_prefix** (`str`): Prefix for feature names (default: 'f')

**num\_thresh** (`int`): Number of thresholds for numerical feature binarization (default: 9)

**negation** (`bool`): Allow negated features in rules (default: True)

**categorical\_features** (`list`): List of categorical feature names (default: \[\])

**binarize\_features** (`bool`): Enable automatic feature binarization (default: True)

**verbose** (`bool`): Enable verbose output (default: False)

## Methods

### fit( X, y,default\_label=None)

```plaintext
fit(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    default_label: Optional[Any] = None
)
```

Learn a rule set from training data.

#### Parameters

**X** (pd.DataFrame or np.ndarray) - Feature matrix of shape (n\_samples, n\_features)

**y** (pd.Series or np.ndarray) - Label vector of shape (n\_samples,)

**default\_label** (Any, optional) - Label for default rule (uses most frequent class if None)

#### Returns

**RuleExplanation** - Object containing learned rules and default rule

### predict(X\_test)

```plaintext
predict(X_test: Union[pd.DataFrame, np.ndarray]) -> pd.Series
```

#### Parameters

**X\_test** (pd.DataFrame or np.ndarray) - Test data of shape (n\_samples, n\_features).

#### Returns

**pd.DataFrame**\- Predicted labels of shape (n\_samples,)

**Basic Usage**

Here's a simple example of how to use RuleSet:

```python
from mindxlib import RuleSet
import pandas as pd
import numpy as np
```

### Create binary classification example data (anomaly detection):

```python
X = pd.DataFrame({
        'age': ['young', 'middle-aged', 'old', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low', 'low', 'medium'],
    })
y = pd.Series([0, 1, 1, 0, 1], name='is_anomaly')  # 1 indicates anomaly
```

### Initialize RuleSetImb:

```python
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
```

### Fit the Model:

```python
explainer.fit(X, y)
```

### Make predictions on new data

```python
test_data = pd.DataFrame({
        'age': ['young', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low']
    })
fraud_predictions = explainer.predict(test_data)
```

### Show the rules in pure text

```python
explainer.rules.show()
```