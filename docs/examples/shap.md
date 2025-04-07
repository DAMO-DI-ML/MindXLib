# SHAP (SHapley Additive exPlanations)

ShapExplainer is a wrapper for the SHAP package that provides:
- Unified interface for all SHAP methods (Kernel, Tree, Linear)
- Simple parameter handling with sensible defaults
- Easy visualization methods
- Flexible baseline handling

SHAP assigns each feature an importance value using game theory principles.

## Class Definition

```python
class ShapExplainer(FeatureImportanceExplainer):
    def __init__(self,
                 model,
                 method="auto",
                 link="identity",
                 **kwargs
                 )
```

### Parameters

- **model**
  - Must have predict() or predict_proba()
  - link="identity": uses predict()
  - link="logit": uses predict_proba()

- **method** : str, default="auto"
  - "kernel": Uses shap.KernelExplainer - Works with any model
  - "tree": Uses shap.TreeExplainer - Optimized for tree models
  - "linear": Uses shap.LinearExplainer - Optimized for linear models
  - "auto": Automatically selects best explainer

- **link** : str, default="identity"
  - "identity": Raw model output
  - "logit": Probability predictions (requires predict_proba)

## Methods

### explain()

```python
def explain(self,
           data,           # Samples to explain
           baseline,       # Background data (required)
           mode="match",   # How to use baseline
           **kwargs       # Additional SHAP arguments
           )
```

**Parameters:**
- **data** : numpy.ndarray or pandas.DataFrame
  - Samples to explain
  - Shape: (n_samples, n_features)

- **baseline** : numpy.ndarray or pandas.DataFrame
  - Background data for computing SHAP values
  - Required parameter
  - Shape requirements:
    - match mode: (n_samples, n_features)
    - origin mode: (any_samples, n_features)

- **mode** : str or int
  - How to use baseline data
  - Options:
    - "match" or 1: Pair each sample with specific baseline
    - "origin" or 0: Use all baseline data for each sample, similar to shap.Explainer

- **kwargs** : dict
  - Additional arguments passed to SHAP explainers:
    - For KernelExplainer: nsamples, silent
    - For TreeExplainer: check_additivity
    - For LinearExplainer: feature_dependence

**Returns:**
- For mode="match":
  - MultipleShapExplanations object containing list of ShapExplanation objects:
    - Each ShapExplanation has:
      - data[i,:]: One sample from input
      - feature_importance[i]: SHAP values for that sample
      - shap_explanation[i]: Original SHAP Explanation for that sample

- For mode="origin":
  - Single ShapExplanation object containing:
    - data: All input samples
    - feature_importance: SHAP values for all samples
    - shap_explanation: Original SHAP Explanation for all samples

### show()

Display SHAP values and visualizations.

```python
def show(self,
         type='waterfall',  # Plot type
         **kwargs          # Plot arguments
         )
```

**Plot Types:**
- 'waterfall': Feature contributions to prediction
- 'bar': Feature importance rankings
- 'scatter': Feature dependence
  - Single feature: SHAP vs feature values
  - All features: Summary plot

**Arguments:**
- scatter: feature, class_index(default=0)
- waterfall/bar: index(default=0), class_index(default=0)

## Examples

### Kernel SHAP with Iris Dataset

```python
import sklearn
import shap  # for loading the dataset
from sklearn.model_selection import train_test_split
from mindxlib import ShapExplainer

# Load and split iris dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    *shap.datasets.iris(), test_size=0.2, random_state=0
)

# Train SVM model
svm = sklearn.svm.SVC(kernel='rbf', probability=True)
svm.fit(X_train, Y_train)

# Initialize Kernel SHAP explainer
explainer = ShapExplainer(svm, link="logit", method='kernel')

# Generate explanations
explanation = explainer.explain(
    X_test[:5], 
    baseline=X_train[:5], 
    mode="origin",
    nsamples=100
)

# Show waterfall plot for specific prediction
explanation.show(type='waterfall', class_index=1)
```

**Waterfall Plot Output:**

![Waterfall plot showing how each feature contributes to the prediction](waterfall.png)

This waterfall plot shows:
- Baseline prediction (E[f(X)]) starts at 0.259
- Petal width contributes -0.22 to the prediction
- Sepal width has a small negative impact (-0.01)
- Petal length and sepal length have minimal effects
- Final prediction f(x) is 0.033

### TreeSHAP with Adult Dataset (Bar Plot)

```python
import xgboost
import shap  # for loading the dataset
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

# Show bar plot
explanation.show(type='bar', class_index=1)
```

**Bar Plot Output:**

![Bar plot showing SHAP values for each feature](bar.png)

This bar plot shows:
- Capital Gain has the strongest negative impact (-3.14)
- Relationship (-0.76) and Workclass (-0.55) also have notable negative effects
- Education-Num (+0.46) and Age (+0.42) have positive contributions
- Sex (+0.33) has a moderate positive effect
- Other features like Hours per week have smaller impacts
- The sum of 3 other features has minimal effect (-0.02)

### TreeSHAP with Adult Dataset (Scatter Plot)

```python
# Using same model and explainer from previous example
# Show scatter plot for Age feature
explanation.show('scatter', feature='Age')
```

**Scatter Plot Output:**

![Scatter plot showing SHAP values vs Age](scatter.png)

This scatter plot shows:
- X-axis: Age values from 20 to 90
- Y-axis: SHAP values for Age feature
- Color: Age value (blue=young, red=old)
- Shows how Age impacts model predictions across different age ranges

### Match Mode with Simple Regression

```python
import numpy as np
from mindxlib import ShapExplainer
import xgboost

# Prepare regression data with more samples
X = np.array([
    [1, 2], [2, 4], [3, 1], [4, 3], [5, 3], [6, 2],[2, 1], [3, 4], [4, 2], 
    [5, 1], [6, 3], [7, 2], [3, 3], [4, 1], [5, 4], [6, 1], [7, 3], [8, 2]
])
y = np.array([1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8])

# Train XGBoost regressor
model = xgboost.XGBRegressor()
model.fit(X, y)

# Initialize explainer
explainer = ShapExplainer(model, link="identity")

# Generate explanations with matched baselines
# Using more samples for both explanation and baseline
explanation = explainer.explain(X[9:18], baseline=X[:9], mode="match")

# Show bar plot for specific sample
explanation.show(type='waterfall', index=0)
```

**Waterfall Plot Output:**

![Waterfall plot showing matched feature contributions](waterfall_match.png)

This plot shows:
- Match mode pairs each sample with its baseline
- Feature 0: 
  - Sample=5, Baseline=1
  - 1:1 relationship with output
  - SHAP value = +4 (pushing prediction up by exactly 4)
- Feature 1: Minimal impact (≈0)
- Base value (E[f(X)]) = 1.001
- Final prediction = 5

## Implementation Notes

Current E[f(X)] behavior:
- Match mode: Shows baseline value (not expectation)
- Origin mode: Shows expectation over all baselines
- Future: Will allow custom y-axis labels instead of E[f(X)]



## References

1. [SHAP Package](https://github.com/slundberg/shap)
2. Lundberg, S.M., Lee, S.I. (2017) "A Unified Approach to Interpreting Model Predictions" 