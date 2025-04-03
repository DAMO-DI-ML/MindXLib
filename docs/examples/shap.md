# SHAP (SHapley Additive exPlanations)

SHAP is an implementation of the SHAP (SHapley Additive exPlanations) method for explaining model predictions. It uses game theoretic approaches to assign each feature an importance value for a particular prediction.

## Class Definition

```python
class ShapExplainer(FeatureImportanceExplainer):
    def __init__(self,
                 model,
                 link="identity",
                 **kwargs
                 )
```

### Parameters

- **model** : callable
  - Model prediction function to explain
  - Should accept numpy array/pandas DataFrame input
  - For classifiers, should return probabilities

- **link** : str, default="identity"
  - The link function used to transform model outputs
  - Options: "identity", "logit"
  - Use "logit" for classification problems

## Methods

### explain(data, baseline=None, mode="match", **kwargs)

Generate SHAP explanations for the given samples.

```python
def explain(self,
           data,              # Samples to explain
           baseline=None,     # Background data for expectations
           mode="match",      # How to use baseline data
           **kwargs          # Additional arguments for KernelExplainer
           )
```

**Parameters:**
- **data** : DataFrame or ndarray
  - Samples to explain
  - Shape should be (n_samples, n_features)

- **baseline** : DataFrame, ndarray or None
  - Background data for computing feature importance
  - If mode="match", should have same number of samples as data
  - Required for proper SHAP value computation

- **mode** : str or int, default="match"
  - How to use the baseline data
  - "match" or 1: Use corresponding baseline for each sample
  - "origin" or 0: Use all baseline data for each sample

- **kwargs** : dict
  - Additional arguments passed to SHAP KernelExplainer
  - Common options: nsamples, l1_reg, silent

**Returns:**
- ShapExplanation or MultipleShapExplanations object containing:
  - Feature importance values
  - Visualization methods
  - Original SHAP explanation objects

### show()

Display SHAP values and visualizations.

```python
def show(self,
         type='waterfall',  # Type of visualization
         **kwargs          # Additional plotting arguments
         )
```

**Parameters:**
- **type** : str, default='waterfall'
  - Type of visualization to display
  - Current options: 'waterfall'

- **kwargs** : dict
  - Additional arguments passed to plotting functions

## Examples

### Binary Classification

```python
from mindxlib.explainers import ShapExplainer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Prepare data
X = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': [4, 5, 6]
})
X_background = pd.DataFrame({
    'feature1': [1.5, 2.5],
    'feature2': [4.5, 5.5]
})

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Initialize explainer
explainer = ShapExplainer(model.predict_proba, link="logit")

# Generate explanations
explanations = explainer.explain(
    data=X,
    baseline=X_background,
    mode="origin",
    nsamples=100
)

# Show waterfall plot for first prediction
explanations.show(type='waterfall', index=0)
```

### Multiple Sample Explanations

```python
# Generate explanations for multiple samples
X_test = pd.DataFrame({
    'feature1': [2.1, 2.2, 2.3],
    'feature2': [5.1, 5.2, 5.3]
})
X_background = pd.DataFrame({
    'feature1': [2.0, 2.1, 2.2],
    'feature2': [5.0, 5.1, 5.2]
})

# Explain with matched baselines
explanations = explainer.explain(
    data=X_test,
    baseline=X_background,
    mode="match"
)

# Show explanations for different samples
explanations.show(type='waterfall', index=0)
explanations.show(type='waterfall', index=1)
```

## Notes

- SHAP values provide a unified measure of feature importance
- Supports both single predictions and batch explanations
- Can handle both regression and classification models
- Requires background data to compute expectations
- Computation can be slow for large datasets or complex models
- Multiple visualization options for interpretation

## References

1. [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap)
2. Lundberg, S.M., Lee, S.I. (2017) "A Unified Approach to Interpreting Model Predictions" 