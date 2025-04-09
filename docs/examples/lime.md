# LIME (Local Interpretable Model-agnostic Explanations)

LimeExplainer is a wrapper for the LIME package that provides:
- Unified interface for different data types (Tabular, Text, Image)
- Simple parameter handling with sensible defaults
- Local interpretable explanations
- Model-agnostic approach

LIME explains individual predictions by learning an interpretable model locally around the prediction.

## Class Definition

```python
class LimeExplainer(FeatureImportanceExplainer):
    def __init__(self,
                 model,
                 *argv,
                 **kwargs
                 )
```

### Parameters

- **model**
  - Must have predict() or predict_proba()
  - For classification: use predict_proba()
  - For regression: use predict()

### Subclasses

- **LimeTabularExplainer**: For structured data
- **LimeTextExplainer**: For text data
- **LimeImageExplainer**: For image data

## Methods

### explain()

```python
def explain(self,
           data,           # Samples to explain
           baseline,       # Training data for sampling
           mode='classification',  # 'classification' or 'regression'
           feature_names=None,    # Names of features
           class_names=None,      # Names of classes
           **kwargs              # Additional LIME arguments
           )
```

**Parameters:**
- **data** : `numpy.ndarray` or `pandas.DataFrame`
  - Samples to explain
  - Shape: (n_samples, n_features)

- **baseline** : `numpy.ndarray` or `pandas.DataFrame`
  - Training data used for sampling around instances
  - Shape: (n_training_samples, n_features)

- **mode** : `str`
  - "classification": For classification tasks
  - "regression": For regression tasks

- **feature_names** : `list`, optional
  - Names of the features
  - Used for more interpretable explanations

- **class_names** : `list`, optional
  - Names of the classes
  - Used for classification tasks

- **kwargs** : 
  - Additional arguments passed to LIME explainers:
    - kernel_width: Size of the neighborhood to consider
    - verbose: Display progress
    - feature_selection: Method for selecting features

**Returns:**
- `LimeExplanation` object containing:
  - `data`: Input samples
  - `feature_importance`: LIME explanations for each sample
  - Original LIME explanation objects

## Examples

### Classification with Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mindxlib import LimeTabularExplainer

# Load and split iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Initialize LIME explainer
explainer = LimeTabularExplainer(model.predict_proba)

# Generate explanations
explanation = explainer.explain(
    X_test[:2],
    baseline=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# Print feature importance for first two samples
print(explanation.feature_importance["feature_importance"][0].as_list())
print(explanation.feature_importance["feature_importance"][1].as_list())
```

**Example Output:**
```python
# First sample explanation
[('petal width (cm)', 0.35), ('petal length (cm)', 0.28),
 ('sepal length (cm)', -0.12), ('sepal width (cm)', 0.08)]

# Second sample explanation
[('petal length (cm)', 0.42), ('petal width (cm)', 0.31),
 ('sepal width (cm)', -0.15), ('sepal length (cm)', 0.05)]
```

This output shows:
- Feature importance scores for each prediction
- Positive values indicate features supporting the predicted class
- Negative values indicate features opposing the predicted class
- Values represent the feature's contribution to the local prediction

## Implementation Notes

The current implementation:
- Supports tabular, text, and image data through specialized explainers
- Uses sampling around individual instances for local explanations
- Provides feature importance scores and visualizations
- Is model-agnostic (works with any black-box model)

## References

1. [LIME Package](https://github.com/marcotcr/lime)
2. Ribeiro, M.T., Singh, S., Guestrin, C. (2016) "Why Should I Trust You?: Explaining the Predictions of Any Classifier"
```

This documentation provides a comprehensive overview of the LIME implementation in mindxlib, following a similar structure to the SHAP documentation while focusing on LIME's specific features and usage patterns.