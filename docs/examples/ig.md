# Integrated Gradients

IntegratedGradients is an attribution method that:
- Provides feature importance for deep learning models
- Satisfies important theoretical properties (completeness, sensitivity)
- Requires access to model gradients
- Supports multiple approximation methods for path integrals

Integrated Gradients assigns attribution scores by accumulating gradients along a straight-line path from a baseline to the input.

## Class Definition

```python
class IntegratedGradients(FeatureImportanceExplainer):
    def __init__(self,
                 model,
                 steps=50,
                 method="gausslegendre"
                 )
```

### Parameters

- **model**
  - Must implement gradient() method
  - Should return gradients of outputs with respect to inputs

- **steps** : `int`, default=50
  - Number of steps for path integral approximation
  - Higher values give more precise approximations

- **method** : `str`, default="gausslegendre"
  - "gausslegendre": Uses Gauss-Legendre quadrature for better approximation
  - "riemann": Uses simple Riemann sum approximation

## Methods

### explain()

```python
def explain(self,
           data,           # Samples to explain
           baseline=None,  # Baseline inputs
           **kwargs       # Additional arguments
           )
```

**Parameters:**
- **data** : `numpy.ndarray`
  - Samples to explain
  - Shape: (n_samples, n_features)

- **baseline** : `numpy.ndarray`, optional
  - Baseline inputs for attribution
  - If None, zero baseline is used
  - Shape requirements:
    - Same shape as data, or
    - One less dimension than data (will be broadcast)

- **kwargs** : 
  - Additional arguments passed to internal methods

**Returns:**
- `FeatureImportanceExplanation` object containing:
  - `data`: Input samples
  - `feature_importance`: Attribution scores for each sample
  - Shape: (n_samples, n_features, n_outputs)

### validate_attributions()

```python
def validate_attributions(self,
                        attributions,
                        inputs,
                        baseline=None
                        )
```

Validates attributions using the completeness axiom.

**Parameters:**
- **attributions** : Attribution scores to validate
- **inputs** : Original input data
- **baseline** : Baseline inputs

**Returns:**
- Completeness score (lower is better)

## Examples

### Simple Multiplicative Model

```python
import numpy as np
from mindxlib import IntegratedGradients

# Define a simple model with known gradients
class MultiplicativeModel:
    def forward(self, X):
        x1, x2, x3, x4, x5 = X.T
        
        output1 = x1 * x2
        output2 = x2 * x3
        output3 = x3 * x4
        output4 = x4 * x5
        
        output = np.stack([output1, output2, output3, output4], axis=1)
        return self.softmax(output)
    
    def gradient(self, X):
        # Return gradients for each output w.r.t. inputs
        x1, x2, x3, x4, x5 = X.T
        zo = np.zeros_like(x1)
        
        grad1 = np.stack([x2, x1, zo, zo, zo], axis=1)
        grad2 = np.stack([zo, x3, x2, zo, zo], axis=1)
        grad3 = np.stack([zo, zo, x4, x3, zo], axis=1)
        grad4 = np.stack([zo, zo, zo, x5, x4], axis=1)
        
        return np.stack((grad1, grad2, grad3, grad4), axis=2)
    
    def __call__(self, X):
        return self.forward(X)

# Create sample data
X = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 2, 3, 2, 1]
])

baseline = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0]
])

# Initialize model and explainer
model = MultiplicativeModel()
explainer = IntegratedGradients(model)

# Generate explanations
explanation = explainer.explain(X, baseline=baseline)

# Print attributions
print("Input data:")
print(X)
print("\nAttributions:")
print(explanation.feature_importance["feature_importance"])
```

**Example Output:**
```python
Input data:
[[1 1 1 1 1]
 [1 2 3 4 5]
 [5 4 3 2 1]
 [1 2 3 2 1]]

Attributions:
[[[0.5  0.5  0.0  0.0  0.0]   # First sample, first output
  [0.0  0.5  0.5  0.0  0.0]   # First sample, second output
  [0.0  0.0  0.5  0.5  0.0]   # First sample, third output
  [0.0  0.0  0.0  0.5  0.5]]  # First sample, fourth output
 ...
]
```

The attributions show:
- How each input feature contributes to each output
- Values represent integrated gradients along the path
- Sum of attributions approximates output difference from baseline

## Implementation Notes

The implementation:
- Supports both Gauss-Legendre quadrature and Riemann sum approximations
- Validates attributions using completeness axiom
- Requires model to provide gradient access
- Handles batched inputs efficiently

## References

1. Sundararajan, M., Taly, A., Yan, Q. (2017) "Axiomatic Attribution for Deep Networks"
2. [Original Implementation](https://github.com/ankurtaly/Integrated-Gradients)