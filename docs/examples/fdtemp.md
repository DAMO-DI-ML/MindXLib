# FDTemp (Functional Decomposition Temperature)

FDTemp is a method for explaining temporal black-box models through functional decomposition. It provides interpretable insights into how different features and their interactions contribute to model predictions over time.

## Usage

```python
from mindxlib import FDTempExplainer

# Initialize the explainer
explainer = FDTempExplainer(model)

# Generate explanations
explanation = explainer.explain(X)

# Access main effects and interaction effects
main_effects = explanation.main_effect
interaction_effects = explanation.interaction_effect
```

## Parameters

Documentation coming soon...

## References

Yang, L., Tong, Y., Gu, X., & Sun, L. (2024). Explain temporal black-box models via functional decomposition. In Forty-first International Conference on Machine Learning. 