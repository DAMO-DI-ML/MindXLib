from __future__ import print_function
import numpy as np
import shap
from mindxlib.base.explainer import FeatureImportanceExplainer

class KernelExplainer(FeatureImportanceExplainer):
    """SHAP Kernel Explainer for black-box models"""
    def __init__(self, model, *argv, **kwargs):
        super().__init__(model)
        self.explainer = shap.KernelExplainer(self.predict, *argv, **kwargs)

    def explain_instance(self, *argv, **kwargs):
        return self.explainer.shap_values(*argv, **kwargs)

class GradientExplainer(FeatureImportanceExplainer):
    """SHAP Gradient Explainer for differentiable models"""
    def __init__(self, model, *argv, **kwargs):
        super().__init__(model)
        self.explainer = shap.GradientExplainer(model, *argv, **kwargs)

class DeepExplainer(FeatureImportanceExplainer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model)
        # ... rest of init

class TreeExplainer(FeatureImportanceExplainer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model)
        # ... rest of init
