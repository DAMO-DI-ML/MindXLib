from __future__ import print_function
import numpy as np
import shap
from mindxlib.base.explainer import FeatureImportanceExplainer

class KernelExplainer(FeatureImportanceExplainer):
    """SHAP Kernel Explainer for black-box models"""
    def __init__(self, model, *argv, **kwargs):
        super().__init__(model)

    def _initial_baseline(self, data, baseline):
        """
        Initialize the baseline for the SHAP Kernel Explainer.
        
        Args:
            data (numpy.ndarray): The dataset used to compute the baseline.
            baseline (numpy.ndarray or None): User-provided baseline. If None, the mean of the data is used.
            
        Returns:
            numpy.ndarray: The initialized baseline.
        """
        if baseline is None:
            return data.mean(axis=0)[np.newaxis, :]
        else:
            return baseline

    def _compute_attributions(self, datas, baseline, *argv, **kwargs):
        explainer = shap.KernelExplainer(self.model, data=baseline, *argv, **kwargs)
        batch_attributions = []
        for data in datas:
            attribution = explainer.shap_values(data)
            batch_attributions.append(attribution)
        return np.array(batch_attributions)

class PermutationExplainer(FeatureImportanceExplainer):
    def __init__(self, model, *argv, **kwargs):
        super().__init__(model)
        self.explainer = shap.PermutationExplainer(model, *argv, **kwargs)
    
    def _compute_attributions(self, datas, *argv, **kwargs):
        batch_attributions = []
        for data in datas:
            attribution = self.explainer.shap_values(data)
            batch_attributions.append(attribution)
        return np.array(batch_attributions)
