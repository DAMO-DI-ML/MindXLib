from __future__ import print_function
import numpy as np
import shap
from mindxlib.base.explainer import FeatureImportanceExplainer



class ShapExplainer(FeatureImportanceExplainer):
    """SHAP Kernel Explainer for black-box models"""
    def __init__(self, model, *argv, **kwargs):
        super().__init__(model)

    def _initial_baseline(self, data, baseline, mode="match", **kwargs):
        """
        Initialize the baseline for the SHAP Kernel Explainer.
        
        Args:
            data (numpy.ndarray): The dataset used to compute the baseline.
            baseline (numpy.ndarray or None): User-provided baseline. If None, the mean of the data is used.
            
        Returns:
            numpy.ndarray: The initialized baseline.
        """
        if baseline is not None:
            if mode == "match":
                if data.shape[0] == baseline.shape[0]:
                    return baseline
                elif data.shape[0] != baseline.shape[0]:
                    raise ValueError("match mode specifies that data and baseline must have the same first dimension")
            if mode == "origin":
                return baseline
        else:
            raise ValueError(f"shap needs to specify a baseline")

    def _compute_attributions(self, datas, baseline, mode="match", *argv, **kwargs):
        attributions = []
        print(f"mode:{mode}")
        if mode == "match":
            n_samples = datas.shape[0]
            for i in range(n_samples):
                data = datas.iloc[i]
                base = baseline.iloc[i]
                explainer = shap.KernelExplainer(self.model, data=base, *argv, **kwargs)
                attributions.append(explainer.shap_values(data))
            return attributions
        elif mode == "origin":
            explainer = shap.KernelExplainer(self.model, data=baseline, *argv, **kwargs)
            attributions.append(explainer.shap_values(datas))
            return attributions
        else:
            raise ValueError(f"mode must be 'match' or 'origin'")

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
