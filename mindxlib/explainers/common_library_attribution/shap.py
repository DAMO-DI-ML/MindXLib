from __future__ import print_function
import numpy as np
import shap
from mindxlib.base.explanation import FeatureImportanceExplanation
from mindxlib.base.explainer import FeatureImportanceExplainer
from mindxlib.visualization.plots import plot_waterfall

#TODO: _compute_attributions return shap Explanation classes. 
# then _get_attribution forms the attribution into FeatureImportanceExplantion 
# show() write all types of plots from shap classes

class MultipleShapExplanations:
    """Wrapper class for handling multiple SHAP explanations"""
    def __init__(self, explanations):
        self.explanations = explanations

    def show(self, type='waterfall', index=0, **kwargs):
        """
        Show visualization for a specific explanation
        
        Args:
            type (str): Type of visualization ('waterfall', etc.)
            index (int): Index of the explanation to show
            **kwargs: Additional arguments passed to the plotting function
        """
        if index >= len(self.explanations):
            raise ValueError(f"Index {index} out of range. Only {len(self.explanations)} explanations available.")
        self.explanations[index].show(type=type, **kwargs)

class ShapExplainer(FeatureImportanceExplainer):
    """SHAP Kernel Explainer for black-box models"""
    def __init__(self, model, *argv, **kwargs):
        super().__init__(model)


    def _initial_baseline(self, data, baseline, mode="match", **kwargs):

        """
        Initialize the baseline for the SHAP Kernel Explainer.
        
        Args:
            data (numpy.ndarray): The dataet used to compute the baseline.
            baseline (numpy.ndarray or None): User-provided baseline. If None, the mean of the data is used.
            
        Returns:
            numpy.ndarray: The initialized baseline.
        """

        if baseline is not None: #TODO: change the following into assert
            if mode == "match":
                if data.shape[0] == baseline.shape[0]:
                    return baseline
                elif data.shape[0] != baseline.shape[0]:
                    raise ValueError("match mode specifies that data and baseline must have the same first dimension")
            if mode == "origin":
                return baseline
        else:
            raise ValueError(f"shap needs to specify a baseline")

    def _compute_attributions(self, data, baseline, mode="match", *argv, **kwargs):
        mode_dict = {1:"match", 0:"origin"}
        if isinstance(mode, int):
            print(f"mode:{mode_dict[mode]}")
        print(f"mode:{mode}")
        if mode == "match" or mode == 1:
            # explainer_list = []
            explanation_list = []
            n_samples = data.shape[0]
            for i in range(n_samples):
                sample = data.iloc[i:(i+1),:]
                base = baseline.iloc[i:(i+1),:]
                explainer = shap.KernelExplainer(self.model, data=base, *argv, **kwargs)
                # explainer_list.append(explainer)
                explanation_list.append(explainer(sample))
            return {'explanation':explanation_list}
        elif mode == "origin" or mode == 0:
            explainer = shap.KernelExplainer(self.model, data=baseline, *argv, **kwargs)
            explanation = explainer(data)
            return {'explanation':explanation}
        else:
            raise ValueError(f"mode must be 'origin','match' or 0,1")
    def _format_explanation(self, data, attribution_dict):
        if isinstance(attribution_dict['explanation'], list):
            explanations = [shapExplanation(data.iloc[i,:], attribution_dict['explanation'][i].values, attribution_dict['explanation'][i]) 
                          for i in range(len(attribution_dict['explanation']))]
            return MultipleShapExplanations(explanations)
        else:
            return shapExplanation(data, attribution_dict['explanation'].values, attribution_dict['explanation'])

# class PermutationExplainer(FeatureImportanceExplainer):
#     def __init__(self, model, *argv, **kwargs):
#         super().__init__(model)
#         self.explainer = shap.PermutationExplainer(model, *argv, **kwargs)
    
#     def _compute_attributions(self, data, *argv, **kwargs):
#         batch_attributions = []
#         for sample in data:
#             attribution = self.explainer.shap_values(sample)
#             batch_attributions.append(attribution)
#         return np.array(batch_attributions)


class shapExplanation(FeatureImportanceExplanation):
    def __init__(self, data, feature_importance, shap_explanation):
        super().__init__(data, feature_importance)
        self.shap_explanation = shap_explanation
    def show(self, type='waterfall', **kwargs):
        """
        Show the SHAP explanation visualization
        
        Args:
            type (str): Type of visualization ('waterfall', etc.)
            **kwargs: Additional arguments passed to the plotting function
        """
        if type == 'waterfall':
            plot_waterfall(self, **kwargs)