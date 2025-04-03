from __future__ import print_function
import numpy as np
import shap
from mindxlib.base.explanation import FeatureImportanceExplanation
from mindxlib.base.explainer import FeatureImportanceExplainer
from mindxlib.visualization.plots import plot_waterfall, plot_bar, plot_scatter
import pandas as pd
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
        if type in ['waterfall', 'bar']:
            self.explanations[index].show(type=type, **kwargs)
        else:
            self.explanations.show(type=type, **kwargs)
        
class ShapExplainer(FeatureImportanceExplainer):
    """SHAP Explainer that supports multiple SHAP methods (Kernel, Tree, etc.)"""
    def __init__(self, model, method="auto", link="identity", **kwargs):
        """
        Initialize SHAP explainer
        
        Args:
            model: The model to explain
            method (str): SHAP explanation method ('kernel', 'tree', 'linear' or 'auto')
            link (str): The link function ('identity' or 'logit') to map between model output and SHAP values
            **kwargs: Additional arguments passed to the SHAP explainer
        """
        super().__init__(model)
        self.method = method.lower()
        self.kwargs = kwargs
        
        # Validate method
        valid_methods = ['kernel', 'tree', 'linear', 'auto']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
            
        # Validate link function
        valid_links = ['identity', 'logit']
        if link not in valid_links:
            raise ValueError(f"link must be one of {valid_links}")
        self.link = link

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
        
        # Choose prediction function based on link parameter
        if self.link == "logit":
            if hasattr(self.model, 'predict_proba'):
                predict_function = lambda x: self.model.predict_proba(x)
            else:
                raise AttributeError("Model must have predict_proba method when link='logit'")
        elif self.method!='tree':
            predict_function = lambda x: self.model.predict(x)
        else:
            predict_function = self.model

        # Get appropriate SHAP link function
        link_function = shap.links.logit if self.link == "logit" else shap.links.identity

        # Create appropriate SHAP explainer factory based on method
        if self.method == 'kernel':
            explainer_factory = shap.KernelExplainer
        elif self.method == 'tree':
            explainer_factory = shap.TreeExplainer
        elif self.method == 'linear':
            explainer_factory = shap.LinearExplainer
        else:  # auto
            explainer_factory = shap.Explainer

        def create_explainer(baseline):
            return explainer_factory(
                predict_function,
                baseline,
                *argv,
                **kwargs
            )

        def get_samples(data, baseline, idx=None):
            if isinstance(data, pd.DataFrame):
                return (data.iloc[idx:(idx+1),:] if idx is not None else data,
                       baseline.iloc[idx:(idx+1),:] if idx is not None else baseline)
            else:
                return (data[idx:(idx+1),:] if idx is not None else data,
                       baseline[idx:(idx+1),:] if idx is not None else baseline)

        # Generate explanations based on mode
        if mode in ["match", 1]:
            explanation_list = []
            for i in range(data.shape[0]):
                sample, base = get_samples(data, baseline, i)
                explainer = create_explainer(base)
                explanation_list.append(explainer(sample))
            return {'explanation': explanation_list}
        elif mode in ["origin", 0]:
            sample, base = get_samples(data, baseline)
            explainer = create_explainer(base)
            return {'explanation': explainer(sample)}
        else:
            raise ValueError(f"mode must be 'origin','match' or 0,1")

    def _format_explanation(self, data, attribution_dict):
        if isinstance(attribution_dict['explanation'], list):
            if isinstance(data, pd.DataFrame):      
                explanations = [shapExplanation(data.iloc[i,:], attribution_dict['explanation'][i].values, attribution_dict['explanation'][i]) 
                              for i in range(len(attribution_dict['explanation']))]
            else:
                explanations = [shapExplanation(data[i,:], attribution_dict['explanation'][i].values, attribution_dict['explanation'][i]) 
                              for i in range(len(attribution_dict['explanation']))]
            return MultipleShapExplanations(explanations)
        else:
            return shapExplanation(data, attribution_dict['explanation'].values, attribution_dict['explanation'])

    def explain(self, data, baseline=None, mode="match", **kwargs):
        """
        Generate SHAP explanations for the given samples.
        
        Args:
            data: Samples to explain
            baseline: Background data for computing feature importance
            mode: How to use baseline data ("match" or "origin")
            **kwargs: Additional arguments for KernelExplainer
            
        Returns:
            ShapExplanation or MultipleShapExplanations object
        """
        baseline = self._initial_baseline(data, baseline, mode)
        attribution_dict = self._compute_attributions(data, baseline, mode, **kwargs)
        return self._format_explanation(data, attribution_dict)



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
        elif type == 'bar':
            plot_bar(self, **kwargs)
        elif type == 'scatter':
            plot_scatter(self, **kwargs)
        else:
            raise ValueError(f"type must be 'waterfall' or 'bar'")
