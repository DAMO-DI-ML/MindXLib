import abc
import sys
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import warnings
from mindxlib.base.explanation import RuleExplanation, FeatureImportanceExplanation
"""
Base classes for explainable AI methods
"""

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})

class ExplainerBase(ABC):
    """
    Base class for all explainers.
    """
    def __init__(self, model, data=None, **kwargs):
        """
        Args:
            model: prediction model
            data: data to train explainer
        """
        self.model = model
        self.data = data
        
    @abstractmethod
    def explain(self, X, y=None, **kwargs):
        """generate explanation results
        Args:
            X: data to explain (tabular or time series)
            y: corresponding labels (optional)
        Returns:
            Explanation results
        """
        pass

    def __call__(self, X, **kwargs):
        """support function call"""
        return self.explain(X, **kwargs)

class RuleExplainerBase(ExplainerBase):
    """Base class for rule-based explainers.
    
    This class provides common functionality for rule-based explanation methods
    like rule lists, rule sets, and decision lists.
    """
    
    def __init__(self, model, data=None, **kwargs):
        """Initialize rule explainer
        
        Args:
            model: The model to explain
            data: Optional training data for learning rules
            **kwargs: Additional arguments for specific rule learners
        """
        super().__init__(model, data, **kwargs)
        self.rules = []
        
    
    @abstractmethod
    def predict(self, X):
        """Make predictions using learned rules
        
        Args:
            X: Input features (DataFrame or ndarray)
            
        Returns:
            Predictions from applying the rules as a 1D array or Series.
            If input is DataFrame, returns Series. If input is ndarray, returns 1D ndarray.
            Shape should be (n_samples,) where n_samples is X.shape[0]
        """
        pass

    def _validate_input(self, X, y=None):
        """Validate and format input data
        
        Args:
            X: Input features (DataFrame or ndarray)
            y: Optional labels (Series, DataFrame or ndarray)
            
        Returns:
            X: Validated/formatted features
            y: Validated/formatted labels (if provided)
        """
        # Validate types
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError(f'X must be DataFrame or numpy array, got {type(X).__name__}')
        if y is not None:
            if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
                raise ValueError(f'y must be Series, DataFrame or numpy array, got {type(y).__name__}')
        
        # Check if y has only one column, otherwise use the first column
        if isinstance(y, pd.DataFrame):
            if len(y.columns) > 1:
                warnings.warn('Multiple label columns found, using first column')
                y = y.iloc[:,0]
        elif isinstance(y, np.ndarray):
            if len(y.shape) > 1 and y.shape[1] > 1:
                warnings.warn('Multiple label columns found, using first column')
                y = y[:,0]
            # Check if X and y are consistent types
            x_is_pandas = isinstance(X, (pd.DataFrame, pd.Series))
            y_is_pandas = isinstance(y, (pd.DataFrame, pd.Series))
            if x_is_pandas != y_is_pandas:
                raise ValueError(f'X and y must be same type - got X: {type(X).__name__}, y: {type(y).__name__}')
        return X, y

    def explain(self, X, y=None, **kwargs):
        """Generate rule-based explanations
        
        Args:
            X: Input features to explain
            y: Optional ground truth labels
            **kwargs: Additional explanation parameters
            
        Returns:
            RuleExplanation object containing the learned rules
        """
        # Validate inputs
        X, y = self._validate_input(X, y)
        
        # Fit if not already fit
        if not self.rules:
            if y is None:
                raise ValueError("Must provide labels (y) when fitting rules")
            self.fit(X, y, **kwargs)
            
        # Format rules for explanation
        rule_texts = []
        coverage = {}
        
        # Specific formatting logic should be implemented by child classes
        return self._format_explanation(X, rule_texts, coverage)
        
    def _format_explanation(self, X, rule_texts, coverage):
        """Format rules into explanation object
        
        Args:
            X: Input data
            rule_texts: List of rule strings 
            coverage: Dict mapping rules to covered examples
            
        Returns:
            RuleExplanation object
        """
        # Convert SSRL rulelist format to explanation format
        rules = []
        for rule in self.rulelist[:-1]:  # Skip default rule
            rules.append({
                'condition': rule['condition'],
                'prediction': rule['label_name'],
                'coverage': rule['covered']
            })
        
        return RuleExplanation(
            data=X,
            rules=rules,
            default_rule=self.defaultRuleName
        )

class FeatureImportanceExplainer(ExplainerBase):
    """Base class for feature importance/attribution explainers.
    
    This class provides common functionality for attribution methods that explain
    feature importance, including both traditional methods (SHAP, LIME) and 
    time series methods (FDTemp).
    """
    
    def __init__(self, model, data=None, **kwargs):
        """Initialize feature importance explainer
        
        Args:
            model: The model to explain
            data: Optional input data for initialization
            **kwargs: Additional arguments for specific explainers
        """
        super().__init__(model, **kwargs)
        self.data = data
        
    def explain(self, data, baseline=None, **kwargs):
        """Generate feature importance explanations
        
        Args:
            data: Input data to explain (array-like)
                For tabular data: shape (n_samples, n_features)
                For time series: shape (n_samples, n_timesteps, n_features)
            baseline: Optional reference values for computing feature importance
                Default is None, in which case method-specific defaults are used
            **kwargs: Additional explanation parameters
            
        Returns:
            Explanation object containing attribution results
        """
        # Validate inputs
        data = self._validate_data(data)
        if baseline is not None:
            baseline = self._validate_baseline(baseline, data)
            
        # Generate attributions (to be implemented by child classes)
        attributions = self._compute_attributions(data, baseline, **kwargs)
        
        return self._format_explanation(data, attributions)
    
    def _validate_data(self, data):
        """Validate and format input data
        
        Args:
            data: Input data (array-like)
            
        Returns:
            Formatted data as numpy array
        """
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            try:
                data = np.array(data)
            except:
                raise ValueError(f"Data must be array-like, got {type(data)}")
                
        return data
    
    def _validate_baseline(self, baseline, data):
        """Validate baseline reference values
        
        Args:
            baseline: Baseline values
            data: Input data for shape reference
            
        Returns:
            Validated baseline array
        """
        baseline = np.array(baseline)
        if baseline.shape[-1] != data.shape[-1]:
            raise ValueError("Baseline must have same number of features as data")
        return baseline
    
    @abstractmethod
    def _compute_attributions(self, data, baseline=None, **kwargs):
        """Compute feature attributions
        
        Args:
            data: Input data
            baseline: Optional reference values
            **kwargs: Additional parameters
            
        Returns:
            Feature attributions
        """
        pass
    
    def _format_explanation(self, data, attributions):
        """Format attributions into explanation object
        
        Args:
            data: Input data
            attributions: Computed feature attributions
            
        Returns:
            Explanation object
        """
        

        return FeatureImportanceExplanation(
            data=data,
            attributions=attributions
        )

# class BlackBoxBase(ExplainerBase):
#     """
#     Base class for black-box explainers that only require
#     access to model inputs and outputs without gradients.
#     """
    
#     def predict(self, inputs):
#         """
#         Get model predictions.

#         Args:
#             inputs: Model inputs

#         Returns:
#             predictions: Model outputs
#         """
#         if hasattr(self.model, 'predict_proba'):
#             return self.model.predict_proba(inputs)
#         elif hasattr(self.model, 'predict'):
#             return self.model.predict(inputs)
#         else:
#             raise ValueError("Model must implement either predict() or predict_proba()")

# class WhiteBoxBase(ExplainerBase):
#     """
#     Base class for white-box explainers that require
#     access to model internals (gradients, layers, etc).
#     """
    
#     def gradient(self, inputs, target_labels=None):
#         """
#         Get gradients of model output with respect to inputs.

#         Args:
#             inputs: Model inputs
#             target_labels: Target classes for gradients

#         Returns:
#             gradients: Gradients of model outputs
#         """
#         if hasattr(self.model, 'gradient'):
#             return self.model.gradient(inputs, target_labels)
#         else:
#             raise ValueError("Model must implement gradient() method for white-box explanations")

#     def get_layer(self, layer_name):
#         """
#         Get intermediate layer by name.

#         Args:
#             layer_name: Name of the layer to retrieve

#         Returns:
#             layer: The requested layer
#         """
#         if hasattr(self.model, 'get_layer'):
#             return self.model.get_layer(layer_name)
#         else:
#             raise ValueError("Model must implement get_layer() method to access internal layers")

