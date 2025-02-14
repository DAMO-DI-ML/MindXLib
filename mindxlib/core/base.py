import abc
import sys
from abc import ABC, abstractmethod
import numpy as np

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
    def __init__(self, model):
        """
        Args:
            model: The model to explain
        """
        self.model = model

    @abstractmethod
    def explain_instance(self, input_tensor, target_label=None):
        """
        Explains a single instance.

        Args:
            input_tensor: Input to explain
            target_label: Target class to explain (for classification)

        Returns:
            explanation: Explanation for the instance
        """
        pass

    @abstractmethod
    def explain_batch(self, inputs, target_labels=None):
        """
        Explains a batch of instances.

        Args:
            inputs: Batch of inputs to explain
            target_labels: Target classes to explain

        Returns:
            explanations: Explanations for the batch
        """
        pass

class BlackBoxBase(ExplainerBase):
    """
    Base class for black-box explainers that only require
    access to model inputs and outputs without gradients.
    """
    
    def predict(self, inputs):
        """
        Get model predictions.

        Args:
            inputs: Model inputs

        Returns:
            predictions: Model outputs
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(inputs)
        elif hasattr(self.model, 'predict'):
            return self.model.predict(inputs)
        else:
            raise ValueError("Model must implement either predict() or predict_proba()")

class WhiteBoxBase(ExplainerBase):
    """
    Base class for white-box explainers that require
    access to model internals (gradients, layers, etc).
    """
    
    def gradient(self, inputs, target_labels=None):
        """
        Get gradients of model output with respect to inputs.

        Args:
            inputs: Model inputs
            target_labels: Target classes for gradients

        Returns:
            gradients: Gradients of model outputs
        """
        if hasattr(self.model, 'gradient'):
            return self.model.gradient(inputs, target_labels)
        else:
            raise ValueError("Model must implement gradient() method for white-box explanations")

    def get_layer(self, layer_name):
        """
        Get intermediate layer by name.

        Args:
            layer_name: Name of the layer to retrieve

        Returns:
            layer: The requested layer
        """
        if hasattr(self.model, 'get_layer'):
            return self.model.get_layer(layer_name)
        else:
            raise ValueError("Model must implement get_layer() method to access internal layers")

