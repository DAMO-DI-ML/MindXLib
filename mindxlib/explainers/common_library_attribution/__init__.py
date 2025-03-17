"""
Common attribution methods from popular interpretability libraries
"""

from .shap import (
    KernelExplainer, PermutationExplainer
)
from .lime import (
    LimeTabularExplainer
)
from .ig import IntegratedGradients

__all__ = [
    # SHAP explainers
    'KernelExplainer', 'PermutationExplainer',
    
    # LIME explainers  
    'LimeTabularExplainer',
    
    # Integrated Gradients
    'IntegratedGradients'
] 