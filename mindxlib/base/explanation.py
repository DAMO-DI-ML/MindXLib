from abc import ABC
import pandas as pd

class BaseExplanation(ABC):
    def __init__(self, data: pd.DataFrame):
        """base class for all explanation results
        Args:
            data: original input data (unified to DataFrame, time series expanded to multiple columns)
        """
        self.data = data

    def validate(self):
        """validate the legitimacy of the explanation results"""
        pass



    def visualize(self):
        """general visualization interface"""
        pass


# feature importance/Shapley value class
class FeatureImportanceExplanation(BaseExplanation):
    """Base class for storing feature importance/attribution explanations"""
    
    def __init__(self, data, attributions, interaction_effects=None):
        """Initialize feature importance explanation
        
        Args:
            data: Input data that was explained
            attributions: Feature attributions/main effects with same shape as input data
            interaction_effects: Optional interaction effects between features
                Shape: (n_samples, n_features, n_features)
        """
        super().__init__(data)
        self.attributions = attributions
        self._interaction_effects = interaction_effects
        

    # 
        
    @property 
    def main_effect(self):
        """Get main/individual feature effects"""
        return self.attributions
        
    @property
    def interaction_effect(self):
        """Get interaction effects between features"""
        return self._interaction_effects

# shape function explanation (GAM)
class GAMShapeFunctionExplanation(BaseExplanation):
    def __init__(self, data, shape_functions, feature_ranges):
        super().__init__(data)
        self.shape_functions = shape_functions  # dict {feature: function}
        self.feature_ranges = feature_ranges    # dict {feature: range}

    def visualize(self, feature):
        """plot single feature shape function"""
        # 实现GAM可视化

# rule explanation
class RuleExplanation(BaseExplanation):
    def __init__(self, data, rules, coverage):
        super().__init__(data)
        self.rules = rules       # e.g. ["IF age>10 THEN risk=0.2", ...]

    def to_text(self):
        """return natural language description"""
        return "\n".join(self.rules)