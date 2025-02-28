from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd

class Explanation(ABC):
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

class RuleExplanation(Explanation):
    """Class for rule-based explanations."""
    
    def __init__(self, data: pd.DataFrame, rules: List[Dict[str, Any]], default_rule: Any):
        """Initialize rule explanation.
        
        Args:
            data: Original input data
            rules: List of rules where each rule is a dictionary containing:
                - condition: List of feature conditions
                - prediction: Predicted class/value
                - coverage: Set of covered example indices
            default_rule: Default prediction when no rules match
        """
        super().__init__(data)
        self.rules = rules
        self.default_rule = default_rule

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule explanation to dictionary format."""
        return {
            "rules": self.rules,
            "default_rule": self.default_rule
        }
        
    def print_rules(self):
        """Print rules in human-readable format."""
        if len(self.rules) > 0:
            print('IF ' + ' AND '.join(self.rules[0]['condition']) + 
                  ' THEN ' + str(self.rules[0]['prediction']))
            for rule in self.rules[1:]:
                print('ELIF ' + ' AND '.join(rule['condition']) + 
                      ' THEN ' + str(rule['prediction']))
        print('ELSE ' + str(self.default_rule))

class FeatureImportanceExplanation(Explanation):
    """Class for feature importance explanations."""
    
    def __init__(self, data: pd.DataFrame, feature_importance: Dict[str, float]):
        """Initialize feature importance explanation.
        
        Args:
            feature_importance (Dict[str, float]): Dictionary mapping feature names to importance scores
        """
        super().__init__(data)
        self.feature_importance = feature_importance

    def to_dict(self) -> Dict[str, Any]:
        """Convert feature importance explanation to dictionary format.
        
        Returns:
            Dict[str, Any]: Dictionary containing the feature importance scores
        """
        return {"feature_importance": self.feature_importance}

# shape function explanation (GAM)
class GAMShapeFunctionExplanation(Explanation):
    def __init__(self, data, shape_functions, feature_ranges):
        super().__init__(data)
        self.shape_functions = shape_functions  # dict {feature: function}
        self.feature_ranges = feature_ranges    # dict {feature: range}

    def visualize(self, feature):
        """plot single feature shape function"""
        # 实现GAM可视化