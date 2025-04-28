from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd

class Explanation(ABC):
    def __init__(self, data):
        """base class for all explanation results
        Args:
            data: original input data (unified to DataFrame, time series expanded to multiple columns)
        """
        # self.data = data

    @abstractmethod
    def validate(self):
        """validate the legitimacy of the explanation results"""
        pass

    @abstractmethod
    def show(self):
        """general visualization interface"""
        pass

class RuleExplanation(Explanation):
    """Class for rule-based explanations."""
    
    def __init__(self, rules: List[Dict[str, Any]], default_rule: Any):
        """Initialize rule explanation.
        
        Args:
            rules: List of rules where each rule is a dictionary containing:
                - condition: List of feature conditions
                - label_name: Predicted class/value
                - covered: Set of covered example indices
                - length: Length of the rule
            default_rule: Default prediction when no rules match
        """
        self.rules = rules  # List of rule dictionaries
        self.default_rule = default_rule

    def validate(self):
        pass
        
    def show(self, save: bool = False, path: Optional[str] = None):
        """Print rules in human-readable format.
        
        Args:
            save: Whether to save the output to a file
            path: Path to save the output file. Required if save=True
        """
        output_lines = []
        N = len(self.rules)
        if N > 0:
            line = 'IF '+' AND '.join(sorted(self.rules[0]['condition']))+', THEN '+str(self.rules[0]['label_name'])
            output_lines.append(line)
            print(line)
            for ii in range(1,N):
                line = 'ELIF '+' AND '.join(sorted(self.rules[ii]['condition']))+', THEN '+str(self.rules[ii]['label_name'])
                output_lines.append(line)
                print(line)
            line = 'ELSE '+str(self.default_rule)
            output_lines.append(line)
            print(line)
        else:
            line = 'No rules found, default rule: '+str(self.default_rule)
            output_lines.append(line)
            print(line)
            
        if save:
            if path is None:
                raise ValueError("path must be provided when save=True")
            with open(path, 'w') as f:
                f.write('\n'.join(output_lines))
            print(f"Rules saved to {path}")


class RuleSetExplanation(RuleExplanation):
    """Class for rule-based explanations where each rule is independent (no ELIF/ELSE).
    Used by RuleSet, RuleSetImb, and Diver explainers."""
    
    def __init__(self, rules: List[str], default_rule: Any, label_map: dict = None):
        """Initialize rule set explanation.
        
        Args:
            rules: List of rules as strings
            default_rule: Default prediction when no rules match
            label_map: Optional mapping from binary (0/1) to original labels
        """
        super().__init__(rules, default_rule)
        self.label_map = label_map

    def show(self, save=False, path=None):
        """Print rules in independent IF-THEN format.
        
        Args:
            save: Whether to save the output to a file
            path: Path to save the output file. Required if save=True
        """
        output_lines = []
        N = len(self.rules)
        if N > 0:
            for rule in self.rules:
                # Handle both string rules and dictionary rules
                if isinstance(rule, str):
                    # If we have a label map, use original labels
                    if self.label_map:
                        then_label = self.label_map[1]
                        else_label = self.label_map[0]
                    else:
                        then_label = 1
                        else_label = 0
                    line = f"IF {rule}, THEN {then_label}, ELSE {else_label}"
                    output_lines.append(line)
                    print(line)
                else:
                    line = f"IF {' AND '.join(sorted(rule['condition']))}, THEN {rule['label_name']}"
                    output_lines.append(line)
                    print(line)
        else:
            line = f"No rules found, default rule: {self.default_rule}"
            output_lines.append(line)
            print(line)
            
        if save:
            if path is None:
                raise ValueError("path must be provided when save=True")
            with open(path, 'w') as f:
                f.write('\n'.join(output_lines))
            print(f"Rules saved to {path}")

class FeatureImportanceExplanation(Explanation):
    """Class for feature importance explanations."""
    
    def __init__(self, data, feature_importance):
        super().__init__(data)
        
        self.data = data

        # if not isinstance(feature_importance, dict):
        #     feature_importance = {"feature_importance": feature_importance}
        self.feature_importance = feature_importance


    
    def validate(self):
        pass
        
    def show(self):
        pass

# shape function explanation (GAM)
class GAMShapeFunctionExplanation(Explanation):
    def __init__(self, data, shape_functions, feature_ranges):
        super().__init__(data)
        self.shape_functions = shape_functions  # dict {feature: function}
        self.feature_ranges = feature_ranges    # dict {feature: range}

    def visualize(self, feature):
        """plot single feature shape function"""
        # 实现GAM可视化