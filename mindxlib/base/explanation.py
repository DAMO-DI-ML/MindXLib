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

    def to_df(self):
        """convert to standard DataFrame output"""
        raise NotImplementedError

    def visualize(self):
        """general visualization interface"""
        pass


# feature importance/Shapley value class
class FeatureImportanceExplanation(BaseExplanation):
    def __init__(self, data, baseline, scores, feature_names=None):
        super().__init__(data)
        self.scores = scores  # importance scores [n_samples, n_features]
        self.baseline = baseline  
        self.feature_names = feature_names

    def to_df(self):
        return pd.DataFrame(self.values, columns=self.feature_names)

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