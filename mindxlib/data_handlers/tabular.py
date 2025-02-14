from ..utils.features import FeatureBinarizer
from ..utils.datautil import DatasetLoader

class TabularDataHandler:
    def __init__(self, binarizer_params=None):
        self.binarizer = FeatureBinarizer(**(binarizer_params or {}))
        
    def preprocess(self, X, y=None):
        # ... preprocessing logic ...
        return self.binarizer.fit_transform(X) 