import pytest
import numpy as np
from mindxlib.explainers.rules import RuleSet

def test_ruleset_fit_predict():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    model = RuleSet()
    model.fit(X, y)
    pred = model.predict(X)
    
    assert len(pred) == len(y) 