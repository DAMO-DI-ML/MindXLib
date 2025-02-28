import numpy as np
import pandas as pd
from mindxlib.explainers.common_library_attribution.ig import IntegratedGradients
from mindxlib.utils.features import FeatureBinarizer

class _model():
    def __init__(self):
        super().__init__()

    def forward(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]

        output1 = x1 + x2
        output2 = x2 + x3
        output3 = x3 + x4
        output4 = x4 + x5
        output = np.concatenate((output1, output2, output3, output4), axis=1)

        return output

    def gradient(self, X, target_label=None):

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        grad_x1 = 2 * np.ones_like(x1)
        grad_x2 = x3
        grad_x3 = x2

        gradient = np.stack((grad_x1, grad_x2, grad_x3), axis=1)
        return gradient


    def __call__(self, X):
        return self.forward(X)

def test_ig_with_numpy():
    X = np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 2, 1]
    ])
    model = test_model()
    Y = np.array([0, 1, 2, 3])
    explainer = IntegratedGradients(model)
    attribution = explainer._compute_attributions(X)
    print(attribution)
    input()

if __name__ == "__main__":
    test_ig_with_numpy()