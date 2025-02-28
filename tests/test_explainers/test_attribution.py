import numpy as np
import pandas as pd
from mindxlib.explainers.common_library_attribution.ig import IntegratedGradients
from mindxlib.utils.features import FeatureBinarizer

class test_model():
    def __init__(self):
        super().__init__()

    def forward(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]

        output1 = x1 * x2
        output2 = x2 * x3
        output3 = x3 * x4
        output4 = x4 * x5
        output = np.concatenate((output1, output2, output3, output4), axis=1)

        return output

    def gradient(self, X, target_label=None):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]

        # 创建与 x1 形状相同的全零数组
        zo = np.zeros_like(x1)

        # 使用 np.stack 创建 grad1, grad2, grad3, grad4
        grad1 = np.stack([x2, x1, zo, zo, zo], axis=1)
        grad2 = np.stack([zo, x3, x2, zo, zo], axis=1)
        grad3 = np.stack([zo, zo, x4, x3, zo], axis=1)
        grad4 = np.stack([zo, zo, zo, x5, x4], axis=1)

        gradient = np.stack((grad1, grad2, grad3, grad4), axis=2)
        
        return gradient[:, :, target_label]


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
    attribution = explainer._compute_attributions(X, Y)
    print(attribution.feature_importance)

if __name__ == "__main__":
    test_ig_with_numpy()