import numpy as np
import pandas as pd
from mindxlib.explainers.common_library_attribution.ig import IntegratedGradients

class test_model():
    def __init__(self):
        super().__init__()

    def forward(self, X):
        x1 = X[:, 0][:, np.newaxis]
        x2 = X[:, 1][:, np.newaxis]
        x3 = X[:, 2][:, np.newaxis]
        x4 = X[:, 3][:, np.newaxis]
        x5 = X[:, 4][:, np.newaxis]

        output1 = x1 * x2
        output2 = x2 * x3
        output3 = x3 * x4
        output4 = x4 * x5
        output = np.concatenate((output1, output2, output3, output4), axis=1)
        output = self.softmax(output, axis=1)

        return output

    def gradient(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]

        zo = np.zeros_like(x1)

        grad1 = np.stack([x2, x1, zo, zo, zo], axis=1)
        grad2 = np.stack([zo, x3, x2, zo, zo], axis=1)
        grad3 = np.stack([zo, zo, x4, x3, zo], axis=1)
        grad4 = np.stack([zo, zo, zo, x5, x4], axis=1)

        # gradient: shape (n_steps, n_samples, output_dim)
        gradient = np.stack((grad1, grad2, grad3, grad4), axis=2)
        
        return gradient

    def softmax(self, x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def __call__(self, X):
        return self.forward(X)

def test_with_ig():
    X = np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 2, 1]
    ])
    Y = np.array([0, 1, 2, 3])
    baseline = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0]])

    model = test_model()
    explainer = IntegratedGradients(model)

    # attribution: shape: (n_samples, n_features, n_class)
    explanation = explainer.explain(X, baseline=baseline)
    print(f"data: ")
    print(X)
    print(f"attribution")
    print(explanation.feature_importance["feature_importance"])

def test_with_lime():
    X = np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 2, 1]
    ])
    Y = np.array([0, 1, 2, 3])
    baseline = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0]])

    model = test_model()
    explainer = LimeTabularExplainer(model, training_data=X)
    # attribution: shape: (n_samples, n_features, n_class)
    explanation = explainer.explain(X, baseline=baseline)


if __name__ == "__main__":

    test_with_ig()