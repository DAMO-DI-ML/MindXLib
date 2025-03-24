import sys
import os

sys.path.append('/jupyter/MindXLib')

print(f"Current sys.path: {sys.path}")

import numpy as np
import pandas as pd
from mindxlib.explainers.common_library_attribution.shap import ShapExplainer, PermutationExplainer
import sklearn
import shap
from sklearn.model_selection import train_test_split


def test_with_shap():
    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    svm = sklearn.svm.SVC(kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = ShapExplainer(svm.predict_proba, link="logit")
    explanation = explainer.explain(X_test[:5], baseline=X_train[:5], mode="match", nsamples=100)

test_with_shap()

