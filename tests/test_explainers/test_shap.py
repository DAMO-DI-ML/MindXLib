import sys
import os

# sys.path.append('/jupyter/MindXLib')

# print(f"Current sys.path: {sys.path}")

# import numpy as np
# import pandas as pd
from mindxlib import ShapExplainer


def test_numpy_regression():
    import numpy as np
    from mindxlib import ShapExplainer
    import xgboost

    # Prepare regression data with more samples
    X = np.array([
        [1, 2], [2, 4], [3, 1], [4, 3], [5, 3], [6, 2],[2, 1], [3, 4], [4, 2], 
        [5, 1], [6, 3], [7, 2], [3, 3], [4, 1], [5, 4], [6, 1], [7, 3], [8, 2]
    ])
    y = np.array([1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8])

    # Train XGBoost regressor
    model = xgboost.XGBRegressor()
    model.fit(X, y)

    # Initialize explainer
    explainer = ShapExplainer(model, link="identity")

    # Generate explanations with matched baselines
    # Using more samples for both explanation and baseline
    explanation = explainer.explain(X[9:18], baseline=X[:9], mode="match")

    # Show bar plot for specific sample
    explanation.show(type='waterfall', index=0)

def test_iris():
    import sklearn
    import shap
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    svm = sklearn.svm.SVC(kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = ShapExplainer(svm, link="logit", method='kernel')
    explanation = explainer.explain(X_test[:5], baseline=X_train[:5], mode="origin", nsamples=100)

    explanation.show(type = 'scatter', class_index=1, feature='petal length (cm)')
# # test_with_shap()

def test_adult_scatter():
    import xgboost
    import shap
    X, y = shap.datasets.adult()
    model = xgboost.XGBClassifier().fit(X, y)
    explainer = ShapExplainer(model, method = 'tree')
    explanation = explainer.explain(X[:1000], baseline=X, mode="origin")
    explanation.show('scatter', feature='Age', save=True, path='./scatter_test.png')

def test_official_tree_shap():
    import xgboost

    import shap

    # train XGBoost model
    X, y = shap.datasets.adult()
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    explainer = ShapExplainer(model, method="tree", link="logit")
    explanation = explainer.explain(X[:3], baseline=X[3:6], mode="match")
    explanation.show(type='waterfall', index=2, class_index=1)  # Show first class

def test_shap_scatter():
    import xgboost
    import shap
    X, y = shap.datasets.adult()
    model = xgboost.XGBClassifier().fit(X, y)
    explainer = shap.Explainer(model, X)
    explanation = explainer(X[:1000])
    shap.plots.scatter(explanation[:, "Age"], color=explanation)


test_adult_scatter()
# test_iris()