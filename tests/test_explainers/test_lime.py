import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mindxlib.explainers.common_library_attribution.lime import LimeTabularExplainer
import lime

def test_with_lime():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names  # ['sepal length (cm)', 'sepal width (cm)', ...]
    class_names = iris.target_names     # ['setosa', 'versicolor', 'virginica']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    explainer = LimeTabularExplainer(model.predict_proba)

    explanation = explainer.explain(X_test[:2], baseline=X_train, feature_names=feature_names, class_names=class_names, mode='classification')

    print(explanation.feature_importance["feature_importance"][0].as_list())
    print(explanation.feature_importance["feature_importance"][1].as_list())


if __name__ == "__main__":
    test_with_lime()