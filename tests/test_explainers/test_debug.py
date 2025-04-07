import xgboost
import shap # for loading the dataset
from mindxlib import ShapExplainer

# Load adult dataset
X, y = shap.datasets.adult()

# Train XGBoost classifier
model = xgboost.XGBClassifier()
model.fit(X, y)

# Initialize Tree SHAP explainer
explainer = ShapExplainer(model, method="tree")

# Generate explanations
explanation = explainer.explain(X[:1000], baseline=X, mode="origin")

# Show waterfall plot for third sample's positive class
explanation.show(type='bar', class_index=1)