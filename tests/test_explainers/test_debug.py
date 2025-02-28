import numpy as np
from mindxlib.explainers.rules.rulelist import SSRL
import pandas as pd
from mindxlib.utils.features import FeatureBinarizer
# Create a more realistic sample DataFrame
data = pd.DataFrame({
    'age': [25, 35, 45, 55, 22],
    'income': [30000, 45000, 60000, 75000, 75000],
    'education_years': [12, 14, 16, 18, 18]
})
y = pd.Series([0, 1, 1, 0, 0], name='label')

# Initialize feature binarizer
# binarizer = FeatureBinarizer(numThresh=3, negations=True, threshStr=True)
# X_binarized = binarizer.fit_transform(data)

# # Clean up column names
# X_binarized.columns = [' '.join(col).strip() for col in X_binarized.columns.values]

# Initialize and fit explainer with specific parameters
explainer = SSRL(
    lambda_1=1.0,
    distorted_step=10,
    cc=10,
    use_multi_pool=False
)

# Fit the model with default rule name
explainer.fit(data, y, defaultRuleName=0)

# Test prediction
test_data = pd.DataFrame({
    'age': [30],
    'income': [50000],
    'education_years': [15]
})
# test_binarized = binarizer.transform(test_data)
explainer.rules.show()
predictions = explainer.predict(test_data)
print(predictions)

# Assertions
assert predictions is not None, "Predictions should not be None"
assert hasattr(explainer, 'defaultRuleName'), "Explainer should have default rule after fitting"
assert len(str(explainer.defaultRuleName)) > 0, "Default rule name should be set"

