import numpy as np
from mindxlib.explainers.rules.rulelist import SSRL
import pandas as pd
from mindxlib.utils.features import FeatureBinarizer

# def test_rulelist_with_numpy():
#     # Create numpy array with meaningful feature names
#     X = np.array([
#         [25, 30000, 12],
#         [35, 45000, 14],
#         [45, 60000, 16],
#         [55, 75000, 18],
#         [22, 75000, 18]
#     ])
#     y = np.array([0, 1, 1, 0, 0])
    
#     # Initialize feature binarizer
#     # binarizer = FeatureBinarizer(numThresh=3, negations=True, threshStr=True)
    
#     # Binarize X while keeping numpy format
#     # X_binarized = binarizer.fit_transform(X).values #not working with numpy arrays, as data[c] is not supported
    
#     # Initialize and fit explainer with same parameters as pandas test
#     explainer = SSRL(
#         lambda_1=1.0,
#         distorted_step=10,
#         cc=10,
#         use_multi_pool=False,
#         binarize_features=True,
#         categorical_features=[],
#         num_thresh=3,
#         negation=True,
#         threshStr=True
#     )
    
#     # Fit the model
#     explanation = explainer.fit(X_binarized, y)
#     explainer.print_rulelist()

#     # Assertions
#     # assert predictions is not None, "Predictions should not be None"
#     assert hasattr(explainer, 'defaultRuleName'), "Explainer should have rules_ attribute after fitting"
#     assert len(str(explainer.defaultRuleName)) > 0, "Default rule name should be set"

# def test_rulelist_invalid_input():
#     explainer = SSRL(0.5)
    

#     X_invalid = np.array([[1, 2], [3, 4]])  # 2D array with wrong shape
#     y_invalid = np.array([1, 2, 3])  # Length mismatch with X_invalid
    
#     try:
#         explainer.fit(X_invalid, y_invalid)
#         assert False, "Should raise an error for invalid input shapes"
#     except ValueError:
#         pass

def test_rulelist_with_pandas():
    # Create a more realistic sample DataFrame
    data = pd.DataFrame({
        'age': [25, 35, 45, 55, 22],
        'income': [30000, 45000, 60000, 75000, 75000],
        'education_years': [12, 14, 16, 18, 18]
    })
    y = pd.Series([0, 1, 1, 0, 0], name='label')
    
    # # Initialize feature binarizer
    # binarizer = FeatureBinarizer(numThresh=3, negations=True, threshStr=True)
    # X_binarized = binarizer.fit_transform(data)
    
    # # Clean up column names
    # X_binarized.columns = [' '.join(col).strip() for col in X_binarized.columns.values]
    
    # Initialize and fit explainer with specific parameters
    explainer = SSRL(
        lambda_1=1.0,
        distorted_step=10,
        cc=10,
        use_multi_pool= False,
        binarize_features=True,
        categorical_features=[],
        num_thresh=3,
        negation=True
    )
    
    # Fit the model with default rule name
    explainer.fit(data, y, defaultRuleName=0)
    # Test prediction
    test_data = pd.DataFrame({
        'age': [30],
        'income': [50000],
        'education_years': [15]
    })

    explainer.predict(test_data)
    # test_binarized = binarizer.transform(test_data)
    # predictions = explainer.predict(test_binarized)
    
    # Assertions
    # assert predictions is not None, "Predictions should not be None"
    assert hasattr(explainer, 'defaultRuleName'), "Explainer should have default rule after fitting"
    assert len(str(explainer.defaultRuleName)) > 0, "Default rule name should be set"
    
    # Test accuracy calculation
    train_predictions = explainer.predict(X_binarized)
    accuracy = np.sum(train_predictions == y) / len(y)
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
