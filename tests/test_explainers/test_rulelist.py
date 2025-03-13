import numpy as np
from mindxlib import SSRL
import pandas as pd
from mindxlib.explainers.rules.rulelist import SSRL
from mindxlib.utils.features import FeatureBinarizer

def test_rulelist_with_numpy():
    # Create numpy array with meaningful feature names
    X = np.array([
        [25, 30000, 12],
        [35, 45000, 14],
        [45, 60000, 16],
        [55, 75000, 18],
        [22, 75000, 18]
    ])
    y = np.array([0, 1, 1, 0, 0])
    
    # Initialize feature binarizer
    # binarizer = FeatureBinarizer(numThresh=3, negations=True, threshStr=True)
    
    # Binarize X while keeping numpy format
    # X_binarized = binarizer.fit_transform(X).values #not working with numpy arrays, as data[c] is not supported
    
    # Initialize and fit explainer with same parameters as pandas test
    explainer = SSRL(
        lambda_1=1.0,
        distorted_step=10,
        cc=10,
        feature_prefix='feature_',
        binarize_features=True,
        categorical_features=[],
        num_thresh=3,
        negation=True
    )
    
    # Fit the model
    explainer.fit(X, y)
    explainer.show()
    X_test = np.array([[25, 30000, 12]])
    a = explainer.predict(X_test)
    print(a)


def test_rulelist_invalid_input():
    explainer = SSRL(0.5)
    
    X_invalid = np.array([[1, 2], [3, 4]])  # 2D array with wrong shape
    y_invalid = np.array([1, 2, 3])  # Length mismatch with X_invalid
    
    try:
        explainer.fit(X_invalid, y_invalid)
        assert False, "Should raise an error for invalid input shapes"
    except ValueError:
        pass

    explainer = SSRL(0.5)
    X = np.array([
        [25, 30000, 12],
        [35, 45000, 14],
        [45, 60000, 16],
        [55, 75000, 18],
        [22, 75000, 18]
    ])
    y = np.array([0, 1, 1, 0, 0])
    try:
        explainer.fit(X,y, default_label=2)
        assert False, "Should raise an error for invalid default_label"
    except ValueError:
        pass

def test_rulelist_with_pandas():
    # Create a more realistic sample DataFrame
    data = pd.DataFrame({
        'age': [25, 35, 45, 55, 22],
        'income': [30000, 45000, 60000, 75000, 75000],
        'education_years': [12, 14, 16, 18, 18]
    })
    y = pd.Series([0, 1, 1, 0, 1], name='label')
    
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
    explainer.fit(data, y, default_label=0)
    explainer.show()
    # Test prediction
    test_data = pd.DataFrame({
        'age': [30],
        'income': [50000],
        'education_years': [15]
    })

    explainer.predict(test_data)


def test_rulelist_from_csv():
    data = pd.read_csv('dataset/tic_tac_toe.csv', header=None)
    y = data.iloc[:,-1]
    X = data.iloc[:,:-1]
    explainer = SSRL(cc=10, lambda_1=1, distorted_step=10, categorical_features=X.columns.tolist())
    explainer.fit(X, y)
    explainer.show()
    predictions = explainer.predict(X)
    acc = np.sum(1.0*(predictions.values==y.values))/y.shape[0]
    print(f'The training acc is {acc:.2f}')
    '''
    IF 1==o AND 4==o AND 7==o, THEN negative
    ELIF 3==o AND 4==o AND 5==o, THEN negative
    ELIF 0==o AND 1==o AND 2==o, THEN negative
    ELIF 6==o AND 7==o AND 8==o, THEN negative
    ELIF 0==o AND 3==o AND 6==o, THEN negative
    ELIF 2==o AND 5==o AND 8==o, THEN negative
    ELIF 0!=x AND 4!=x AND 8!=x, THEN negative
    ELIF 2!=x AND 4!=x AND 6!=x, THEN negative
    ELSE positive
    The training acc is 0.98
    '''

def test_rulelist_multiclass_with_pandas():
    # Create a sample DataFrame with a multi-class target
    data = pd.DataFrame({
        'age': [25, 35, 45, 55, 22, 28, 32, 42, 50, 38, 48, 52],
        'income': [30000, 45000, 60000, 75000, 75000, 50000, 65000, 40000, 80000, 70000, 55000, 62000],
        'education_years': [12, 14, 16, 18, 18, 15, 16, 14, 19, 17, 15, 16]
    })
    # Three classes with more balanced distribution: 0 (low risk), 1 (medium risk), 2 (high risk)
    y = pd.Series([0, 1, 2, 0, 1, 2, 1, 0, 2, 2, 1, 2], name='risk_level')
    
    # Initialize and fit explainer
    explainer = SSRL(
        lambda_1=1.0,
        distorted_step=10,
        cc=10,
        use_multi_pool=False,
        binarize_features=True,
        categorical_features=[],
        num_thresh=3,
        negation=True
    )
    
    # Fit the model
    explainer.fit(data, y)
    
    # Show the rules
    explainer.show()
    
    # Calculate training accuracy
    train_predictions = explainer.predict(data)
    train_acc = np.mean(train_predictions == y)
    print(f'Training accuracy: {train_acc:.2f}')
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f'Class distribution: {dict(zip(unique, counts))}')
    
    # Test predictions
    test_data = pd.DataFrame({
        'age': [30, 50],
        'income': [50000, 70000],
        'education_years': [15, 17]
    })
    
    predictions = explainer.predict(test_data)
    
    # Verify predictions are within valid classes
    assert all(pred in [0, 1, 2] for pred in predictions), "Predictions should be in [0, 1, 2]"
    
    # Test with different default label
    explainer.fit(data, y, default_label=1)  # Set medium risk as default
    explainer.show()
    
    # Calculate accuracy with new default label
    train_predictions_default = explainer.predict(data)
    train_acc_default = np.mean(train_predictions_default == y)
    print(f'Training accuracy with default label=1: {train_acc_default:.2f}')

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

    print(f"accuracy: {accuracy}")
    
    # Test rulelist printing
    explainer.print_rulelist()

if __name__ == '__main__':
    test_rulelist_with_pandas()
