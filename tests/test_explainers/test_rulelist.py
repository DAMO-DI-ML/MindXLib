import numpy as np
from mindxlib import SSRL
import pandas as pd
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