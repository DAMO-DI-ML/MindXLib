import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mindxlib.explainers.rules.ruleset import RuleSet


def test_rulelset_with_numpy():
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
    explainer = RuleSet(
        max_num_rules=3, 
        time_limit=60, 
        verbose=True,
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



def test_ruleset_from_csv():
    data = pd.read_csv('dataset/tic_tac_toe.csv', header=None)
    y = data.iloc[:,-1]
    print(y.value_counts())
    # y = y.map({'negative': 0, 'positive': 1})
    X = data.iloc[:,:-1]
    explainer = RuleSet(
        max_num_rules=15, 
        time_limit=120, 
        verbose=True,
        feature_prefix='feature_',
        binarize_features=True,
        categorical_features=[],
        num_thresh=25,
        negation=True
    )
    explainer.fit(X, y)
    
    explainer.show()
    predictions = explainer.predict(X)
    acc = np.sum(1.0*(predictions.values==y.values))/y.shape[0]
    print(f'The training acc is {acc:.2f}')
    '''
    IF 2!=o AND 4!=o AND 6!=o, THEN 1
    ELIF 0!=o AND 4!=o AND 8!=o, THEN 1
    ELIF 1!=o AND 2!=o AND 3!=o AND 8!=o, THEN 1
    ELIF 1!=o AND 2==o AND 4!=o AND 5!=o, THEN 1
    ELIF 0!=o AND 2!=o AND 3!=o AND 7!=o, THEN 1
    ELIF 4==o AND 5!=o AND 6!=o AND 7!=o AND 8!=o, THEN 1
    ELIF 0!=o AND 5!=o AND 6!=o AND 7!=o, THEN 1
    ELSE 0
    The training acc is 0.91
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
    explainer = RuleSet(
        max_num_rules=15, 
        time_limit=120, 
        verbose=True,
        feature_prefix='feature_',
        binarize_features=True,
        categorical_features=[],
        num_thresh=25,
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

# test_rulelset_with_numpy()
test_ruleset_from_csv()